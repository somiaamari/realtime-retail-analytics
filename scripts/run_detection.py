"""Run YOLO people/object detection on a video source."""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

from vision_pipeline.config import AppConfig, load_config
from vision_pipeline.detection.yolo_detector import YoloDetector
from vision_pipeline.logging_utils import setup_logging
from vision_pipeline.streaming.sources import VideoSourceError, create_source
from vision_pipeline.streaming.writer import VideoWriter, VideoWriterError
from vision_pipeline.utils.profiling import FPSMeter, LatencyStats, StageTimer
from vision_pipeline.utils.visualization import draw_detections, draw_overlay_stats

logger = logging.getLogger(__name__)


def _positive_int(value: str) -> int:
    """Parse a positive integer CLI argument."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--source",
        help="Override the configured file/camera/RTSP source",
    )
    parser.add_argument("--weights", type=Path, help="Override configured YOLO weights")
    parser.add_argument("--conf", type=float, help="Override confidence threshold")
    parser.add_argument("--device", help="Override device (auto, cpu, cuda, or mps)")
    parser.add_argument("--output", type=Path, help="Annotated output video path")
    parser.add_argument("--max-frames", type=_positive_int)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display output; press q to quit",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write an output video",
    )
    return parser


def _configuration(args: argparse.Namespace) -> AppConfig:
    """Load config and apply explicit CLI overrides."""
    config = load_config(args.config)
    video = config.video
    model = config.model
    if args.source is not None:
        video = video.model_copy(update={"source": args.source})
    updates: dict[str, object] = {}
    if args.weights is not None:
        updates["weights_path"] = args.weights
    if args.conf is not None:
        if not 0 <= args.conf <= 1:
            raise ValueError("--conf must be between 0 and 1.")
        updates["conf_threshold"] = args.conf
    if args.device is not None:
        updates["device"] = args.device
    if updates:
        model = model.model_copy(update=updates)
    return config.model_copy(update={"video": video, "model": model})


def run(args: argparse.Namespace) -> int:
    """Execute detection, annotate frames, and optionally save/display them."""
    config = _configuration(args)
    output = args.output or config.paths.output_dir / "detected.mp4"
    detector = YoloDetector(config.model)
    detector.warmup()
    stats = LatencyStats()
    fps_meter = FPSMeter()
    frames_processed = 0
    started_at = time.perf_counter()
    source = create_source(config.video)
    writer: VideoWriter | None = None
    try:
        with source:
            if not args.no_save:
                writer = VideoWriter.from_source(output, source)
            while args.max_frames is None or frames_processed < args.max_frames:
                frame_started = time.perf_counter()
                with StageTimer("read", stats):
                    frame_data = source.read()
                if frame_data is None:
                    break
                fps = fps_meter.tick()
                with StageTimer("preprocess", stats):
                    frame = frame_data.frame
                with StageTimer("inference", stats):
                    raw_detections = detector.detect(frame)
                with StageTimer("postprocess", stats):
                    detections = list(raw_detections)
                latency_ms = (time.perf_counter() - frame_started) * 1000
                with StageTimer("draw", stats):
                    annotated = draw_detections(frame, detections)
                    annotated = draw_overlay_stats(
                        annotated,
                        fps,
                        latency_ms,
                        frame_data.frame_id,
                    )
                if writer is not None:
                    with StageTimer("write", stats):
                        writer.write(annotated)
                if args.show:
                    cv2.imshow("Vision Pipeline Detection", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                frames_processed += 1
    finally:
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started_at
    average_fps = frames_processed / elapsed if elapsed > 0 else 0.0
    logger.info(
        "Detection complete: %d frames, %.2f average FPS, output=%s",
        frames_processed,
        average_fps,
        "disabled" if args.no_save else output,
    )
    for stage, values in stats.to_dict().items():
        logger.info("%s latency: mean %.2f ms", stage, values["mean"])
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (
        ImportError,
        OSError,
        ValueError,
        cv2.error,
        VideoSourceError,
        VideoWriterError,
    ) as error:
        logger.error("%s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
