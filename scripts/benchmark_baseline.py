"""Measure baseline video detection latency and throughput."""

import argparse
import json
import logging
import platform
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

from vision_pipeline.config import AppConfig, load_config
from vision_pipeline.detection.yolo_detector import YoloDetector
from vision_pipeline.logging_utils import setup_logging
from vision_pipeline.streaming.sources import VideoSourceError, create_source
from vision_pipeline.streaming.writer import VideoWriter, VideoWriterError
from vision_pipeline.utils.profiling import LatencyStats, StageTimer
from vision_pipeline.utils.visualization import draw_detections

logger = logging.getLogger(__name__)


def _positive_int(value: str) -> int:
    """Parse a positive integer CLI argument."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the benchmark command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--source", help="Override the configured video source")
    parser.add_argument("--weights", type=Path, help="Override configured YOLO weights")
    parser.add_argument("--device", help="Override device (auto, cpu, cuda, or mps)")
    parser.add_argument(
        "--imgsz",
        type=_positive_int,
        help="Override inference image size",
    )
    parser.add_argument("--frames", type=_positive_int, default=100)
    parser.add_argument("--warmup", type=_positive_int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks"))
    return parser


def _get_config(args: argparse.Namespace) -> AppConfig:
    """Load config and apply benchmark overrides."""
    config = load_config(args.config)
    video = config.video
    model = config.model
    if args.source is not None:
        video = video.model_copy(update={"source": args.source})
    updates: dict[str, object] = {}
    if args.weights is not None:
        updates["weights_path"] = args.weights
    if args.device is not None:
        updates["device"] = args.device
    if args.imgsz is not None:
        updates["imgsz"] = args.imgsz
    if updates:
        model = model.model_copy(update=updates)
    return config.model_copy(update={"video": video, "model": model})


def _hardware_description() -> str:
    """Describe the host and optional torch runtime without requiring torch."""
    try:
        import torch
    except ImportError:
        torch_version = "not installed"
    else:
        torch_version = torch.__version__
    return (
        f"{platform.platform()}; Python {platform.python_version()}; "
        f"torch {torch_version}"
    )


def _append_markdown_row(
    readme: Path,
    *,
    model: str,
    device: str,
    imgsz: int,
    fps: float,
    stage_stats: dict[str, dict[str, float | int]],
    date: str,
    hardware: str,
) -> None:
    """Append one result row to the benchmark markdown table."""
    readme.parent.mkdir(parents=True, exist_ok=True)
    if not readme.exists():
        readme.write_text(
            "# Baseline benchmarks\n\n"
            "| Model | Device | imgsz | FPS | Stage latency mean/p95 ms | Date "
            "| Hardware |\n"
            "|---|---|---:|---:|---|---|---|\n",
            encoding="utf-8",
        )
    stage_latency = "; ".join(
        f"{name}: {values['mean']:.2f}/{values['p95']:.2f}"
        for name, values in sorted(stage_stats.items())
    )
    escaped_hardware = hardware.replace("|", "\\|")
    with readme.open("a", encoding="utf-8") as output:
        output.write(
            f"| {model} | {device} | {imgsz} | {fps:.2f} | "
            f"{stage_latency} | {date} | {escaped_hardware} |\n"
        )


def run(args: argparse.Namespace) -> Path:
    """Run warmup and measured frames, then persist JSON and table results."""
    config = _get_config(args)
    detector = YoloDetector(config.model)
    detector.warmup()
    source = create_source(config.video)
    stats = LatencyStats()
    processed = 0
    started_at = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="vision-benchmark-") as directory:
        temp_output = Path(directory) / "benchmark.avi"
        with source:
            with VideoWriter.from_source(
                temp_output,
                source,
                codec="MJPG",
            ) as writer:
                for _ in range(args.warmup):
                    with StageTimer("warmup_read", stats):
                        frame_data = source.read()
                    if frame_data is None:
                        break
                    detector.detect(frame_data.frame)
                started_at = time.perf_counter()
                for _ in range(args.frames):
                    with StageTimer("read", stats):
                        frame_data = source.read()
                    if frame_data is None:
                        break
                    with StageTimer("preprocess", stats):
                        frame = frame_data.frame
                    with StageTimer("inference", stats):
                        raw_detections = detector.detect(frame)
                    with StageTimer("postprocess", stats):
                        detections = list(raw_detections)
                    with StageTimer("draw", stats):
                        annotated = draw_detections(frame, detections)
                    with StageTimer("write", stats):
                        writer.write(annotated)
                    processed += 1
    if processed == 0:
        raise ValueError("No video frames were processed; no benchmark was written.")
    elapsed = time.perf_counter() - started_at
    fps = processed / elapsed if elapsed > 0 else 0.0
    stage_stats = {
        name: values
        for name, values in stats.to_dict().items()
        if name != "warmup_read"
    }
    date = datetime.now(timezone.utc).date().isoformat()
    hardware = _hardware_description()
    model_name = config.model.weights_path.stem
    device_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", detector.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / f"baseline_{device_name}_{model_name}.json"
    result = {
        "model": str(config.model.weights_path),
        "device": detector.device,
        "imgsz": config.model.imgsz,
        "frames_processed": processed,
        "warmup_frames": args.warmup,
        "fps": fps,
        "elapsed_s": elapsed,
        "stage_latency_ms": stage_stats,
        "date": date,
        "hardware": hardware,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _append_markdown_row(
        args.output_dir / "README.md",
        model=model_name,
        device=detector.device,
        imgsz=config.model.imgsz,
        fps=fps,
        stage_stats=stage_stats,
        date=date,
        hardware=hardware,
    )
    logger.info(
        "Benchmark complete: %d frames, %.2f FPS; results saved to %s",
        processed,
        fps,
        result_path,
    )
    return result_path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    setup_logging()
    args = build_parser().parse_args(argv)
    try:
        run(args)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
