"""Run tracking and retail analytics on a video source."""

import argparse
import json
import logging
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import cv2

from vision_pipeline.analytics.engine import AnalyticsEngine
from vision_pipeline.analytics.events import CsvSink, JsonlSink
from vision_pipeline.analytics.geometry import load_zones
from vision_pipeline.config import AppConfig, load_config
from vision_pipeline.detection.yolo_detector import YoloDetector
from vision_pipeline.logging_utils import setup_logging
from vision_pipeline.pipeline import AnalyticsPipeline
from vision_pipeline.streaming.sources import VideoSourceError, create_source
from vision_pipeline.streaming.writer import VideoWriter, VideoWriterError
from vision_pipeline.tracking.bytetrack import ByteTracker

logger = logging.getLogger(__name__)


def _positive_int(value: str) -> int:
    """Parse a positive integer CLI argument."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the analytics command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--zones", type=Path, default=Path("configs/zones.yaml"))
    parser.add_argument(
        "--source",
        required=True,
        help="Video file, camera index, or RTSP URL",
    )
    parser.add_argument("--weights", type=Path, help="Override configured YOLO weights")
    parser.add_argument("--output", type=Path, help="Annotated output video")
    parser.add_argument("--events-out", type=Path, default=Path("outputs/events.jsonl"))
    parser.add_argument("--heatmap-out", type=Path, default=Path("outputs/heatmap.png"))
    parser.add_argument("--report-out", type=Path, default=Path("outputs/report"))
    parser.add_argument("--max-frames", type=_positive_int)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display output; press q to quit",
    )
    return parser


def _configuration(args: argparse.Namespace) -> AppConfig:
    """Load config and replace CLI-selected source, weights, and zones."""
    config = load_config(args.config)
    model = config.model
    if args.weights is not None:
        model = model.model_copy(update={"weights_path": args.weights})
    analytics = load_zones(args.zones)
    return config.model_copy(
        update={
            "video": config.video.model_copy(update={"source": args.source}),
            "model": model,
            "analytics": analytics,
        }
    )


def _plain(value: Any) -> Any:
    """Convert dataclasses and common containers into JSON-compatible values."""
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value


def _report_paths(path: Path) -> tuple[Path, Path]:
    """Resolve JSON and Markdown report destinations."""
    if path.suffix.lower() == ".json":
        return path, path.with_suffix(".md")
    path.mkdir(parents=True, exist_ok=True)
    return path / "report.json", path / "report.md"


def _write_report(
    path: Path,
    summary: Any,
    records: list[dict[str, Any]],
    latency_ms: dict[str, float | int],
) -> None:
    """Write the final summary and time-series records in JSON and Markdown."""
    summary_data = _plain(summary)
    payload = {
        "summary": summary_data,
        "time_series": _plain(records),
        "analytics_latency_ms": latency_ms,
    }
    json_path, markdown_path = _report_paths(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    markdown = [
        "# Retail analytics report",
        "",
        f"- Analytics mean latency: {latency_ms.get('mean', 0):.4f} ms/frame",
        f"- Analytics p95 latency: {latency_ms.get('p95', 0):.4f} ms/frame",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary_data, indent=2, default=str),
        "```",
        "",
        "## Time series",
        "",
        f"Buckets: {len(records)}",
        "",
    ]
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(markdown), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    """Execute detection, tracking, analytics, and output generation."""
    config = _configuration(args)
    sinks = []
    if args.events_out.suffix.lower() == ".csv":
        sinks.append(CsvSink(args.events_out))
    else:
        sinks.append(JsonlSink(args.events_out))
    engine = AnalyticsEngine(config.analytics, sinks=sinks)
    detector = YoloDetector(config.model)
    detector.warmup()
    tracker = ByteTracker(config.tracker)
    source = create_source(config.video)
    writer: VideoWriter | None = None
    last_frame = None
    summary = None
    try:
        with source:
            if args.output is not None:
                writer = VideoWriter.from_source(args.output, source)
            pipeline = AnalyticsPipeline(
                source,
                detector,
                tracker,
                engine,
                writer=writer,
                max_frames=args.max_frames,
            )
            for result in pipeline.run():
                last_frame = result.frame_data.frame
                if args.show:
                    cv2.imshow("Retail Analytics", result.annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            summary = engine.finalize()
            latency = pipeline.latency_stats.summary("analytics")
            records = engine.aggregator.to_records()
            engine.close()
    finally:
        if writer is not None:
            writer.release()
        if summary is None:
            summary = engine.finalize()
            engine.close()
        if args.show:
            cv2.destroyAllWindows()

    if last_frame is not None and engine.heatmap is not None:
        heatmap = engine.heatmap.render(last_frame, alpha=0.5)
        args.heatmap_out.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.heatmap_out), heatmap):
            raise OSError(f"Unable to write heatmap image: {args.heatmap_out}")
    _write_report(args.report_out, summary, records, latency)
    logger.info("Analytics report written to %s", args.report_out)
    logger.info(
        "Analytics latency: mean %.4f ms/frame, p95 %.4f ms/frame",
        latency.get("mean", 0),
        latency.get("p95", 0),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    setup_logging()
    args = build_parser().parse_args(argv)
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
