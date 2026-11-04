"""Compare synchronous, threaded, and adaptive analytics throughput."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from vision_pipeline.analytics.engine import AnalyticsEngine
from vision_pipeline.config import AppConfig, load_config
from vision_pipeline.detection.factory import create_detector
from vision_pipeline.detection.yolo_detector import BaseDetector
from vision_pipeline.logging_utils import setup_logging
from vision_pipeline.pipeline import AnalyticsPipeline
from vision_pipeline.streaming.controller import ControllerConfig
from vision_pipeline.streaming.engine import StreamingConfig, StreamingPipeline
from vision_pipeline.streaming.sources import (
    VideoSource,
    VideoSourceError,
    create_source,
)
from vision_pipeline.tracking.bytetrack import ByteTracker
from vision_pipeline.types import FrameData

logger = logging.getLogger(__name__)


class _LimitedSource(VideoSource):
    """Expose a bounded prefix of a source for reproducible benchmarks."""

    def __init__(self, source: VideoSource, limit: int) -> None:
        self.source = source
        self.limit = limit
        self.read_count = 0

    def open(self) -> None:
        """Open the wrapped source."""
        self.source.open()

    def read(self) -> FrameData | None:
        """Read no more than the configured frame limit."""
        if self.read_count >= self.limit:
            return None
        frame = self.source.read()
        if frame is not None:
            self.read_count += 1
        return frame

    def release(self) -> None:
        """Release the wrapped source."""
        self.source.release()

    @property
    def fps(self) -> float:
        """Return the wrapped source frame rate."""
        return self.source.fps

    @property
    def width(self) -> int:
        """Return the wrapped source width."""
        return self.source.width

    @property
    def height(self) -> int:
        """Return the wrapped source height."""
        return self.source.height

    @property
    def frame_count(self) -> int | None:
        """Report a finite size for the benchmark prefix."""
        return self.limit


def build_parser() -> argparse.ArgumentParser:
    """Create the benchmark command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--source", help="Override the configured video source")
    parser.add_argument("--backend", choices=("auto", "pytorch", "onnx", "tensorrt"))
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--latency-budget-ms", type=float, default=100.0)
    parser.add_argument("--output", type=Path, help="Optional JSON result path")
    return parser


def _configured(args: argparse.Namespace) -> AppConfig:
    """Load config and apply source/backend overrides."""
    config = load_config(args.config)
    updates: dict[str, object] = {}
    if args.source is not None:
        updates["video"] = config.video.model_copy(update={"source": args.source})
    if args.backend is not None:
        updates["detector"] = config.detector.model_copy(
            update={"backend": args.backend}
        )
    return config.model_copy(update=updates) if updates else config


def _components(
    config: AppConfig,
    limit: int,
) -> tuple[VideoSource, BaseDetector, ByteTracker, AnalyticsEngine]:
    """Create independent source, detector, tracker, and analytics instances."""
    source = _LimitedSource(create_source(config.video), limit)
    return (
        source,
        create_detector(config),
        ByteTracker(config.tracker),
        AnalyticsEngine(config.analytics),
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    """Run the three benchmark modes and return machine-readable results."""
    if args.frames < 1:
        raise ValueError("--frames must be positive")
    if args.latency_budget_ms <= 0:
        raise ValueError("--latency-budget-ms must be positive")
    config = _configured(args)
    results: dict[str, object] = {}

    source, detector, tracker, analytics = _components(config, args.frames)
    sync = AnalyticsPipeline(
        source, detector, tracker, analytics, max_frames=args.frames
    )
    started = time.perf_counter()
    processed = 0
    try:
        detector.warmup()
        for _ in sync.run():
            processed += 1
    finally:
        sync.close()
        close = getattr(detector, "close", None)
        if callable(close):
            close()
    elapsed = time.perf_counter() - started
    results["sync"] = {
        "frames_processed": processed,
        "elapsed_seconds": elapsed,
        "fps": processed / elapsed if elapsed else 0.0,
    }

    for mode in ("threaded", "adaptive"):
        source, detector, tracker, analytics = _components(config, args.frames)
        adaptive = mode == "adaptive"
        pipeline = StreamingPipeline(
            StreamingConfig(
                stream_id=f"benchmark-{mode}",
                live=False,
                controller=ControllerConfig(
                    latency_budget_ms=args.latency_budget_ms,
                    adaptive=adaptive,
                    max_stride=8,
                ),
            ),
            source,
            detector,
            tracker,
            analytics,
        )
        started = time.perf_counter()
        pipeline.start()
        pipeline.join()
        elapsed = time.perf_counter() - started
        result_metrics = pipeline.get_metrics()
        results[mode] = {
            "frames_captured": result_metrics["frames_captured"],
            "frames_processed": result_metrics["frames_processed"],
            "frames_output": result_metrics["frames_output"],
            "elapsed_seconds": elapsed,
            "fps": result_metrics["frames_output"] / elapsed if elapsed else 0.0,
            "p95_end_to_end_latency_ms": result_metrics["end_to_end_latency"]["p95_ms"],
            "frames_dropped": result_metrics["frames_dropped"],
            "final_stride": result_metrics["current_stride"],
        }
        pipeline.stop()
        close = getattr(detector, "close", None)
        if callable(close):
            close()
        analytics.finalize()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for name, result in results.items():
        logger.info("%s: %s", name, result)
    return results


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    setup_logging()
    try:
        run(build_parser().parse_args(argv))
    except (ImportError, OSError, ValueError, RuntimeError, VideoSourceError) as error:
        logger.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
