"""Run the supervised threaded video analytics pipeline."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from pathlib import Path
from typing import Any

from vision_pipeline.analytics.engine import AnalyticsEngine
from vision_pipeline.config import load_config
from vision_pipeline.detection.factory import create_detector
from vision_pipeline.detection.yolo_detector import BaseDetector
from vision_pipeline.logging_utils import setup_logging
from vision_pipeline.streaming.controller import ControllerConfig
from vision_pipeline.streaming.engine import (
    HealthState,
    StreamingConfig,
    StreamingPipeline,
)
from vision_pipeline.streaming.sinks import (
    JpegStreamSink,
    OutputConsumer,
    VideoFileSink,
    WindowSink,
)
from vision_pipeline.streaming.sources import VideoSourceError, create_source
from vision_pipeline.tracking.bytetrack import ByteTracker

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the live-pipeline command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--source", help="Override file, camera, or RTSP source")
    parser.add_argument("--backend", choices=("auto", "pytorch", "onnx", "tensorrt"))
    parser.add_argument("--output", type=Path, help="Optional annotated video output")
    parser.add_argument(
        "--show", action="store_true", help="Display output in a window"
    )
    parser.add_argument(
        "--realtime", action="store_true", help="Pace file frames at source FPS"
    )
    parser.add_argument(
        "--adaptive", action="store_true", help="Enable adaptive detection stride"
    )
    parser.add_argument("--stream-id", default="default")
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--latency-budget-ms", type=float, default=100.0)
    parser.add_argument("--detect-every-n", type=int, default=1)
    return parser


def run(args: argparse.Namespace) -> int:
    """Configure and run one streaming pipeline until EOF or interruption."""
    config = load_config(args.config)
    updates: dict[str, object] = {}
    if args.source is not None:
        updates["video"] = config.video.model_copy(update={"source": args.source})
    if args.backend is not None:
        updates["detector"] = config.detector.model_copy(
            update={"backend": args.backend}
        )
    if updates:
        config = config.model_copy(update=updates)

    source = create_source(config.video)
    detector: BaseDetector | None = None
    tracker = ByteTracker(config.tracker)
    analytics = AnalyticsEngine(config.analytics)
    sinks: list[OutputConsumer] = []
    window: WindowSink | None = None
    pipeline: StreamingPipeline | None = None
    interrupted = threading.Event()
    previous_handlers: dict[signal.Signals, Any] = {}

    def request_shutdown(_signal: int, _frame: object) -> None:
        interrupted.set()

    try:
        detector = create_detector(config)
        source.open()
        window = WindowSink() if args.show else None
        sinks.append(JpegStreamSink())
        if window is not None:
            sinks.append(window)
        if args.output is not None:
            sinks.append(
                VideoFileSink(
                    str(args.output),
                    source.fps,
                    source.width,
                    source.height,
                )
            )
        streaming_config = StreamingConfig(
            stream_id=args.stream_id,
            realtime=args.realtime,
            controller=ControllerConfig(
                target_fps=args.target_fps,
                latency_budget_ms=args.latency_budget_ms,
                detect_every_n=args.detect_every_n,
                adaptive=args.adaptive,
            ),
        )
        pipeline = StreamingPipeline(
            streaming_config, source, detector, tracker, analytics, sinks
        )
        for handled_signal in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[handled_signal] = signal.signal(
                handled_signal, request_shutdown
            )
        pipeline.start()
        while not interrupted.wait(0.05):
            if window is not None and not window.pump():
                break
            health = pipeline.get_health()["state"]
            if health in (HealthState.STOPPED.value, HealthState.FAILED.value):
                break
        pipeline.stop()
        logger.info("Stream finished: %s", pipeline.get_metrics())
        if pipeline.get_health()["state"] == HealthState.FAILED.value:
            logger.error("Pipeline failed: %s", pipeline.get_health()["error"])
            return 1
        return 0
    finally:
        for handled_signal, previous_handler in previous_handlers.items():
            signal.signal(handled_signal, previous_handler)
        if pipeline is not None:
            pipeline.stop()
        else:
            source.release()
            for sink in sinks:
                sink.close()
        if window is not None:
            window.close()
        if detector is not None:
            close = getattr(detector, "close", None)
            if callable(close):
                close()
        analytics.finalize()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    setup_logging()
    try:
        return run(build_parser().parse_args(argv))
    except (ImportError, OSError, ValueError, RuntimeError, VideoSourceError) as error:
        logger.error("%s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
