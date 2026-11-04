"""Fast deterministic tests for the threaded streaming system."""

from __future__ import annotations

import queue
import threading
import time
import importlib.util
from pathlib import Path

import cv2
import numpy as np

from vision_pipeline.analytics.engine import AnalyticsEngine
from vision_pipeline.analytics.events import Event, EventType, JsonlSink
from vision_pipeline.config import TrackerConfig
from vision_pipeline.detection.yolo_detector import FakeDetector
from vision_pipeline.pipeline import AnalyticsFrame, AnalyticsPipeline
from vision_pipeline.streaming.clock import FakeClock
from vision_pipeline.streaming.controller import AdaptiveController, ControllerConfig
from vision_pipeline.streaming.engine import (
    HealthState,
    StreamingConfig,
    StreamingPipeline,
)
from vision_pipeline.streaming.metrics import StreamMetrics
from vision_pipeline.streaming.queues import (
    BoundedQueue,
    LatestValue,
    OverflowPolicy,
    SENTINEL,
)
from vision_pipeline.streaming.sinks import EventLogSink, JpegStreamSink, VideoFileSink
from vision_pipeline.streaming.sources import FakeSource
from vision_pipeline.streaming.stage import PipelineStage
from vision_pipeline.tracking.bytetrack import ByteTracker
from vision_pipeline.types import Detection, FrameData

ROOT = Path(__file__).resolve().parents[1]
_benchmark_spec = importlib.util.spec_from_file_location(
    "benchmark_streaming", ROOT / "scripts" / "benchmark_streaming.py"
)
assert _benchmark_spec is not None and _benchmark_spec.loader is not None
benchmark_streaming = importlib.util.module_from_spec(_benchmark_spec)
_benchmark_spec.loader.exec_module(benchmark_streaming)


def _frames(count: int = 8) -> list[FrameData]:
    """Create a short sequence of identical synthetic frames."""
    return [
        FrameData(
            np.zeros((32, 48, 3), dtype=np.uint8),
            frame_id,
            frame_id / 10.0,
            10.0,
        )
        for frame_id in range(count)
    ]


def _components(frames: list[FrameData]):
    """Create independent dependency-free components for each pipeline run."""
    return (
        FakeSource(frames=frames, fps=10.0, width=48, height=32),
        FakeDetector([Detection((5, 5, 20, 28), 0.9, 0, "person")]),
        ByteTracker(TrackerConfig(min_hits=1)),
        AnalyticsEngine(),
    )


def test_adaptive_controller_hysteresis_and_detection_age() -> None:
    """Stride changes need sustained pressure and forced detection bounds age."""
    controller = AdaptiveController(
        ControllerConfig(
            latency_budget_ms=100,
            adaptive=True,
            min_stride=1,
            max_stride=4,
            ema_alpha=1,
            hysteresis=0.1,
            hysteresis_samples=2,
            max_detection_age_frames=5,
        )
    )
    assert controller.observe(120) == 1
    assert controller.observe(130) == 2
    controller.record_detection(0)
    assert not controller.should_detect(1)
    assert controller.should_detect(5)

    assert controller.observe(20) == 2
    assert controller.observe(20) == 1
    assert controller.observe(20, queue_depth=1) == 1

    rate_limited = AdaptiveController(ControllerConfig(target_fps=10))
    assert rate_limited.should_detect(0, capture_ts=0.0)
    rate_limited.record_detection(0, capture_ts=0.0)
    assert not rate_limited.should_detect(1, capture_ts=0.05)
    assert rate_limited.should_detect(2, capture_ts=0.1)


def test_queue_overflow_close_and_latest_value() -> None:
    """Queue overflow policies and latest-value version waits are deterministic."""
    queue_oldest: BoundedQueue[int] = BoundedQueue(2, policy=OverflowPolicy.DROP_OLDEST)
    queue_oldest.put(1)
    queue_oldest.put(2)
    queue_oldest.put(3)
    assert queue_oldest.get() == 2
    assert queue_oldest.get() == 3
    assert queue_oldest.stats.dropped_count == 1

    queue_newest: BoundedQueue[int] = BoundedQueue(1, policy=OverflowPolicy.DROP_NEWEST)
    queue_newest.put(1)
    assert queue_newest.put(2) is False
    queue_newest.close()
    assert queue_newest.get() == 1
    assert queue_newest.get() is SENTINEL

    latest: LatestValue[str] = LatestValue()
    assert latest.set("first") == 1
    assert latest.wait_for_update(0, timeout=0) == 1
    assert latest.get() == (1, "first")

    waiting: BoundedQueue[int] = BoundedQueue(1)
    received: list[object] = []
    waiter = threading.Thread(target=lambda: received.append(waiting.get()))
    waiter.start()
    waiting.close()
    waiter.join(1)
    assert not waiter.is_alive()
    assert received == [SENTINEL]


def test_pipeline_stage_lifecycle_and_error_capture() -> None:
    """A stage drains and closes normally and captures processing exceptions."""

    class DoubleStage(PipelineStage[int]):
        def process(self, item: int) -> int:
            return item * 2

    input_queue: BoundedQueue[int] = BoundedQueue(2, policy=OverflowPolicy.BLOCK)
    output_queue: BoundedQueue[int] = BoundedQueue(2, policy=OverflowPolicy.BLOCK)
    stage = DoubleStage("double", input_queue=input_queue, output_queue=output_queue)
    stage.start()
    input_queue.put(4)
    input_queue.close()
    stage.join(2)
    assert not stage.is_alive()
    assert output_queue.get() == 8
    assert output_queue.get() is SENTINEL
    assert stage.metrics.processed == 1

    class BrokenStage(PipelineStage[int]):
        def process(self, item: int) -> None:
            raise ValueError(f"bad item: {item}")

    broken_input: BoundedQueue[int] = BoundedQueue(1)
    broken = BrokenStage("broken", input_queue=broken_input)
    broken.start()
    broken_input.put(1)
    broken.join(2)
    assert isinstance(broken.exception, ValueError)
    assert broken.metrics.errors == 1


def test_sync_and_threaded_results_match_and_threads_stop() -> None:
    """Threaded output preserves frame-level analytics for the no-drop case."""
    frames = _frames()
    sync_source, sync_detector, sync_tracker, sync_analytics = _components(frames)
    sync = AnalyticsPipeline(
        sync_source, sync_detector, sync_tracker, sync_analytics, max_frames=len(frames)
    )
    sync_results = list(sync.run())
    sync_snapshot = sync_analytics.snapshot().to_dict()
    sync.close()

    source, detector, tracker, analytics = _components(frames)
    events: queue.Queue[object] = queue.Queue(maxsize=8)
    jpeg_sink = JpegStreamSink(queue_size=2)
    pipeline = StreamingPipeline(
        StreamingConfig(stream_id="integration", live=False, queue_size=2),
        source,
        detector,
        tracker,
        analytics,
        [jpeg_sink],
        clock=FakeClock(),
    )
    pipeline.subscribe_events(events)
    pipeline.start()
    pipeline.join(timeout=5)

    metrics = pipeline.get_metrics()
    assert metrics["frames_captured"] == len(frames)
    assert metrics["frames_processed"] == len(frames)
    assert metrics["frames_output"] == len(frames)
    assert metrics["frames_dropped"] == 0
    assert pipeline.get_snapshot().to_dict() == sync_snapshot
    assert len(sync_results) == len(frames)
    latest = pipeline.get_latest_frame()
    assert latest is not None
    assert latest.stream_id == "integration"
    assert latest.jpeg.startswith(b"\xff\xd8")
    _, sink_frame = jpeg_sink.latest.get()
    assert sink_frame is not None and sink_frame.frame_id == len(frames) - 1
    assert pipeline.wait_for_frame(0, timeout=0) is not None
    assert events.get(timeout=1).frame_id == 0
    assert pipeline.get_health()["state"] == HealthState.STOPPED.value
    assert all(not stage.is_alive() for stage in pipeline._stages)
    assert not jpeg_sink.alive
    assert "vision_pipeline_frames_total" in pipeline.metrics.to_prometheus()
    pipeline.stop()


def test_video_and_event_sinks_drain_before_close(tmp_path) -> None:
    """Asynchronous file sinks flush queued frames and events during shutdown."""
    frame_data = _frames(1)[0]
    analytics = AnalyticsEngine()
    event = Event(EventType.TRACK_STARTED, 0.0, 0, 1)
    result = AnalyticsFrame(
        frame_data=frame_data,
        detections=[],
        tracks=[],
        events=[event],
        snapshot=analytics.snapshot(),
        annotated=frame_data.frame.copy(),
    )
    video_path = tmp_path / "async.avi"
    event_path = tmp_path / "events.jsonl"
    video_sink = VideoFileSink(
        str(video_path),
        10.0,
        frame_data.frame.shape[1],
        frame_data.frame.shape[0],
        codec="MJPG",
    )
    event_sink = EventLogSink(JsonlSink(event_path))
    video_sink.submit(result)
    event_sink.submit(result)
    video_sink.close()
    event_sink.close()

    capture = cv2.VideoCapture(str(video_path))
    try:
        ok, decoded = capture.read()
    finally:
        capture.release()
    assert ok and decoded.shape == frame_data.frame.shape
    assert '"TRACK_STARTED"' in event_path.read_text(encoding="utf-8")


def test_live_pipeline_drops_old_frames_and_stops_cleanly() -> None:
    """Slow inference cannot build an unbounded live backlog or leak threads."""

    class SlowDetector(FakeDetector):
        def detect(self, frame):
            time.sleep(0.03)
            return super().detect(frame)

    source = FakeSource(fps=1000.0, width=8, height=8, infinite=True)
    pipeline = StreamingPipeline(
        StreamingConfig(
            stream_id="live-drop",
            live=True,
            queue_size=1,
            output_queue_size=1,
        ),
        source,
        SlowDetector(),
        ByteTracker(TrackerConfig(min_hits=1)),
        AnalyticsEngine(),
    )
    pipeline.start()
    time.sleep(0.3)
    pipeline.stop(timeout=3)

    metrics = pipeline.get_metrics()
    assert metrics["frames_dropped"] > 0
    assert metrics["dropped_by_policy"]["drop_oldest"] > 0
    assert metrics["queue_policies"]["q1"] == "drop_oldest"
    assert metrics["queue_depths"]["q1"] <= 1
    assert pipeline.get_health()["state"] == HealthState.STOPPED.value
    assert all(not stage.is_alive() for stage in pipeline._stages)


def test_stage_failure_marks_pipeline_failed_and_stops_downstream() -> None:
    """Detector exceptions reach the supervisor and close every stage queue."""

    class BrokenDetector(FakeDetector):
        def detect(self, frame):
            raise RuntimeError("inference failed")

    source, _, tracker, analytics = _components(_frames(3))
    pipeline = StreamingPipeline(
        StreamingConfig(stream_id="failure", live=False, queue_size=1),
        source,
        BrokenDetector(),
        tracker,
        analytics,
    )
    pipeline.start()
    pipeline.join(timeout=3)

    health = pipeline.get_health()
    assert health["state"] == HealthState.FAILED.value
    assert "inference failed" in health["error"]
    assert all(not stage.is_alive() for stage in pipeline._stages)
    pipeline.stop()


def test_stream_metrics_and_fake_clock() -> None:
    """Metrics export rates and latency summaries from an injected clock."""
    clock = FakeClock()
    metrics = StreamMetrics(clock=clock)
    metrics.record_capture()
    clock.advance(0.5)
    metrics.record_capture()
    metrics.record_processed()
    metrics.record_output(20.0)
    snapshot = metrics.snapshot()
    assert snapshot["input_fps"] == 2.0
    assert snapshot["frames_captured"] == 2
    assert snapshot["end_to_end_latency"]["p95_ms"] == 20.0
    assert '"frames_output": 1' in metrics.to_json()


def test_streaming_benchmark_runs_all_modes_with_fake_detector(
    tmp_path, monkeypatch
) -> None:
    """The benchmark executable reports all comparison modes without a model."""
    video_path = tmp_path / "benchmark.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (48, 32),
    )
    assert writer.isOpened()
    for _ in range(4):
        writer.write(np.zeros((32, 48, 3), dtype=np.uint8))
    writer.release()

    monkeypatch.setattr(
        benchmark_streaming,
        "create_detector",
        lambda _config: FakeDetector([Detection((5, 5, 20, 28), 0.9, 0, "person")]),
    )
    output = tmp_path / "benchmark-results.json"
    args = benchmark_streaming.build_parser().parse_args(
        [
            "--source",
            str(video_path),
            "--frames",
            "4",
            "--output",
            str(output),
        ]
    )
    results = benchmark_streaming.run(args)

    assert set(results) == {"sync", "threaded", "adaptive"}
    assert results["sync"]["frames_processed"] == 4
    assert results["threaded"]["frames_output"] == 4
    assert results["adaptive"]["frames_output"] == 4
    assert output.is_file()
