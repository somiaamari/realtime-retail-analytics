"""Supervised threaded capture, inference, tracking, and output pipeline."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
from queue import Full, Queue
from typing import Any

import cv2

from vision_pipeline.analytics.engine import AnalyticsEngine, AnalyticsSnapshot
from vision_pipeline.detection.yolo_detector import BaseDetector
from vision_pipeline.logging_utils import get_logger
from vision_pipeline.pipeline import AnalyticsFrame
from vision_pipeline.streaming.clock import Clock, SystemClock
from vision_pipeline.streaming.controller import AdaptiveController, ControllerConfig
from vision_pipeline.streaming.metrics import StreamMetrics
from vision_pipeline.streaming.queues import (
    BoundedQueue,
    EmptyQueueError,
    LatestValue,
    OverflowPolicy,
    QueueClosedError,
    SENTINEL,
    _Sentinel,
)
from vision_pipeline.streaming.sinks import EncodedFrame, OutputConsumer
from vision_pipeline.streaming.sources import (
    FileVideoSource,
    RTSPSource,
    VideoSource,
    VideoSourceError,
    WebcamSource,
)
from vision_pipeline.streaming.stage import PipelineStage
from vision_pipeline.tracking.base import BaseTracker
from vision_pipeline.types import FrameData
from vision_pipeline.utils.profiling import FPSMeter
from vision_pipeline.utils.visualization import (
    draw_lines,
    draw_overlay_stats,
    draw_tracks,
    draw_zones,
)

logger = get_logger(__name__)


class HealthState(str, Enum):
    """Public health state for a streaming instance."""

    STARTING = "STARTING"
    RUNNING = "RUNNING"
    DEGRADED = "DEGRADED"
    STALLED = "STALLED"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class StreamingConfig:
    """Queue, pacing, watchdog, and controller settings."""

    stream_id: str = "default"
    live: bool | None = None
    realtime: bool = False
    queue_size: int = 4
    output_queue_size: int = 2
    watchdog_interval_s: float = 0.25
    stalled_after_s: float = 10.0
    reconnect_delay_s: float = 0.1
    controller: ControllerConfig = ControllerConfig()
    metrics_log_interval_s: float = 0.0

    def __post_init__(self) -> None:
        """Validate queue and timeout settings."""
        if self.queue_size < 1 or self.output_queue_size < 1:
            raise ValueError("queue sizes must be positive")
        if self.watchdog_interval_s <= 0 or self.stalled_after_s <= 0:
            raise ValueError("watchdog intervals must be positive")
        if self.reconnect_delay_s < 0 or self.metrics_log_interval_s < 0:
            raise ValueError("delays must be non-negative")
        if not self.stream_id:
            raise ValueError("stream_id must not be empty")


@dataclass(frozen=True)
class InferenceResult:
    """Frame and detector output passed to tracking."""

    frame_data: FrameData
    detections: list[Any]
    detected: bool


class _Subscriber:
    """One asynchronous event subscriber."""

    def __init__(
        self,
        callback: Callable[[Any], None] | None = None,
        output_queue: Queue[Any] | BoundedQueue[Any] | None = None,
        maxsize: int = 64,
    ) -> None:
        self.callback = callback
        self.external_queue = output_queue
        self.queue: BoundedQueue[Any] | None = None
        self.closed = False
        self.thread: threading.Thread | None = None
        if callback is not None:
            self.queue = BoundedQueue(maxsize, policy=OverflowPolicy.DROP_OLDEST)
            self.thread = threading.Thread(
                target=self._run, name="stream-event-subscriber", daemon=True
            )
            self.thread.start()

    def _run(self) -> None:
        """Deliver callback events outside of the pipeline threads."""
        assert self.queue is not None and self.callback is not None
        while True:
            try:
                event = self.queue.get(timeout=0.1)
            except EmptyQueueError:
                continue
            if event is SENTINEL:
                return
            try:
                self.callback(event)
            except Exception:
                logger.exception("Streaming event subscriber callback failed")

    def publish(self, event: Any) -> None:
        """Offer one event without blocking on a subscriber."""
        if self.closed:
            return
        if self.queue is not None:
            self.queue.put_nowait(event)
            return
        if isinstance(self.external_queue, BoundedQueue):
            self.external_queue.put_nowait(event)
        elif self.external_queue is not None:
            try:
                self.external_queue.put_nowait(event)
            except Full:
                try:
                    self.external_queue.get_nowait()
                except Exception:
                    return
                try:
                    self.external_queue.put_nowait(event)
                except Full:
                    return

    def close(self) -> None:
        """Stop callback delivery."""
        self.closed = True
        if self.queue is not None:
            self.queue.close()
        if self.thread is not None and threading.current_thread() is not self.thread:
            self.thread.join(timeout=1.0)


class EventBus:
    """Small in-process event bus with isolated bounded subscribers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: dict[int, _Subscriber] = {}
        self._next_id = 1

    def subscribe(
        self,
        consumer: Callable[[Any], None] | Queue[Any] | BoundedQueue[Any],
        *,
        maxsize: int = 64,
    ) -> Callable[[], None]:
        """Subscribe a callback or queue and return an unsubscribe function."""
        if maxsize < 1:
            raise ValueError("subscriber maxsize must be positive")
        if callable(consumer):
            subscriber = _Subscriber(callback=consumer, maxsize=maxsize)
        else:
            subscriber = _Subscriber(output_queue=consumer)
        with self._lock:
            subscription_id = self._next_id
            self._next_id += 1
            self._subscribers[subscription_id] = subscriber

        def unsubscribe() -> None:
            with self._lock:
                target = self._subscribers.pop(subscription_id, None)
            if target is not None:
                target.close()

        return unsubscribe

    def publish(self, event: Any) -> None:
        """Fan one event out to all subscribers without blocking."""
        with self._lock:
            subscribers = list(self._subscribers.values())
        for subscriber in subscribers:
            subscriber.publish(event)

    def close(self) -> None:
        """Stop all callback workers."""
        with self._lock:
            subscribers = list(self._subscribers.values())
            self._subscribers.clear()
        for subscriber in subscribers:
            subscriber.close()


class CaptureStage(PipelineStage[FrameData]):
    """Read from a source and stamp each frame with monotonic capture time."""

    def __init__(
        self,
        source: VideoSource,
        output_queue: BoundedQueue[Any],
        *,
        stream_id: str,
        clock: Clock,
        realtime: bool,
        live: bool | None,
        reconnect_delay_s: float,
        metrics: StreamMetrics,
        on_error: Callable[[PipelineStage[Any], Exception], None],
    ) -> None:
        super().__init__(
            "capture", output_queue=output_queue, on_error=on_error, clock=clock
        )
        self.source = source
        self.stream_id = stream_id
        self.realtime = realtime
        self.live = live
        self.reconnect_delay_s = reconnect_delay_s
        self.stream_metrics = metrics
        self.reconnecting = False
        self._reconnect_attempts = 0
        self._next_capture_at: float | None = None

    def on_start(self) -> None:
        """Open the source on the capture thread."""
        self.source.open()
        if self.live is None:
            self.live = self.source.frame_count is None

    def next_item(self) -> FrameData | _Sentinel:
        """Read a source frame or signal EOF for a finite input."""
        while not self.stop_event.is_set():
            try:
                frame = self.source.read()
            except VideoSourceError:
                if not self.live:
                    raise
                if self.stop_event.is_set():
                    return SENTINEL
                self._mark_reconnecting()
                delay = min(
                    self.reconnect_delay_s * (2**self._reconnect_attempts), 10.0
                )
                self.clock.sleep(delay)
                self._reconnect_attempts += 1
                continue
            if frame is None:
                if self.stop_event.is_set():
                    return SENTINEL
                if not self.live:
                    return SENTINEL
                self._mark_reconnecting()
                delay = min(
                    self.reconnect_delay_s * (2**self._reconnect_attempts), 10.0
                )
                self.clock.sleep(delay)
                self._reconnect_attempts += 1
                continue
            if self.reconnecting:
                logger.info("Stream %s resumed after source dropout", self.stream_id)
                self.reconnecting = False
                self._reconnect_attempts = 0
            now = self.clock.monotonic()
            if self.realtime and not self.live:
                interval = 1.0 / max(self.source.fps, 1e-6)
                if self._next_capture_at is None:
                    self._next_capture_at = now
                self._next_capture_at += interval
                delay = self._next_capture_at - now
                if delay > 0:
                    self.clock.sleep(delay)
                    now = self.clock.monotonic()
            metadata = dict(frame.stage_timestamps)
            metadata["capture"] = now
            stamped = replace(
                frame,
                stream_id=self.stream_id,
                capture_ts=now,
                stage_timestamps=metadata,
            )
            self.stream_metrics.record_capture(now)
            return stamped
        return SENTINEL

    def _mark_reconnecting(self) -> None:
        """Record one disconnect episode for health and monitoring."""
        if not self.reconnecting:
            self.reconnecting = True
            if not hasattr(self.source, "reconnect_count"):
                self.stream_metrics.record_reconnect()

    def process(self, item: FrameData) -> FrameData:
        """Capture items are already read and stamped by :meth:`next_item`."""
        return item

    def on_stop(self) -> None:
        """Release source resources when the stage exits."""
        self.source.release()


class InferenceStage(PipelineStage[FrameData]):
    """Run detector inference for selected frames."""

    def __init__(
        self,
        input_queue: BoundedQueue[Any],
        output_queue: BoundedQueue[Any],
        detector: BaseDetector,
        controller: AdaptiveController,
        queue_for_depth: BoundedQueue[Any],
        *,
        target_rate_control: bool,
        on_error: Callable[[PipelineStage[Any], Exception], None],
        clock: Clock,
    ) -> None:
        super().__init__(
            "inference",
            input_queue=input_queue,
            output_queue=output_queue,
            on_error=on_error,
            clock=clock,
        )
        self.detector = detector
        self.controller = controller
        self.queue_for_depth = queue_for_depth
        self.target_rate_control = target_rate_control

    def on_start(self) -> None:
        """Warm up detector before accepting the first frame."""
        self.detector.warmup()

    def process(self, frame_data: FrameData) -> InferenceResult:
        """Apply detection when required by the configured controller."""
        should_detect = self.controller.should_detect(
            frame_data.frame_id,
            len(self.queue_for_depth),
            frame_data.capture_ts if self.target_rate_control else None,
        )
        metadata = dict(frame_data.stage_timestamps)
        start = self.clock.monotonic()
        metadata["inference_start"] = start
        detections = self.detector.detect(frame_data.frame) if should_detect else []
        metadata["inference_end"] = self.clock.monotonic()
        if should_detect:
            self.controller.record_detection(
                frame_data.frame_id,
                frame_data.capture_ts if self.target_rate_control else None,
            )
        return InferenceResult(
            replace(frame_data, stage_timestamps=metadata),
            detections,
            should_detect,
        )


class TrackingAnalyticsStage(PipelineStage[InferenceResult]):
    """Advance tracking and analytics and render an annotated frame."""

    def __init__(
        self,
        input_queue: BoundedQueue[Any],
        output_queue: BoundedQueue[Any],
        tracker: BaseTracker,
        analytics: AnalyticsEngine,
        *,
        stream_metrics: StreamMetrics,
        latest_snapshot: LatestValue[AnalyticsSnapshot],
        on_error: Callable[[PipelineStage[Any], Exception], None],
        clock: Clock,
    ) -> None:
        super().__init__(
            "tracking_analytics",
            input_queue=input_queue,
            output_queue=output_queue,
            on_error=on_error,
            clock=clock,
        )
        self.tracker = tracker
        self.analytics = analytics
        self.stream_metrics = stream_metrics
        self._fps_meter = FPSMeter()
        self.latest_snapshot = latest_snapshot

    def process(self, item: InferenceResult) -> AnalyticsFrame:
        """Update tracker and analytics, then draw the same overlays as sync mode."""
        frame_data = item.frame_data
        stage_times = dict(frame_data.stage_timestamps)
        stage_times["tracking_start"] = self.clock.monotonic()
        if item.detected:
            tracks = self.tracker.update(item.detections, frame_data.frame_id)
        else:
            predict = getattr(self.tracker, "predict_only", None)
            tracks = predict(frame_data.frame_id) if callable(predict) else []
        analytics_started = self.clock.monotonic()
        events = self.analytics.update(tracks, frame_data)
        analytics_duration_ms = (self.clock.monotonic() - analytics_started) * 1000.0
        fps = self._fps_meter.tick(self.clock.monotonic())
        snapshot = self.analytics.snapshot()
        self.latest_snapshot.set(snapshot)
        annotated = draw_zones(
            frame_data.frame, self.analytics.config.zones, snapshot.occupancy
        )
        annotated = draw_tracks(annotated, tracks)
        annotated = draw_lines(
            annotated, self.analytics.config.lines, snapshot.line_counts
        )
        totals = {
            "in": sum(value.get("in", 0) for value in snapshot.line_counts.values()),
            "out": sum(value.get("out", 0) for value in snapshot.line_counts.values()),
        }
        totals["net"] = totals["in"] - totals["out"]
        busiest_zone = (
            max(snapshot.occupancy, key=snapshot.occupancy.__getitem__)
            if snapshot.occupancy
            else None
        )
        annotated = draw_overlay_stats(
            annotated,
            fps,
            analytics_duration_ms,
            frame_data.frame_id,
            total_unique=snapshot.unique_visitors,
            line_totals=totals,
            busiest_zone=busiest_zone,
        )
        stage_times["tracking_end"] = self.clock.monotonic()
        frame_data = replace(frame_data, stage_timestamps=stage_times)
        self.stream_metrics.record_processed(self.clock.monotonic())
        return AnalyticsFrame(
            frame_data=frame_data,
            detections=item.detections,
            tracks=tracks,
            events=events,
            snapshot=snapshot,
            annotated=annotated,
        )


class _OutputStage(PipelineStage[AnalyticsFrame]):
    """Publish results and fan out work to independent sinks."""

    def __init__(
        self,
        input_queue: BoundedQueue[Any],
        sinks: list[OutputConsumer],
        latest: LatestValue[AnalyticsFrame],
        event_bus: EventBus,
        metrics: StreamMetrics,
        controller: AdaptiveController,
        latency_queue: BoundedQueue[Any],
        *,
        clock: Clock,
        on_error: Callable[[PipelineStage[Any], Exception], None],
    ) -> None:
        super().__init__(
            "output",
            input_queue=input_queue,
            on_error=on_error,
            clock=clock,
        )
        self.sinks = sinks
        self.latest = latest
        self.event_bus = event_bus
        self.stream_metrics = metrics
        self.controller = controller
        self.latency_queue = latency_queue

    def process(self, result: AnalyticsFrame) -> None:
        """Publish the latest frame and dispatch sinks/events."""
        self.latest.set(result)
        now = self.clock.monotonic()
        capture_ts = result.frame_data.capture_ts
        latency_ms = (
            max(0.0, now - capture_ts) * 1000 if capture_ts is not None else 0.0
        )
        self.stream_metrics.record_output(latency_ms, now)
        self.controller.observe(latency_ms, len(self.latency_queue))
        for event in result.events:
            self.event_bus.publish(event)
        for sink in self.sinks:
            sink.submit(result)

    def on_stop(self) -> None:
        """Close all sinks after the result queue drains."""
        for sink in self.sinks:
            sink.close()


class StreamingPipeline:
    """Threaded video analytics engine with bounded latency and live APIs.

    Threads are appropriate for OpenCV and ONNX Runtime workloads, which release
    the GIL during native execution. Python-heavy workloads may not scale across
    CPU cores; a future multiprocessing implementation would need explicit
    frame serialization, process-safe queues, and model lifecycle management.
    """

    def __init__(
        self,
        config: StreamingConfig | None,
        source: VideoSource,
        detector: BaseDetector,
        tracker: BaseTracker,
        analytics: AnalyticsEngine,
        sinks: list[OutputConsumer] | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.config = config or StreamingConfig()
        self.source = source
        self.detector = detector
        self.tracker = tracker
        self.analytics = analytics
        self.sinks = list(sinks or [])
        self.clock = clock or SystemClock()
        self.controller = AdaptiveController(self.config.controller)
        self.metrics = StreamMetrics(clock=self.clock)
        live = self.config.live
        if live is None:
            live = isinstance(source, (WebcamSource, RTSPSource)) or (
                source.frame_count is None and not isinstance(source, FileVideoSource)
            )
        policy = OverflowPolicy.DROP_OLDEST if live else OverflowPolicy.BLOCK
        self.q1: BoundedQueue[Any] = BoundedQueue(
            self.config.queue_size, policy=policy, clock=self.clock
        )
        self.q2: BoundedQueue[Any] = BoundedQueue(
            self.config.queue_size, policy=policy, clock=self.clock
        )
        self.q3: BoundedQueue[Any] = BoundedQueue(
            self.config.output_queue_size,
            policy=policy,
            clock=self.clock,
        )
        self._latest: LatestValue[AnalyticsFrame] = LatestValue()
        self._latest_snapshot: LatestValue[AnalyticsSnapshot] = LatestValue()
        self._latest_snapshot.set(analytics.snapshot())
        self._encoded_latest: EncodedFrame | None = None
        self._encoded_version = 0
        self._encode_lock = threading.Lock()
        self.event_bus = EventBus()
        self._state_lock = threading.RLock()
        self._state = HealthState.STARTING
        self._health_error: str | None = None
        self._started_at: float | None = None
        self._drop_lock = threading.Lock()
        self._queue_drop_seen = {"q1": 0, "q2": 0, "q3": 0}
        self._source_reconnect_seen = int(getattr(source, "reconnect_count", 0))
        self._stop_requested = threading.Event()
        self._watchdog: threading.Thread | None = None
        self._stages: list[PipelineStage[Any]] = []
        self._capture = CaptureStage(
            source,
            self.q1,
            stream_id=self.config.stream_id,
            clock=self.clock,
            realtime=self.config.realtime,
            live=live,
            reconnect_delay_s=self.config.reconnect_delay_s,
            metrics=self.metrics,
            on_error=self._on_stage_error,
        )
        inference = InferenceStage(
            self.q1,
            self.q2,
            detector,
            self.controller,
            self.q1,
            target_rate_control=bool(live or self.config.realtime),
            on_error=self._on_stage_error,
            clock=self.clock,
        )
        tracking = TrackingAnalyticsStage(
            self.q2,
            self.q3,
            tracker,
            analytics,
            stream_metrics=self.metrics,
            latest_snapshot=self._latest_snapshot,
            on_error=self._on_stage_error,
            clock=self.clock,
        )
        output = _OutputStage(
            self.q3,
            self.sinks,
            self._latest,
            self.event_bus,
            self.metrics,
            self.controller,
            self.q2,
            clock=self.clock,
            on_error=self._on_stage_error,
        )
        self._stages = [self._capture, inference, tracking, output]

    def start(self) -> StreamingPipeline:
        """Start all stage threads and the health watchdog."""
        with self._state_lock:
            if self._started_at is not None:
                raise RuntimeError("streaming pipeline has already been started")
            self._started_at = self.clock.monotonic()
            self._state = HealthState.STARTING
        for stage in reversed(self._stages):
            stage.start()
        self._watchdog = threading.Thread(
            target=self._watch, name=f"watchdog-{self.config.stream_id}", daemon=True
        )
        self._watchdog.start()
        return self

    def stop(self, timeout: float | None = 5.0) -> None:
        """Request graceful shutdown and wait up to ``timeout`` seconds."""
        self._stop_requested.set()
        self._capture.stop()
        try:
            self.source.release()
        except Exception:
            logger.exception("Failed to release source during shutdown")
        deadline = None if timeout is None else time.monotonic() + timeout
        for stage in self._stages:
            if stage.ident is None:
                continue
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            stage.join(remaining)
        if any(stage.is_alive() for stage in self._stages):
            logger.warning(
                "Pipeline %s did not stop before timeout", self.config.stream_id
            )
            self.q1.close()
            self.q2.close()
            self.q3.close()
            for stage in self._stages:
                stage.stop_event.set()
            for stage in self._stages:
                if stage.ident is not None:
                    stage.join(0.1)
        if (
            self._watchdog is not None
            and threading.current_thread() is not self._watchdog
        ):
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            self._watchdog.join(remaining)
        with self._state_lock:
            if self._state is not HealthState.FAILED:
                self._state = HealthState.STOPPED
        self.event_bus.close()

    def join(self, timeout: float | None = None) -> None:
        """Wait for natural end-of-stream or a requested shutdown."""
        deadline = None if timeout is None else time.monotonic() + timeout
        for stage in self._stages:
            if stage.ident is not None:
                remaining = (
                    None if deadline is None else max(0.0, deadline - time.monotonic())
                )
                stage.join(remaining)
        if self._watchdog is not None:
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            self._watchdog.join(remaining)
        if all(not stage.is_alive() for stage in self._stages):
            self.event_bus.close()
            with self._state_lock:
                if self._state is not HealthState.FAILED:
                    self._state = HealthState.STOPPED

    def _on_stage_error(self, stage: PipelineStage[Any], error: Exception) -> None:
        """Mark pipeline failed and close queues to wake downstream stages."""
        if self._stop_requested.is_set() and isinstance(error, QueueClosedError):
            return
        with self._state_lock:
            self._state = HealthState.FAILED
            self._health_error = f"{stage.name}: {error}"
        self._stop_requested.set()
        for queue in (self.q1, self.q2, self.q3):
            queue.close()
        for other in self._stages:
            if other is not stage:
                other.stop_event.set()

    def _watch(self) -> None:
        """Monitor stage progress, exceptions, drops, and source reconnects."""
        last_metrics_log = time.monotonic()
        while not self._stop_requested.wait(self.config.watchdog_interval_s):
            sink_errors = [
                (sink, getattr(sink, "error", None))
                for sink in self.sinks
                if getattr(sink, "error", None) is not None
            ]
            if sink_errors:
                with self._state_lock:
                    self._state = HealthState.FAILED
                    self._health_error = f"output sink failed: {sink_errors[0][1]}"
                self._stop_requested.set()
                break
            if any(stage.exception is not None for stage in self._stages):
                with self._state_lock:
                    self._state = HealthState.FAILED
                self._stop_requested.set()
                break
            source_reconnect_count = int(getattr(self.source, "reconnect_count", 0))
            if source_reconnect_count > self._source_reconnect_seen:
                self.metrics.record_reconnect(
                    source_reconnect_count - self._source_reconnect_seen
                )
                self._source_reconnect_seen = source_reconnect_count
            queues = self._queue_map()
            total_dropped = 0
            for name, queue in queues.items():
                stats = queue.stats
                self.metrics.record_queue(
                    name,
                    stats.current_depth,
                    stats.high_water_mark,
                    queue.policy.value,
                )
                self._record_new_drops(name, stats.dropped_count)
                total_dropped += stats.dropped_count
            active = [stage for stage in self._stages if stage.is_alive()]
            now = self.clock.monotonic()
            stale = [
                stage
                for stage in active
                if stage.metrics.last_progress_at is not None
                and now - stage.metrics.last_progress_at > self.config.stalled_after_s
            ]
            pending = any(len(queue) > 0 for queue in queues.values())
            capture_stale = (
                self._capture.live
                and self._capture.metrics.last_progress_at is not None
                and now - self._capture.metrics.last_progress_at
                > self.config.stalled_after_s
            )
            if (pending and stale) or capture_stale:
                state = HealthState.STALLED
            elif total_dropped or self._is_reconnecting():
                state = HealthState.DEGRADED
            elif active and all(stage.started_event.is_set() for stage in active):
                state = HealthState.RUNNING
            else:
                state = HealthState.STARTING
            with self._state_lock:
                if self._state not in (HealthState.FAILED, HealthState.STOPPED):
                    self._state = state
            if (
                self.config.metrics_log_interval_s > 0
                and time.monotonic() - last_metrics_log
                >= self.config.metrics_log_interval_s
            ):
                snapshot = self.get_metrics()
                logger.info(
                    "Stream %s: input %.2f FPS, output %.2f FPS, p95 %.2f ms, "
                    "dropped %d, stride %d",
                    self.config.stream_id,
                    snapshot["input_fps"],
                    snapshot["output_fps"],
                    snapshot["end_to_end_latency"]["p95_ms"],
                    snapshot["frames_dropped"],
                    snapshot["current_stride"],
                )
                last_metrics_log = time.monotonic()
            if not active:
                with self._state_lock:
                    if self._state is not HealthState.FAILED:
                        self._state = HealthState.STOPPED
                self._stop_requested.set()
                break

    def _record_new_drops(self, name: str, dropped_count: int) -> None:
        """Record queue-drop counter deltas exactly once across readers."""
        with self._drop_lock:
            previous = self._queue_drop_seen.get(name, 0)
            self._queue_drop_seen[name] = dropped_count
        if dropped_count > previous:
            self.metrics.record_drop(name, dropped_count - previous)

    def _queue_map(self) -> dict[str, BoundedQueue[Any]]:
        """Return pipeline and sink queues for consistent metric collection."""
        queues: dict[str, BoundedQueue[Any]] = {
            "q1": self.q1,
            "q2": self.q2,
            "q3": self.q3,
        }
        for index, sink in enumerate(self.sinks):
            sink_queue = getattr(sink, "queue", None)
            if isinstance(sink_queue, BoundedQueue):
                name = getattr(sink, "name", f"sink_{index}")
                queues[f"sink_{name}"] = sink_queue
        return queues

    def _is_reconnecting(self) -> bool:
        """Return whether capture or the source itself is reconnecting."""
        return self._capture.reconnecting or bool(
            getattr(self.source, "reconnecting", False)
        )

    def get_latest_frame(self) -> EncodedFrame | None:
        """Lazily JPEG-encode and cache the latest annotated frame."""
        version, result = self._latest.get()
        if result is None:
            return None
        with self._encode_lock:
            if self._encoded_version == version and self._encoded_latest is not None:
                return self._encoded_latest
            ok, data = cv2.imencode(".jpg", result.annotated)
            if not ok:
                raise RuntimeError(
                    f"Unable to encode stream frame {result.frame_data.frame_id}"
                )
            capture_ts = result.frame_data.capture_ts
            if capture_ts is None:
                capture_ts = result.frame_data.timestamp_s
            self._encoded_latest = EncodedFrame(
                stream_id=result.frame_data.stream_id,
                frame_id=result.frame_data.frame_id,
                capture_ts=capture_ts,
                jpeg=data.tobytes(),
                encoded_ts=self.clock.monotonic(),
            )
            self._encoded_version = version
            return self._encoded_latest

    def wait_for_frame(
        self, version: int, timeout: float | None = None
    ) -> EncodedFrame | None:
        """Wait for a newer annotated frame and return its encoded form."""
        new_version = self._latest.wait_for_update(version, timeout)
        return self.get_latest_frame() if new_version > version else None

    def subscribe_events(
        self,
        consumer: Callable[[Any], None] | Queue[Any] | BoundedQueue[Any],
        *,
        maxsize: int = 64,
    ) -> Callable[[], None]:
        """Subscribe a callback or queue to analytics events."""
        return self.event_bus.subscribe(consumer, maxsize=maxsize)

    def get_snapshot(self) -> Any:
        """Return the analytics engine's current immutable snapshot."""
        _, snapshot = self._latest_snapshot.get()
        return snapshot

    def get_metrics(self) -> dict[str, Any]:
        """Return stream and stage metrics as a thread-safe dictionary."""
        stages = {stage.name: stage.metrics_snapshot() for stage in self._stages}
        for stage in self._stages:
            stage_metrics = stages[stage.name]
            self.metrics.record_stage_summary(
                stage.name,
                {
                    "mean_ms": float(stage_metrics["avg_ms"] or 0.0),
                    "p50_ms": float(stage_metrics["p50_ms"] or 0.0),
                    "p95_ms": float(stage_metrics["p95_ms"] or 0.0),
                    "p99_ms": float(stage_metrics["p99_ms"] or 0.0),
                },
                float(stage_metrics["utilization_pct"] or 0.0),
            )
        self.metrics.set_stride(self.controller.stride)
        for name, queue in self._queue_map().items():
            stats = queue.stats
            self.metrics.record_queue(
                name,
                stats.current_depth,
                stats.high_water_mark,
                queue.policy.value,
            )
            self._record_new_drops(name, stats.dropped_count)
        return self.metrics.snapshot(stages)

    def get_health(self) -> dict[str, Any]:
        """Return state, stream identifier, progress, and failure context."""
        with self._state_lock:
            state = self._state
            error = self._health_error
        return {
            "stream_id": self.config.stream_id,
            "state": state.value,
            "error": error,
            "started_at": self._started_at,
            "stages": {
                stage.name: {
                    "alive": stage.is_alive(),
                    "processed": stage.metrics_snapshot()["processed"],
                    "last_progress_at": stage.metrics_snapshot()["last_progress_at"],
                    "reconnecting": self._is_reconnecting()
                    if stage is self._capture
                    else False,
                }
                for stage in self._stages
            },
        }

    def __enter__(self) -> StreamingPipeline:
        """Start and return the pipeline."""
        return self.start()

    def __exit__(self, *_: object) -> None:
        """Stop the pipeline on leaving its context."""
        self.stop()
