"""Threaded stage framework and per-stage timing metrics."""

from __future__ import annotations

import threading
import time
import traceback
from math import ceil
from abc import ABC
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from vision_pipeline.logging_utils import get_logger
from vision_pipeline.streaming.clock import Clock, SystemClock
from vision_pipeline.streaming.queues import (
    BoundedQueue,
    EmptyQueueError,
    _Sentinel,
)

logger = get_logger(__name__)
T = TypeVar("T")


@dataclass
class StageMetrics:
    """Thread-safe bounded-window metrics for a pipeline stage."""

    window_size: int = 256
    processed: int = 0
    errors: int = 0
    idle_time_ms: float = 0.0
    last_progress_at: float | None = None
    last_error: str | None = None
    _samples: deque[float] = field(default_factory=deque, repr=False)
    _active_seconds: float = 0.0
    _started_at: float | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, duration_ms: float, started_at: float) -> None:
        """Record one completed item."""
        with self._lock:
            if self.window_size < 1:
                self.window_size = 1
            if self._samples.maxlen != self.window_size:
                self._samples = deque(self._samples, maxlen=self.window_size)
            self._samples.append(max(0.0, duration_ms))
            self.processed += 1
            self._active_seconds += max(0.0, duration_ms) / 1000.0
            self.last_progress_at = time.monotonic()
            if self._started_at is None:
                self._started_at = started_at

    def record_idle(self, duration_ms: float) -> None:
        """Add time spent waiting for input."""
        with self._lock:
            self.idle_time_ms += max(0.0, duration_ms)

    def record_error(self, error_text: str) -> None:
        """Record one stage failure and its traceback."""
        with self._lock:
            self.errors += 1
            self.last_error = error_text

    def snapshot(self) -> dict[str, float | int | None | str]:
        """Return a consistent JSON-compatible metric snapshot."""
        with self._lock:
            samples = sorted(self._samples)
            elapsed = (
                max(time.monotonic() - self._started_at, 1e-9)
                if self._started_at is not None
                else 0.0
            )
            mean = sum(samples) / len(samples) if samples else 0.0

            def percentile(fraction: float) -> float:
                if not samples:
                    return 0.0
                index = max(0, min(len(samples) - 1, ceil(len(samples) * fraction) - 1))
                return samples[index]

            return {
                "processed": self.processed,
                "errors": self.errors,
                "avg_ms": mean,
                "p50_ms": percentile(0.50),
                "p95_ms": percentile(0.95),
                "p99_ms": percentile(0.99),
                "idle_time_ms": self.idle_time_ms,
                "utilization_pct": min(100.0, self._active_seconds / elapsed * 100.0)
                if elapsed
                else 0.0,
                "last_progress_at": self.last_progress_at,
                "last_error": self.last_error,
            }


class PipelineStage(threading.Thread, ABC, Generic[T]):
    """Run a ``process`` function on items from a bounded queue.

    Exceptions are captured and reported to ``on_error``; downstream queues
    are closed in all exit paths so a supervisor can shut down cleanly.
    """

    def __init__(
        self,
        name: str,
        *,
        input_queue: BoundedQueue[Any] | None = None,
        output_queue: BoundedQueue[Any] | None = None,
        on_error: Any | None = None,
        queue_timeout: float = 0.1,
        clock: Clock | None = None,
    ) -> None:
        super().__init__(name=name, daemon=True)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.on_error = on_error
        self.queue_timeout = queue_timeout
        self.clock = clock or SystemClock()
        self.stop_event = threading.Event()
        self.started_event = threading.Event()
        self.metrics = StageMetrics()
        self._exception: Exception | None = None

    def run(self) -> None:
        """Execute the stage lifecycle and close downstream on exit."""
        try:
            self.on_start()
            self.started_event.set()
            while not self.stop_event.is_set():
                wait_started = time.monotonic()
                try:
                    item = self.next_item()
                except EmptyQueueError:
                    self.metrics.record_idle((time.monotonic() - wait_started) * 1000)
                    continue
                if isinstance(item, _Sentinel):
                    break
                if item is None:
                    continue
                started = time.monotonic()
                result = self.process(item)
                self.metrics.record((time.monotonic() - started) * 1000, started)
                if result is not None and self.output_queue is not None:
                    self.output_queue.put(result)
        except Exception as error:
            self._exception = error
            error_text = traceback.format_exc()
            self.metrics.record_error(error_text)
            logger.error("Stage %s failed\n%s", self.name, error_text)
            if self.on_error is not None:
                self.on_error(self, error)
        finally:
            try:
                self.on_stop()
            except Exception:
                logger.exception("Stage %s failed during shutdown", self.name)
            if self.output_queue is not None:
                self.output_queue.close()
            self.started_event.set()

    def next_item(self) -> T | _Sentinel:
        """Read an item from the input queue; subclasses may supply their own."""
        if self.input_queue is None:
            raise RuntimeError(f"{self.name} has no input queue")
        try:
            return self.input_queue.get(timeout=self.queue_timeout)
        except EmptyQueueError:
            raise

    def process(self, item: T) -> Any | None:
        """Process one item and optionally return an output item."""
        raise NotImplementedError

    def on_start(self) -> None:
        """Hook called in the stage thread before processing begins."""

    def on_stop(self) -> None:
        """Hook called in the stage thread after processing ends."""

    def stop(self) -> None:
        """Request stage shutdown and wake a queue waiter if possible."""
        self.stop_event.set()
        if self.input_queue is not None:
            self.input_queue.close()

    @property
    def exception(self) -> Exception | None:
        """Return the exception that stopped this stage, if any."""
        return self._exception

    def metrics_snapshot(self) -> dict[str, float | int | None | str]:
        """Return the current stage metrics."""
        return self.metrics.snapshot()
