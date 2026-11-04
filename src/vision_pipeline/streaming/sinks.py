"""Asynchronous output sinks for annotated streaming results."""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

import cv2

from vision_pipeline.analytics.events import EventSink
from vision_pipeline.logging_utils import get_logger
from vision_pipeline.streaming.queues import (
    BoundedQueue,
    EmptyQueueError,
    LatestValue,
    OverflowPolicy,
    SENTINEL,
)
from vision_pipeline.streaming.writer import VideoWriter

logger = get_logger(__name__)


@dataclass(frozen=True)
class EncodedFrame:
    """JPEG representation of one annotated source frame."""

    stream_id: str
    frame_id: int
    capture_ts: float
    jpeg: bytes
    encoded_ts: float


class OutputConsumer(Protocol):
    """Public sink contract consumed by :class:`StreamingPipeline`."""

    def submit(self, result: Any) -> None:
        """Queue one analytics result for output."""

    def close(self) -> None:
        """Finish consuming results and release output resources."""


class OutputSink(ABC):
    """Independent worker with a bounded drop-oldest submission queue."""

    def __init__(self, name: str, *, queue_size: int = 2) -> None:
        if queue_size < 1:
            raise ValueError("queue_size must be positive")
        self.name = name
        self.queue: BoundedQueue[Any] = BoundedQueue(
            queue_size, policy=OverflowPolicy.DROP_OLDEST
        )
        self._closed = threading.Event()
        self._error: Exception | None = None
        self._thread = threading.Thread(
            target=self._run, name=f"sink-{name}", daemon=True
        )
        self._thread.start()

    def submit(self, result: Any) -> None:
        """Submit a result without allowing slow I/O to block the pipeline."""
        if self._closed.is_set():
            return
        try:
            self.queue.put_nowait(result)
        except Exception:
            logger.exception("Unable to submit result to sink %s", self.name)
            raise

    def _run(self) -> None:
        """Consume queued results and close sink resources."""
        try:
            while True:
                try:
                    result = self.queue.get(timeout=0.1)
                except EmptyQueueError:
                    continue
                if result is SENTINEL:
                    break
                self.handle(result)
        except Exception as error:
            self._error = error
            logger.exception("Output sink %s failed", self.name)
        finally:
            try:
                self.on_close()
            except Exception:
                logger.exception("Output sink %s failed during close", self.name)
            self._closed.set()

    @abstractmethod
    def handle(self, result: Any) -> None:
        """Write one result to this sink."""

    def on_close(self) -> None:
        """Release resources after queued results have drained."""

    def close(self, timeout: float | None = 5.0) -> None:
        """Stop the worker after draining its bounded queue."""
        if self._closed.is_set():
            return
        self.queue.close()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout)

    @property
    def error(self) -> Exception | None:
        """Return a worker exception, if one occurred."""
        return self._error

    @property
    def alive(self) -> bool:
        """Return whether the sink worker is still running."""
        return self._thread.is_alive()


class VideoFileSink(OutputSink):
    """Asynchronously write annotated images with the shared video writer."""

    def __init__(
        self,
        path: str,
        fps: float,
        width: int,
        height: int,
        *,
        codec: str = "mp4v",
        queue_size: int = 4,
    ) -> None:
        self.writer = VideoWriter(path, fps, width, height, codec=codec)
        super().__init__("video", queue_size=queue_size)

    def handle(self, result: Any) -> None:
        """Write the annotated result image."""
        self.writer.write(result.annotated)

    def on_close(self) -> None:
        """Release the OpenCV video writer."""
        self.writer.release()


class JpegStreamSink(OutputSink):
    """Publish the latest JPEG and provide a multipart MJPEG iterator."""

    def __init__(
        self,
        *,
        max_width: int | None = None,
        quality: int = 85,
        queue_size: int = 1,
        clock: Any | None = None,
    ) -> None:
        if max_width is not None and max_width < 1:
            raise ValueError("max_width must be positive")
        if not 0 <= quality <= 100:
            raise ValueError("JPEG quality must be between 0 and 100")
        self.max_width = max_width
        self.quality = quality
        self.clock = clock
        self.latest: LatestValue[EncodedFrame] = LatestValue()
        super().__init__("jpeg", queue_size=queue_size)

    def handle(self, result: Any) -> None:
        """Resize and encode a result as JPEG."""
        frame = result.annotated
        if self.max_width is not None and frame.shape[1] > self.max_width:
            height = max(1, round(frame.shape[0] * self.max_width / frame.shape[1]))
            frame = cv2.resize(
                frame, (self.max_width, height), interpolation=cv2.INTER_AREA
            )
        ok, encoded = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        )
        if not ok:
            raise RuntimeError(
                f"Unable to encode frame {result.frame_data.frame_id} as JPEG"
            )
        capture_ts = result.frame_data.capture_ts
        if capture_ts is None:
            capture_ts = result.frame_data.timestamp_s
        now = self.clock.monotonic() if self.clock is not None else time.monotonic()
        self.latest.set(
            EncodedFrame(
                stream_id=result.frame_data.stream_id,
                frame_id=result.frame_data.frame_id,
                capture_ts=capture_ts,
                jpeg=encoded.tobytes(),
                encoded_ts=now,
            )
        )

    def multipart_chunks(
        self,
        *,
        boundary: str = "frame",
        timeout: float | None = None,
    ) -> Iterator[bytes]:
        """Yield MJPEG multipart chunks whenever a newer encoded frame exists."""
        version = 0
        while not self._closed.is_set():
            new_version = self.latest.wait_for_update(version, timeout)
            if new_version <= version:
                continue
            version, frame = self.latest.get()
            if frame is None:
                continue
            yield (
                (
                    f"--{boundary}\r\n"
                    "Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(frame.jpeg)}\r\n\r\n"
                ).encode("ascii")
                + frame.jpeg
                + b"\r\n"
            )


class EventLogSink(OutputSink):
    """Write analytics events asynchronously through an existing event sink."""

    def __init__(self, event_sink: EventSink, *, queue_size: int = 8) -> None:
        self.event_sink = event_sink
        super().__init__("events", queue_size=queue_size)

    def handle(self, result: Any) -> None:
        """Append all events emitted for one processed frame."""
        for event in result.events:
            self.event_sink.write(event)

    def on_close(self) -> None:
        """Flush and close the wrapped event sink."""
        self.event_sink.close()


class WindowSink:
    """Display frames on the main thread; callers must invoke ``pump`` there."""

    def __init__(self, window_name: str = "Vision Pipeline", *, queue_size: int = 1):
        self.window_name = window_name
        self.queue: BoundedQueue[Any] = BoundedQueue(
            queue_size, policy=OverflowPolicy.DROP_OLDEST
        )
        self._closed = False
        self._displayed = False

    def submit(self, result: Any) -> None:
        """Queue a result for display without waiting for the UI thread."""
        if not self._closed:
            self.queue.put_nowait(result)

    def pump(self, delay_ms: int = 1) -> bool:
        """Display the most recent pending image; return false when q is pressed."""
        newest: Any | None = None
        while True:
            try:
                item = self.queue.get(block=False)
            except EmptyQueueError:
                break
            if item is SENTINEL:
                break
            newest = item
        if newest is not None:
            cv2.imshow(self.window_name, newest.annotated)
            self._displayed = True
        return (cv2.waitKey(delay_ms) & 0xFF) != ord("q")

    def close(self) -> None:
        """Close the UI window; call from the thread that owns the window."""
        if not self._closed:
            self._closed = True
            self.queue.close()
            if self._displayed:
                cv2.destroyWindow(self.window_name)
