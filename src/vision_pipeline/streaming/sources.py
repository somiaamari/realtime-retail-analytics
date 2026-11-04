"""Video source implementations for files, webcams, and RTSP streams."""

from __future__ import annotations

import os
import random
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np
from numpy.typing import NDArray

from vision_pipeline.config import VideoConfig
from vision_pipeline.logging_utils import get_logger
from vision_pipeline.types import FrameData

logger = get_logger(__name__)
_FFMPEG_OPTIONS_LOCK = threading.Lock()
_LOW_LATENCY_FFMPEG_OPTIONS = "rtsp_transport;udp|fflags;nobuffer|flags;low_delay"


def _create_cv_capture(
    source: str | int,
    *,
    low_latency: bool = False,
) -> cv2.VideoCapture:
    """Create a capture handle while safely scoping optional FFmpeg settings."""
    with _FFMPEG_OPTIONS_LOCK:
        previous = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
        if low_latency and not previous:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _LOW_LATENCY_FFMPEG_OPTIONS
        try:
            return cv2.VideoCapture(source)
        finally:
            if low_latency and not previous:
                os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)


class VideoSourceError(RuntimeError):
    """Raised when a video source cannot be opened or decoded."""


class VideoSource(ABC):
    """Abstract context-managed and iterable video source."""

    @abstractmethod
    def open(self) -> None:
        """Open the underlying video source."""

    @abstractmethod
    def read(self) -> FrameData | None:
        """Read a frame, returning ``None`` when a finite source reaches EOF."""

    @abstractmethod
    def release(self) -> None:
        """Release source resources."""

    @property
    @abstractmethod
    def fps(self) -> float:
        """Nominal source frames per second."""

    @property
    @abstractmethod
    def width(self) -> int:
        """Decoded frame width in pixels."""

    @property
    @abstractmethod
    def height(self) -> int:
        """Decoded frame height in pixels."""

    @property
    @abstractmethod
    def frame_count(self) -> int | None:
        """Number of source frames, or ``None`` for live/unknown sources."""

    def __enter__(self) -> VideoSource:
        """Open the source and return this instance."""
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        """Release the source on leaving its context."""
        self.release()

    def __iter__(self) -> Iterator[FrameData]:
        """Yield decoded frames, opening the source if necessary."""
        self.open()
        return self

    def __next__(self) -> FrameData:
        """Return the next decoded frame or stop at end-of-stream."""
        frame_data = self.read()
        if frame_data is None:
            self.release()
            raise StopIteration
        return frame_data


class _CaptureVideoSource(VideoSource):
    """Shared OpenCV capture and frame metadata behavior."""

    def __init__(self, config: VideoConfig) -> None:
        self.config = config
        self._capture: cv2.VideoCapture | None = None
        self._fps = float(config.fps)
        self._width = 0
        self._height = 0
        self._frame_count: int | None = None
        self._frame_id = 0
        self._started_at: float | None = None

    @abstractmethod
    def _open_capture(self) -> cv2.VideoCapture:
        """Create and open the OpenCV capture handle."""

    def open(self) -> None:
        """Open and cache stream properties."""
        if self._capture is not None and self._capture.isOpened():
            return
        try:
            capture = self._open_capture()
        except cv2.error as error:
            raise VideoSourceError(f"Unable to open video source: {self!s}") from error
        if not capture.isOpened():
            capture.release()
            raise VideoSourceError(f"Unable to open video source: {self!s}")
        if self.config.low_latency:
            set_property = getattr(capture, "set", None)
            if callable(set_property):
                set_property(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._capture = capture
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if np.isfinite(source_fps) and source_fps > 0:
            self._fps = source_fps
        self._width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self._frame_count = count if count > 0 else None
        self._frame_id = 0
        self._started_at = time.perf_counter()
        logger.info(
            "Opened video source %s (%dx%d at %.3f FPS)",
            self,
            self._width,
            self._height,
            self._fps,
        )

    def read(self) -> FrameData | None:
        """Read and validate one frame from the capture handle."""
        if self._capture is None:
            self.open()
        assert self._capture is not None
        try:
            ok, frame = self._capture.read()
        except cv2.error as error:
            raise VideoSourceError(
                f"Unable to read frame from video source: {self!s}"
            ) from error
        if not ok:
            return None
        if (
            not isinstance(frame, np.ndarray)
            or frame.ndim != 3
            or frame.shape[2] != 3
            or frame.size == 0
        ):
            raise VideoSourceError(f"Corrupted frame from video source: {self!s}")
        frame = np.asarray(frame, dtype=np.uint8)
        timestamp = float(self._capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
        if timestamp <= 0:
            timestamp = self._frame_id / self._fps
        if self.config.resize:
            frame = self._resize(frame)
        data = FrameData(
            frame=frame,
            frame_id=self._frame_id,
            timestamp_s=timestamp,
            source_fps=self._fps,
        )
        self._frame_id += 1
        return data

    def _resize(self, frame: NDArray[Any]) -> NDArray[Any]:
        """Fit a frame within configured dimensions without changing its ratio."""
        frame_height, frame_width = frame.shape[:2]
        scale = min(
            self.config.width / frame_width,
            self.config.height / frame_height,
        )
        if scale == 1:
            return frame
        new_size = (
            max(1, round(frame_width * scale)),
            max(1, round(frame_height * scale)),
        )
        resized = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
        return np.asarray(resized, dtype=np.uint8)

    def release(self) -> None:
        """Release the capture handle and clear its state."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            logger.info("Released video source %s", self)

    @property
    def fps(self) -> float:
        """Nominal source frames per second."""
        return self._fps

    @property
    def width(self) -> int:
        """Decoded frame width in pixels."""
        if self.config.resize and self._width > 0 and self._height > 0:
            scale = min(
                self.config.width / self._width,
                self.config.height / self._height,
            )
            return max(1, round(self._width * scale))
        return self._width

    @property
    def height(self) -> int:
        """Decoded frame height in pixels."""
        if self.config.resize and self._width > 0 and self._height > 0:
            scale = min(
                self.config.width / self._width,
                self.config.height / self._height,
            )
            return max(1, round(self._height * scale))
        return self._height

    @property
    def frame_count(self) -> int | None:
        """Number of source frames, or ``None`` when unknown."""
        return self._frame_count


class FileVideoSource(_CaptureVideoSource):
    """Read frames from a local video file."""

    def __init__(
        self,
        path: str | Path,
        config: VideoConfig | None = None,
    ) -> None:
        super().__init__(config or VideoConfig(source=str(path)))
        self.path = Path(path)

    def _open_capture(self) -> cv2.VideoCapture:
        if not self.path.is_file():
            raise VideoSourceError(f"Video file does not exist: {self.path}")
        return _create_cv_capture(str(self.path))

    def __str__(self) -> str:
        """Return the source path for logs and errors."""
        return str(self.path)


class WebcamSource(_CaptureVideoSource):
    """Read frames from a webcam device index."""

    def __init__(
        self,
        device_index: int,
        config: VideoConfig | None = None,
    ) -> None:
        if device_index < 0:
            raise ValueError("Webcam device index must be non-negative.")
        super().__init__(config or VideoConfig(source=device_index))
        self.device_index = device_index

    def _open_capture(self) -> cv2.VideoCapture:
        return _create_cv_capture(self.device_index)

    @property
    def frame_count(self) -> None:
        """Live camera sources have no known frame count."""
        return None

    def __str__(self) -> str:
        """Return a human-readable webcam identifier."""
        return f"webcam:{self.device_index}"


class RTSPSource(_CaptureVideoSource):
    """Read frames from an RTSP URL with bounded exponential reconnection."""

    def __init__(
        self,
        url: str,
        config: VideoConfig | None = None,
        *,
        max_retries: int | None = None,
        reconnect_delay_s: float | None = None,
    ) -> None:
        if not url.lower().startswith(("rtsp://", "rtsps://")):
            raise ValueError("RTSP source URL must start with rtsp:// or rtsps://.")
        source_config = config or VideoConfig(source=url)
        super().__init__(source_config)
        self.url = url
        self.max_retries = (
            source_config.max_retries if max_retries is None else max_retries
        )
        self.reconnect_delay_s = (
            source_config.reconnect_delay_s
            if reconnect_delay_s is None
            else reconnect_delay_s
        )
        self.reconnecting = False
        self.reconnect_count = 0
        if self.max_retries < 0 or self.reconnect_delay_s < 0:
            raise ValueError("Retry count and reconnect delay must be non-negative.")

    def _open_capture(self) -> cv2.VideoCapture:
        last_error: cv2.error | None = None
        for attempt in range(self.max_retries + 1):
            try:
                capture = self._create_capture()
            except cv2.error as error:
                last_error = error
                capture = None
            if capture is not None and capture.isOpened():
                self.reconnecting = False
                return capture
            if capture is not None:
                capture.release()
            self._mark_reconnecting()
            if attempt < self.max_retries:
                delay = self.reconnect_delay_s * (2**attempt)
                logger.warning(
                    "RTSP open attempt %d/%d failed; retrying in %.2f seconds",
                    attempt + 1,
                    self.max_retries + 1,
                    delay,
                )
                time.sleep(delay)
        source_error = VideoSourceError(
            f"Unable to open RTSP stream after {self.max_retries + 1} attempts: "
            f"{self!s}"
        )
        if last_error is not None:
            raise source_error from last_error
        raise source_error

    def _create_capture(self) -> cv2.VideoCapture:
        """Create an RTSP capture with optional low-latency FFmpeg options."""
        return _create_cv_capture(self.url, low_latency=self.config.low_latency)

    def read(self) -> FrameData | None:
        """Read a frame, reconnecting after a failed live-stream read."""
        for attempt in range(self.max_retries + 1):
            try:
                frame = super().read()
            except VideoSourceError:
                self.release()
                self._mark_reconnecting()
                if attempt >= self.max_retries:
                    raise
                frame = None
            if frame is not None:
                self.reconnecting = False
                return frame
            self.release()
            self._mark_reconnecting()
            if attempt < self.max_retries:
                delay = self.reconnect_delay_s * (2**attempt)
                logger.warning("RTSP read failed; reconnecting in %.2f seconds", delay)
                time.sleep(delay)
                next_frame_id = self._frame_id
                self.open()
                self._frame_id = next_frame_id
        raise VideoSourceError(
            f"RTSP stream stopped after {self.max_retries + 1} read attempts: "
            f"{self.url}"
        )

    def _mark_reconnecting(self) -> None:
        """Track an RTSP disconnect episode for health and metrics."""
        if not self.reconnecting:
            self.reconnecting = True
            self.reconnect_count += 1

    @property
    def frame_count(self) -> None:
        """Live RTSP sources have no known frame count."""
        return None

    def __str__(self) -> str:
        """Return a credential-free stream URL for logs and errors."""
        parsed = urlsplit(self.url)
        netloc = parsed.netloc.rsplit("@", maxsplit=1)[-1]
        return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


class FakeSource(VideoSource):
    """Synthetic source with deterministic timing for tests and demos."""

    def __init__(
        self,
        *,
        frames: list[FrameData] | None = None,
        fps: float = 10.0,
        width: int = 64,
        height: int = 48,
        infinite: bool = False,
        frame_generator: Any | None = None,
        jitter: float = 0.0,
        burst: int = 0,
        dropout_windows: list[tuple[float, float]] | None = None,
        clock: Any | None = None,
    ) -> None:
        self._frames = list(frames or [])
        self._fps = float(fps)
        self._width = int(width)
        self._height = int(height)
        self._infinite = infinite
        self._frame_generator = frame_generator
        self._jitter = float(jitter)
        self._burst = max(0, int(burst))
        self._dropout_windows = list(dropout_windows or [])
        self._clock = clock
        self._index = 0
        self._frame_id = 0
        self._time = 0.0
        self._random = random.Random(0)
        self._stop_event = threading.Event()

    def open(self) -> None:
        """Reset the fake source so it can be iterated from the start."""
        self._index = 0
        self._frame_id = 0
        self._time = self._clock.monotonic() if self._clock is not None else 0.0
        self._random.seed(0)
        self._stop_event.clear()

    def read(self) -> FrameData | None:
        """Return the next synthetic frame or signal EOF for finite sources."""
        if self._frame_generator is not None:
            generated = self._frame_generator(self._frame_id)
            if generated is None:
                return None
            if isinstance(generated, FrameData):
                result = generated
            else:
                result = FrameData(
                    frame=np.asarray(generated, dtype=np.uint8),
                    frame_id=self._frame_id,
                    timestamp_s=self._time,
                    source_fps=self._fps,
                )
            self._frame_id += 1
            self._time += 1.0 / self._fps
            return result

        if self._infinite:
            interval = max(
                0.0,
                1.0 / self._fps + self._random.uniform(-self._jitter, self._jitter),
            )
            if self._burst > 1:
                interval = (
                    interval * self._burst if self._frame_id % self._burst == 0 else 0.0
                )
            if self._clock is not None:
                self._clock.sleep(interval)
                self._time = self._clock.monotonic()
            else:
                if self._stop_event.wait(interval):
                    return None
                self._time += interval
            if self._dropout_windows:
                for start, end in self._dropout_windows:
                    if start <= self._time < end:
                        return None
            frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
            if self._burst > 1 and self._frame_id % self._burst == 0:
                frame[...] = 255
            data = FrameData(
                frame=frame,
                frame_id=self._frame_id,
                timestamp_s=self._time,
                source_fps=self._fps,
            )
            self._frame_id += 1
            return data

        if self._index >= len(self._frames):
            return None
        source_frame = self._frames[self._index]
        self._index += 1
        self._frame_id = source_frame.frame_id + 1
        return FrameData(
            frame=np.asarray(source_frame.frame, dtype=np.uint8),
            frame_id=source_frame.frame_id,
            timestamp_s=source_frame.timestamp_s,
            source_fps=source_frame.source_fps,
        )

    def release(self) -> None:
        """Release the fake source; there are no external resources to clean up."""
        self._stop_event.set()

    @property
    def fps(self) -> float:
        """Return the configured nominal fake-source frame rate."""
        return self._fps

    @property
    def width(self) -> int:
        """Return the configured synthetic frame width."""
        return self._width

    @property
    def height(self) -> int:
        """Return the configured synthetic frame height."""
        return self._height

    @property
    def frame_count(self) -> int | None:
        """Return the finite frame count for non-infinite fake sources."""
        if self._infinite:
            return None
        return len(self._frames)

    def __str__(self) -> str:
        """Return a readable fake-source identifier."""
        return "fake-source"


def create_source(config: VideoConfig) -> VideoSource:
    """Create a video source based on ``config.source``.

    Integer values and numeric strings select a webcam; RTSP URLs select an
    RTSP source; all other strings are interpreted as local file paths.
    """
    source = config.source
    if isinstance(source, int):
        return WebcamSource(source, config)
    source_text = source.strip()
    if source_text.isdecimal():
        return WebcamSource(int(source_text), config)
    if source_text.lower().startswith(("rtsp://", "rtsps://")):
        return RTSPSource(source_text, config)
    return FileVideoSource(Path(source_text), config)
