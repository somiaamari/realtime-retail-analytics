"""Video source implementations for files, webcams, and RTSP streams."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from vision_pipeline.config import VideoConfig
from vision_pipeline.logging_utils import get_logger
from vision_pipeline.types import FrameData

logger = get_logger(__name__)


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
        capture = self._open_capture()
        if not capture.isOpened():
            capture.release()
            raise VideoSourceError(f"Unable to open video source: {self!s}")
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
        ok, frame = self._capture.read()
        if not ok:
            return None
        if (
            not isinstance(frame, np.ndarray)
            or frame.ndim != 3
            or frame.shape[2] != 3
            or frame.size == 0
        ):
            raise VideoSourceError(f"Corrupted frame from video source: {self!s}")
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

    def _resize(self, frame: np.ndarray) -> np.ndarray:
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
        return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

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
        return cv2.VideoCapture(str(self.path))

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
        return cv2.VideoCapture(self.device_index)

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
        if self.max_retries < 0 or self.reconnect_delay_s < 0:
            raise ValueError("Retry count and reconnect delay must be non-negative.")

    def _open_capture(self) -> cv2.VideoCapture:
        for attempt in range(self.max_retries + 1):
            capture = cv2.VideoCapture(self.url)
            if capture.isOpened():
                return capture
            capture.release()
            if attempt < self.max_retries:
                delay = self.reconnect_delay_s * (2**attempt)
                logger.warning(
                    "RTSP open attempt %d/%d failed; retrying in %.2f seconds",
                    attempt + 1,
                    self.max_retries + 1,
                    delay,
                )
                time.sleep(delay)
        raise VideoSourceError(
            f"Unable to open RTSP stream after {self.max_retries + 1} attempts: "
            f"{self.url}"
        )

    def read(self) -> FrameData | None:
        """Read a frame, reconnecting after a failed live-stream read."""
        for attempt in range(self.max_retries + 1):
            try:
                frame = super().read()
            except VideoSourceError:
                self.release()
                if attempt >= self.max_retries:
                    raise
                frame = None
            if frame is not None:
                return frame
            self.release()
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

    @property
    def frame_count(self) -> None:
        """Live RTSP sources have no known frame count."""
        return None

    def __str__(self) -> str:
        """Return the stream URL for logs and errors."""
        return self.url


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
