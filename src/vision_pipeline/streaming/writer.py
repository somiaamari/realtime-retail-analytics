"""Context-managed OpenCV video writer."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from vision_pipeline.logging_utils import get_logger
from vision_pipeline.streaming.sources import VideoSource

logger = get_logger(__name__)


class VideoWriterError(RuntimeError):
    """Raised when an output video cannot be written."""


class VideoWriter:
    """Write BGR frames to a video file using OpenCV."""

    def __init__(
        self,
        path: str | Path,
        fps: float,
        width: int,
        height: int,
        *,
        codec: str = "mp4v",
    ) -> None:
        if fps <= 0 or width <= 0 or height <= 0:
            raise ValueError("FPS and output dimensions must be positive.")
        if len(codec) != 4:
            raise ValueError("Video codec must be a four-character code.")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.width = width
        self.height = height
        self.codec = codec
        fourcc = cv2.VideoWriter.fourcc(*codec)
        self._writer = cv2.VideoWriter(
            str(self.path),
            fourcc,
            fps,
            (width, height),
        )
        if not self._writer.isOpened():
            self._writer.release()
            raise VideoWriterError(
                f"Unable to open video output for writing: {self.path}"
            )
        logger.info(
            "Opened video writer %s (%dx%d at %.3f FPS)",
            self.path,
            width,
            height,
            fps,
        )

    @classmethod
    def from_source(
        cls,
        path: str | Path,
        source: VideoSource,
        *,
        codec: str = "mp4v",
    ) -> VideoWriter:
        """Create a writer using the source's frame rate and dimensions."""
        return cls(path, source.fps, source.width, source.height, codec=codec)

    def write(self, frame: np.ndarray) -> None:
        """Write a frame matching the configured output dimensions."""
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise VideoWriterError("Output frames must be three-channel BGR images.")
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            raise VideoWriterError(
                f"Frame size {frame.shape[1]}x{frame.shape[0]} does not match "
                f"writer size {self.width}x{self.height}."
            )
        if not self._writer.isOpened():
            raise VideoWriterError(f"Video writer is closed: {self.path}")
        self._writer.write(frame)

    def release(self) -> None:
        """Flush and release the underlying OpenCV writer."""
        if self._writer.isOpened():
            self._writer.release()
            logger.info("Released video writer %s", self.path)

    def __enter__(self) -> VideoWriter:
        """Return this writer for use in a ``with`` block."""
        return self

    def __exit__(self, *_: object) -> None:
        """Release the writer when leaving its context."""
        self.release()
