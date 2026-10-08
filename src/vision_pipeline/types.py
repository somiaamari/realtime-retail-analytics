"""Core types shared by video ingestion and object detection."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrameData:
    """A decoded video frame and its source timing metadata.

    Attributes:
        frame: Image in OpenCV BGR channel order.
        frame_id: Zero-based index of this frame from the source.
        timestamp_s: Timestamp in seconds from the beginning of the source.
        source_fps: Nominal source frame rate.
    """

    frame: np.ndarray
    frame_id: int
    timestamp_s: float
    source_fps: float


@dataclass(frozen=True)
class Detection:
    """A single object detection with an xyxy bounding box."""

    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str

    @property
    def center(self) -> tuple[float, float]:
        """Return the bounding-box center as ``(x, y)``."""
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Return the center of the box's bottom edge as ``(x, y)``."""
        x1, _, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, y2)

    @property
    def width(self) -> float:
        """Return the bounding-box width."""
        x1, _, x2, _ = self.bbox_xyxy
        return x2 - x1

    @property
    def height(self) -> float:
        """Return the bounding-box height."""
        _, y1, _, y2 = self.bbox_xyxy
        return y2 - y1

    @property
    def area(self) -> float:
        """Return the bounding-box area."""
        return self.width * self.height
