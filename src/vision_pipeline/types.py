"""Core types shared by video ingestion and object detection."""

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from numpy.typing import NDArray


@dataclass(frozen=True)
class FrameData:
    """A decoded video frame and its source timing metadata.

    Attributes:
        frame: Image in OpenCV BGR channel order.
        frame_id: Zero-based index of this frame from the source.
        timestamp_s: Timestamp in seconds from the beginning of the source.
        source_fps: Nominal source frame rate.
    """

    frame: NDArray[Any]
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


class TrackState(Enum):
    """Lifecycle state for a tracked object."""

    TENTATIVE = auto()
    TRACKED = auto()
    LOST = auto()
    REMOVED = auto()


@dataclass
class Track:
    """A persistent object identity and its current tracking metadata.

    Attributes:
        track_id: Unique ID assigned for the current tracker run.
        bbox_xyxy: Current bounding box in ``(x1, y1, x2, y2)`` format.
        confidence: Confidence of the most recently associated detection.
        class_id: Class index of the most recently associated detection.
        class_name: Human-readable class name.
        state: Current lifecycle state.
        age: Number of frames elapsed since creation, including its first frame.
        hits: Number of frames associated with a detection.
        time_since_update: Number of frames since the last matched detection.
        history: Bounded deque of bottom-center points for trail rendering.
        reidentified: Whether this update recovered the track from the lost state.
    """

    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    state: TrackState = TrackState.TENTATIVE
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    history: deque[tuple[float, float]] | None = None
    reidentified: bool = False

    def __post_init__(self) -> None:
        """Create a default bounded trail if one was not supplied."""
        if self.history is None:
            self.history = deque(maxlen=30)
        self.history.append(self.bottom_center)

    def as_detection(self) -> Detection:
        """Return the current box as a :class:`Detection` for geometry helpers."""
        return Detection(
            bbox_xyxy=self.bbox_xyxy,
            confidence=self.confidence,
            class_id=self.class_id,
            class_name=self.class_name,
        )

    @property
    def center(self) -> tuple[float, float]:
        """Return the bounding-box center as ``(x, y)``."""
        return self.as_detection().center

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Return the center of the box's bottom edge as ``(x, y)``."""
        return self.as_detection().bottom_center

    @property
    def width(self) -> float:
        """Return the bounding-box width."""
        return self.as_detection().width

    @property
    def height(self) -> float:
        """Return the bounding-box height."""
        return self.as_detection().height

    @property
    def area(self) -> float:
        """Return the bounding-box area."""
        return self.as_detection().area
