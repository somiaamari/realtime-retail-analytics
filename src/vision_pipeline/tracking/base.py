"""Common tracker interface."""

from abc import ABC, abstractmethod

from vision_pipeline.types import Detection, Track


class BaseTracker(ABC):
    """Interface implemented by trackers that consume detector outputs."""

    @abstractmethod
    def update(self, detections: list[Detection], frame_id: int) -> list[Track]:
        """Update tracker state and return currently confirmed tracks.

        Args:
            detections: Detections for the current frame.
            frame_id: Monotonically increasing source-frame index.

        Returns:
            Confirmed tracks matched to detections in the current frame.
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear tracking state and begin a new ID sequence."""
