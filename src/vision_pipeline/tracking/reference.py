"""Optional adapter for Supervision's ByteTrack reference implementation."""

import importlib
from collections import deque
from typing import Any

import numpy as np

from vision_pipeline.config import TrackerConfig
from vision_pipeline.tracking.base import BaseTracker
from vision_pipeline.tracking.stats import TrackingStats
from vision_pipeline.types import Detection, Track, TrackState


class SupervisionByteTracker(BaseTracker):
    """Adapt ``supervision.ByteTrack`` to the package's tracker interface.

    Supervision is imported only when this adapter is constructed. Install the
    optional dependency with ``pip install vision-pipeline[tracking-ref]``.
    """

    def __init__(self, config: TrackerConfig | None = None) -> None:
        """Load the optional reference and configure its ByteTrack instance.

        Args:
            config: Shared tracker settings.

        Raises:
            ImportError: If the optional ``tracking-ref`` extra is unavailable.
        """
        self.config = config or TrackerConfig()
        try:
            sv = importlib.import_module("supervision")
        except ImportError as exc:
            raise ImportError(
                "SupervisionByteTracker requires the optional extra; install "
                "'vision-pipeline[tracking-ref]'."
            ) from exc
        self._sv: Any = sv
        self._tracker: Any = self._new_tracker()
        self._tracks: dict[int, Track] = {}
        self._last_frame_id: int | None = None
        self.stats = TrackingStats()

    def _new_tracker(self) -> Any:
        """Create a fresh reference tracker."""
        return self._sv.ByteTrack(
            track_activation_threshold=self.config.effective_new_track_thresh,
            lost_track_buffer=self.config.track_buffer,
            minimum_matching_threshold=self.config.match_thresh,
            frame_rate=round(self.config.frame_rate),
            minimum_consecutive_frames=self.config.min_hits,
        )

    def reset(self) -> None:
        """Clear adapter state and restart the reference tracker."""
        self._tracker = self._new_tracker()
        self._tracks.clear()
        self._last_frame_id = None
        self.stats.reset()

    def update(self, detections: list[Detection], frame_id: int) -> list[Track]:
        """Update the reference tracker and convert its results to ``Track`` objects.

        Args:
            detections: Detections for the current frame.
            frame_id: Monotonically increasing source-frame index.

        Returns:
            Currently matched tracks emitted by Supervision.
        """
        if self._last_frame_id is not None and frame_id <= self._last_frame_id:
            raise ValueError("frame_id must increase on every tracker update")
        frame_delta = (
            1 if self._last_frame_id is None else frame_id - self._last_frame_id
        )
        self._last_frame_id = frame_id
        selected = [
            detection
            for detection in detections
            if detection.area >= self.config.min_box_area
        ]
        sv_detections = self._sv.Detections(
            xyxy=np.asarray(
                [item.bbox_xyxy for item in selected], dtype=np.float32
            ).reshape(-1, 4),
            confidence=np.asarray(
                [item.confidence for item in selected], dtype=np.float32
            ),
            class_id=np.asarray([item.class_id for item in selected], dtype=int),
        )
        tracked = self._tracker.update_with_detections(sv_detections)
        tracker_ids = tracked.tracker_id
        if tracker_ids is None:
            self._age_missing_tracks(frame_delta, set())
            self.stats.update(list(self._tracks.values()))
            return []
        names = {item.class_id: item.class_name for item in selected}
        visible: list[Track] = []
        tracked_confidence = tracked.confidence
        for index, (box, class_id, tracker_id) in enumerate(
            zip(tracked.xyxy, tracked.class_id, tracker_ids, strict=True)
        ):
            if tracker_id is None or class_id is None:
                continue
            track_id = int(tracker_id)
            bbox = (
                float(box[0]),
                float(box[1]),
                float(box[2]),
                float(box[3]),
            )
            class_index = int(class_id)
            prior = self._tracks.get(track_id)
            confidence = (
                float(tracked_confidence[index])
                if tracked_confidence is not None
                else prior.confidence
                if prior is not None
                else 1.0
            )
            if prior is None:
                track = Track(
                    track_id=track_id,
                    bbox_xyxy=bbox,
                    confidence=confidence,
                    class_id=class_index,
                    class_name=names.get(class_index, str(class_index)),
                    state=TrackState.TRACKED,
                    history=deque(maxlen=self.config.history_len),
                )
                self._tracks[track_id] = track
            else:
                was_lost = prior.state is TrackState.LOST
                prior.age += frame_delta
                prior.hits += 1
                prior.time_since_update = 0
                prior.reidentified = was_lost
                prior.state = TrackState.TRACKED
                prior.bbox_xyxy = bbox
                prior.confidence = confidence
                prior.class_id = class_index
                prior.class_name = names.get(class_index, prior.class_name)
                if prior.history is not None:
                    prior.history.append(prior.bottom_center)
                track = prior
            visible.append(track)
        active_ids = {track.track_id for track in visible}
        self._age_missing_tracks(frame_delta, active_ids)
        self.stats.update(list(self._tracks.values()))
        return sorted(visible, key=lambda track: track.track_id)

    def _age_missing_tracks(self, frame_delta: int, active_ids: set[int]) -> None:
        """Age absent reference tracks and retire those beyond the buffer."""
        for track in self._tracks.values():
            if track.track_id not in active_ids:
                track.age += frame_delta
                track.time_since_update += frame_delta
                track.state = TrackState.LOST
                track.reidentified = False
        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if track.time_since_update > self.config.track_buffer
        ]
        for track_id in expired:
            self._tracks[track_id].state = TrackState.REMOVED
            del self._tracks[track_id]
