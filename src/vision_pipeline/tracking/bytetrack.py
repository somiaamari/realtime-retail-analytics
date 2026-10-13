"""ByteTrack multi-object tracking implementation."""

from collections import deque

import numpy as np

from vision_pipeline.config import TrackerConfig
from vision_pipeline.logging_utils import get_logger
from vision_pipeline.tracking.base import BaseTracker
from vision_pipeline.tracking.kalman import FloatArray, KalmanFilter
from vision_pipeline.tracking.matching import iou_distance, linear_assignment
from vision_pipeline.tracking.stats import TrackingStats
from vision_pipeline.types import Detection, Track, TrackState

logger = get_logger(__name__)


def _measurement(detection: Detection) -> FloatArray:
    """Convert an xyxy box to center-x, center-y, aspect-ratio, and height."""
    x1, y1, x2, y2 = detection.bbox_xyxy
    height = max(y2 - y1, 1e-3)
    width = max(x2 - x1, 1e-3)
    return np.asarray([(x1 + x2) / 2, (y1 + y2) / 2, width / height, height])


def _state_box(mean: FloatArray) -> tuple[float, float, float, float]:
    """Convert a Kalman state to a valid xyxy box."""
    center_x, center_y, aspect_ratio, height = (float(value) for value in mean[:4])
    height = max(height, 1e-3)
    width = max(aspect_ratio * height, 1e-3)
    return (
        center_x - width / 2,
        center_y - height / 2,
        center_x + width / 2,
        center_y + height / 2,
    )


class ByteTracker(BaseTracker):
    """Track detections using a two-stage, confidence-aware ByteTrack method."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        """Create a tracker using the supplied validated settings.

        Args:
            config: Thresholds, retention, and trail settings. Defaults to
                :class:`TrackerConfig`.
        """
        self.config = config or TrackerConfig()
        self._kalman = KalmanFilter()
        self.stats = TrackingStats()
        self.reset()

    def reset(self) -> None:
        """Clear all tracks and restart IDs at one for a new tracker run."""
        self._tracks: list[Track] = []
        self._states: dict[int, tuple[FloatArray, FloatArray]] = {}
        self._next_id = 1
        self._last_frame_id: int | None = None
        self.stats.reset()

    def update(self, detections: list[Detection], frame_id: int) -> list[Track]:
        """Associate detections with existing tracks and return confirmed tracks.

        Args:
            detections: Detector results for this frame.
            frame_id: Monotonically increasing source-frame index.

        Returns:
            Confirmed tracks observed in this frame.

        Raises:
            ValueError: If frame IDs do not increase between updates.
        """
        if self._last_frame_id is not None and frame_id <= self._last_frame_id:
            raise ValueError("frame_id must increase on every tracker update")
        frame_delta = (
            1 if self._last_frame_id is None else frame_id - self._last_frame_id
        )
        self._last_frame_id = frame_id
        usable = [
            detection
            for detection in detections
            if detection.area >= self.config.min_box_area
        ]
        high_detections = [
            detection
            for detection in usable
            if detection.confidence > self.config.effective_high_thresh
        ]
        low_detections = [
            detection
            for detection in usable
            if self.config.effective_low_thresh
            <= detection.confidence
            <= self.config.effective_high_thresh
        ]

        current = [
            track for track in self._tracks if track.state is not TrackState.REMOVED
        ]
        for track in current:
            track.reidentified = False
            track.age += frame_delta
            track.time_since_update += frame_delta
            mean, covariance = self._states[track.track_id]
            for _ in range(frame_delta):
                mean, covariance = self._kalman.predict(mean, covariance)
            self._states[track.track_id] = (mean, covariance)
            track.bbox_xyxy = _state_box(mean)

        association_tracks = sorted(
            (
                track
                for track in current
                if track.state
                in (TrackState.TRACKED, TrackState.LOST, TrackState.TENTATIVE)
            ),
            key=lambda track: track.track_id,
        )
        first_matches, unmatched_pool, unmatched_high = self._associate(
            association_tracks, high_detections
        )
        for track_index, detection_index in first_matches:
            track = association_tracks[track_index]
            was_lost = track.state is TrackState.LOST
            self._update_track(track, high_detections[detection_index])
            track.reidentified = was_lost

        remaining_tracked = [
            association_tracks[index]
            for index in unmatched_pool
            if association_tracks[index].state is TrackState.TRACKED
        ]
        # ByteTrack's key idea is a second IoU pass against low-score boxes:
        # weak detections can preserve an identity but can never create one.
        second_matches, unmatched_after_low, _ = self._associate(
            remaining_tracked, low_detections
        )
        for track_index, detection_index in second_matches:
            self._update_track(
                remaining_tracked[track_index],
                low_detections[detection_index],
            )

        matched_track_ids = {
            association_tracks[index].track_id for index, _ in first_matches
        }
        matched_track_ids.update(
            remaining_tracked[index].track_id for index, _ in second_matches
        )
        for track_index in unmatched_after_low:
            track = remaining_tracked[track_index]
            if track.track_id not in matched_track_ids:
                track.state = TrackState.LOST

        for index in unmatched_pool:
            track = association_tracks[index]
            if track.state is TrackState.TENTATIVE:
                track.state = TrackState.REMOVED
                self._states.pop(track.track_id, None)

        for index in unmatched_high:
            detection = high_detections[index]
            if detection.confidence >= self.config.effective_new_track_thresh:
                self._start_track(detection)

        for track in current:
            if (
                track.state is TrackState.LOST
                and track.time_since_update > self.config.track_buffer
            ):
                track.state = TrackState.REMOVED
                self._states.pop(track.track_id, None)
            elif (
                track.state is TrackState.TRACKED
                and track.track_id not in matched_track_ids
            ):
                track.state = TrackState.LOST
            if (
                track.state is TrackState.LOST
                and track.track_id not in matched_track_ids
                and track.history is not None
            ):
                track.history.append(track.bottom_center)

        self._tracks = [
            track for track in self._tracks if track.state is not TrackState.REMOVED
        ] + [track for track in current if track.state is TrackState.REMOVED]
        self.stats.update(self._tracks)
        result = [
            track
            for track in self._tracks
            if track.state is TrackState.TRACKED and track.time_since_update == 0
        ]
        return sorted(result, key=lambda track: track.track_id)

    def _associate(
        self,
        tracks: list[Track],
        detections: list[Detection],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Associate boxes using IoU while preventing cross-class matches."""
        costs = iou_distance(tracks, detections, fuse_score=self.config.fuse_score)
        for row, track in enumerate(tracks):
            for column, detection in enumerate(detections):
                if track.class_id != detection.class_id:
                    costs[row, column] = np.inf
        return linear_assignment(costs, self.config.match_thresh)

    def _update_track(self, track: Track, detection: Detection) -> None:
        """Apply a matched detector result to a track."""
        mean, covariance = self._states[track.track_id]
        mean, covariance = self._kalman.update(
            mean, covariance, _measurement(detection)
        )
        self._states[track.track_id] = (mean, covariance)
        track.bbox_xyxy = detection.bbox_xyxy
        track.confidence = detection.confidence
        track.class_id = detection.class_id
        track.class_name = detection.class_name
        track.hits += 1
        track.time_since_update = 0
        if track.hits >= self.config.min_hits:
            track.state = TrackState.TRACKED
        else:
            track.state = TrackState.TENTATIVE
        if track.history is not None:
            track.history.append(track.bottom_center)

    def _start_track(self, detection: Detection) -> None:
        """Initialize a new Kalman state and tentative identity."""
        mean, covariance = self._kalman.initiate(_measurement(detection))
        track_id = self._next_id
        self._next_id += 1
        track = Track(
            track_id=track_id,
            bbox_xyxy=detection.bbox_xyxy,
            confidence=detection.confidence,
            class_id=detection.class_id,
            class_name=detection.class_name,
            state=(
                TrackState.TRACKED
                if self.config.min_hits <= 1
                else TrackState.TENTATIVE
            ),
            history=deque(maxlen=self.config.history_len),
        )
        self._states[track_id] = (mean, covariance)
        self._tracks.append(track)
        logger.debug("Started track %d at frame %s", track_id, self._last_frame_id)
