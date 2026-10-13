"""Offline tests for ByteTrack, Kalman filtering, and tracking visualization."""

from collections import deque

import numpy as np
import pytest
from pydantic import ValidationError

from vision_pipeline.config import TrackerConfig
from vision_pipeline.tracking.bytetrack import ByteTracker
from vision_pipeline.tracking.kalman import KalmanFilter
from vision_pipeline.tracking.matching import (
    iou_distance,
    linear_assignment,
    pairwise_iou,
)
from vision_pipeline.tracking.stats import TrackingStats
from vision_pipeline.types import Detection, Track, TrackState
from vision_pipeline.utils.visualization import draw_tracks


def detection(x: float, confidence: float = 0.9) -> Detection:
    """Create a fixed-size synthetic person detection."""
    return Detection((x, 10.0, x + 20.0, 60.0), confidence, 0, "person")


def test_kalman_filter_initiates_predicts_and_updates_measurement() -> None:
    """The state model preserves a box and estimates its constant velocity."""
    kalman = KalmanFilter()
    measurement = np.asarray([20.0, 35.0, 0.4, 50.0])
    mean, covariance = kalman.initiate(measurement)

    assert mean.shape == (8,)
    assert covariance.shape == (8, 8)
    predicted, predicted_covariance = kalman.predict(mean, covariance)
    assert predicted_covariance.shape == (8, 8)
    assert np.all(np.diag(predicted_covariance) > 0)

    updated, updated_covariance = kalman.update(
        predicted, predicted_covariance, measurement + np.asarray([2, 0, 0, 0])
    )
    assert updated[0] > predicted[0]
    assert updated_covariance.shape == (8, 8)
    assert (
        kalman.gating_distance(updated, updated_covariance, measurement[None, :])[0]
        >= 0
    )


def test_iou_and_assignment_support_empty_and_thresholded_inputs() -> None:
    """Pairwise IoU and Hungarian assignment return predictable edge cases."""
    boxes = np.asarray([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=np.float64)
    overlap = pairwise_iou(boxes, boxes[:1])
    assert overlap.shape == (2, 1)
    assert overlap[0, 0] == pytest.approx(1.0)
    assert overlap[1, 0] == pytest.approx(25 / 175)

    matches, missing_tracks, missing_detections = linear_assignment(
        np.asarray([[0.1, 0.9], [0.8, 0.2]]), 0.3
    )
    assert matches == [(0, 0), (1, 1)]
    assert missing_tracks == []
    assert missing_detections == []
    matches, missing_tracks, missing_detections = linear_assignment(
        np.asarray([[0.1, 0.49], [0.49, 0.51]]), 0.5
    )
    assert matches == [(0, 1), (1, 0)]
    assert missing_tracks == []
    assert missing_detections == []
    assert linear_assignment(np.empty((0, 1)), 0.5) == ([], [], [0])
    assert linear_assignment(np.asarray([[np.inf]]), 0.5) == ([], [0], [0])


def test_iou_distance_score_fusion_penalizes_low_confidence() -> None:
    """Optional confidence fusion lowers similarity for weak detections."""
    tracker = Track(1, detection(10).bbox_xyxy, 0.9, 0, "person")
    strong = detection(10, 0.9)
    weak = detection(10, 0.2)

    assert iou_distance([tracker], [strong], fuse_score=True)[0, 0] == pytest.approx(
        0.1
    )
    assert iou_distance([tracker], [weak], fuse_score=True)[0, 0] == pytest.approx(0.8)


def test_bytetrack_keeps_id_with_low_confidence_detection() -> None:
    """A low-score box maintains an existing ID but cannot start a new one."""
    tracker = ByteTracker(
        TrackerConfig(track_thresh=0.5, low_thresh=0.1, min_hits=1, track_buffer=3)
    )
    first = tracker.update([detection(10, 0.9)], frame_id=0)
    second = tracker.update([detection(12, 0.3), detection(100, 0.3)], frame_id=1)

    assert [track.track_id for track in first] == [1]
    assert [track.track_id for track in second] == [1]
    assert tracker.stats.total_unique_ids == 1
    assert tracker.stats.per_frame_active_counts == [1, 1]


def test_bytetrack_tentative_confirmation_expiry_and_id_reset() -> None:
    """Confirmation waits for min_hits and IDs restart only after explicit reset."""
    tracker = ByteTracker(TrackerConfig(min_hits=2, track_buffer=1))

    assert tracker.update([detection(10)], 0) == []
    confirmed = tracker.update([detection(11)], 1)
    assert [track.track_id for track in confirmed] == [1]
    assert confirmed[0].hits == 2
    assert len(confirmed[0].history) <= 30

    assert tracker.update([], 2) == []
    assert tracker.update([], 3) == []
    assert tracker.stats.total_unique_ids == 1
    tracker.reset()
    assert tracker.update([detection(30)], 0) == []
    assert [track.track_id for track in tracker.update([detection(31)], 1)] == [1]


def test_bytetrack_reidentifies_lost_track_without_double_counting() -> None:
    """A short occlusion recovers the same confirmed identity."""
    tracker = ByteTracker(TrackerConfig(track_buffer=3, min_hits=1))
    tracker.update([detection(10)], 0)
    assert tracker.update([], 1) == []
    recovered = tracker.update([detection(11)], 2)

    assert [track.track_id for track in recovered] == [1]
    assert recovered[0].reidentified
    assert tracker.stats.total_unique_ids == 1
    assert tracker.stats.reidentifications == 1


def test_bytetrack_filters_small_boxes_and_separates_classes() -> None:
    """Small boxes are discarded and distinct classes cannot share an ID."""
    tracker = ByteTracker(TrackerConfig(min_box_area=100, min_hits=1))
    assert tracker.update([Detection((0, 0, 5, 5), 0.9, 0, "person")], 0) == []
    person = tracker.update([detection(10)], 1)[0]
    other = Detection((10, 10, 30, 60), 0.9, 2, "car")

    new_track = tracker.update([other], 2)[0]
    assert new_track.track_id != person.track_id


def test_config_rejects_invalid_confidence_bands() -> None:
    """Low-score matching and high-score creation bands must be ordered."""
    with pytest.raises(ValidationError, match="low_thresh"):
        TrackerConfig(high_thresh=0.1, low_thresh=0.2)
    assert TrackerConfig(track_thresh=0.05).effective_low_thresh == 0.05


def test_track_geometry_history_stats_and_visualization() -> None:
    """Tracks reuse detection geometry, retain a bounded trail, and draw."""
    track = Track(
        7,
        detection(10).bbox_xyxy,
        0.9,
        0,
        "person",
        history=deque(maxlen=2),
    )
    track.history.append((20.0, 60.0))
    track.bbox_xyxy = detection(12).bbox_xyxy
    track.history.append(track.bottom_center)
    track.history.append((32.0, 60.0))

    assert track.center == (22.0, 35.0)
    assert track.bottom_center == (22.0, 60.0)
    assert track.width == 20.0
    assert track.height == 50.0
    assert track.area == 1000.0
    assert len(track.history) == 2

    stats = TrackingStats()
    track.state = TrackState.TRACKED
    stats.update([track])
    assert stats.total_unique_ids == 1
    assert stats.active_count == 1
    assert stats.average_track_lifetime == 1.0

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    annotated = draw_tracks(frame, [track])
    assert annotated.shape == frame.shape
    assert np.any(annotated != frame)
