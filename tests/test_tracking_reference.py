"""Optional smoke comparison against Supervision's ByteTrack reference."""

import pytest

pytest.importorskip("supervision")

from vision_pipeline.config import TrackerConfig
from vision_pipeline.tracking.bytetrack import ByteTracker
from vision_pipeline.tracking.reference import SupervisionByteTracker
from vision_pipeline.types import Detection


def test_local_tracker_and_supervision_reference_preserve_single_identity() -> None:
    """Both implementations keep one identity through a smooth synthetic path."""
    config = TrackerConfig(
        track_thresh=0.5,
        high_thresh=0.5,
        low_thresh=0.1,
        match_thresh=0.8,
        min_hits=1,
    )
    local = ByteTracker(config)
    reference = SupervisionByteTracker(config)
    sequence = [
        [Detection((x, 10, x + 20, 60), 0.9, 0, "person")] for x in (10, 12, 14)
    ]

    local_results = [
        local.update(items, frame_id) for frame_id, items in enumerate(sequence)
    ]
    reference_results = [
        reference.update(items, frame_id) for frame_id, items in enumerate(sequence)
    ]

    assert [len(frame) for frame in local_results] == [1, 1, 1]
    assert [len(frame) for frame in reference_results] == [1, 1, 1]
    assert len({frame[0].track_id for frame in local_results}) == 1
    assert len({frame[0].track_id for frame in reference_results}) == 1
