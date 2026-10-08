"""Tests for core detection types, fake detections, and visualization."""

import numpy as np

from vision_pipeline.detection.yolo_detector import FakeDetector
from vision_pipeline.types import Detection
from vision_pipeline.utils.visualization import draw_detections, draw_overlay_stats


def test_detection_geometry_helpers() -> None:
    """Detection geometry is calculated from xyxy coordinates."""
    detection = Detection((10.0, 20.0, 50.0, 80.0), 0.9, 0, "person")

    assert detection.center == (30.0, 50.0)
    assert detection.bottom_center == (30.0, 80.0)
    assert detection.width == 40.0
    assert detection.height == 60.0
    assert detection.area == 2400.0


def test_fake_detector_and_drawing_modify_a_copy() -> None:
    """Fake detections draw visible pixels without changing frame dimensions."""
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    detections = [Detection((10, 10, 80, 80), 0.95, 0, "person")]
    assert FakeDetector(detections).detect(frame) == detections

    annotated = draw_detections(frame, detections)
    assert annotated.shape == frame.shape
    assert np.any(annotated != frame)
    assert not np.any(frame)


def test_overlay_draws_on_a_copy() -> None:
    """HUD output maintains the input dimensions and adds visible pixels."""
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    overlay = draw_overlay_stats(frame, 12.3, 45.6, 7)

    assert overlay.shape == frame.shape
    assert np.any(overlay != frame)
    assert not np.any(frame)
