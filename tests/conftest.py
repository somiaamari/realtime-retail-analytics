"""Shared test fixtures and optional-dependency marker handling."""

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Iterator[Path]:
    """Create a short, self-contained video with moving colored rectangles."""
    path = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (160, 120),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG video encoding is unavailable.")
    for frame_id in range(5):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.rectangle(
            frame,
            (10 + frame_id * 4, 20),
            (40 + frame_id * 4, 80),
            (0, 255, 0),
            cv2.FILLED,
        )
        writer.write(frame)
    writer.release()
    yield path


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip tests marked for the optional detection extra when absent."""
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        skip = pytest.mark.skip(
            reason="requires the optional detection extra (ultralytics)"
        )
        for item in items:
            if "requires_detection" in item.keywords:
                item.add_marker(skip)
