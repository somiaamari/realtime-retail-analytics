"""Tests for output video writing."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from vision_pipeline.streaming.writer import VideoWriter, VideoWriterError


def test_video_writer_round_trip_and_parent_creation(tmp_path: Path) -> None:
    """The writer creates directories and emits a readable video."""
    output = tmp_path / "nested" / "output.avi"
    with VideoWriter(output, 10.0, 64, 48, codec="MJPG") as writer:
        writer.write(np.full((48, 64, 3), 80, dtype=np.uint8))

    capture = cv2.VideoCapture(str(output))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    assert ok
    assert frame.shape == (48, 64, 3)


def test_video_writer_rejects_wrong_frame_size(tmp_path: Path) -> None:
    """A wrong-sized frame fails explicitly instead of silently corrupting output."""
    with VideoWriter(tmp_path / "output.avi", 10.0, 64, 48, codec="MJPG") as writer:
        with pytest.raises(VideoWriterError, match="does not match"):
            writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
