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


def test_video_writer_validates_frames_and_arguments(tmp_path: Path) -> None:
    """Invalid dimensions, codecs, and frame shapes are rejected explicitly."""
    with pytest.raises(ValueError, match="positive"):
        VideoWriter(tmp_path / "bad.avi", 0, 64, 48)
    with pytest.raises(ValueError, match="four-character"):
        VideoWriter(tmp_path / "bad.avi", 10, 64, 48, codec="bad")

    writer = VideoWriter(tmp_path / "closed.avi", 10.0, 64, 48, codec="MJPG")
    with pytest.raises(VideoWriterError, match="three-channel"):
        writer.write(np.zeros((48, 64), dtype=np.uint8))
    writer.release()
    with pytest.raises(VideoWriterError, match="closed"):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
