"""Tests for local and factory-created video sources."""

from pathlib import Path

import pytest

from vision_pipeline.config import VideoConfig
from vision_pipeline.streaming.sources import (
    FileVideoSource,
    RTSPSource,
    VideoSourceError,
    WebcamSource,
    create_source,
)


def test_file_source_metadata_iteration_and_release(synthetic_video: Path) -> None:
    """File sources expose metadata and release after iteration."""
    source = FileVideoSource(synthetic_video)
    with source:
        assert source.frame_count == 5
        assert source.fps == pytest.approx(10.0)
        assert (source.width, source.height) == (160, 120)
        frames = list(source)
    assert len(frames) == 5
    assert [frame.frame_id for frame in frames] == list(range(5))
    assert frames[0].frame.shape == (120, 160, 3)
    assert frames[-1].timestamp_s > frames[0].timestamp_s
    assert source.read() is not None


def test_file_source_reports_missing_file(tmp_path: Path) -> None:
    """Opening a nonexistent file raises a helpful source exception."""
    source = FileVideoSource(tmp_path / "missing.mp4")
    with pytest.raises(VideoSourceError, match="does not exist"):
        source.open()


def test_source_factory_selects_expected_implementation(tmp_path: Path) -> None:
    """The factory maps numeric, RTSP, and filesystem sources correctly."""
    assert isinstance(create_source(VideoConfig(source=1)), WebcamSource)
    assert isinstance(create_source(VideoConfig(source="0")), WebcamSource)
    assert isinstance(
        create_source(VideoConfig(source="rtsp://camera.local/stream")),
        RTSPSource,
    )
    file_source = create_source(VideoConfig(source=str(tmp_path / "input.mp4")))
    assert isinstance(file_source, FileVideoSource)
    assert file_source.path == tmp_path / "input.mp4"


def test_resize_preserves_aspect_ratio(synthetic_video: Path) -> None:
    """Opt-in frame resizing fits within configured bounds without distortion."""
    source = FileVideoSource(
        synthetic_video,
        VideoConfig(source=str(synthetic_video), width=80, height=80, resize=True),
    )
    with source:
        frame = source.read()
        assert frame is not None
        assert frame.frame.shape[:2] == (60, 80)
        assert (source.width, source.height) == (80, 60)
