"""Tests for local and factory-created video sources."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from vision_pipeline.config import VideoConfig
from vision_pipeline.streaming.sources import (
    FileVideoSource,
    RTSPSource,
    VideoSourceError,
    WebcamSource,
    create_source,
)


class StubCapture:
    """Minimal OpenCV capture stand-in for live-source edge cases."""

    def __init__(
        self,
        opened: bool,
        reads: list[tuple[bool, np.ndarray | None]],
    ) -> None:
        self.opened = opened
        self.reads = iter(reads)
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def get(self, prop: int) -> float:
        return {
            cv2.CAP_PROP_FPS: 25.0,
            cv2.CAP_PROP_FRAME_WIDTH: 64.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 48.0,
            cv2.CAP_PROP_FRAME_COUNT: 0.0,
            cv2.CAP_PROP_POS_MSEC: 1000.0,
        }.get(prop, 0.0)

    def read(self) -> tuple[bool, np.ndarray | None]:
        return next(self.reads, (False, None))

    def release(self) -> None:
        self.released = True


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


def test_webcam_source_reads_a_live_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """Webcam properties and frame metadata work for a live source."""
    capture = StubCapture(
        True,
        [(True, np.zeros((48, 64, 3), dtype=np.uint8))],
    )
    monkeypatch.setattr(
        "vision_pipeline.streaming.sources.cv2.VideoCapture",
        lambda _source: capture,
    )
    source = WebcamSource(2)

    source.open()
    frame = source.read()
    assert frame is not None
    assert frame.frame_id == 0
    assert frame.timestamp_s == 1.0
    assert source.frame_count is None
    assert (source.width, source.height, source.fps) == (64, 48, 25.0)
    source.release()
    assert capture.released
    with pytest.raises(ValueError, match="non-negative"):
        WebcamSource(-1)


def test_rtsp_source_reconnects_after_failed_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An RTSP read failure triggers one bounded reconnect attempt."""
    first = StubCapture(
        True,
        [
            (True, np.zeros((48, 64, 3), dtype=np.uint8)),
            (False, None),
        ],
    )
    second = StubCapture(
        True,
        [(True, np.zeros((48, 64, 3), dtype=np.uint8))],
    )
    captures = iter((first, second))
    monkeypatch.setattr(
        "vision_pipeline.streaming.sources.cv2.VideoCapture",
        lambda _source: next(captures),
    )
    monkeypatch.setattr("vision_pipeline.streaming.sources.time.sleep", lambda _: None)
    source = RTSPSource(
        "rtsp://camera/stream",
        max_retries=1,
        reconnect_delay_s=0,
    )

    initial_frame = source.read()
    frame = source.read()

    assert frame is not None
    assert initial_frame is not None
    assert initial_frame.frame_id == 0
    assert frame.frame_id == 1
    assert first.released
    assert source.frame_count is None
    source.release()


def test_rtsp_source_redacts_credentials_and_query_from_display() -> None:
    """RTSP log/error representations omit URL credentials and query tokens."""
    source = RTSPSource("rtsp://user:secret@camera.local:8554/live?token=private")

    assert str(source) == "rtsp://camera.local:8554/live"


def test_rtsp_open_exhaustion_and_corrupt_webcam_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Open retries are bounded and invalid decoded frames raise clearly."""
    failed = StubCapture(False, [])
    monkeypatch.setattr(
        "vision_pipeline.streaming.sources.cv2.VideoCapture",
        lambda _source: failed,
    )
    monkeypatch.setattr("vision_pipeline.streaming.sources.time.sleep", lambda _: None)
    with pytest.raises(VideoSourceError, match="after 2 attempts"):
        RTSPSource(
            "rtsp://camera/stream",
            max_retries=1,
            reconnect_delay_s=0,
        ).open()
    assert failed.released

    corrupt = StubCapture(True, [(True, None)])
    monkeypatch.setattr(
        "vision_pipeline.streaming.sources.cv2.VideoCapture",
        lambda _source: corrupt,
    )
    with pytest.raises(VideoSourceError, match="Corrupted frame"):
        WebcamSource(0).read()
