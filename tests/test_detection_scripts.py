"""Dependency-free end-to-end tests for detection and benchmark CLIs."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from vision_pipeline.detection.yolo_detector import FakeDetector
from vision_pipeline.streaming.sources import VideoSource
from vision_pipeline.streaming.writer import VideoWriter
from vision_pipeline.types import Detection

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def _load_script(name: str) -> object:
    """Load a CLI module from its source path without packaging scripts."""
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load CLI script from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


benchmark_baseline = _load_script("benchmark_baseline")
run_detection = _load_script("run_detection")


class StubDetector(FakeDetector):
    """Fake detector accepting the same constructor input as the YOLO wrapper."""

    def __init__(self, _config: object) -> None:
        super().__init__([Detection((15, 15, 60, 90), 0.9, 0, "person")])
        self.device = "cpu"


class MjpegVideoWriter(VideoWriter):
    """Use broadly supported AVI/MJPEG encoding in CLI integration tests."""

    @classmethod
    def from_source(
        cls,
        path: str | Path,
        source: VideoSource,
        *,
        codec: str = "mp4v",
    ) -> MjpegVideoWriter:
        del codec
        return cls(path, source.fps, source.width, source.height, codec="MJPG")


def test_detection_cli_writes_annotated_video(
    synthetic_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The detection CLI processes synthetic frames with an injected fake model."""
    monkeypatch.setattr(run_detection, "YoloDetector", StubDetector)
    monkeypatch.setattr(run_detection, "VideoWriter", MjpegVideoWriter)
    output = tmp_path / "detected.avi"
    args = run_detection.build_parser().parse_args(
        [
            "--config",
            str(DEFAULT_CONFIG),
            "--source",
            str(synthetic_video),
            "--output",
            str(output),
            "--max-frames",
            "3",
        ]
    )

    assert run_detection.run(args) == 0
    capture = cv2.VideoCapture(str(output))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    assert ok
    assert frame.shape == (120, 160, 3)
    assert np.any(frame)


def test_benchmark_cli_persists_json_and_markdown_row(
    synthetic_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The benchmark CLI records measured stages without a real YOLO install."""
    monkeypatch.setattr(benchmark_baseline, "YoloDetector", StubDetector)
    output_dir = tmp_path / "benchmarks"
    args = benchmark_baseline.build_parser().parse_args(
        [
            "--config",
            str(DEFAULT_CONFIG),
            "--source",
            str(synthetic_video),
            "--frames",
            "3",
            "--warmup",
            "1",
            "--output-dir",
            str(output_dir),
        ]
    )

    result_path = benchmark_baseline.run(args)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    readme = (output_dir / "README.md").read_text(encoding="utf-8")

    assert result["frames_processed"] == 3
    assert {"read", "preprocess", "inference", "postprocess", "draw", "write"} <= (
        result["stage_latency_ms"].keys()
    )
    assert "yolov8n" in readme
    assert "cpu" in readme
