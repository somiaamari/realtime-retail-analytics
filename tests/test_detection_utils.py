"""Tests for core detection types, fake detections, and visualization."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vision_pipeline.config import ModelConfig
from vision_pipeline.detection.yolo_detector import FakeDetector, YoloDetector
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


def test_yolo_detector_converts_and_filters_without_ultralytics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wrapper converts mocked Ultralytics output and filters by name."""
    person = SimpleNamespace(
        cls=np.array([0.0]),
        conf=np.array([0.9]),
        xyxy=np.array([[1.0, 2.0, 30.0, 40.0]]),
    )
    car = SimpleNamespace(
        cls=np.array([2.0]),
        conf=np.array([0.8]),
        xyxy=np.array([[5.0, 6.0, 20.0, 25.0]]),
    )

    class MockYolo:
        names = {0: "person", 2: "car"}

        def predict(self, *_args: object, **_kwargs: object) -> list[object]:
            return [SimpleNamespace(names=self.names, boxes=[person, car])]

    class MockUltralytics:
        YOLO = staticmethod(lambda _weights: MockYolo())

    real_import = __import__("importlib").import_module

    def mock_import(name: str) -> object:
        if name == "ultralytics":
            return MockUltralytics
        return real_import(name)

    def mock_download(_url: str, destination: str | Path) -> tuple[str, None]:
        Path(destination).write_bytes(b"test weights")
        return str(destination), None

    monkeypatch.setattr(
        "vision_pipeline.detection.yolo_detector.importlib.import_module",
        mock_import,
    )
    monkeypatch.setattr(
        "vision_pipeline.detection.yolo_detector.urlretrieve",
        mock_download,
    )
    weights = tmp_path / "models" / "yolov8n.pt"
    detector = YoloDetector(
        ModelConfig(weights_path=weights, device="cpu", classes=["PERSON", 2])
    )

    detections = detector.detect(np.zeros((64, 64, 3), dtype=np.uint8))
    detector.warmup()

    assert weights.is_file()
    assert detections == [
        Detection((1.0, 2.0, 30.0, 40.0), 0.9, 0, "person"),
        Detection((5.0, 6.0, 20.0, 25.0), 0.8, 2, "car"),
    ]


def test_yolo_device_selection_prefers_available_accelerators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto device selection prefers CUDA, then MPS, then CPU."""
    import vision_pipeline.detection.yolo_detector as yolo_module

    monkeypatch.setattr(
        yolo_module.importlib,
        "import_module",
        lambda _name: SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        ),
    )
    assert YoloDetector._select_device("auto") == "cuda"

    monkeypatch.setattr(
        yolo_module.importlib,
        "import_module",
        lambda _name: SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        ),
    )
    assert YoloDetector._select_device("auto") == "mps"

    monkeypatch.setattr(
        yolo_module.importlib,
        "import_module",
        lambda _name: SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        ),
    )
    assert YoloDetector._select_device("auto") == "cpu"
    assert YoloDetector._select_device("CPU") == "CPU"


def test_yolo_class_names_and_existing_weights(tmp_path: Path) -> None:
    """String selectors resolve against list names and existing weights persist."""
    weights = tmp_path / "custom.pt"
    weights.write_bytes(b"weights")

    assert YoloDetector._ensure_weights(weights) == weights
    assert YoloDetector._ensure_weights(tmp_path / "custom-missing.pt").name == (
        "custom-missing.pt"
    )
    assert YoloDetector._resolve_classes(["person"], ["person", "bike"]) == (
        {0},
        {"person"},
    )


def test_yolo_detector_reports_missing_optional_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing Ultralytics raises an actionable install instruction."""

    def missing_import(_name: str) -> object:
        raise ModuleNotFoundError("no ultralytics")

    monkeypatch.setattr(
        "vision_pipeline.detection.yolo_detector.importlib.import_module",
        missing_import,
    )
    with pytest.raises(ImportError, match=r"\.\[detection\]"):
        YoloDetector(ModelConfig(device="cpu"))
