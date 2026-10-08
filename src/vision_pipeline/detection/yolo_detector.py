"""Detector interfaces and Ultralytics YOLO integration."""

import importlib
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import numpy as np

from vision_pipeline.config import ModelConfig
from vision_pipeline.logging_utils import get_logger
from vision_pipeline.types import Detection

logger = get_logger(__name__)
_WEIGHT_URLS = {
    "yolov8n.pt": (
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
    ),
    "yolo11n.pt": (
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
    ),
}


class BaseDetector(ABC):
    """Interface implemented by object detectors."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect objects in a BGR image."""

    @abstractmethod
    def warmup(self) -> None:
        """Run any one-time initialization inference."""


class FakeDetector(BaseDetector):
    """Deterministic detector for tests and dependency-free examples."""

    def __init__(self, detections: list[Detection] | None = None) -> None:
        self.detections = list(detections or [])

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return the configured detections without changing the input frame."""
        del frame
        return list(self.detections)

    def warmup(self) -> None:
        """Perform no work for the deterministic fake detector."""


class YoloDetector(BaseDetector):
    """Ultralytics YOLO detector with lazy optional-dependency loading."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        try:
            ultralytics = importlib.import_module("ultralytics")
        except ImportError as error:
            raise ImportError(
                "YOLO detection requires the optional dependencies. Install them "
                "with `pip install -e '.[detection]'`."
            ) from error
        self.device = self._select_device(self.config.device)
        weights_path = self._ensure_weights(self.config.weights_path)
        self._model = ultralytics.YOLO(str(weights_path))
        self._class_ids, self._class_names = self._resolve_classes(
            self.config.classes,
            self._model.names,
        )
        logger.info(
            "Loaded YOLO weights %s on %s",
            self.config.weights_path,
            self.device,
        )

    @staticmethod
    def _ensure_weights(weights_path: Path) -> Path:
        """Download supported nano weights directly to their configured path."""
        path = weights_path
        if path.is_file():
            return path
        download_url = _WEIGHT_URLS.get(path.name)
        if download_url is None:
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f"{path.name}.download")
        try:
            logger.info("Downloading YOLO weights from %s to %s", download_url, path)
            urlretrieve(download_url, temporary_path)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return path

    @staticmethod
    def _select_device(requested: str) -> str:
        """Resolve ``auto`` to the best available torch device."""
        if requested.lower() != "auto":
            return requested
        try:
            torch = importlib.import_module("torch")
        except ImportError as error:
            raise ImportError(
                "Automatic device selection requires PyTorch. Install the "
                "optional dependencies with `pip install -e '.[detection]'`."
            ) from error
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _resolve_classes(
        requested: list[str | int],
        names: dict[int, str] | list[str],
    ) -> tuple[set[int], set[str]]:
        """Split configured class selectors into IDs and normalized names."""
        ids = {value for value in requested if isinstance(value, int)}
        name_selectors = {
            value.casefold() for value in requested if isinstance(value, str)
        }
        if isinstance(names, dict):
            ids.update(
                class_id
                for class_id, class_name in names.items()
                if class_name.casefold() in name_selectors
            )
        else:
            ids.update(
                class_id
                for class_id, class_name in enumerate(names)
                if class_name.casefold() in name_selectors
            )
        return ids, name_selectors

    @staticmethod
    def _number(value: Any) -> float:
        """Convert scalar/tensor values returned by Ultralytics to floats."""
        item = getattr(value, "item", None)
        if callable(item):
            value = item()
        return float(value)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run inference and return configured classes as typed detections."""
        results = self._model.predict(
            frame,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.imgsz,
            device=self.device,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            names: dict[int, str] | list[str] = result.names
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(self._number(box.cls[0]))
                class_name = (
                    names[class_id] if isinstance(names, dict) else names[class_id]
                )
                if (
                    (self._class_ids or self._class_names)
                    and class_id not in self._class_ids
                    and class_name.casefold() not in self._class_names
                ):
                    continue
                coordinates = box.xyxy[0].tolist()
                x1, y1, x2, y2 = (float(value) for value in coordinates)
                detections.append(
                    Detection(
                        bbox_xyxy=(x1, y1, x2, y2),
                        confidence=self._number(box.conf[0]),
                        class_id=class_id,
                        class_name=class_name,
                    )
                )
        return detections

    def warmup(self) -> None:
        """Run one inference on a black image matching the configured size."""
        dummy = np.zeros((self.config.imgsz, self.config.imgsz, 3), dtype=np.uint8)
        self.detect(dummy)
