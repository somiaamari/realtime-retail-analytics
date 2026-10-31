"""ONNX Runtime detector with numpy/OpenCV-only preprocessing and decoding."""

import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vision_pipeline.config import DetectorConfig, ModelConfig
from vision_pipeline.detection.yolo_detector import BaseDetector
from vision_pipeline.logging_utils import get_logger
from vision_pipeline.optimization.postprocess import (
    decode_nms_output,
    decode_yolo_output,
)
from vision_pipeline.optimization.preprocess import LetterboxInfo, letterbox
from vision_pipeline.types import Detection

logger = get_logger(__name__)


class OnnxDetector(BaseDetector):
    """Run a YOLO ONNX model without importing torch or Ultralytics."""

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        detector_config: DetectorConfig | None = None,
    ) -> None:
        self.model_config = model_config or ModelConfig()
        self.detector_config = detector_config or DetectorConfig(backend="onnx")
        self.model_path = Path(
            self.detector_config.model_path
            or self.model_config.weights_path.parent
            / "exports"
            / f"{self.model_config.weights_path.stem}.onnx"
        )
        if not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        try:
            runtime = importlib.import_module("onnxruntime")
        except ImportError as error:
            raise ImportError(
                "ONNX inference requires the CPU or GPU ONNX Runtime package. "
                "Install CPU dependencies with `pip install -e '.[onnx]'`."
            ) from error

        available = set(runtime.get_available_providers())
        unavailable = [
            provider
            for provider in self.detector_config.providers
            if provider not in available
        ]
        if unavailable:
            raise RuntimeError(
                f"Requested ONNX Runtime provider(s) are unavailable: {unavailable}. "
                f"Available providers: {sorted(available)}"
            )
        self.providers = list(self.detector_config.providers)
        self._session = runtime.InferenceSession(
            str(self.model_path), providers=self.providers
        )
        self._input = self._session.get_inputs()[0]
        self._outputs = self._session.get_outputs()
        sidecar_path = self.model_path.with_suffix(self.model_path.suffix + ".json")
        if sidecar_path.is_file():
            self.metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
        else:
            logger.warning(
                "ONNX metadata sidecar not found at %s; using inferred defaults",
                sidecar_path,
            )
            self.metadata = {}
        self.class_names = self.metadata.get("class_names", ["person"])
        if isinstance(self.class_names, dict):
            self.class_names = {
                int(class_id): str(class_name)
                for class_id, class_name in self.class_names.items()
            }
        self.nms_embedded = bool(self.metadata.get("nms_embedded", False))
        declared_precision = self.metadata.get("precision")
        if declared_precision and declared_precision != self.detector_config.precision:
            raise ValueError(
                f"ONNX model {self.model_path} declares {declared_precision} "
                f"precision, but detector.precision is "
                f"{self.detector_config.precision}."
            )
        self.input_height, self.input_width = self._resolve_input_size(
            self._input.shape, self.model_config.imgsz
        )
        self._input_dtype = np.float16 if "float16" in self._input.type else np.float32
        self._class_ids = {
            value for value in self.model_config.classes if isinstance(value, int)
        }
        name_selectors = {
            value.casefold()
            for value in self.model_config.classes
            if isinstance(value, str)
        }
        if isinstance(self.class_names, dict):
            self._class_ids.update(
                int(class_id)
                for class_id, class_name in self.class_names.items()
                if str(class_name).casefold() in name_selectors
            )
        else:
            self._class_ids.update(
                class_id
                for class_id, class_name in enumerate(self.class_names)
                if str(class_name).casefold() in name_selectors
            )
        logger.info(
            "Loaded ONNX model %s with providers %s",
            self.model_path,
            self._session.get_providers(),
        )

    @staticmethod
    def _resolve_input_size(shape: list[Any], imgsz: int) -> tuple[int, int]:
        if len(shape) != 4:
            raise ValueError(f"Expected ONNX NCHW image input, got shape {shape}")
        height = shape[2] if isinstance(shape[2], int) else imgsz
        width = shape[3] if isinstance(shape[3], int) else imgsz
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid ONNX input dimensions: {shape}")
        return int(height), int(width)

    def _preprocess(self, frame: NDArray[Any]) -> tuple[NDArray[Any], LetterboxInfo]:
        return letterbox(
            frame,
            (self.input_height, self.input_width),
            dtype=np.dtype(self._input_dtype),
        )

    def raw_output(self, frame: NDArray[Any]) -> NDArray[Any]:
        """Return the primary model output for numerical parity checks."""
        tensor, _ = self._preprocess(frame)
        output = self._session.run([self._outputs[0].name], {self._input.name: tensor})[
            0
        ]
        return np.asarray(output)

    def detect(self, frame: NDArray[Any]) -> list[Detection]:
        """Run inference and return detections in source-frame pixel coordinates."""
        tensor, geometry = self._preprocess(frame)
        raw_outputs = self._session.run(
            [output.name for output in self._outputs],
            {self._input.name: tensor},
        )
        output = np.asarray(raw_outputs[0])
        common = {
            "class_names": self.class_names,
            "confidence_threshold": self.model_config.conf_threshold,
            "class_ids": self._class_ids or None,
        }
        if self.nms_embedded:
            return decode_nms_output(output, geometry, **common)
        if output.ndim >= 2 and output.shape[-1] == 6:
            return decode_nms_output(output, geometry, **common)
        return decode_yolo_output(
            output,
            geometry,
            iou_threshold=self.model_config.iou_threshold,
            **common,
        )

    def warmup(self) -> None:
        """Run a zero-image inference to initialize the selected execution provider."""
        self.detect(
            np.zeros(
                (self.input_height, self.input_width, 3),
                dtype=np.uint8,
            )
        )
