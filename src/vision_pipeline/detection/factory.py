"""Configuration-driven detector backend selection."""

import importlib.util
from collections.abc import Mapping
from pathlib import Path

from vision_pipeline.config import AppConfig
from vision_pipeline.detection.yolo_detector import BaseDetector
from vision_pipeline.logging_utils import get_logger

logger = get_logger(__name__)


def _available_backends() -> dict[str, bool]:
    """Check optional runtime modules without importing heavy backends."""
    return {
        "pytorch": importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("ultralytics") is not None,
        "onnx": importlib.util.find_spec("onnxruntime") is not None,
        "tensorrt": importlib.util.find_spec("tensorrt") is not None
        and importlib.util.find_spec("pycuda") is not None,
    }


def _candidate_path(config: AppConfig, backend: str) -> Path:
    if config.detector.model_path is not None:
        return config.detector.model_path
    weights = config.model.weights_path
    if backend == "onnx":
        return weights.parent / "exports" / f"{weights.stem}.onnx"
    if backend == "tensorrt":
        return weights.parent / "exports" / f"{weights.stem}.engine"
    return weights


def select_backend(
    config: AppConfig,
    available_backends: Mapping[str, bool] | None = None,
) -> str:
    """Select the fastest configured backend with its model artifact available."""
    requested = config.detector.backend
    if requested != "auto":
        return requested
    available = dict(available_backends or _available_backends())
    preference: tuple[str, ...]
    if config.detector.precision == "fp32":
        preference = ("onnx", "pytorch")
    elif config.detector.precision == "fp16":
        preference = ("tensorrt", "onnx")
    else:
        preference = ("onnx",)
    for backend in preference:
        if available.get(backend, False) and _candidate_path(config, backend).is_file():
            return backend
    raise RuntimeError(
        "No usable detector backend/model artifact was found for backend=auto. "
        "Install the requested runtime and configure detector.model_path."
    )


def create_detector(config: AppConfig) -> BaseDetector:
    """Instantiate the detector selected by the validated application config."""
    backend = select_backend(config)
    if backend == "pytorch":
        if config.detector.precision != "fp32":
            raise ValueError("The PyTorch backend currently supports fp32 only")
        from vision_pipeline.detection.yolo_detector import YoloDetector

        return YoloDetector(config.model)
    if backend == "onnx":
        from vision_pipeline.detection.onnx_detector import OnnxDetector

        settings = config.detector.model_copy(
            update={"model_path": _candidate_path(config, backend)}
        )
        return OnnxDetector(config.model, settings)
    if backend == "tensorrt":
        from vision_pipeline.detection.tensorrt_detector import TensorRTDetector

        settings = config.detector.model_copy(
            update={"model_path": _candidate_path(config, backend)}
        )
        return TensorRTDetector(config.model, settings)
    raise ValueError(f"Unsupported detector backend: {backend}")
