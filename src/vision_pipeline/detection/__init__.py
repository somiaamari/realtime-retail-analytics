"""Detector interfaces, implementations, and configuration-based construction."""

from vision_pipeline.detection.factory import create_detector, select_backend
from vision_pipeline.detection.onnx_detector import OnnxDetector
from vision_pipeline.detection.tensorrt_detector import TensorRTDetector
from vision_pipeline.detection.yolo_detector import (
    BaseDetector,
    FakeDetector,
    YoloDetector,
)

__all__ = [
    "BaseDetector",
    "FakeDetector",
    "OnnxDetector",
    "TensorRTDetector",
    "YoloDetector",
    "create_detector",
    "select_backend",
]
