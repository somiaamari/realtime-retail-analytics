"""Numerical and detection parity checks for exported ONNX models."""

from collections.abc import Iterable
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from vision_pipeline.types import Detection


class RawDetector(Protocol):
    """Detector protocol required for parity checks."""

    def raw_output(self, frame: NDArray[Any]) -> NDArray[Any]:
        """Return unfiltered network output for one frame."""

    def detect(self, frame: NDArray[Any]) -> list[Detection]:
        """Return postprocessed source-coordinate detections."""


def _iou(first: Detection, second: Detection) -> float:
    a = first.bbox_xyxy
    b = second.bbox_xyxy
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    union = first.area + second.area - intersection
    return intersection / union if union > 0 else 0.0


def compare_outputs(
    torch_detector: RawDetector,
    onnx_detector: RawDetector,
    images: Iterable[NDArray[Any]],
) -> dict[str, float | int]:
    """Compare raw tensors and detection parity across the provided images.

    A detection is a parity match only when its class agrees, IoU is at least
    0.95, and confidence differs by less than 0.01.
    """
    max_differences: list[float] = []
    mean_differences: list[float] = []
    matched_total = 0
    torch_total = 0
    onnx_total = 0
    image_count = 0
    for image in images:
        torch_raw = np.asarray(torch_detector.raw_output(image), dtype=np.float32)
        onnx_raw = np.asarray(onnx_detector.raw_output(image), dtype=np.float32)
        if torch_raw.shape != onnx_raw.shape:
            raise ValueError(
                "Raw model output shapes differ: "
                f"PyTorch {torch_raw.shape} versus ONNX {onnx_raw.shape}"
            )
        difference = np.abs(torch_raw - onnx_raw)
        max_differences.append(float(difference.max(initial=0.0)))
        mean_differences.append(float(difference.mean()))
        torch_detections = torch_detector.detect(image)
        onnx_detections = onnx_detector.detect(image)
        torch_total += len(torch_detections)
        onnx_total += len(onnx_detections)
        used: set[int] = set()
        for reference in sorted(
            torch_detections, key=lambda detection: detection.confidence, reverse=True
        ):
            candidates = [
                (index, candidate)
                for index, candidate in enumerate(onnx_detections)
                if index not in used
                and candidate.class_id == reference.class_id
                and abs(candidate.confidence - reference.confidence) < 0.01
            ]
            if candidates:
                index, candidate = max(
                    candidates, key=lambda pair: _iou(reference, pair[1])
                )
                if _iou(reference, candidate) >= 0.95:
                    used.add(index)
                    matched_total += 1
        image_count += 1
    if image_count == 0:
        raise ValueError("At least one validation image is required")
    denominator = max(torch_total, onnx_total, 1)
    return {
        "images": image_count,
        "max_absolute_difference": max(max_differences, default=0.0),
        "mean_absolute_difference": float(np.mean(mean_differences)),
        "parity_matches": matched_total,
        "torch_detections": torch_total,
        "onnx_detections": onnx_total,
        "detection_parity": matched_total / denominator,
    }
