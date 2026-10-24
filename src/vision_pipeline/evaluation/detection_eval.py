"""Optional held-out Ultralytics detection evaluation."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from vision_pipeline.config import ModelConfig


def evaluate_detector(
    model_config: ModelConfig,
    data_yaml: str | Path,
) -> dict[str, Any]:
    """Run Ultralytics validation on a local labeled dataset.

    Dataset paths and annotations remain external to the repository. This
    reports a true held-out evaluation only when the caller provides a valid
    dataset YAML; Phase 5's stored validation results are not substituted.
    """
    dataset = Path(data_yaml)
    if not dataset.is_file():
        raise FileNotFoundError(f"Detection dataset YAML not found: {dataset}")
    if not model_config.weights_path.is_file():
        raise FileNotFoundError(
            f"Detector weights not found: {model_config.weights_path}"
        )
    try:
        ultralytics = importlib.import_module("ultralytics")
    except ImportError as error:
        raise ImportError(
            "Detection evaluation requires the detection extra: "
            "`python -m pip install -e '.[detection]'`."
        ) from error
    model = ultralytics.YOLO(str(model_config.weights_path))
    metrics = model.val(
        data=str(dataset),
        imgsz=model_config.imgsz,
        conf=model_config.conf_threshold,
        iou=model_config.iou_threshold,
        device=model_config.device,
        verbose=False,
    )
    box_metrics = metrics.box
    return {
        "status": "complete",
        "dataset_yaml": str(dataset),
        "weights_path": str(model_config.weights_path),
        "imgsz": model_config.imgsz,
        "metrics": {
            "map50": float(box_metrics.map50),
            "map50_95": float(box_metrics.map),
            "precision": float(box_metrics.mp),
            "recall": float(box_metrics.mr),
        },
    }


def evaluate_hota(
    *,
    sequence_directory: str | Path,
    predictions_path: str | Path,
) -> dict[str, Any]:
    """Report whether optional TrackEval HOTA evaluation can be run.

    TrackEval requires a dataset-specific adapter and benchmark directory
    layout. The core project currently has no pinned TrackEval adapter, so the
    function explicitly skips instead of returning success-shaped metrics.
    """
    try:
        importlib.import_module("trackeval")
    except ImportError:
        return {
            "status": "skipped",
            "reason": "TrackEval is not installed.",
        }
    return {
        "status": "skipped",
        "reason": (
            "TrackEval is installed, but no dataset adapter is configured for "
            f"sequence '{sequence_directory}' and predictions '{predictions_path}'."
        ),
    }
