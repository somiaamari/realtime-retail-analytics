"""MOT metrics and frame-by-frame tracker evaluation."""

from __future__ import annotations

import hashlib
import importlib
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from vision_pipeline.detection.yolo_detector import BaseDetector
from vision_pipeline.evaluation.mot_io import (
    MOTRecord,
    SequenceInfo,
    iter_sequence_frames,
    read_mot,
    write_mot,
)
from vision_pipeline.logging_utils import get_logger
from vision_pipeline.tracking.base import BaseTracker
from vision_pipeline.types import Detection

logger = get_logger(__name__)

_METRIC_NAMES = (
    "mota",
    "motp",
    "idf1",
    "idp",
    "idr",
    "precision",
    "recall",
    "num_switches",
    "num_fragmentations",
    "mostly_tracked",
    "mostly_lost",
    "num_false_positives",
    "num_misses",
)


def _optional_motmetrics() -> Any:
    """Import motmetrics on demand and explain how to install it."""
    try:
        if not hasattr(np, "asfarray"):
            # motmetrics 1.4 still calls this alias removed in NumPy 2.
            np.asfarray = (  # type: ignore[attr-defined]
                lambda values, dtype=float: np.asarray(values, dtype=dtype)
            )
        return importlib.import_module("motmetrics")
    except ImportError as error:
        raise ImportError(
            "Tracking metrics require the evaluation extra: "
            "`python -m pip install -e '.[eval]'`."
        ) from error


def evaluate_tracking(
    ground_truth: Iterable[MOTRecord],
    predictions: Iterable[MOTRecord],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, float | int | None]:
    """Calculate standard MOT metrics using an IoU matching threshold."""
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be between zero and one")
    motmetrics = _optional_motmetrics()
    gt_by_frame: dict[int, list[MOTRecord]] = defaultdict(list)
    pred_by_frame: dict[int, list[MOTRecord]] = defaultdict(list)
    for row in ground_truth:
        gt_by_frame[row.frame].append(row)
    for row in predictions:
        pred_by_frame[row.frame].append(row)
    accumulator = motmetrics.MOTAccumulator(auto_id=False)
    for frame in sorted(gt_by_frame.keys() | pred_by_frame.keys()):
        gt = gt_by_frame[frame]
        pred = pred_by_frame[frame]
        distances = motmetrics.distances.iou_matrix(
            np.asarray([item.xyxy for item in gt], dtype=float).reshape(-1, 4),
            np.asarray([item.xyxy for item in pred], dtype=float).reshape(-1, 4),
            max_iou=iou_threshold,
        )
        accumulator.update(
            [item.track_id for item in gt],
            [item.track_id for item in pred],
            distances,
            frameid=frame,
        )
    metrics = motmetrics.metrics.create()
    names = list(_METRIC_NAMES)
    frame = metrics.compute(
        accumulator,
        metrics=names,
        name="sequence",
    )
    raw = frame.loc["sequence"].to_dict()
    result: dict[str, float | int | None] = {}
    for name in names:
        value = raw.get(name)
        if value is None or not np.isfinite(value):
            result[name] = None
        elif name in {
            "num_switches",
            "num_fragmentations",
            "mostly_tracked",
            "mostly_lost",
            "num_false_positives",
            "num_misses",
        }:
            result[name] = int(value)
        else:
            result[name] = float(value)
    return result


def _hash_file(path: Path) -> str:
    """Return a content hash used to invalidate stale detection caches."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DetectionCache:
    """Compressed per-frame detection cache keyed by inference parameters."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def cache_path(
        self,
        sequence_name: str,
        *,
        weights_path: str | Path,
        imgsz: int,
        conf_threshold: float,
        iou_threshold: float,
    ) -> Path:
        """Build the cache filename from model and inference configuration."""
        weight_path = Path(weights_path)
        if not weight_path.is_file():
            raise FileNotFoundError(f"Detection cache weights not found: {weight_path}")
        weight_hash = _hash_file(weight_path)
        key = hashlib.sha256(
            f"{sequence_name}|{weight_hash}|{imgsz}|{conf_threshold}|{iou_threshold}".encode()
        ).hexdigest()[:20]
        return self.directory / f"{sequence_name}_{key}.npz"

    @staticmethod
    def load(path: Path) -> dict[int, list[Detection]]:
        """Read detections from a cache archive."""
        with np.load(path, allow_pickle=False) as saved:
            frame_ids = saved["frame_ids"]
            offsets = saved["offsets"]
            boxes = saved["boxes"]
            confidence = saved["confidence"]
            class_ids = saved["class_ids"]
            names = saved["class_names"].astype(str)
        output: dict[int, list[Detection]] = {}
        for index, frame_id in enumerate(frame_ids):
            start, end = int(offsets[index]), int(offsets[index + 1])
            output[int(frame_id)] = [
                Detection(
                    (
                        float(boxes[row, 0]),
                        float(boxes[row, 1]),
                        float(boxes[row, 2]),
                        float(boxes[row, 3]),
                    ),
                    float(confidence[row]),
                    int(class_ids[row]),
                    str(names[row]),
                )
                for row in range(start, end)
            ]
        return output

    @staticmethod
    def save(path: Path, items: dict[int, list[Detection]]) -> None:
        """Atomically write detections as a compressed NumPy archive."""
        frame_ids = sorted(items)
        offsets = [0]
        flat: list[Detection] = []
        for frame_id in frame_ids:
            flat.extend(items[frame_id])
            offsets.append(len(flat))
        boxes = np.asarray([item.bbox_xyxy for item in flat], dtype=np.float64).reshape(
            -1, 4
        )
        confidence = np.asarray([item.confidence for item in flat], dtype=np.float64)
        class_ids = np.asarray([item.class_id for item in flat], dtype=np.int32)
        names = np.asarray([item.class_name for item in flat], dtype=np.str_)
        temporary = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(
            temporary,
            frame_ids=np.asarray(frame_ids, dtype=np.int32),
            offsets=np.asarray(offsets, dtype=np.int32),
            boxes=boxes,
            confidence=confidence,
            class_ids=class_ids,
            class_names=names,
        )
        temporary.replace(path)


def _records_to_detections(
    rows: Iterable[MOTRecord],
) -> dict[int, list[Detection]]:
    """Convert one-based MOT records into zero-based frame detections."""
    output: dict[int, list[Detection]] = defaultdict(list)
    for row in rows:
        output[row.frame - 1].append(
            Detection(row.xyxy, row.confidence, row.class_id, "person")
        )
    return output


def evaluate_tracker_on_sequence(
    detector: BaseDetector | None,
    tracker: BaseTracker,
    sequence: SequenceInfo,
    *,
    ground_truth_path: str | Path | None = None,
    predictions_path: str | Path | None = None,
    detections_cache: DetectionCache | None = None,
    mode: str = "detector",
    public_detections_path: str | Path | None = None,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Run a tracker over a sequence and optionally compute MOT metrics.

    ``mode='public'`` uses supplied public detections; ``mode='oracle'`` feeds
    filtered ground-truth boxes to the tracker to isolate association quality.
    """
    if mode not in {"detector", "public", "oracle"}:
        raise ValueError("mode must be 'detector', 'public', or 'oracle'")
    tracker.reset()
    mode_detections: dict[int, list[Detection]] = {}
    cached_detections: dict[int, list[Detection]] | None = None
    cache_output: Path | None = None
    detections_to_save: dict[int, list[Detection]] = {}
    if mode == "oracle":
        if ground_truth_path is None:
            raise ValueError("oracle mode requires ground_truth_path")
        mode_detections = _records_to_detections(
            read_mot(ground_truth_path, ground_truth=True)
        )
    elif mode == "public":
        if public_detections_path is None:
            raise ValueError("public mode requires public_detections_path")
        mode_detections = _records_to_detections(read_mot(public_detections_path))
    else:
        if detector is None:
            raise ValueError("detector mode requires a detector")
        if detections_cache is not None:
            config = getattr(detector, "config", None)
            cache_output = detections_cache.cache_path(
                sequence.name,
                weights_path=getattr(config, "weights_path", "unknown"),
                imgsz=int(getattr(config, "imgsz", 0)),
                conf_threshold=float(getattr(config, "conf_threshold", 0.0)),
                iou_threshold=float(getattr(config, "iou_threshold", 0.0)),
            )
            if cache_output.is_file():
                logger.info("Loading cached detections from %s", cache_output)
                cached_detections = detections_cache.load(cache_output)

    predictions: list[MOTRecord] = []
    frame_count = 0
    for frame in iter_sequence_frames(sequence):
        frame_count += 1
        if mode == "detector":
            assert detector is not None
            if cached_detections is None:
                frame_detections = detector.detect(frame.frame)
                if detections_cache is not None:
                    detections_to_save[frame.frame_id] = frame_detections
            else:
                frame_detections = cached_detections.get(frame.frame_id, [])
        else:
            frame_detections = mode_detections.get(frame.frame_id, [])
        for track in tracker.update(frame_detections, frame.frame_id):
            x1, y1, x2, y2 = track.bbox_xyxy
            predictions.append(
                MOTRecord(
                    frame.frame_id + 1,
                    track.track_id,
                    x1,
                    y1,
                    x2 - x1,
                    y2 - y1,
                    track.confidence,
                    track.class_id,
                )
            )
    if (
        mode == "detector"
        and detections_cache is not None
        and cached_detections is None
        and cache_output is not None
    ):
        detections_cache.save(cache_output, detections_to_save)
    if predictions_path is not None:
        write_mot(predictions, predictions_path)
    result: dict[str, Any] = {
        "sequence": sequence.name,
        "mode": mode,
        "frames": frame_count,
        "predictions": len(predictions),
        "predictions_path": str(predictions_path) if predictions_path else None,
        "metrics": None,
    }
    if ground_truth_path is not None:
        ground_truth = read_mot(ground_truth_path, ground_truth=True)
        result["metrics"] = evaluate_tracking(
            ground_truth, predictions, iou_threshold=iou_threshold
        )
    return result
