"""Numpy YOLO output decoding and class-aware non-maximum suppression."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vision_pipeline.optimization.preprocess import LetterboxInfo, undo_letterbox
from vision_pipeline.types import Detection


def numpy_nms(
    boxes_xyxy: NDArray[Any],
    scores: NDArray[Any],
    iou_threshold: float,
) -> NDArray[np.int64]:
    """Return greedy NMS indices, using vectorized overlap calculations."""
    boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
    confidence = np.asarray(scores, dtype=np.float32).reshape(-1)
    if boxes.shape[0] != confidence.shape[0]:
        raise ValueError("boxes and scores must have the same number of rows")
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be between zero and one")
    if boxes.size == 0:
        return np.empty(0, dtype=np.int64)
    areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0, boxes[:, 3] - boxes[:, 1]
    )
    order = np.argsort(confidence, kind="stable")[::-1]
    kept: list[int] = []
    while order.size:
        current = int(order[0])
        kept.append(current)
        remaining = order[1:]
        if remaining.size == 0:
            break
        left_top = np.maximum(boxes[current, :2], boxes[remaining, :2])
        right_bottom = np.minimum(boxes[current, 2:], boxes[remaining, 2:])
        intersection_size = np.maximum(0, right_bottom - left_top)
        intersection = intersection_size[:, 0] * intersection_size[:, 1]
        union = areas[current] + areas[remaining] - intersection
        overlaps = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0,
        )
        order = remaining[overlaps <= iou_threshold]
    return np.asarray(kept, dtype=np.int64)


def _class_name(class_id: int, names: Mapping[int, str] | Sequence[str]) -> str:
    if isinstance(names, Mapping):
        return str(names.get(class_id, class_id))
    return str(names[class_id]) if class_id < len(names) else str(class_id)


def decode_yolo_output(
    output: NDArray[Any],
    info: LetterboxInfo,
    *,
    class_names: Mapping[int, str] | Sequence[str],
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    class_ids: set[int] | None = None,
    max_detections: int = 300,
) -> list[Detection]:
    """Decode a YOLOv8/v11 ``(1, 4 + nc, N)`` raw prediction tensor."""
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between zero and one")
    prediction = np.asarray(output)
    if prediction.ndim == 3:
        if prediction.shape[0] != 1:
            raise ValueError("decode_yolo_output accepts batch size one only")
        prediction = prediction[0]
    if prediction.ndim != 2:
        raise ValueError("raw YOLO output must have shape (1, 4 + nc, N)")
    expected_channels = (
        len(class_names) + 4
        if not isinstance(class_names, Mapping)
        else max(class_names, default=-1) + 5
    )
    if prediction.shape[0] != expected_channels and prediction.shape[1] == (
        expected_channels
    ):
        prediction = prediction.T
    if prediction.shape[0] < 5:
        raise ValueError("raw YOLO output must contain four box and class scores")
    prediction = prediction.T

    class_scores = prediction[:, 4:]
    class_indices = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_indices]
    selected = scores >= confidence_threshold
    if class_ids is not None:
        selected &= np.isin(class_indices, list(class_ids))
    if not np.any(selected):
        return []

    centers = prediction[selected, :2]
    sizes = prediction[selected, 2:4]
    boxes = np.concatenate((centers - sizes / 2, centers + sizes / 2), axis=1)
    scores = scores[selected].astype(np.float32, copy=False)
    class_indices = class_indices[selected]
    keep: list[int] = []
    for class_id in np.unique(class_indices):
        indices = np.flatnonzero(class_indices == class_id)
        keep.extend(indices[numpy_nms(boxes[indices], scores[indices], iou_threshold)])
    keep_array = np.asarray(keep, dtype=np.int64)
    keep_array = keep_array[np.argsort(scores[keep_array], kind="stable")[::-1]]
    keep_array = keep_array[:max_detections]
    mapped_boxes = undo_letterbox(boxes[keep_array], info)
    detections: list[Detection] = []
    for index, box in zip(keep_array, mapped_boxes, strict=True):
        class_id = int(class_indices[index])
        detections.append(
            Detection(
                (
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                ),
                float(scores[index]),
                class_id,
                _class_name(class_id, class_names),
            )
        )
    return detections


def decode_nms_output(
    output: NDArray[Any],
    info: LetterboxInfo,
    *,
    class_names: Mapping[int, str] | Sequence[str],
    confidence_threshold: float = 0.25,
    class_ids: set[int] | None = None,
) -> list[Detection]:
    """Decode embedded-NMS rows with ``xyxy, confidence, class_id`` columns."""
    rows = np.asarray(output)
    if rows.ndim == 3:
        if rows.shape[0] != 1:
            raise ValueError("decode_nms_output accepts batch size one only")
        rows = rows[0]
    if rows.ndim != 2 or rows.shape[1] < 6:
        raise ValueError("embedded-NMS output must have shape (1, N, 6)")
    rows = rows[:, :6]
    valid = np.isfinite(rows).all(axis=1) & (rows[:, 4] >= confidence_threshold)
    class_values = rows[:, 5].astype(np.int64)
    if class_ids is not None:
        valid &= np.isin(class_values, list(class_ids))
    rows = rows[valid]
    class_values = class_values[valid]
    mapped_boxes = undo_letterbox(rows[:, :4], info)
    return [
        Detection(
            (
                float(box[0]),
                float(box[1]),
                float(box[2]),
                float(box[3]),
            ),
            float(row[4]),
            int(class_id),
            _class_name(int(class_id), class_names),
        )
        for row, box, class_id in zip(rows, mapped_boxes, class_values, strict=True)
        if box[2] > box[0] and box[3] > box[1]
    ]
