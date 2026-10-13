"""Pairwise box distances and thresholded linear assignment."""

from collections.abc import Sequence
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from vision_pipeline.types import Detection, Track


def pairwise_iou(
    boxes_a: NDArray[np.float64],
    boxes_b: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute pairwise IoU for two ``(N, 4)`` and ``(M, 4)`` xyxy arrays."""
    boxes_a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    boxes_b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)

    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    intersection_size = np.maximum(bottom_right - top_left, 0.0)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    area_a = np.maximum(boxes_a[:, 2] - boxes_a[:, 0], 0.0) * np.maximum(
        boxes_a[:, 3] - boxes_a[:, 1], 0.0
    )
    area_b = np.maximum(boxes_b[:, 2] - boxes_b[:, 0], 0.0) * np.maximum(
        boxes_b[:, 3] - boxes_b[:, 1], 0.0
    )
    union = area_a[:, None] + area_b[None, :] - intersection
    return cast(
        NDArray[np.float64],
        np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0.0,
        ),
    )


def iou_distance(
    tracks: Sequence[Track],
    detections: Sequence[Detection],
    *,
    fuse_score: bool = False,
) -> NDArray[np.float64]:
    """Return ``1 - IoU`` costs, optionally weighted by detection confidence."""
    track_boxes = np.asarray(
        [track.bbox_xyxy for track in tracks], dtype=np.float64
    ).reshape(-1, 4)
    detection_boxes = np.asarray(
        [detection.bbox_xyxy for detection in detections], dtype=np.float64
    ).reshape(-1, 4)
    similarities = pairwise_iou(track_boxes, detection_boxes)
    if fuse_score and detections:
        scores = np.asarray([item.confidence for item in detections], dtype=np.float64)
        similarities = similarities * scores[None, :]
    return 1.0 - similarities


def linear_assignment(
    cost_matrix: NDArray[np.float64],
    thresh: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Solve minimum-cost assignment and discard pairs above ``thresh``.

    Args:
        cost_matrix: Matrix with tracks in rows and detections in columns.
        thresh: Maximum accepted assignment cost.

    Returns:
        Accepted ``(track_index, detection_index)`` pairs, unmatched track
        indices, and unmatched detection indices.
    """
    costs = np.asarray(cost_matrix, dtype=np.float64)
    if costs.ndim != 2:
        raise ValueError("cost_matrix must be two-dimensional")
    track_count, detection_count = costs.shape
    unmatched_tracks = list(range(track_count))
    unmatched_detections = list(range(detection_count))
    if track_count == 0 or detection_count == 0:
        return [], unmatched_tracks, unmatched_detections

    finite = np.isfinite(costs)
    valid = finite & (costs <= thresh)
    if not valid.any():
        return [], unmatched_tracks, unmatched_detections
    sentinel = max(1e6, float(np.max(np.abs(costs[finite]))) + 1e6)
    safe_costs = np.where(valid, costs, sentinel)
    row_indices, column_indices = linear_sum_assignment(safe_costs)
    matches: list[tuple[int, int]] = []
    for row, column in zip(
        row_indices.tolist(),
        column_indices.tolist(),
        strict=True,
    ):
        if valid[row, column]:
            matches.append((row, column))
    matched_tracks = {row for row, _ in matches}
    matched_detections = {column for _, column in matches}
    unmatched_tracks = [
        index for index in unmatched_tracks if index not in matched_tracks
    ]
    unmatched_detections = [
        index for index in unmatched_detections if index not in matched_detections
    ]
    return matches, unmatched_tracks, unmatched_detections
