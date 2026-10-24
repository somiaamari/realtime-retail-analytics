"""Frame-level attribution of detector and tracker failures."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2

from vision_pipeline.evaluation.mot_io import MOTRecord


def _frame_matches(
    truth: list[MOTRecord],
    predictions: list[MOTRecord],
    threshold: float,
) -> list[tuple[int, int]]:
    """Return greedy one-to-one IoU matches in a single frame."""
    candidates: list[tuple[float, int, int]] = []
    for truth_index, gt in enumerate(truth):
        gx1, gy1, gx2, gy2 = gt.xyxy
        for pred_index, pred in enumerate(predictions):
            px1, py1, px2, py2 = pred.xyxy
            intersection = max(0.0, min(gx2, px2) - max(gx1, px1)) * max(
                0.0, min(gy2, py2) - max(gy1, py1)
            )
            union = gt.width * gt.height + pred.width * pred.height - intersection
            iou = intersection / union if union > 0 else 0.0
            if iou >= threshold:
                candidates.append((iou, truth_index, pred_index))
    used_truth: set[int] = set()
    used_predictions: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _iou, truth_index, pred_index in sorted(candidates, reverse=True):
        if truth_index not in used_truth and pred_index not in used_predictions:
            used_truth.add(truth_index)
            used_predictions.add(pred_index)
            matches.append((truth_index, pred_index))
    return matches


def analyze_tracking_errors(
    ground_truth: Iterable[MOTRecord],
    detector_detections: Iterable[MOTRecord],
    tracker_predictions: Iterable[MOTRecord],
    *,
    oracle_predictions: Iterable[MOTRecord] | None = None,
    iou_threshold: float = 0.5,
    image_paths: dict[int, str | Path] | None = None,
    crops_directory: str | Path | None = None,
    top_n: int = 10,
) -> dict[str, Any]:
    """Count frame-level failure classes and optionally save annotated crops.

    MOT IDs are evaluated per sequence. A detector miss is a GT box unmatched
    to a detector box; oracle-recovered misses are candidates for association
    or tracking failures rather than detector recall failures.
    """
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between zero and one")
    if top_n < 1:
        raise ValueError("top_n must be at least one")
    gt_by_frame: dict[int, list[MOTRecord]] = {}
    det_by_frame: dict[int, list[MOTRecord]] = {}
    pred_by_frame: dict[int, list[MOTRecord]] = {}
    oracle_by_frame: dict[int, list[MOTRecord]] = {}
    for row in ground_truth:
        gt_by_frame.setdefault(row.frame, []).append(row)
    for row in detector_detections:
        det_by_frame.setdefault(row.frame, []).append(row)
    for row in tracker_predictions:
        pred_by_frame.setdefault(row.frame, []).append(row)
    for row in oracle_predictions or ():
        oracle_by_frame.setdefault(row.frame, []).append(row)
    frame_failures: Counter[int] = Counter()
    false_negatives = false_positives = id_switches = fragmentations = 0
    oracle_recovered = 0
    last_identity: dict[int, int] = {}
    last_seen: dict[int, int] = {}
    for frame in sorted(set(gt_by_frame) | set(det_by_frame) | set(pred_by_frame)):
        gt = gt_by_frame.get(frame, [])
        detections = det_by_frame.get(frame, [])
        detector_matches = _frame_matches(gt, detections, iou_threshold)
        detector_fn = len(gt) - len(detector_matches)
        detector_fp = len(detections) - len(detector_matches)
        false_negatives += detector_fn
        false_positives += detector_fp
        frame_failures[frame] += detector_fn + detector_fp

        predictions = pred_by_frame.get(frame, [])
        matches = _frame_matches(gt, predictions, iou_threshold)
        frame_failures[frame] += (
            len(gt) - len(matches) + len(predictions) - len(matches)
        )
        for gt_index, pred_index in matches:
            gt_id = gt[gt_index].track_id
            pred_id = predictions[pred_index].track_id
            if gt_id in last_identity and last_identity[gt_id] != pred_id:
                id_switches += 1
                frame_failures[frame] += 1
            if gt_id in last_seen and frame - last_seen[gt_id] > 1:
                fragmentations += 1
                frame_failures[frame] += 1
            last_identity[gt_id] = pred_id
            last_seen[gt_id] = frame
        if oracle_by_frame:
            oracle_matches = _frame_matches(
                gt, oracle_by_frame.get(frame, []), iou_threshold
            )
            oracle_recovered += max(0, len(oracle_matches) - len(matches))
    result: dict[str, Any] = {
        "detector_false_negatives": false_negatives,
        "detector_false_positives": false_positives,
        "id_switches": id_switches,
        "fragmentations": fragmentations,
        "oracle_recovered_associations": oracle_recovered,
        "top_failure_frames": [
            {"frame": frame, "failure_count": count}
            for frame, count in frame_failures.most_common(top_n)
        ],
        "limitations": [
            "Automatic attribution cannot reliably label occlusion without explicit "
            "occlusion annotations.",
            "Oracle comparison estimates association loss; it is not a causal proof.",
        ],
    }
    if crops_directory is not None and image_paths is not None:
        result["saved_crops"] = _save_failure_crops(
            result["top_failure_frames"],
            image_paths,
            Path(crops_directory),
            gt_by_frame,
            pred_by_frame,
        )
    return result


def _save_failure_crops(
    failures: list[dict[str, int]],
    image_paths: dict[int, str | Path],
    destination: Path,
    ground_truth: dict[int, list[MOTRecord]],
    predictions: dict[int, list[MOTRecord]],
) -> list[str]:
    """Draw ground-truth/prediction boxes and save crops around failure regions."""
    destination.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for item in failures:
        frame_number = item["frame"]
        source = image_paths.get(frame_number)
        if source is None:
            continue
        frame = cv2.imread(str(source))
        if frame is None:
            raise OSError(f"Unable to decode failure frame image: {source}")
        boxes = [
            ("GT", row, (0, 200, 0)) for row in ground_truth.get(frame_number, [])
        ] + [("Pred", row, (0, 0, 255)) for row in predictions.get(frame_number, [])]
        if boxes:
            height, width = frame.shape[:2]
            x1 = max(0, int(min(row.x for _, row, _ in boxes)) - 12)
            y1 = max(0, int(min(row.y for _, row, _ in boxes)) - 12)
            x2 = min(
                width,
                int(max(row.x + row.width for _, row, _ in boxes)) + 12,
            )
            y2 = min(
                height,
                int(max(row.y + row.height for _, row, _ in boxes)) + 12,
            )
            if x2 <= x1 or y2 <= y1:
                x1, y1, x2, y2 = 0, 0, width, height
            frame = frame[y1:y2, x1:x2].copy()
            for label, row, color in boxes:
                left, top, right, bottom = (
                    int(row.x - x1),
                    int(row.y - y1),
                    int(row.x + row.width - x1),
                    int(row.y + row.height - y1),
                )
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(
                    frame,
                    f"{label} {row.track_id}",
                    (left, max(14, top - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                )
        cv2.putText(
            frame,
            f"Failures: {item['failure_count']}",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        output_path = destination / f"failure_frame_{frame_number:06d}.jpg"
        if not cv2.imwrite(str(output_path), frame):
            raise OSError(f"Unable to save failure crop: {output_path}")
        saved.append(str(output_path))
    return saved
