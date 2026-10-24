"""Evaluation of line-crossing and zone-dwell analytics against annotations."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from vision_pipeline.analytics.events import Event, EventType


def read_line_ground_truth(path: str | Path) -> list[dict[str, Any]]:
    """Read line annotations with ``timestamp_s`` and ``direction`` columns."""
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for index, row in enumerate(rows, start=2):
        try:
            row["timestamp_s"] = float(row["timestamp_s"])
            row["direction"] = str(row["direction"]).strip().lower()
            if row["direction"] not in {"in", "out"}:
                raise ValueError("direction must be 'in' or 'out'")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{path}:{index}: invalid line ground-truth row"
            ) from error
    return rows


def _event_value(event: Event | dict[str, Any], key: str) -> Any:
    """Read an attribute from either a typed event or a serialized event."""
    if isinstance(event, dict):
        return event.get(key)
    return getattr(event, key)


def _line_event(event: Event | dict[str, Any]) -> dict[str, Any] | None:
    """Normalize line-crossing events into timestamp, direction, and line ID."""
    event_type = _event_value(event, "event_type")
    event_type = getattr(event_type, "value", event_type)
    if event_type not in {
        EventType.LINE_CROSS_IN.value,
        EventType.LINE_CROSS_OUT.value,
    }:
        return None
    return {
        "timestamp_s": float(_event_value(event, "timestamp_s")),
        "direction": "in" if event_type == EventType.LINE_CROSS_IN.value else "out",
        "line_id": _event_value(event, "line_id"),
    }


def _match_timestamps(
    truth: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
    tolerance_s: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Match timestamp events one-to-one using minimum-cost Hungarian matching."""
    if not truth or not predicted:
        return [], list(range(len(truth))), list(range(len(predicted)))
    costs = np.asarray(
        [[abs(a["timestamp_s"] - b["timestamp_s"]) for b in predicted] for a in truth],
        dtype=float,
    )
    costs[costs > tolerance_s] = tolerance_s + 1_000_000.0
    rows, columns = linear_sum_assignment(costs)
    matches = [
        (int(row), int(column))
        for row, column in zip(rows, columns, strict=True)
        if costs[row, column] <= tolerance_s
    ]
    matched_truth = {row for row, _ in matches}
    matched_predicted = {column for _, column in matches}
    return (
        matches,
        [index for index in range(len(truth)) if index not in matched_truth],
        [index for index in range(len(predicted)) if index not in matched_predicted],
    )


def evaluate_line_counts(
    predicted_events: Iterable[Event | dict[str, Any]],
    ground_truth: str | Path | Iterable[dict[str, Any]],
    *,
    tolerance_s: float = 1.5,
    timeline_path: str | Path | None = None,
) -> dict[str, Any]:
    """Match crossing events by line, direction, and time tolerance."""
    if tolerance_s < 0:
        raise ValueError("tolerance_s must be non-negative")
    truth_rows = (
        read_line_ground_truth(ground_truth)
        if isinstance(ground_truth, (str, Path))
        else [dict(row) for row in ground_truth]
    )
    normalized_truth = [
        {
            "timestamp_s": float(row["timestamp_s"]),
            "direction": str(row["direction"]).lower(),
            "line_id": row.get("line_id"),
        }
        for row in truth_rows
    ]
    if any(row["direction"] not in {"in", "out"} for row in normalized_truth):
        raise ValueError("ground-truth direction must be 'in' or 'out'")
    has_line_ids = any(row.get("line_id") not in (None, "") for row in normalized_truth)
    normalized_predicted = [
        row for event in predicted_events if (row := _line_event(event)) is not None
    ]
    if not has_line_ids:
        for row in normalized_predicted:
            row["line_id"] = None
    grouped_truth: dict[tuple[Any, str], list[dict[str, Any]]] = defaultdict(list)
    grouped_pred: dict[tuple[Any, str], list[dict[str, Any]]] = defaultdict(list)
    for row in normalized_truth:
        grouped_truth[(row["line_id"], row["direction"])].append(row)
    for row in normalized_predicted:
        grouped_pred[(row["line_id"], row["direction"])].append(row)
    total_tp = total_fp = total_fn = 0
    direction_metrics: dict[str, dict[str, int | float | None]] = {}
    matched_times: list[tuple[float, str]] = []
    false_positive_times: list[tuple[float, str]] = []
    false_negative_times: list[tuple[float, str]] = []
    for direction in ("in", "out"):
        tp = fp = fn = 0
        for key in sorted(
            set(grouped_truth) | set(grouped_pred),
            key=lambda item: (str(item[0]), item[1]),
        ):
            if key[1] != direction:
                continue
            truths, preds = grouped_truth[key], grouped_pred[key]
            matches, missing_truth, extra_pred = _match_timestamps(
                truths, preds, tolerance_s
            )
            tp += len(matches)
            fn += len(missing_truth)
            fp += len(extra_pred)
            matched_times.extend(
                (preds[column]["timestamp_s"], direction) for _, column in matches
            )
            false_positive_times.extend(
                (preds[index]["timestamp_s"], direction) for index in extra_pred
            )
            false_negative_times.extend(
                (truths[index]["timestamp_s"], direction) for index in missing_truth
            )
        total_tp += tp
        total_fp += fp
        total_fn += fn
        denominator = tp + fp
        ground_truth_count = sum(
            len(rows)
            for (line_id, item_direction), rows in grouped_truth.items()
            if item_direction == direction
        )
        predicted_count = sum(
            len(rows)
            for (line_id, item_direction), rows in grouped_pred.items()
            if item_direction == direction
        )
        direction_count_error = abs(predicted_count - ground_truth_count)
        direction_metrics[direction] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / denominator if denominator else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "predicted_count": predicted_count,
            "ground_truth_count": ground_truth_count,
            "count_error_absolute": direction_count_error,
            "count_error_percent": (
                100.0 * direction_count_error / ground_truth_count
                if ground_truth_count
                else None
            ),
        }
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    count_error = abs(len(normalized_predicted) - len(normalized_truth))
    result = {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_count": len(normalized_predicted),
        "ground_truth_count": len(normalized_truth),
        "count_error_absolute": count_error,
        "count_error_percent": (
            100.0 * count_error / len(normalized_truth) if normalized_truth else None
        ),
        "by_direction": direction_metrics,
    }
    if timeline_path is not None:
        write_event_timeline(
            matched_times,
            false_positive_times,
            false_negative_times,
            timeline_path,
        )
    return result


def evaluate_zone_dwell(
    predicted_events: Iterable[Event | dict[str, Any]],
    ground_truth: str | Path | Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Compare predicted zone visit durations with track-independent labels."""
    if isinstance(ground_truth, (str, Path)):
        with Path(ground_truth).open(encoding="utf-8", newline="") as stream:
            truth_rows = list(csv.DictReader(stream))
    else:
        truth_rows = [dict(row) for row in ground_truth]
    visits: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(
        lambda: {"enter": [], "exit": []}
    )
    for event in predicted_events:
        event_type = _event_value(event, "event_type")
        event_type = getattr(event_type, "value", event_type)
        if event_type not in {EventType.ZONE_ENTER.value, EventType.ZONE_EXIT.value}:
            continue
        zone_id = _event_value(event, "zone_id")
        track_id = int(_event_value(event, "track_id"))
        timestamp = float(_event_value(event, "timestamp_s"))
        edge = "enter" if event_type == EventType.ZONE_ENTER.value else "exit"
        visits[(str(zone_id), track_id)][edge].append(timestamp)
    predicted: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (zone_id, _track_id), values in visits.items():
        enters = sorted(values["enter"])
        exits = sorted(values["exit"])
        for enter, exit_time in zip(enters, exits, strict=False):
            if exit_time >= enter:
                predicted[zone_id].append((enter, exit_time))

    errors: list[float] = []
    matched = 0
    for line_number, row in enumerate(truth_rows, start=2):
        try:
            zone_id = str(row["zone_id"])
            enter = float(row["enter_s"])
            exit_time = float(row["exit_s"])
            if exit_time < enter:
                raise ValueError("exit_s precedes enter_s")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid zone ground-truth row {line_number}") from error
        candidates = predicted.get(zone_id, [])
        if not candidates:
            continue
        best_index = min(
            range(len(candidates)),
            key=lambda index: abs(candidates[index][0] - enter),
        )
        pred_enter, pred_exit = candidates.pop(best_index)
        errors.append(abs((pred_exit - pred_enter) - (exit_time - enter)))
        matched += 1
    return {
        "matched_visits": matched,
        "ground_truth_visits": len(truth_rows),
        "unmatched_ground_truth_visits": len(truth_rows) - matched,
        "unmatched_predicted_visits": sum(map(len, predicted.values())),
        "mean_absolute_error_seconds": float(np.mean(errors)) if errors else None,
    }


def write_event_timeline(
    true_positive: list[tuple[float, str]],
    false_positive: list[tuple[float, str]],
    false_negative: list[tuple[float, str]],
    path: str | Path,
) -> None:
    """Save a compact timeline of matched, extra, and missed line events."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "Timeline charts require matplotlib; install `.[eval]`."
        ) from error
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(10, 3))
    series = (
        (true_positive, "TP", "tab:green", 2),
        (false_positive, "FP", "tab:red", 1),
        (false_negative, "FN", "tab:orange", 0),
    )
    for points, label, color, y in series:
        if points:
            axis.scatter(
                [item[0] for item in points],
                [y] * len(points),
                label=label,
                color=color,
            )
    axis.set_yticks((0, 1, 2), ("FN", "FP", "TP"))
    axis.set_xlabel("Video time (seconds)")
    axis.set_title("Line-crossing events vs ground truth")
    axis.grid(axis="x", alpha=0.25)
    if any(points for points, *_ in series):
        axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(destination, dpi=140)
    plt.close(fig)
