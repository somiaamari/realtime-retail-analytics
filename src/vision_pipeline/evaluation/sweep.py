"""Deterministic ByteTracker hyperparameter search on cached detections."""

from __future__ import annotations

import csv
import itertools
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import yaml

from vision_pipeline.config import TrackerConfig
from vision_pipeline.evaluation.mot_io import MOTRecord
from vision_pipeline.evaluation.tracking_eval import evaluate_tracking
from vision_pipeline.tracking.bytetrack import ByteTracker
from vision_pipeline.types import Detection


def _run_trial(
    job: tuple[
        dict[str, Any],
        dict[str, tuple[list[MOTRecord], dict[int, list[Detection]]]],
    ],
) -> dict[str, Any]:
    """Evaluate one tracker parameter set across a sequence mapping."""
    parameters, sequences = job
    per_sequence: dict[str, dict[str, Any]] = {}
    for name, (ground_truth, detections) in sequences.items():
        tracker = ByteTracker(TrackerConfig(**parameters))
        predictions: list[MOTRecord] = []
        for frame_id in sorted(
            set(detections) | {row.frame - 1 for row in ground_truth}
        ):
            for track in tracker.update(detections.get(frame_id, []), frame_id):
                x1, y1, x2, y2 = track.bbox_xyxy
                predictions.append(
                    MOTRecord(
                        frame_id + 1,
                        track.track_id,
                        x1,
                        y1,
                        x2 - x1,
                        y2 - y1,
                        track.confidence,
                        track.class_id,
                    )
                )
        per_sequence[name] = evaluate_tracking(ground_truth, predictions)
    available_idf1 = [
        result["idf1"] for result in per_sequence.values() if result["idf1"] is not None
    ]
    available_mota = [
        result["mota"] for result in per_sequence.values() if result["mota"] is not None
    ]
    return {
        "parameters": parameters,
        "idf1": sum(available_idf1) / len(available_idf1) if available_idf1 else None,
        "mota": sum(available_mota) / len(available_mota) if available_mota else None,
        "id_switches": sum(
            int(result["num_switches"] or 0) for result in per_sequence.values()
        ),
        "per_sequence": per_sequence,
    }


def sweep_tracker(
    sequences: dict[str, tuple[list[MOTRecord], dict[int, list[Detection]]]],
    *,
    train_sequences: list[str],
    validation_sequences: list[str],
    parameter_grid: dict[str, list[Any]],
    output_directory: str | Path,
    workers: int = 1,
    random_trials: int | None = None,
    seed: int = 1729,
) -> dict[str, Any]:
    """Search tracker parameters and evaluate the winner on held-out sequences.

    ``sequences`` contains already cached detections, so each trial only runs
    ByteTracker. Sequence split names must be disjoint to prevent leakage.
    """
    if workers < 1:
        raise ValueError("workers must be at least one")
    if set(train_sequences) & set(validation_sequences):
        raise ValueError("train and validation sequences must be disjoint")
    missing = (set(train_sequences) | set(validation_sequences)) - set(sequences)
    if missing:
        raise ValueError(f"Unknown benchmark sequences: {sorted(missing)}")
    if not train_sequences or not validation_sequences:
        raise ValueError("at least one train and one validation sequence are required")
    if not parameter_grid:
        raise ValueError("parameter_grid cannot be empty")
    names = list(parameter_grid)
    combinations = [
        dict(zip(names, values, strict=True))
        for values in itertools.product(*(parameter_grid[name] for name in names))
    ]
    if random_trials is not None:
        if random_trials < 1:
            raise ValueError("random_trials must be positive")
        random.Random(seed).shuffle(combinations)
        combinations = combinations[:random_trials]
    train_data = {name: sequences[name] for name in train_sequences}
    jobs = [(parameters, train_data) for parameters in combinations]
    if workers == 1:
        trials = [_run_trial(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            trials = list(executor.map(_run_trial, jobs))
    valid_trials = [trial for trial in trials if trial["idf1"] is not None]
    if not valid_trials:
        raise ValueError("No valid trials produced IDF1 metrics")
    best = max(
        valid_trials,
        key=lambda trial: (
            float(trial["idf1"]),
            float(trial["mota"] or -1),
            -int(trial["id_switches"]),
        ),
    )
    validation = _run_trial(
        (best["parameters"], {name: sequences[name] for name in validation_sequences})
    )
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "tracker_sweep.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["parameters", "idf1", "mota", "id_switches"],
        )
        writer.writeheader()
        for trial in trials:
            writer.writerow(
                {
                    "parameters": yaml.safe_dump(
                        trial["parameters"], default_flow_style=True
                    ).strip(),
                    "idf1": trial["idf1"],
                    "mota": trial["mota"],
                    "id_switches": trial["id_switches"],
                }
            )
    (output / "tracker.tuned.yaml").write_text(
        yaml.safe_dump({"tracker": best["parameters"]}, sort_keys=False),
        encoding="utf-8",
    )
    chart_path = output / "tracker_sensitivity.png"
    _write_sensitivity_chart(trials, names, chart_path)
    return {
        "objective": (
            "mean train-sequence IDF1; tie-break by MOTA then fewer ID switches"
        ),
        "seed": seed,
        "train_sequences": train_sequences,
        "validation_sequences": validation_sequences,
        "trial_count": len(trials),
        "best_train": best,
        "held_out_validation": validation,
        "artifacts": {
            "csv": str(output / "tracker_sweep.csv"),
            "config": str(output / "tracker.tuned.yaml"),
            "chart": str(chart_path) if chart_path.is_file() else None,
        },
        "interpretation_template": (
            "Describe which parameters changed IDF1 most, whether validation "
            "matched training, and likely trade-offs in MOTA and ID switches."
        ),
    }


def _write_sensitivity_chart(
    trials: list[dict[str, Any]],
    parameter_names: list[str],
    destination: Path,
) -> None:
    """Plot trial scores and parameter-level sensitivity when matplotlib exists."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.2, 1]}
    )
    axes[0].plot(
        range(1, len(trials) + 1),
        [trial["idf1"] for trial in trials],
        marker="o",
        linewidth=1,
    )
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel("Mean train IDF1")
    axes[0].set_title("Tracker sweep scores")
    axis = axes[1]
    colors = plt.get_cmap("viridis")
    scores = [float(trial["idf1"] or 0.0) for trial in trials]
    low_score, high_score = min(scores, default=0.0), max(scores, default=1.0)
    value_maps: dict[str, dict[str, float]] = {}
    for parameter in parameter_names:
        unique = sorted({str(trial["parameters"].get(parameter)) for trial in trials})
        value_maps[parameter] = {
            value: index / max(len(unique) - 1, 1) for index, value in enumerate(unique)
        }
    for trial, score in zip(trials, scores, strict=True):
        y_values = [
            value_maps[parameter][str(trial["parameters"].get(parameter))]
            for parameter in parameter_names
        ]
        fraction = (
            (score - low_score) / (high_score - low_score)
            if high_score > low_score
            else 0.5
        )
        axis.plot(
            range(len(parameter_names)),
            y_values,
            color=colors(fraction),
            alpha=0.65,
        )
    axis.set_xticks(
        range(len(parameter_names)), parameter_names, rotation=30, ha="right"
    )
    axis.set_yticks([])
    axis.set_ylabel("Normalized parameter value")
    axis.set_title("Parameter sensitivity (line color = IDF1)")
    fig.tight_layout()
    fig.savefig(destination, dpi=140)
    plt.close(fig)
