"""Run configured Phase 6 evaluation and write a combined report."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from vision_pipeline.analytics.engine import AnalyticsEngine
from vision_pipeline.config import load_config
from vision_pipeline.detection.yolo_detector import YoloDetector
from vision_pipeline.evaluation.analytics_eval import (
    evaluate_line_counts,
    evaluate_zone_dwell,
)
from vision_pipeline.evaluation.detection_eval import evaluate_detector, evaluate_hota
from vision_pipeline.evaluation.error_analysis import analyze_tracking_errors
from vision_pipeline.evaluation.mot_io import (
    MOTRecord,
    iter_sequence_frames,
    load_sequence,
    read_mot,
)
from vision_pipeline.evaluation.perf import benchmark_pipeline
from vision_pipeline.evaluation.report import (
    collect_environment,
    load_targets,
    set_reproducible_seed,
    write_phase6_report,
)
from vision_pipeline.evaluation.sweep import sweep_tracker
from vision_pipeline.evaluation.tracking_eval import (
    DetectionCache,
    evaluate_tracker_on_sequence,
    evaluate_tracking,
)
from vision_pipeline.logging_utils import setup_logging
from vision_pipeline.streaming.sources import create_source
from vision_pipeline.tracking.bytetrack import ByteTracker
from vision_pipeline.types import Detection

logger = logging.getLogger(__name__)


def _path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _mean_metrics(sequence_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Average available per-sequence scalar metrics and sum count metrics."""
    metrics = [item.get("metrics") or {} for item in sequence_results]
    output: dict[str, Any] = {"sequences": sequence_results}
    for name in (
        "mota",
        "motp",
        "idf1",
        "idp",
        "idr",
        "precision",
        "recall",
    ):
        values = [item[name] for item in metrics if item.get(name) is not None]
        output[name] = sum(values) / len(values) if values else None
    for name in (
        "num_switches",
        "num_fragmentations",
        "num_false_positives",
        "num_misses",
    ):
        output[name] = sum(int(item.get(name) or 0) for item in metrics)
    return output


def main(argv: list[str] | None = None) -> int:
    """Execute available Phase 6 evaluations and clearly mark missing data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/eval.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/phase6"))
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args(argv)
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max-frames must be positive")
    setup_logging()
    root = Path.cwd()
    with args.config.open(encoding="utf-8") as stream:
        config_data: dict[str, Any] = yaml.safe_load(stream) or {}
        seed = int(config_data.get("seed", 1729))
        set_reproducible_seed(seed)
    app_config_path = _path(root, config_data.get("app_config", "configs/default.yaml"))
    app_config = load_config(app_config_path)
    targets_path = _path(root, config_data.get("targets", "configs/eval_targets.yaml"))
    targets = load_targets(targets_path)
    cache = DetectionCache(
        _path(root, config_data.get("detection_cache", "data/benchmarks/cache"))
    )
    results: dict[str, Any] = {
        "detection": {
            "status": "pending",
            "reason": "Phase 6 held-out detection labels not configured.",
        },
        "tracking": {
            "status": "pending",
            "reason": "No evaluation sequences configured.",
        },
        "analytics": {
            "status": "pending",
            "reason": "Manual line/zone ground truth not configured.",
        },
        "performance": {
            "status": "pending",
            "reason": "No performance video configured.",
        },
        "sweep": {
            "status": "pending",
            "reason": "No disjoint tracker sweep split configured.",
        },
        "error_analysis": {
            "status": "pending",
            "reason": "No sequence-level predictions available.",
        },
    }
    if isinstance(config_data.get("phase5_detection_metrics"), dict):
        results["detection"] = {
            "metrics": config_data["phase5_detection_metrics"],
            "note": "Phase 5 validation values; not a Phase 6 held-out evaluation.",
        }
    detection_yaml = config_data.get("detection_data_yaml")
    if detection_yaml:
        results["detection"] = evaluate_detector(
            app_config.model,
            _path(root, detection_yaml),
        )
    sequence_config = config_data.get("sequences", [])
    performance_source = config_data.get("performance_source")
    requires_detector = bool(performance_source) or any(
        item.get("mode", "detector") == "detector" for item in sequence_config
    )
    detector = YoloDetector(app_config.model) if requires_detector else None
    if detector is not None:
        detector.warmup()
    sequence_results: list[dict[str, Any]] = []
    error_results: list[dict[str, Any]] = []
    sweep_sequences: dict[str, Any] = {}
    line_results: list[dict[str, Any]] = []
    zone_results: list[dict[str, Any]] = []
    line_charts: list[str] = []
    combined_ground_truth: list[MOTRecord] = []
    combined_predictions: list[MOTRecord] = []
    frame_offset = 0
    identity_offset = 0
    for item in sequence_config:
        name = item["name"]
        sequence = load_sequence(_path(root, item["directory"]))
        gt_path = _path(root, item["ground_truth"])
        prediction_path = args.output_dir / "predictions" / f"{name}.txt"
        tracker = ByteTracker(app_config.tracker)
        mode = item.get("mode", "detector")
        tracking = evaluate_tracker_on_sequence(
            detector,
            tracker,
            sequence,
            ground_truth_path=gt_path,
            predictions_path=prediction_path,
            detections_cache=cache,
            mode=mode,
            public_detections_path=(
                _path(root, item["public_detections"])
                if item.get("public_detections")
                else None
            ),
        )
        sequence_results.append(tracking)
        truth = read_mot(gt_path, ground_truth=True)
        predictions = read_mot(prediction_path)
        if mode == "oracle":
            oracle_predictions = predictions
        else:
            oracle_path = args.output_dir / "predictions" / f"{name}_oracle.txt"
            oracle_result = evaluate_tracker_on_sequence(
                None,
                ByteTracker(app_config.tracker),
                sequence,
                ground_truth_path=gt_path,
                predictions_path=oracle_path,
                mode="oracle",
            )
            tracking["oracle_metrics"] = oracle_result["metrics"]
            oracle_predictions = read_mot(oracle_path)
        tracking["hota"] = evaluate_hota(
            sequence_directory=sequence.directory,
            predictions_path=prediction_path,
        )
        combined_ground_truth.extend(
            replace(
                row,
                frame=row.frame + frame_offset,
                track_id=row.track_id + identity_offset,
            )
            for row in truth
        )
        combined_predictions.extend(
            replace(
                row,
                frame=row.frame + frame_offset,
                track_id=row.track_id + identity_offset,
            )
            for row in predictions
        )
        max_frame = max((row.frame for row in truth), default=0)
        max_identity = max(
            (row.track_id for row in (*truth, *predictions)),
            default=0,
        )
        frame_offset += max_frame + 1
        identity_offset += max_identity + 1
        if mode == "detector":
            cache_path = cache.cache_path(
                name,
                weights_path=app_config.model.weights_path,
                imgsz=app_config.model.imgsz,
                conf_threshold=app_config.model.conf_threshold,
                iou_threshold=app_config.model.iou_threshold,
            )
            cached = cache.load(cache_path)
        else:
            detection_source = (
                _path(root, item["public_detections"]) if mode == "public" else gt_path
            )
            detection_records = read_mot(
                detection_source,
                ground_truth=mode == "oracle",
            )
            cached_from_file: dict[int, list[Detection]] = {}
            for row in detection_records:
                cached_from_file.setdefault(row.frame - 1, []).append(
                    Detection(row.xyxy, row.confidence, row.class_id, "person")
                )
            cached = cached_from_file
        if mode != "oracle":
            sweep_sequences[name] = (truth, cached)
        analytics = AnalyticsEngine(app_config.analytics)
        tracker.reset()
        local_events: list[Any] = []
        for frame in iter_sequence_frames(sequence):
            tracks = tracker.update(cached.get(frame.frame_id, []), frame.frame_id)
            local_events.extend(analytics.update(tracks, frame))
        analytics.finalize()
        line_truth_path = item.get("line_ground_truth")
        if line_truth_path:
            chart_relative = f"assets/{name}_events.png"
            line_results.append(
                evaluate_line_counts(
                    local_events,
                    _path(root, line_truth_path),
                    timeline_path=args.output_dir / chart_relative,
                )
            )
            line_charts.append(chart_relative)
        zone_truth_path = item.get("zone_ground_truth")
        if zone_truth_path:
            zone_results.append(
                evaluate_zone_dwell(local_events, _path(root, zone_truth_path))
            )
        detection_rows = []
        for frame_id, frame_detections in cached.items():
            for detection_id, detection in enumerate(frame_detections, start=1):
                x1, y1, x2, y2 = detection.bbox_xyxy
                detection_rows.append(
                    MOTRecord(
                        frame_id + 1,
                        frame_id * 1_000_000 + detection_id,
                        x1,
                        y1,
                        x2 - x1,
                        y2 - y1,
                        detection.confidence,
                        detection.class_id,
                    )
                )
        image_paths: dict[int, str | Path] = {}
        if sequence.image_directory is not None:
            for row in truth:
                candidates = (
                    sequence.image_directory / f"{row.frame:06d}.jpg",
                    sequence.image_directory / f"{row.frame:06d}.png",
                )
                image_path = next(
                    (candidate for candidate in candidates if candidate.is_file()),
                    None,
                )
                if image_path is not None:
                    image_paths[row.frame] = image_path
        error_results.append(
            {
                "sequence": name,
                **analyze_tracking_errors(
                    truth,
                    detection_rows,
                    predictions,
                    oracle_predictions=oracle_predictions,
                    image_paths=image_paths or None,
                    crops_directory=(
                        args.output_dir / "crops" / name if image_paths else None
                    ),
                ),
            }
        )

    if sequence_results:
        results["tracking"] = {
            "metrics": evaluate_tracking(combined_ground_truth, combined_predictions),
            "sequence_macro_metrics": _mean_metrics(sequence_results),
            "sequences": sequence_results,
        }
        results["error_analysis"] = {
            "known_weaknesses": [
                "Review the per-sequence frame failure counts; automatic attribution "
                "does not establish occlusion as the cause without occlusion labels."
            ],
            "sequences": error_results,
            "detector_false_negatives": sum(
                item["detector_false_negatives"] for item in error_results
            ),
            "detector_false_positives": sum(
                item["detector_false_positives"] for item in error_results
            ),
            "id_switches": sum(item["id_switches"] for item in error_results),
            "fragmentations": sum(item["fragmentations"] for item in error_results),
        }
    if line_results:
        results["analytics"] = {
            "line_counts": line_results,
            "zone_dwell": zone_results,
            "line_f1_mean": sum(item["f1"] for item in line_results)
            / len(line_results),
            "line_count_error_percent_mean": (
                sum(item["count_error_percent"] or 0 for item in line_results)
                / len(line_results)
            ),
            "zone_dwell_mae_mean_seconds": (
                sum(item["mean_absolute_error_seconds"] or 0 for item in zone_results)
                / len(zone_results)
                if zone_results
                else None
            ),
            "charts": line_charts,
        }
    if performance_source:
        assert detector is not None
        source = create_source(
            app_config.video.model_copy(
                update={"source": str(_path(root, performance_source))}
            )
        )
        tracker = ByteTracker(app_config.tracker)
        analytics = AnalyticsEngine(app_config.analytics)
        with source:
            perf_result = benchmark_pipeline(
                source,
                detector,
                tracker,
                analytics,
                warmup_frames=int(config_data.get("warmup_frames", 5)),
                realtime_fps=float(config_data.get("realtime_fps", 30)),
                max_frames=args.max_frames,
            )
        results["performance"] = {"metrics": perf_result}
    sweep_config = config_data.get("tracker_sweep")
    if sweep_config:
        results["sweep"] = sweep_tracker(
            sweep_sequences,
            train_sequences=list(sweep_config.get("train_sequences", [])),
            validation_sequences=list(sweep_config.get("validation_sequences", [])),
            parameter_grid=sweep_config.get("parameter_grid", {}),
            output_directory=args.output_dir / "sweep",
            workers=int(sweep_config.get("workers", 1)),
            random_trials=sweep_config.get("random_trials"),
            seed=int(sweep_config.get("seed", 1729)),
        )
    environment = collect_environment(
        repository_root=root,
        config_paths=[args.config, app_config_path, targets_path],
    )
    environment["seed"] = seed
    report_path = write_phase6_report(
        args.output_dir,
        environment=environment,
        results=results,
        targets=targets,
    )
    logger.info("Phase 6 report written to %s", report_path)
    print(
        json.dumps(
            {"report": str(report_path), "results": results}, indent=2, default=str
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
