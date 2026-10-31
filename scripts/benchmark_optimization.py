"""Benchmark model backends and enforce Phase 7 accuracy tolerances."""

import argparse
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import yaml

from vision_pipeline.analytics.engine import AnalyticsEngine
from vision_pipeline.config import (
    AppConfig,
    DetectorConfig,
    ModelConfig,
    load_config,
)
from vision_pipeline.detection.factory import create_detector
from vision_pipeline.evaluation.analytics_eval import evaluate_line_counts
from vision_pipeline.evaluation.detection_eval import evaluate_detector
from vision_pipeline.evaluation.mot_io import iter_sequence_frames, load_sequence
from vision_pipeline.evaluation.tracking_eval import evaluate_tracker_on_sequence
from vision_pipeline.logging_utils import get_logger, setup_logging
from vision_pipeline.tracking.bytetrack import ByteTracker

logger = get_logger(__name__)


@dataclass(frozen=True)
class Variant:
    """One named backend model artifact."""

    name: str
    backend: str
    precision: str
    path: Path


@dataclass
class VariantResult:
    """Measured values and evaluation metrics for a backend variant."""

    variant: Variant
    latency_ms: list[float] = field(default_factory=list)
    map50: float | None = None
    map50_95: float | None = None
    idf1: float | None = None
    mota: float | None = None
    line_count_error_percent: float | None = None
    size_bytes: int | None = None
    reason: str | None = None
    accuracy_status: str = "NOT EVALUATED"
    accuracy_failures: list[str] = field(default_factory=list)


def build_parser() -> argparse.ArgumentParser:
    """Build the optimization benchmark CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--weights", type=Path, default=Path("models/retail_people_clean.pt")
    )
    parser.add_argument("--video", type=Path, default=Path("data/raw/vid.mp4"))
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--data-yaml", type=Path, help="Held-out Ultralytics detection dataset"
    )
    parser.add_argument(
        "--mot-sequence", type=Path, help="MOT sequence directory with seqinfo.ini"
    )
    parser.add_argument(
        "--mot-ground-truth", type=Path, help="MOT gt/gt.txt annotation path"
    )
    parser.add_argument(
        "--line-ground-truth", type=Path, help="Line-event ground-truth CSV"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/OPTIMIZATION.md")
    )
    parser.add_argument(
        "--targets", type=Path, default=Path("configs/eval_targets.yaml")
    )
    parser.add_argument(
        "--onnx-fp32",
        type=Path,
        default=Path("models/exports/retail_people_clean.onnx"),
    )
    parser.add_argument(
        "--onnx-fp16",
        type=Path,
        default=Path("models/exports/retail_people_clean_fp16.onnx"),
    )
    parser.add_argument(
        "--onnx-int8",
        type=Path,
        default=Path("models/exports/retail_people_clean_int8.onnx"),
    )
    parser.add_argument(
        "--tensorrt-engine",
        type=Path,
        default=Path("models/exports/retail_people_clean.engine"),
    )
    return parser


def _variants(args: argparse.Namespace) -> list[Variant]:
    return [
        Variant("PyTorch FP32", "pytorch", "fp32", args.weights),
        Variant("ONNX FP32", "onnx", "fp32", args.onnx_fp32),
        Variant("ONNX FP16", "onnx", "fp16", args.onnx_fp16),
        Variant("ONNX INT8 (CPU)", "onnx", "int8", args.onnx_int8),
        Variant("TensorRT FP16", "tensorrt", "fp16", args.tensorrt_engine),
    ]


def _benchmark_latency(
    app_config: AppConfig,
    variant: Variant,
    video_path: Path,
    *,
    frames: int,
    warmup: int,
    imgsz: int,
) -> list[float]:
    """Measure detector-only per-frame latency from a local video."""
    config = app_config.model.model_copy(
        update={"weights_path": variant.path, "imgsz": imgsz}
    )
    providers = (
        ["CPUExecutionProvider"]
        if variant.name == "ONNX INT8 (CPU)"
        else app_config.detector.providers
    )
    detector_config = DetectorConfig(
        backend=variant.backend,
        precision=variant.precision,
        providers=providers,
        model_path=variant.path,
    )
    detector = create_detector(
        app_config.model_copy(update={"model": config, "detector": detector_config})
    )
    detector.warmup()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"Could not open benchmark video: {video_path}")
    durations: list[float] = []
    try:
        for index in range(warmup + frames):
            ok, frame = capture.read()
            if not ok:
                break
            if index < warmup:
                detector.detect(frame)
                continue
            started = time.perf_counter()
            detector.detect(frame)
            durations.append((time.perf_counter() - started) * 1000)
    finally:
        capture.release()
        close = getattr(detector, "close", None)
        if callable(close):
            close()
    return durations


def _detection_metrics(result: VariantResult, data_yaml: Path, imgsz: int) -> None:
    """Measure held-out mAP for one local model artifact."""
    measured = evaluate_detector(
        ModelConfig(weights_path=result.variant.path, imgsz=imgsz),
        data_yaml,
    )
    metrics = measured["metrics"]
    result.map50 = float(metrics["map50"])
    result.map50_95 = float(metrics["map50_95"])


def _tracking_metrics(
    result: VariantResult,
    sequence_dir: Path,
    ground_truth: Path,
    app_config: AppConfig,
    imgsz: int,
) -> None:
    """Run the same ByteTrack settings and MOT annotations for each variant."""
    sequence = load_sequence(sequence_dir)
    model_config = app_config.model.model_copy(
        update={"weights_path": result.variant.path, "imgsz": imgsz}
    )
    detector_config = DetectorConfig(
        backend=result.variant.backend,
        precision=result.variant.precision,
        providers=app_config.detector.providers,
        model_path=result.variant.path,
    )
    detector = create_detector(
        app_config.model_copy(
            update={"model": model_config, "detector": detector_config}
        )
    )
    try:
        result_data = evaluate_tracker_on_sequence(
            detector,
            ByteTracker(app_config.tracker),
            sequence,
            ground_truth_path=ground_truth,
        )
    finally:
        close = getattr(detector, "close", None)
        if callable(close):
            close()
    metrics = result_data["metrics"]
    result.idf1 = float(metrics["idf1"]) if metrics["idf1"] is not None else None
    result.mota = float(metrics["mota"]) if metrics["mota"] is not None else None


def _line_count_error(
    result: VariantResult,
    sequence_dir: Path,
    app_config: AppConfig,
    line_ground_truth: Path,
    imgsz: int,
) -> None:
    """Run configured analytics and compare emitted crossings with annotations."""
    sequence = load_sequence(sequence_dir)
    model_config = app_config.model.model_copy(
        update={"weights_path": result.variant.path, "imgsz": imgsz}
    )
    detector_config = DetectorConfig(
        backend=result.variant.backend,
        precision=result.variant.precision,
        providers=app_config.detector.providers,
        model_path=result.variant.path,
    )
    detector = create_detector(
        app_config.model_copy(
            update={"model": model_config, "detector": detector_config}
        )
    )
    tracker = ByteTracker(app_config.tracker)
    analytics = AnalyticsEngine(app_config.analytics)
    emitted = []
    try:
        for frame_data in iter_sequence_frames(sequence):
            tracks = tracker.update(
                detector.detect(frame_data.frame),
                frame_data.frame_id,
            )
            emitted.extend(analytics.update(tracks, frame_data))
        analytics.finalize()
    finally:
        close = getattr(detector, "close", None)
        if callable(close):
            close()
    metrics = evaluate_line_counts(emitted, line_ground_truth)
    value = metrics["count_error_percent"]
    result.line_count_error_percent = float(value) if value is not None else None


def _accuracy_status(
    results: list[VariantResult],
    thresholds: dict[str, float],
) -> None:
    """Apply configured per-metric tolerances against the measured PyTorch row."""
    baseline = next(
        (result for result in results if result.variant.backend == "pytorch"),
        None,
    )
    if baseline is None:
        return
    metric_rules = (
        ("mAP50", "map50", thresholds["max_map50_drop"]),
        ("mAP50-95", "map50_95", thresholds["max_map50_95_drop"]),
        ("IDF1", "idf1", thresholds["max_idf1_drop"]),
    )
    for result in results:
        if result is baseline or result.reason is not None:
            continue
        evaluated = 0
        failures: list[str] = []
        for label, attribute, tolerance in metric_rules:
            baseline_value = getattr(baseline, attribute)
            variant_value = getattr(result, attribute)
            if baseline_value is None or variant_value is None:
                continue
            evaluated += 1
            drop = baseline_value - variant_value
            limit = tolerance
            if result.variant.precision == "fp16" and attribute == "map50":
                limit = thresholds["fp16_map50_failure_drop"]
                if drop > thresholds["fp16_map50_ideal_drop"]:
                    logger.warning(
                        "%s mAP50 drop %.4f exceeds the FP16 ideal %.4f",
                        result.variant.name,
                        drop,
                        thresholds["fp16_map50_ideal_drop"],
                    )
            if drop > limit:
                failures.append(
                    f"{label} drop {drop:.4f} exceeds tolerance {limit:.4f}"
                )
        if (
            baseline.line_count_error_percent is not None
            and result.line_count_error_percent is not None
        ):
            evaluated += 1
            increase = (
                result.line_count_error_percent - baseline.line_count_error_percent
            )
            if increase > thresholds["max_line_count_error_increase_percent"]:
                failures.append(
                    "line-count error increase "
                    f"{increase:.2f} pp exceeds tolerance "
                    f"{thresholds['max_line_count_error_increase_percent']:.2f} pp"
                )
        result.accuracy_failures = failures
        if failures:
            result.accuracy_status = "FAILED ACCURACY TOLERANCE"
        elif evaluated:
            result.accuracy_status = "PASS"
        else:
            result.accuracy_status = "NOT EVALUATED"


def _percentile(samples: list[float], value: float) -> float | None:
    return float(__import__("numpy").percentile(samples, value)) if samples else None


def _markdown(results: list[VariantResult], hardware: str) -> str:
    """Render a speed, size, and accuracy comparison."""
    baseline = next(
        (
            result
            for result in results
            if result.variant.backend == "pytorch" and result.latency_ms
        ),
        None,
    )
    lines = [
        "# Phase 7 model optimization benchmark",
        "",
        f"- Hardware/software: {hardware}",
        "- Latency measures detector inference only; video decode and drawing "
        "are excluded.",
        "- Accuracy is reported only when held-out annotations are supplied.",
        "",
        "| Variant | Mean ms | p50 ms | p95 ms | p99 ms | FPS | Speedup | "
        "Size MB | mAP50 | mAP50-95 | IDF1 | MOTA | Line error % | "
        "Accuracy status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        samples = result.latency_ms
        mean = statistics.fmean(samples) if samples else None
        fps = 1000 / mean if mean else None
        speedup = (
            baseline.latency_ms and statistics.fmean(baseline.latency_ms) / mean
            if baseline is not None and samples and mean
            else None
        )
        size = result.size_bytes / 1_000_000 if result.size_bytes is not None else None
        values = (
            result.variant.name,
            f"{mean:.2f}" if mean is not None else "N/A",
            f"{_percentile(samples, 50):.2f}" if samples else "N/A",
            f"{_percentile(samples, 95):.2f}" if samples else "N/A",
            f"{_percentile(samples, 99):.2f}" if samples else "N/A",
            f"{fps:.2f}" if fps is not None else "N/A",
            f"{speedup:.2f}x" if isinstance(speedup, float) else "N/A",
            f"{size:.2f}" if size is not None else "N/A",
            f"{result.map50:.4f}" if result.map50 is not None else "N/A",
            f"{result.map50_95:.4f}" if result.map50_95 is not None else "N/A",
            f"{result.idf1:.4f}" if result.idf1 is not None else "N/A",
            f"{result.mota:.4f}" if result.mota is not None else "N/A",
            (
                f"{result.line_count_error_percent:.2f}"
                if result.line_count_error_percent is not None
                else "N/A"
            ),
            result.accuracy_status,
        )
        lines.append("| " + " | ".join(values) + " |")
        if result.reason:
            lines.append(f"\n> **{result.variant.name}:** {result.reason}\n")
        if result.accuracy_failures:
            lines.append(
                "\n> **FAILED ACCURACY TOLERANCE** — do not recommend "
                f"{result.variant.name} for production: "
                + "; ".join(result.accuracy_failures)
                + ".\n"
            )
    lines.extend(
        [
            "",
            "## Decision",
            "",
        ]
    )
    failures = [
        result
        for result in results
        if result.accuracy_status == "FAILED ACCURACY TOLERANCE"
    ]
    lines.append(
        "- Avoid production deployment of: "
        + (
            ", ".join(result.variant.name for result in failures)
            if failures
            else "no measured variant failed the configured tolerances"
        )
        + "."
    )
    lines.append(
        "- Rows with `NOT EVALUATED` are not accuracy-certified and must not be "
        "treated as passing."
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> Path:
    """Run every available variant and write the optimization report."""
    if args.frames <= 0 or args.warmup < 0 or args.imgsz <= 0:
        raise ValueError("frames/imgsz must be positive and warmup non-negative")
    if args.mot_sequence is not None and args.mot_ground_truth is None:
        raise ValueError("--mot-sequence requires --mot-ground-truth")
    if args.line_ground_truth is not None and args.mot_sequence is None:
        raise ValueError("--line-ground-truth requires --mot-sequence")
    app_config = load_config(args.config)
    target_document = yaml.safe_load(args.targets.read_text(encoding="utf-8"))
    thresholds = target_document["optimization_accuracy"]
    results: list[VariantResult] = []
    for variant in _variants(args):
        result = VariantResult(variant)
        results.append(result)
        if not variant.path.is_file():
            result.reason = f"model artifact is missing: {variant.path}"
            continue
        result.size_bytes = variant.path.stat().st_size
        try:
            result.latency_ms = _benchmark_latency(
                app_config,
                variant,
                args.video,
                frames=args.frames,
                warmup=args.warmup,
                imgsz=args.imgsz,
            )
        except ImportError as error:
            result.reason = str(error)
            continue
        if args.data_yaml is not None:
            _detection_metrics(result, args.data_yaml, args.imgsz)
        if args.mot_sequence is not None and args.mot_ground_truth is not None:
            _tracking_metrics(
                result,
                args.mot_sequence,
                args.mot_ground_truth,
                app_config,
                args.imgsz,
            )
        if args.line_ground_truth is not None and args.mot_sequence is not None:
            _line_count_error(
                result,
                args.mot_sequence,
                app_config,
                args.line_ground_truth,
                args.imgsz,
            )
    _accuracy_status(results, thresholds)
    hardware = f"{platform.platform()}; Python {platform.python_version()}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_markdown(results, hardware), encoding="utf-8")
    logger.info("Optimization report written to %s", args.output)
    return args.output


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; return nonzero when any measured accuracy gate fails."""
    setup_logging()
    try:
        args = build_parser().parse_args(argv)
        report = run(args)
    except (ImportError, OSError, RuntimeError, ValueError, KeyError) as error:
        logger.error("%s", error)
        return 1
    contents = report.read_text(encoding="utf-8")
    return 2 if "FAILED ACCURACY TOLERANCE" in contents else 0


if __name__ == "__main__":
    sys.exit(main())
