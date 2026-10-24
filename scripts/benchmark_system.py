"""Benchmark the end-to-end detection, tracking, analytics, and drawing path."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

from vision_pipeline.analytics.engine import AnalyticsEngine
from vision_pipeline.config import load_config
from vision_pipeline.detection.yolo_detector import YoloDetector
from vision_pipeline.evaluation.perf import benchmark_pipeline, run_benchmark_sweep
from vision_pipeline.evaluation.report import (
    collect_environment,
    save_hardware_result,
    set_reproducible_seed,
    update_baseline_markdown,
)
from vision_pipeline.logging_utils import setup_logging
from vision_pipeline.streaming.sources import create_source
from vision_pipeline.streaming.writer import VideoWriter
from vision_pipeline.tracking.bytetrack import ByteTracker

logger = logging.getLogger(__name__)


def _available_devices() -> list[str]:
    """Return available inference devices without requiring a GPU."""
    devices = ["cpu"]
    try:
        import torch

        if torch.cuda.is_available():
            devices.append("cuda")
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            devices.append("mps")
    except ImportError:
        pass
    return devices


def _write_sweep_artifacts(
    cases: list[dict[str, Any]],
    output_directory: Path,
) -> dict[str, str | None]:
    """Write a compact CSV and a latency chart for measured conditions."""
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "system_sweep.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "model",
                "imgsz",
                "device",
                "draw",
                "write",
                "fps",
                "mean_ms",
                "p95_ms",
            ],
        )
        writer.writeheader()
        for row in cases:
            condition = row["case"]
            result = row["result"]
            end_to_end = result["end_to_end"]
            writer.writerow(
                {
                    "model": condition["model"],
                    "imgsz": condition["imgsz"],
                    "device": condition["device"],
                    "draw": condition["draw"],
                    "write": condition["write"],
                    "fps": end_to_end["fps"],
                    "mean_ms": end_to_end["mean_ms"],
                    "p95_ms": end_to_end["p95_ms"],
                }
            )
    chart_path: Path | None = None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        chart_path = None
    else:
        labels = [
            (
                f"{row['case']['model']} {row['case']['imgsz']} "
                f"{row['case']['device']} "
                f"{'draw' if row['case']['draw'] else 'no-draw'}"
            )
            for row in cases
        ]
        values = [row["result"]["end_to_end"]["p95_ms"] or 0 for row in cases]
        figure, axis = plt.subplots(figsize=(max(8, len(labels) * 0.8), 4))
        axis.bar(range(len(values)), values)
        axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        axis.set_ylabel("End-to-end p95 (ms)")
        axis.set_title("System benchmark conditions")
        figure.tight_layout()
        chart_path = output_directory / "system_sweep.png"
        figure.savefig(chart_path, dpi=140)
        plt.close(figure)
    return {
        "csv": str(csv_path),
        "chart": str(chart_path) if chart_path else None,
    }


def build_parser() -> argparse.ArgumentParser:
    """Create command-line options for a system benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--realtime-fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--no-draw", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--imgsz", type=int)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep supported image sizes, available devices, and draw on/off",
    )
    parser.add_argument(
        "--model-s",
        type=Path,
        help="Optional local YOLO small-model checkpoint to include in the sweep",
    )
    parser.add_argument(
        "--stability-minutes",
        type=float,
        default=2.0,
        help="Bound the run by duration for a long-run stability measurement",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        help="Write an annotated video (adds a writing stage to the benchmark)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run configured benchmark cases and save machine-readable artifacts."""
    setup_logging()
    args = build_parser().parse_args(argv)
    set_reproducible_seed(args.seed)
    if args.warmup_frames < 0 or args.realtime_fps <= 0:
        raise SystemExit("warmup frames must be non-negative and FPS must be positive")
    if args.stability_minutes is not None and args.stability_minutes <= 0:
        raise SystemExit("--stability-minutes must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max-frames must be positive")
    config = load_config(args.config)
    sizes = [320, 480, 640, 960] if args.sweep else [args.imgsz or config.model.imgsz]
    devices = (
        _available_devices() if args.sweep else [args.device or config.model.device]
    )
    model_paths = {"n": config.model.weights_path}
    if args.model_s is not None:
        model_paths["s"] = args.model_s
    cases = [
        {
            "model": model_name,
            "weights_path": weights_path,
            "imgsz": size,
            "device": device,
            "draw": draw,
            "write": write,
        }
        for model_name, weights_path in model_paths.items()
        for size in sizes
        for device in devices
        for draw in ([False, True] if args.sweep else [not args.no_draw])
        for write in ([False, True] if args.sweep else [False])
    ]

    def run_case(
        *,
        model: str,
        weights_path: Path,
        imgsz: int,
        device: str,
        draw: bool,
        write: bool,
    ) -> dict[str, Any]:
        """Construct fresh state for each condition and measure its pipeline."""
        model_config = config.model.model_copy(
            update={
                "weights_path": weights_path,
                "imgsz": imgsz,
                "device": device,
            }
        )
        detector = YoloDetector(model_config)
        detector.warmup()
        source = create_source(
            config.video.model_copy(update={"source": str(args.source)})
        )
        tracker = ByteTracker(config.tracker)
        analytics = AnalyticsEngine(config.analytics)
        writer: VideoWriter | None = None
        with source:
            if write:
                output_path = (
                    args.output_video
                    if args.output_video is not None and not args.sweep
                    else args.output_dir / f"annotated_{model}_{imgsz}_{device}.mp4"
                )
                writer = VideoWriter.from_source(output_path, source)
            try:
                result = benchmark_pipeline(
                    source,
                    detector,
                    tracker,
                    analytics,
                    warmup_frames=args.warmup_frames,
                    realtime_fps=args.realtime_fps,
                    draw=draw,
                    writer=writer,
                    max_frames=args.max_frames,
                    max_duration_s=(
                        args.stability_minutes * 60
                        if args.stability_minutes is not None
                        else None
                    ),
                )
            finally:
                if writer is not None:
                    writer.release()
        return result

    if args.sweep:
        results = run_benchmark_sweep(run_case, cases)
    else:
        one_case = cases[0]
        if args.output_video is not None:
            one_case["write"] = True
        results = run_benchmark_sweep(run_case, [one_case])
    artifacts = _write_sweep_artifacts(
        results,
        args.output_dir / "artifacts",
    )
    environment = collect_environment(
        repository_root=Path.cwd(),
        config_paths=[args.config],
    )
    environment["seed"] = args.seed
    for row in results:
        case = row["case"]
        if row["result"]["frames_measured"] > 0:
            update_baseline_markdown(
                Path("benchmarks/BASELINE.md"),
                model=Path(case["weights_path"]).stem,
                device=case["device"],
                imgsz=case["imgsz"],
                result=row["result"],
                environment=environment,
            )
    payload = {
        "conditions": results,
        "artifacts": artifacts,
        "notes": {
            "batch_size": 1,
            "detection_count_buckets": "0-4, 5-9, and 10+ detections per frame",
            "stability_duration_minutes": args.stability_minutes,
            "model_s_included": args.model_s is not None,
        },
    }
    output_path = save_hardware_result(payload, args.output_dir, environment)
    logger.info("System benchmark saved to %s", output_path)
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
