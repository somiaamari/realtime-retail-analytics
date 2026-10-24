"""End-to-end timing and resource measurements for the retail pipeline."""

from __future__ import annotations

import importlib
import statistics
import time
from collections.abc import Iterable
from typing import Any

import numpy as np

from vision_pipeline.analytics.engine import AnalyticsEngine
from vision_pipeline.detection.yolo_detector import BaseDetector
from vision_pipeline.tracking.base import BaseTracker
from vision_pipeline.types import FrameData
from vision_pipeline.utils.profiling import LatencyStats, StageTimer
from vision_pipeline.utils.visualization import draw_tracks


def benchmark_pipeline(
    frames: Iterable[FrameData],
    detector: BaseDetector,
    tracker: BaseTracker,
    analytics: AnalyticsEngine | None = None,
    *,
    warmup_frames: int = 5,
    realtime_fps: float = 30.0,
    draw: bool = True,
    writer: Any | None = None,
    max_frames: int | None = None,
    max_duration_s: float | None = None,
) -> dict[str, Any]:
    """Measure per-stage latency, throughput, jitter, and process resources."""
    if warmup_frames < 0:
        raise ValueError("warmup_frames must be non-negative")
    if realtime_fps <= 0:
        raise ValueError("realtime_fps must be positive")
    if max_duration_s is not None and max_duration_s <= 0:
        raise ValueError("max_duration_s must be positive")
    iterator = iter(frames)
    stats = LatencyStats()
    frame_times: list[float] = []
    process = None
    try:
        psutil = importlib.import_module("psutil")
        process = psutil.Process()
    except ImportError:
        psutil = None
    peak_rss = 0
    cpu_samples: list[float] = []
    gpu_peak_bytes: int | None = None
    torch = None
    try:
        torch = importlib.import_module("torch")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass

    processed = 0
    measured = 0
    started_at = time.perf_counter()
    measured_detection_counts: list[int] = []
    memory_samples: list[int] = []
    while max_frames is None or processed < max_frames:
        if (
            max_duration_s is not None
            and time.perf_counter() - started_at >= max_duration_s
        ):
            break
        frame_start = time.perf_counter()
        frame_stats = LatencyStats()
        decode_timer = StageTimer("decode", frame_stats)
        try:
            with decode_timer:
                frame_data = next(iterator)
        except StopIteration:
            break
        preprocess_timer = StageTimer("preprocess", frame_stats)
        with preprocess_timer:
            frame = np.ascontiguousarray(frame_data.frame)
        detect_timer = StageTimer("detect", frame_stats)
        with detect_timer:
            detections = detector.detect(frame)
        track_timer = StageTimer("track", frame_stats)
        with track_timer:
            tracks = tracker.update(detections, frame_data.frame_id)
        if analytics is not None:
            analytics_timer = StageTimer("analytics", frame_stats)
            with analytics_timer:
                analytics.update(tracks, frame_data)
        annotated = frame
        if draw:
            draw_timer = StageTimer("draw", frame_stats)
            with draw_timer:
                annotated = draw_tracks(frame, tracks)
        if writer is not None:
            write_timer = StageTimer("write", frame_stats)
            with write_timer:
                writer.write(annotated)
        processed += 1
        if processed > warmup_frames:
            elapsed_ms = (time.perf_counter() - frame_start) * 1000
            frame_times.append(elapsed_ms)
            measured_detection_counts.append(len(detections))
            measured += 1
            for stage, summary in frame_stats.to_dict().items():
                stats.add(stage, float(summary["mean"]))
            if process is not None:
                current_rss = int(process.memory_info().rss)
                peak_rss = max(peak_rss, current_rss)
                memory_samples.append(current_rss)
                cpu_samples.append(float(process.cpu_percent(interval=None)))

    if analytics is not None:
        analytics.finalize()
    if torch is not None:
        try:
            if torch.cuda.is_available():
                gpu_peak_bytes = int(torch.cuda.max_memory_allocated())
        except (AttributeError, RuntimeError):
            gpu_peak_bytes = None
    stage_summaries = stats.to_dict()
    total_ms = sum(frame_times)
    frame_budget_ms = 1000.0 / realtime_fps
    bucket_counts: dict[str, list[int]] = {"0-4": [], "5-9": [], "10+": []}
    for index, count in enumerate(measured_detection_counts):
        bucket = "0-4" if count < 5 else "5-9" if count < 10 else "10+"
        bucket_counts[bucket].append(index)
    detection_buckets = {}
    for name, indices in bucket_counts.items():
        detection_buckets[name] = {
            "frames": len(indices),
            "mean_end_to_end_ms": (
                statistics.fmean(frame_times[index] for index in indices)
                if indices
                else None
            ),
        }
    midpoint = len(frame_times) // 2
    first_half_fps = (
        1000.0 / statistics.fmean(frame_times[:midpoint])
        if midpoint and frame_times
        else None
    )
    second_half_fps = (
        1000.0 / statistics.fmean(frame_times[midpoint:])
        if len(frame_times) - midpoint and midpoint
        else None
    )
    result: dict[str, Any] = {
        "frames_total": processed,
        "frames_measured": measured,
        "warmup_frames": min(processed, warmup_frames),
        "stage_latency_ms": stage_summaries,
        "detection_count_buckets": detection_buckets,
        "end_to_end": {
            "mean_ms": statistics.fmean(frame_times) if frame_times else None,
            "median_ms": statistics.median(frame_times) if frame_times else None,
            "p95_ms": _percentile(frame_times, 95),
            "p99_ms": _percentile(frame_times, 99),
            "max_ms": max(frame_times) if frame_times else None,
            "fps": measured / (total_ms / 1000.0) if total_ms > 0 else None,
            "jitter_std_ms": statistics.pstdev(frame_times)
            if len(frame_times) > 1
            else 0.0,
            "realtime_budget_ms": frame_budget_ms,
            "over_budget_frames": sum(value > frame_budget_ms for value in frame_times),
            "over_budget_percent": (
                100.0 * sum(value > frame_budget_ms for value in frame_times) / measured
                if measured
                else None
            ),
        },
        "resources": {
            "peak_rss_bytes": peak_rss or None,
            "average_cpu_percent": statistics.fmean(cpu_samples)
            if cpu_samples
            else None,
            "gpu_peak_memory_bytes": gpu_peak_bytes,
        },
        "stability": {
            "first_half_fps": first_half_fps,
            "second_half_fps": second_half_fps,
            "fps_drift_percent": (
                100.0 * (second_half_fps - first_half_fps) / first_half_fps
                if first_half_fps and second_half_fps is not None
                else None
            ),
            "memory_growth_bytes": (
                memory_samples[-1] - memory_samples[0]
                if len(memory_samples) > 1
                else None
            ),
            "duration_s": time.perf_counter() - started_at,
        },
        "draw_enabled": draw,
        "write_enabled": writer is not None,
        "batch_size": 1,
    }
    return result


def _percentile(values: list[float], percentile: float) -> float | None:
    """Return an interpolated percentile, or None for no samples."""
    if not values:
        return None
    return float(np.percentile(values, percentile, method="linear"))


def run_benchmark_sweep(
    runner: Any,
    cases: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run named benchmark cases and return a machine-readable result list."""
    results: list[dict[str, Any]] = []
    for case in cases:
        results.append({"case": dict(case), "result": runner(**case)})
    return results
