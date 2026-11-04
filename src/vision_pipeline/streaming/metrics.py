"""Thread-safe stream counters, latency summaries, and Prometheus rendering."""

from __future__ import annotations

import json
import math
import statistics
import threading
from collections import defaultdict, deque
from typing import Any


class StreamMetrics:
    """Collect live throughput, queue, stage, and end-to-end measurements."""

    def __init__(
        self,
        *,
        clock: Any,
        window_seconds: float = 10.0,
        latency_window: int = 1024,
    ) -> None:
        if window_seconds <= 0 or latency_window < 1:
            raise ValueError("metric windows must be positive")
        self.clock = clock
        self.window_seconds = window_seconds
        self._lock = threading.RLock()
        self._started_at = clock.monotonic()
        self._captured = 0
        self._processed = 0
        self._output = 0
        self._queue_dropped: dict[str, int] = defaultdict(int)
        self._frame_times: dict[str, deque[float]] = {
            "input": deque(),
            "processed": deque(),
            "output": deque(),
        }
        self._end_to_end: deque[float] = deque(maxlen=latency_window)
        self._stage_latency: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=latency_window)
        )
        self._queue_depths: dict[str, int] = {}
        self._queue_high_water: dict[str, int] = {}
        self._queue_policies: dict[str, str] = {}
        self._stage_utilization: dict[str, float] = {}
        self._stage_latency_summary: dict[str, dict[str, float]] = {}
        self._current_stride = 1
        self._reconnect_count = 0
        self._ema_fps: dict[str, float] = {
            "input": 0.0,
            "processed": 0.0,
            "output": 0.0,
        }

    def record_capture(self, timestamp: float | None = None) -> None:
        """Record one captured frame."""
        self._record_rate("input", timestamp)
        with self._lock:
            self._captured += 1

    def record_processed(self, timestamp: float | None = None) -> None:
        """Record one tracking/analytics result."""
        self._record_rate("processed", timestamp)
        with self._lock:
            self._processed += 1

    def record_output(self, latency_ms: float, timestamp: float | None = None) -> None:
        """Record a published frame and its capture-to-output latency."""
        if latency_ms >= 0:
            with self._lock:
                self._end_to_end.append(float(latency_ms))
        self._record_rate("output", timestamp)
        with self._lock:
            self._output += 1

    def record_stage(
        self, name: str, duration_ms: float, utilization_pct: float
    ) -> None:
        """Record a stage latency and its current utilization."""
        with self._lock:
            self._stage_latency[name].append(max(0.0, float(duration_ms)))
            self._stage_utilization[name] = max(0.0, min(100.0, utilization_pct))

    def record_queue(
        self,
        name: str,
        depth: int,
        high_water_mark: int,
        policy: str | None = None,
    ) -> None:
        """Update queue occupancy and high-water measurements."""
        with self._lock:
            self._queue_depths[name] = depth
            self._queue_high_water[name] = max(
                self._queue_high_water.get(name, 0), high_water_mark
            )
            if policy is not None:
                self._queue_policies[name] = policy

    def record_stage_summary(
        self,
        name: str,
        latency: dict[str, float],
        utilization_pct: float,
    ) -> None:
        """Store current stage latency percentiles and utilization."""
        with self._lock:
            self._stage_latency_summary[name] = dict(latency)
            self._stage_utilization[name] = max(0.0, min(100.0, utilization_pct))

    def record_drop(self, queue_name: str, count: int = 1) -> None:
        """Add frames dropped by a queue overflow policy."""
        with self._lock:
            self._queue_dropped[queue_name] += max(0, count)

    def set_stride(self, stride: int) -> None:
        """Set the current inference stride."""
        with self._lock:
            self._current_stride = stride

    def record_reconnect(self, count: int = 1) -> None:
        """Increment the source reconnect counter."""
        with self._lock:
            self._reconnect_count += max(0, count)

    def _record_rate(self, name: str, timestamp: float | None) -> None:
        now = self.clock.monotonic() if timestamp is None else timestamp
        with self._lock:
            times = self._frame_times[name]
            times.append(now)
            self._trim(times, now)
            if len(times) > 1:
                duration = times[-1] - times[0]
                rate = (len(times) - 1) / duration if duration > 0 else 0.0
                previous = self._ema_fps[name]
                self._ema_fps[name] = (
                    rate if previous == 0 else 0.2 * rate + 0.8 * previous
                )

    def _trim(self, times: deque[float], now: float) -> None:
        while len(times) > 1 and now - times[0] > self.window_seconds:
            times.popleft()

    @staticmethod
    def _percentile(samples: list[float], percentile: float) -> float:
        if not samples:
            return 0.0
        ordered = sorted(samples)
        index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * percentile) - 1))
        return ordered[index]

    def snapshot(self, stages: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return a point-in-time metrics dictionary."""
        now = self.clock.monotonic()
        with self._lock:
            for samples in self._frame_times.values():
                self._trim(samples, now)
            output = list(self._end_to_end)
            end_to_end = {
                "mean_ms": statistics.fmean(output) if output else 0.0,
                "p95_ms": self._percentile(output, 0.95),
                "max_ms": max(output, default=0.0),
            }
            stage_latency = {
                name: {
                    "mean_ms": statistics.fmean(samples) if samples else 0.0,
                    "p50_ms": self._percentile(list(samples), 0.50),
                    "p95_ms": self._percentile(list(samples), 0.95),
                    "p99_ms": self._percentile(list(samples), 0.99),
                }
                for name, samples in self._stage_latency.items()
            }
            stage_latency.update(self._stage_latency_summary)
            stages = stages or {}
            utilization = dict(self._stage_utilization)
            bottleneck = (
                max(utilization, key=lambda stage: utilization[stage])
                if utilization and max(utilization.values()) > 0
                else None
            )
            dropped_by_policy: dict[str, int] = defaultdict(int)
            for name, count in self._queue_dropped.items():
                dropped_by_policy[self._queue_policies.get(name, "unknown")] += count
            elapsed = max(0.0, now - self._started_at)
            return {
                "uptime_seconds": elapsed,
                "input_fps": self._window_rate("input"),
                "processed_fps": self._window_rate("processed"),
                "output_fps": self._window_rate("output"),
                "input_fps_ema": self._ema_fps["input"],
                "processed_fps_ema": self._ema_fps["processed"],
                "output_fps_ema": self._ema_fps["output"],
                "frames_captured": self._captured,
                "frames_processed": self._processed,
                "frames_output": self._output,
                "frames_dropped": sum(self._queue_dropped.values()),
                "dropped_by_queue": dict(self._queue_dropped),
                "dropped_by_policy": dict(dropped_by_policy),
                "queue_depths": dict(self._queue_depths),
                "queue_high_water_marks": dict(self._queue_high_water),
                "queue_policies": dict(self._queue_policies),
                "stage_latency": stage_latency,
                "stage_utilization_pct": utilization,
                "stages": dict(stages),
                "end_to_end_latency": end_to_end,
                "current_stride": self._current_stride,
                "reconnect_count": self._reconnect_count,
                "bottleneck_stage": bottleneck,
            }

    def _window_rate(self, name: str) -> float:
        times = self._frame_times[name]
        if len(times) < 2:
            return 0.0
        duration = times[-1] - times[0]
        return (len(times) - 1) / duration if duration > 0 else 0.0

    def to_dict(self, stages: dict[str, Any] | None = None) -> dict[str, Any]:
        """Alias for :meth:`snapshot` for convenient integration use."""
        return self.snapshot(stages)

    def to_json(self, stages: dict[str, Any] | None = None, *, indent: int = 2) -> str:
        """Serialize metrics as JSON."""
        return json.dumps(self.snapshot(stages), indent=indent)

    def to_prometheus(self, stages: dict[str, Any] | None = None) -> str:
        """Render a plain-text Prometheus exposition."""
        data = self.snapshot(stages)
        lines = [
            "# TYPE vision_pipeline_frames_total counter",
            f'vision_pipeline_frames_total{{kind="captured"}} '
            f"{data['frames_captured']}",
            f'vision_pipeline_frames_total{{kind="processed"}} '
            f"{data['frames_processed']}",
            f'vision_pipeline_frames_total{{kind="output"}} {data["frames_output"]}',
            f'vision_pipeline_frames_total{{kind="dropped"}} {data["frames_dropped"]}',
            "# TYPE vision_pipeline_fps gauge",
        ]
        for label in ("input", "processed", "output"):
            lines.append(
                f'vision_pipeline_fps{{kind="{label}"}} {data[f"{label}_fps"]}'
            )
        lines.extend(
            [
                "# TYPE vision_pipeline_end_to_end_latency_ms gauge",
                "vision_pipeline_end_to_end_latency_ms "
                f"{data['end_to_end_latency']['p95_ms']}",
                "# TYPE vision_pipeline_queue_depth gauge",
            ]
        )
        for name, depth in data["queue_depths"].items():
            lines.append(f'vision_pipeline_queue_depth{{queue="{name}"}} {depth}')
        lines.append("# TYPE vision_pipeline_queue_high_water_mark gauge")
        for name, high_water in data["queue_high_water_marks"].items():
            lines.append(
                f'vision_pipeline_queue_high_water_mark{{queue="{name}"}} {high_water}'
            )
        lines.append("# TYPE vision_pipeline_stage_utilization_percent gauge")
        for name, value in data["stage_utilization_pct"].items():
            lines.append(
                f'vision_pipeline_stage_utilization_percent{{stage="{name}"}} {value}'
            )
        lines.append("# TYPE vision_pipeline_stage_latency_ms gauge")
        for name, values in data["stage_latency"].items():
            for percentile, value in values.items():
                lines.append(
                    f'vision_pipeline_stage_latency_ms{{stage="{name}",'
                    f'percentile="{percentile}"}} {value}'
                )
        lines.append(f"vision_pipeline_detection_stride {data['current_stride']}")
        lines.append(f"vision_pipeline_reconnects_total {data['reconnect_count']}")
        return "\n".join(lines) + "\n"
