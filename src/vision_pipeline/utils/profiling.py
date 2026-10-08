"""Small, dependency-free timing and throughput measurement utilities."""

from collections import defaultdict, deque
from collections.abc import Callable
from functools import wraps
import json
import statistics
import time
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


class LatencyStats:
    """Collect latency samples in milliseconds and summarize by stage."""

    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = defaultdict(list)

    def add(self, stage: str, duration_ms: float) -> None:
        """Record one non-negative latency sample."""
        if duration_ms < 0:
            raise ValueError("Stage duration must be non-negative.")
        self._samples[stage].append(float(duration_ms))

    def summary(self, stage: str | None = None) -> dict[str, object]:
        """Return summary statistics for one stage or all recorded stages."""
        if stage is not None:
            samples = self._samples.get(stage, [])
            return self._summarize(samples)
        return {
            name: self._summarize(samples)
            for name, samples in sorted(self._samples.items())
        }

    @staticmethod
    def _summarize(samples: list[float]) -> dict[str, float | int]:
        if not samples:
            return {
                "count": 0,
                "mean": 0.0,
                "median": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
        ordered = sorted(samples)
        return {
            "count": len(samples),
            "mean": statistics.fmean(samples),
            "median": statistics.median(samples),
            "p95": _percentile(ordered, 95),
            "p99": _percentile(ordered, 99),
            "min": min(samples),
            "max": max(samples),
        }

    def to_dict(self) -> dict[str, dict[str, float | int]]:
        """Export summaries in a JSON-serializable mapping."""
        return {
            name: self._summarize(samples)
            for name, samples in sorted(self._samples.items())
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize stage summaries as JSON."""
        return json.dumps(self.to_dict(), indent=indent)


def _percentile(ordered: list[float], percentile: float) -> float:
    """Compute a linearly interpolated percentile from sorted values."""
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


class StageTimer:
    """Measure a named stage as a context manager or function decorator."""

    def __init__(self, stage: str, stats: LatencyStats) -> None:
        self.stage = stage
        self.stats = stats
        self.duration_ms = 0.0
        self._started_at = 0.0

    def __enter__(self) -> StageTimer:
        """Start timing the stage."""
        self._started_at = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        """Record elapsed time even when the stage raises."""
        self.duration_ms = (time.perf_counter() - self._started_at) * 1000
        self.stats.add(self.stage, self.duration_ms)

    def __call__(self, function: Callable[P, R]) -> Callable[P, R]:
        """Decorate a function to record each call's execution duration."""

        @wraps(function)
        def timed(*args: P.args, **kwargs: P.kwargs) -> R:
            with StageTimer(self.stage, self.stats):
                return function(*args, **kwargs)

        return timed


class FPSMeter:
    """Calculate rolling average FPS from frame arrival times."""

    def __init__(self, window: int = 30) -> None:
        if window < 1:
            raise ValueError("FPS rolling window must be at least 1.")
        self.window = window
        self._times: deque[float] = deque(maxlen=window + 1)

    def tick(self, timestamp: float | None = None) -> float:
        """Record a frame time and return the current rolling FPS."""
        now = time.perf_counter() if timestamp is None else timestamp
        if self._times and now < self._times[-1]:
            raise ValueError("FPS timestamps must be non-decreasing.")
        self._times.append(now)
        return self.fps

    @property
    def fps(self) -> float:
        """Return the rolling average FPS, or zero before two frames."""
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._times) - 1) / elapsed

    def reset(self) -> None:
        """Clear all recorded frame times."""
        self._times.clear()
