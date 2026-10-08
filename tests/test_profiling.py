"""Tests for FPS and latency measurements."""

import json

import pytest

from vision_pipeline.utils.profiling import FPSMeter, LatencyStats, StageTimer


def test_fps_meter_uses_rolling_window() -> None:
    """FPS is computed from only the most recent window intervals."""
    meter = FPSMeter(window=2)

    assert meter.tick(0.0) == 0.0
    assert meter.tick(0.5) == pytest.approx(2.0)
    assert meter.tick(1.5) == pytest.approx(4.0 / 3.0)
    meter.reset()
    assert meter.fps == 0.0


def test_latency_stats_percentiles_and_json() -> None:
    """Latency summaries expose standard statistics in a JSON-safe shape."""
    stats = LatencyStats()
    for value in (1.0, 2.0, 3.0, 4.0):
        stats.add("inference", value)

    result = stats.summary("inference")
    assert result["mean"] == pytest.approx(2.5)
    assert result["median"] == pytest.approx(2.5)
    assert result["p95"] == pytest.approx(3.85)
    assert result["p99"] == pytest.approx(3.97)
    assert json.loads(stats.to_json())["inference"]["count"] == 4
    assert stats.summary("missing")["count"] == 0


def test_stage_timer_context_and_decorator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both timer APIs record controlled perf-counter durations."""
    values = iter((10.0, 10.025, 20.0, 20.010))
    monkeypatch.setattr(
        "vision_pipeline.utils.profiling.time.perf_counter",
        lambda: next(values),
    )
    stats = LatencyStats()
    with StageTimer("read", stats):
        pass

    @StageTimer("draw", stats)
    def draw() -> str:
        return "done"

    assert draw() == "done"
    assert stats.summary("read")["mean"] == pytest.approx(25.0)
    assert stats.summary("draw")["mean"] == pytest.approx(10.0)
