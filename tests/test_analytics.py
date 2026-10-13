"""Synthetic tests for the analytics geometry, engine, and accumulators."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from scripts.replay_events import _replay
from vision_pipeline.analytics.aggregator import TimeSeriesAggregator
from vision_pipeline.analytics.engine import AnalyticsEngine
from vision_pipeline.analytics.events import (
    CsvSink,
    Event,
    EventType,
    InMemorySink,
    JsonlSink,
)
from vision_pipeline.analytics.geometry import (
    AnalyticsConfig,
    CountingLine,
    PolygonZone,
    load_zones,
    point_in_polygon,
    segment_intersects,
    side_of_line,
)
from vision_pipeline.analytics.heatmap import HeatmapAccumulator
from vision_pipeline.analytics.lines import LineCounter
from vision_pipeline.analytics.zones import ZoneAnalyzer
from vision_pipeline.config import TrackerConfig, VideoConfig
from vision_pipeline.detection.yolo_detector import FakeDetector
from vision_pipeline.pipeline import AnalyticsPipeline
from vision_pipeline.streaming.sources import create_source
from vision_pipeline.tracking.bytetrack import ByteTracker
from vision_pipeline.types import Detection, FrameData, Track
from vision_pipeline.utils.visualization import draw_lines, draw_zones


def _track(track_id: int, x: float, y: float) -> Track:
    """Create a synthetic 2x2 track with feet at ``(x, y)``."""
    return Track(
        track_id=track_id,
        bbox_xyxy=(x - 1, y - 2, x + 1, y),
        confidence=1.0,
        class_id=0,
        class_name="person",
    )


def _frame(frame_id: int, timestamp_s: float) -> FrameData:
    """Create a synthetic source frame and video timestamp."""
    return FrameData(
        np.zeros((32, 32, 3), dtype=np.uint8),
        frame_id,
        timestamp_s,
        10.0,
    )


def test_geometry_polygon_edges_line_sign_and_intersection() -> None:
    """Geometry handles polygon boundaries, directed sides, and finite segments."""
    polygon = PolygonZone(
        id="zone",
        name="Zone",
        points=[(1, 1), (8, 1), (8, 8), (1, 8)],
    )

    assert point_in_polygon((1, 5), polygon)
    assert point_in_polygon((5, 5), polygon)
    assert not point_in_polygon((9, 5), polygon)
    assert side_of_line((2, 1), (0, 0), (0, 2)) < 0
    assert side_of_line((-2, 1), (0, 0), (0, 2)) > 0
    assert segment_intersects((4, 5), (6, 5), (5, 0), (5, 10))
    assert not segment_intersects((4, 15), (6, 15), (5, 0), (5, 10))


def test_geometry_validation_normalized_scaling_and_yaml(tmp_path: Path) -> None:
    """Geometry validation and normalized coordinates work at runtime size."""
    with pytest.raises(ValidationError):
        PolygonZone(id="bad", name="Bad", points=[(0, 0), (1, 1)])
    with pytest.raises(ValidationError):
        AnalyticsConfig(
            zones=[
                PolygonZone(id="same", name="Zone", points=[(0, 0), (1, 0), (0, 1)])
            ],
            lines=[CountingLine(id="same", name="Line", start=(0, 0), end=(1, 1))],
        )
    path = tmp_path / "zones.yaml"
    path.write_text(
        "normalized: true\n"
        "zones:\n"
        "  - id: q\n"
        "    name: Queue\n"
        "    points: [[0, 0], [1, 0], [1, 1]]\n"
        "lines:\n"
        "  - id: door\n"
        "    name: Door\n"
        "    start: [0.5, 0]\n"
        "    end: [0.5, 1]\n",
        encoding="utf-8",
    )

    config = load_zones(path)

    assert config.zones[0].to_pixels((100, 50)) == [(0, 0), (99, 0), (99, 49)]
    assert config.lines[0].to_pixels((100, 50)) == ((50, 0), (50, 49))


def test_zone_debounce_dwell_and_track_loss() -> None:
    """Zone state debounces borders and closes a lost track with its dwell."""
    zone = PolygonZone(
        id="queue",
        name="Queue",
        type="queue",
        points=[(0, 0), (10, 0), (10, 10), (0, 10)],
    )
    analyzer = ZoneAnalyzer(
        [zone],
        dwell_min_seconds=1.5,
        inside_debounce_frames=2,
        outside_debounce_frames=2,
    )
    assert analyzer.update([_track(1, 5, 5)], _frame(0, 0.0)) == []
    entered = analyzer.update([_track(1, 5, 5)], _frame(1, 0.1))
    assert [event.event_type for event in entered] == [EventType.ZONE_ENTER]
    assert analyzer.occupancy()["queue"] == 1
    assert analyzer.update([_track(1, 20, 20)], _frame(2, 1.0)) == []
    exited = analyzer.update([_track(1, 20, 20)], _frame(3, 2.0))
    assert [event.event_type for event in exited] == [
        EventType.ZONE_EXIT,
        EventType.DWELL_COMPLETE,
    ]
    assert analyzer.occupancy()["queue"] == 0
    assert analyzer.stats["queue"].completed_dwell_seconds == [1.9]

    analyzer.update([_track(2, 5, 5)], _frame(4, 3.0))
    analyzer.update([_track(2, 5, 5)], _frame(5, 3.1))
    lost_events = analyzer.end_track(2, 6.2, 6)
    assert lost_events[0].event_type is EventType.ZONE_EXIT
    assert lost_events[0].extra["terminated_by_track_loss"] is True
    assert lost_events[1].event_type is EventType.DWELL_COMPLETE
    assert lost_events[1].extra["dwell_seconds"] == pytest.approx(3.1)


def test_line_counter_finite_segment_direction_and_cooldown() -> None:
    """Crossings use finite segments and apply independent directional cooldown."""
    line = CountingLine(
        id="door",
        name="Door",
        start=(5, 0),
        end=(5, 10),
    )
    counter = LineCounter([line], cooldown_s=1.0)

    assert counter.update([_track(1, 6, 5)], _frame(0, 0.0)) == []
    enter = counter.update([_track(1, 4, 5)], _frame(1, 0.1))
    assert [event.event_type for event in enter] == [EventType.LINE_CROSS_IN]
    out_jitter = counter.update([_track(1, 6, 5)], _frame(2, 0.2))
    assert [event.event_type for event in out_jitter] == [EventType.LINE_CROSS_OUT]
    assert counter.update([_track(1, 4, 5)], _frame(3, 0.3)) == []
    assert counter.update([_track(1, 4, 5)], _frame(4, 1.2)) == []
    out = counter.update([_track(1, 6, 5)], _frame(5, 1.3))
    assert [event.event_type for event in out] == [EventType.LINE_CROSS_OUT]
    assert counter.totals["door"] == {"in": 1, "out": 2, "net": -1}

    assert counter.update([_track(2, 4, 15)], _frame(6, 2.0)) == []
    assert counter.update([_track(2, 6, 15)], _frame(7, 2.1)) == []

    on_line_counter = LineCounter([line])
    on_line_counter.update([_track(3, 6, 5)], _frame(8, 3.0))
    assert on_line_counter.update([_track(3, 5, 5)], _frame(9, 3.1)) == []
    crossed = on_line_counter.update([_track(3, 4, 5)], _frame(10, 3.2))
    assert [event.event_type for event in crossed] == [EventType.LINE_CROSS_IN]


def test_heatmap_accumulation_decay_and_exports(tmp_path: Path) -> None:
    """Heat concentrates at visited points, decays, and exports valid images."""
    heatmap = HeatmapAccumulator((64, 48), grid_scale=4, sigma=1.5, decay=1.0)
    heatmap.add([(32, 24), (0, 0)], timestamp_s=0.0)
    center_value = heatmap.grid[6, 8]
    assert center_value > heatmap.grid[2, 8]
    heatmap.add([], timestamp_s=1.0)
    assert heatmap.grid[6, 8] < center_value
    background = np.zeros((48, 64, 3), dtype=np.uint8)
    rendered = heatmap.render(background)
    assert rendered.shape == background.shape
    assert rendered.dtype == np.uint8
    heatmap.export_png(tmp_path / "heat.png", background)
    heatmap.export_npy(tmp_path / "heat.npy")
    assert (tmp_path / "heat.png").is_file()
    assert np.load(tmp_path / "heat.npy").shape == heatmap.grid.shape
    assert heatmap.normalize().max() == 255

    rolling = HeatmapAccumulator(
        (64, 48),
        grid_scale=4,
        sigma=1.5,
        mode="rolling",
        window_seconds=1.0,
    )
    rolling.add([(8, 8)], timestamp_s=0.0)
    rolling.add([(48, 32)], timestamp_s=2.0)
    assert rolling.grid[2, 2] == 0
    assert rolling.grid[8, 12] > 0


def test_aggregator_bucket_boundary_and_csv(tmp_path: Path) -> None:
    """Bucket boundaries are half-open and CSV exports records."""
    aggregator = TimeSeriesAggregator(bucket_seconds=10.0)
    aggregator.add_frame(9.99, {"queue": 2}, {1, 2})
    aggregator.add_frame(10.0, {"queue": 1}, {1})
    assert [row["bucket_start_s"] for row in aggregator.to_records()] == [0.0, 10.0]
    assert aggregator.peak_bucket()["peak_occupancy"] == 2
    csv_path = tmp_path / "buckets.csv"
    aggregator.to_csv(csv_path)
    assert "bucket_start_s" in csv_path.read_text(encoding="utf-8")


def test_engine_events_heatmap_snapshot_summary_and_reset(tmp_path: Path) -> None:
    """Engine produces deterministic lifecycle events and flushes configured sinks."""
    config = AnalyticsConfig(
        zones=[
            PolygonZone(
                id="queue",
                name="Queue",
                type="queue",
                points=[(0, 0), (10, 0), (10, 10), (0, 10)],
            )
        ],
        dwell_min_seconds=1.0,
    )
    sink = JsonlSink(tmp_path / "events.jsonl")
    memory_sink = InMemorySink()
    engine = AnalyticsEngine(config, [sink, memory_sink])
    events = engine.update([_track(1, 5, 5)], _frame(0, 0.0))

    assert [event.event_type for event in events] == [
        EventType.TRACK_STARTED,
        EventType.ZONE_ENTER,
    ]
    assert engine.snapshot().occupancy == {"queue": 1}
    assert engine.heatmap is not None
    summary = engine.finalize()
    assert summary.total_unique_visitors == 1
    assert memory_sink.events[-1].event_type is EventType.TRACK_ENDED
    sink.close()
    assert "ZONE_ENTER" in (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    replayed_summary, replayed_records = _replay(
        [event.to_dict() for event in memory_sink.events],
        bucket_seconds=config.bucket_seconds,
    )
    assert replayed_summary == summary.to_dict()
    assert replayed_records == engine.aggregator.to_records()
    engine.reset()
    assert engine.snapshot().unique_visitors == 0


def test_csv_sink_serializes_events_and_flushes(tmp_path: Path) -> None:
    """CSV event output flushes a tabular record with JSON extra data."""
    path = tmp_path / "events.csv"
    sink = CsvSink(path)
    sink.write(
        Event(
            EventType.LINE_CROSS_IN,
            1.25,
            4,
            7,
            line_id="door",
            extra={"direction": "in"},
        )
    )
    sink.flush()
    sink.close()

    with path.open(encoding="utf-8", newline="") as stream:
        record = next(csv.DictReader(stream))
    assert record["event_type"] == "LINE_CROSS_IN"
    assert json.loads(record["extra"]) == {"direction": "in"}


def test_pipeline_composes_video_detection_tracking_and_analytics(
    synthetic_video: Path,
) -> None:
    """Injected synthetic video components produce per-frame analytics outputs."""
    detector = FakeDetector(
        [
            Detection(
                bbox_xyxy=(20, 10, 40, 80),
                confidence=0.9,
                class_id=0,
                class_name="person",
            )
        ]
    )
    tracker = ByteTracker(
        TrackerConfig(
            track_thresh=0.1,
            high_thresh=0.1,
            low_thresh=0.05,
            new_track_thresh=0.1,
        )
    )
    engine = AnalyticsEngine(AnalyticsConfig())
    pipeline = AnalyticsPipeline(
        create_source(VideoConfig(source=str(synthetic_video))),
        detector,
        tracker,
        engine,
    )

    results = list(pipeline.run())

    assert len(results) == 5
    assert results[0].tracks
    assert results[0].events
    assert results[-1].snapshot.unique_visitors == 1
    assert pipeline.latency_stats.summary("analytics")["count"] == 5
    pipeline.close()


def test_zone_line_visualization_returns_bgr_frames() -> None:
    """Analytics geometry can be drawn with normalized coordinates and counts."""
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    zone = PolygonZone(
        id="zone",
        name="Zone",
        points=[(0.1, 0.1), (0.8, 0.1), (0.8, 0.8), (0.1, 0.8)],
        normalized=True,
    )
    line = CountingLine(
        id="door",
        name="Door",
        start=(0.5, 0.0),
        end=(0.5, 1.0),
        normalized=True,
    )

    zone_frame = draw_zones(frame, [zone], {"zone": 2}, thresholds={"zone": 1})
    annotated = draw_lines(zone_frame, [line], {"door": {"in": 2, "out": 1}})

    assert annotated.shape == frame.shape
    assert annotated.dtype == np.uint8
    assert np.any(annotated)
