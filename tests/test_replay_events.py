"""Tests for offline event replay."""

from scripts.replay_events import _replay


def test_replay_recomputes_zone_line_and_occupancy_totals() -> None:
    """Event replay reproduces core counts and time-bucket statistics."""
    events = [
        {
            "event_type": "TRACK_STARTED",
            "timestamp_s": 0.0,
            "track_id": 7,
            "extra": {},
        },
        {
            "event_type": "ZONE_ENTER",
            "timestamp_s": 0.0,
            "track_id": 7,
            "zone_id": "queue",
            "extra": {},
        },
        {
            "event_type": "LINE_CROSS_IN",
            "timestamp_s": 2.0,
            "track_id": 7,
            "line_id": "door",
            "extra": {},
        },
        {
            "event_type": "ZONE_EXIT",
            "timestamp_s": 10.0,
            "track_id": 7,
            "zone_id": "queue",
            "extra": {"dwell_seconds": 10.0},
        },
        {
            "event_type": "DWELL_COMPLETE",
            "timestamp_s": 10.0,
            "track_id": 7,
            "zone_id": "queue",
            "extra": {"dwell_seconds": 10.0},
        },
    ]

    summary, rows = _replay(events, bucket_seconds=5.0)

    assert summary["total_unique_visitors"] == 1
    assert summary["total_entries"] == 1
    assert summary["total_exits"] == 1
    assert summary["line_totals"]["door"] == {"in": 1, "out": 0, "net": 1}
    assert summary["per_zone_stats"]["queue"]["average_dwell_seconds"] == 10.0
    assert rows[0]["mean_occupancy"] == 1.0
    assert rows[1]["mean_occupancy"] == 1.0
    assert rows[2]["zone_queue_mean_dwell_s"] == 10.0
