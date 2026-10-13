"""Video-time buckets and analytics summary models."""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any

from vision_pipeline.analytics.events import Event, EventType


@dataclass
class _Bucket:
    """Mutable accumulation state for a single source-time interval."""

    entries: int = 0
    exits: int = 0
    visitor_ids: set[int] = field(default_factory=set)
    occupancy_sum: float = 0.0
    occupancy_duration: float = 0.0
    peak_occupancy: int = 0
    dwell_by_zone: dict[str, list[float]] = field(default_factory=dict)


@dataclass
class AnalyticsSummary:
    """Final visitors, bucket, zone, line, and dwell statistics."""

    total_unique_visitors: int
    total_entries: int
    total_exits: int
    busiest_bucket: dict[str, Any] | None
    per_zone_stats: dict[str, dict[str, Any]]
    line_totals: dict[str, dict[str, int]]
    average_dwell_seconds: float
    top_5_longest_dwell_tracks: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "total_unique_visitors": self.total_unique_visitors,
            "total_entries": self.total_entries,
            "total_exits": self.total_exits,
            "busiest_bucket": self.busiest_bucket,
            "per_zone_stats": self.per_zone_stats,
            "line_totals": self.line_totals,
            "average_dwell_seconds": self.average_dwell_seconds,
            "top_5_longest_dwell_tracks": self.top_5_longest_dwell_tracks,
        }


class TimeSeriesAggregator:
    """Aggregate observed tracks and events into fixed video-time buckets."""

    def __init__(self, bucket_seconds: float = 60.0) -> None:
        if bucket_seconds <= 0:
            raise ValueError("bucket_seconds must be positive.")
        self.bucket_seconds = bucket_seconds
        self._buckets: dict[float, _Bucket] = {}
        self._visitors: set[int] = set()
        self.total_entries = 0
        self.total_exits = 0
        self.line_totals: dict[str, dict[str, int]] = {}
        self.dwell_records: list[dict[str, Any]] = []
        self._last_frame_time_s = 0.0
        self._has_frame = False
        self._last_event_time_s: float | None = None
        self._active_by_zone: dict[str, set[int]] = {}

    def add_frame(
        self,
        timestamp_s: float,
        occupancy: dict[str, int],
        track_ids: set[int],
    ) -> None:
        """Record frame occupancy and visitors in the bucket containing its time."""
        row = self._bucket(timestamp_s)
        row.peak_occupancy = max(row.peak_occupancy, sum(occupancy.values()))
        row.visitor_ids.update(track_ids)
        self._visitors.update(track_ids)
        self._last_frame_time_s = timestamp_s
        self._has_frame = True

    def add_event(self, event: Event) -> None:
        """Add event-derived counts and dwell values to a bucket."""
        if self._last_event_time_s is not None:
            if event.timestamp_s < self._last_event_time_s:
                raise ValueError("Events must be added in non-decreasing time order.")
            self._integrate_occupancy(self._last_event_time_s, event.timestamp_s)
        self._last_event_time_s = event.timestamp_s
        bucket = self._bucket(event.timestamp_s)
        bucket.visitor_ids.add(event.track_id)
        self._visitors.add(event.track_id)
        if event.event_type is EventType.ZONE_ENTER:
            self.total_entries += 1
            bucket.entries += 1
            if event.zone_id is not None:
                self._active_by_zone.setdefault(event.zone_id, set()).add(
                    event.track_id
                )
        elif event.event_type is EventType.ZONE_EXIT:
            self.total_exits += 1
            bucket.exits += 1
            if event.zone_id is not None:
                self._active_by_zone.setdefault(event.zone_id, set()).discard(
                    event.track_id
                )
                dwell = event.extra.get("dwell_seconds")
                if dwell is not None:
                    if not isinstance(dwell, (int, float)):
                        raise ValueError("Event dwell_seconds must be numeric.")
                    dwell_seconds = float(dwell)
                    bucket.dwell_by_zone.setdefault(event.zone_id, []).append(
                        dwell_seconds
                    )
                    self.dwell_records.append(
                        {
                            "track_id": event.track_id,
                            "zone_id": event.zone_id,
                            "dwell_seconds": dwell_seconds,
                        }
                    )
        elif event.event_type in (EventType.LINE_CROSS_IN, EventType.LINE_CROSS_OUT):
            if event.line_id is None:
                return
            totals = self.line_totals.setdefault(
                event.line_id,
                {"in": 0, "out": 0, "net": 0},
            )
            direction = "in" if event.event_type is EventType.LINE_CROSS_IN else "out"
            totals[direction] += 1
            totals["net"] = totals["in"] - totals["out"]

    def to_records(self) -> list[dict[str, Any]]:
        """Return sorted bucket records as plain dictionaries."""
        rows: list[dict[str, Any]] = []
        for start in sorted(self._buckets):
            bucket = self._buckets[start]
            row: dict[str, Any] = {
                "bucket_start_s": start,
                "entries": bucket.entries,
                "exits": bucket.exits,
                "unique_visitors": len(bucket.visitor_ids),
                "mean_occupancy": (
                    bucket.occupancy_sum / bucket.occupancy_duration
                    if bucket.occupancy_duration
                    else 0.0
                ),
                "peak_occupancy": bucket.peak_occupancy,
            }
            for zone_id, values in sorted(bucket.dwell_by_zone.items()):
                row[f"zone_{zone_id}_mean_dwell_s"] = fmean(values)
            rows.append(row)
        return rows

    def to_dataframe(self) -> list[dict[str, Any]]:
        """Return records without requiring a pandas dependency."""
        return self.to_records()

    def to_csv(self, path: str | Path) -> None:
        """Write bucket rows to a CSV file."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.to_records()
        columns = list(
            dict.fromkeys(
                [
                    "bucket_start_s",
                    "entries",
                    "exits",
                    "unique_visitors",
                    "mean_occupancy",
                    "peak_occupancy",
                    *(key for row in rows for key in row),
                ]
            )
        )
        with output_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def peak_bucket(self) -> dict[str, Any] | None:
        """Return the busiest recorded bucket, or ``None`` when there are none."""
        return max(
            self.to_records(),
            key=lambda row: (row["peak_occupancy"], row["entries"]),
            default=None,
        )

    def summary(
        self,
        zone_stats: dict[str, Any],
    ) -> AnalyticsSummary:
        """Build final summary values using current zone-history statistics."""
        dwell_values = [item["dwell_seconds"] for item in self.dwell_records]
        zone_data: dict[str, dict[str, Any]] = {}
        for zone_id, stats in zone_stats.items():
            zone_data[zone_id] = {
                "current_occupancy": stats.current_occupancy,
                "max_occupancy": stats.max_occupancy,
                "total_visits": stats.total_visits,
                "average_dwell_seconds": stats.average_dwell_seconds,
                "median_dwell_seconds": stats.median_dwell_seconds,
                "completed_dwell_seconds": list(stats.completed_dwell_seconds),
            }
        return AnalyticsSummary(
            total_unique_visitors=len(self._visitors),
            total_entries=self.total_entries,
            total_exits=self.total_exits,
            busiest_bucket=self.peak_bucket(),
            per_zone_stats=zone_data,
            line_totals={
                key: dict(value) for key, value in sorted(self.line_totals.items())
            },
            average_dwell_seconds=fmean(dwell_values) if dwell_values else 0.0,
            top_5_longest_dwell_tracks=sorted(
                self.dwell_records,
                key=lambda item: item["dwell_seconds"],
                reverse=True,
            )[:5],
        )

    def reset(self) -> None:
        """Clear all bucket and summary state."""
        self._buckets.clear()
        self._visitors.clear()
        self.total_entries = 0
        self.total_exits = 0
        self.line_totals.clear()
        self.dwell_records.clear()
        self._last_frame_time_s = 0.0
        self._has_frame = False
        self._last_event_time_s = None
        self._active_by_zone.clear()

    def _bucket(self, timestamp_s: float) -> _Bucket:
        """Get or create the half-open bucket containing a video timestamp."""
        if timestamp_s < 0:
            raise ValueError("Video timestamps must be non-negative.")
        start = int(timestamp_s // self.bucket_seconds) * self.bucket_seconds
        return self._buckets.setdefault(start, _Bucket())

    def _integrate_occupancy(self, start_s: float, end_s: float) -> None:
        """Integrate current occupancy over buckets crossed between two events."""
        cursor = start_s
        occupancy = sum(len(track_ids) for track_ids in self._active_by_zone.values())
        while cursor < end_s:
            bucket_start = int(cursor // self.bucket_seconds) * self.bucket_seconds
            segment_end = min(end_s, bucket_start + self.bucket_seconds)
            row = self._buckets.setdefault(bucket_start, _Bucket())
            duration = segment_end - cursor
            row.occupancy_sum += occupancy * duration
            row.occupancy_duration += duration
            row.peak_occupancy = max(row.peak_occupancy, occupancy)
            cursor = segment_end
