"""Rebuild summary statistics and time-series CSV from JSON Lines events."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


def _positive_float(value: str) -> float:
    """Parse a positive floating-point CLI argument."""
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the event-replay command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True, help="Input JSONL events")
    parser.add_argument(
        "--timeseries-out",
        type=Path,
        default=Path("outputs/replay_timeseries.csv"),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("outputs/replay_summary.json"),
    )
    parser.add_argument("--bucket-seconds", type=_positive_float, default=60.0)
    return parser


def _read_events(path: Path) -> list[dict[str, Any]]:
    """Read and validate JSONL event records."""
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Event line {line_number} must be a JSON object.")
            try:
                record["timestamp_s"] = float(record["timestamp_s"])
                record["track_id"] = int(record["track_id"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Event line {line_number} has invalid timestamp_s or track_id."
                ) from error
            event_type = record.get("event_type")
            if not isinstance(event_type, str):
                raise ValueError(f"Event line {line_number} has no event_type.")
            record["event_type"] = event_type.rsplit(".", maxsplit=1)[-1].upper()
            extra = record.get("extra") or {}
            if not isinstance(extra, dict):
                raise ValueError(f"Event line {line_number} extra must be an object.")
            record["extra"] = extra
            events.append(record)
    return sorted(events, key=lambda record: record["timestamp_s"])


def _replay(
    events: list[dict[str, Any]],
    bucket_seconds: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Recompute visitor, zone, line, dwell, and occupancy statistics."""
    entries = 0
    exits = 0
    visitors: set[int] = set()
    line_totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"in": 0, "out": 0, "net": 0}
    )
    zone_visits: dict[str, int] = defaultdict(int)
    zone_dwells: dict[str, list[float]] = defaultdict(list)
    bucket_dwells: dict[tuple[float, str], list[float]] = defaultdict(list)
    track_dwells: list[dict[str, Any]] = []
    track_intervals: dict[int, list[float]] = {}
    zone_peak_occupancy: dict[str, int] = defaultdict(int)
    buckets: dict[float, dict[str, Any]] = {}
    active_by_zone: dict[str, set[int]] = defaultdict(set)

    def bucket(start: float) -> dict[str, Any]:
        return buckets.setdefault(
            start,
            {
                "bucket_start_s": start,
                "entries": 0,
                "exits": 0,
                "_visitors": set(),
                "mean_occupancy": 0.0,
                "peak_occupancy": 0,
                "_occupancy_seconds": 0.0,
                "_duration": 0.0,
            },
        )

    for event in events:
        timestamp = event["timestamp_s"]
        bucket_start = int(timestamp // bucket_seconds) * bucket_seconds
        row = bucket(bucket_start)
        track_id = event["track_id"]
        event_type = event["event_type"]
        visitors.add(track_id)
        row["_visitors"].add(track_id)
        zone_id = event.get("zone_id")
        line_id = event.get("line_id")

        if event_type == "ZONE_ENTER":
            entries += 1
            row["entries"] += 1
            if zone_id is not None:
                zone_visits[str(zone_id)] += 1
                active_by_zone[str(zone_id)].add(track_id)
        elif event_type == "ZONE_EXIT":
            exits += 1
            row["exits"] += 1
            if zone_id is not None:
                active_by_zone[str(zone_id)].discard(track_id)
                dwell = event["extra"].get("dwell_seconds")
                if dwell is not None:
                    normalized_zone_id = str(zone_id)
                    dwell_value = float(dwell)
                    zone_dwells[normalized_zone_id].append(dwell_value)
                    bucket_dwells[(bucket_start, normalized_zone_id)].append(
                        dwell_value
                    )
                    track_dwells.append(
                        {
                            "track_id": track_id,
                            "zone_id": zone_id,
                            "dwell_seconds": dwell_value,
                        }
                    )
        elif event_type in ("LINE_CROSS_IN", "LINE_CROSS_OUT"):
            if line_id is not None:
                direction = "in" if event_type == "LINE_CROSS_IN" else "out"
                line_totals[str(line_id)][direction] += 1
        elif event_type == "TRACK_STARTED":
            track_intervals[track_id] = [timestamp, timestamp]
        elif event_type == "TRACK_ENDED":
            interval = track_intervals.setdefault(track_id, [timestamp, timestamp])
            interval[1] = max(interval[1], timestamp)

    if events:
        active_by_zone.clear()
        cursor = events[0]["timestamp_s"]
        occupancy = 0
        for event in events:
            timestamp = event["timestamp_s"]
            while cursor < timestamp:
                start = int(cursor // bucket_seconds) * bucket_seconds
                end = min(timestamp, start + bucket_seconds)
                row = bucket(float(start))
                duration = end - cursor
                row["_occupancy_seconds"] += occupancy * duration
                row["_duration"] += duration
                row["peak_occupancy"] = max(row["peak_occupancy"], occupancy)
                cursor = end
            if event["event_type"] == "ZONE_ENTER":
                zone_id = event.get("zone_id")
                if zone_id is not None:
                    active_by_zone[str(zone_id)].add(event["track_id"])
                    zone_peak_occupancy[str(zone_id)] = max(
                        zone_peak_occupancy[str(zone_id)],
                        len(active_by_zone[str(zone_id)]),
                    )
            elif event["event_type"] == "ZONE_EXIT":
                zone_id = event.get("zone_id")
                if zone_id is not None:
                    active_by_zone[str(zone_id)].discard(event["track_id"])
            occupancy = sum(len(track_ids) for track_ids in active_by_zone.values())
            event_bucket = int(timestamp // bucket_seconds) * bucket_seconds
            bucket(float(event_bucket))["peak_occupancy"] = max(
                bucket(float(event_bucket))["peak_occupancy"],
                occupancy,
            )
        final_timestamp = events[-1]["timestamp_s"]
        while cursor < final_timestamp:
            start = int(cursor // bucket_seconds) * bucket_seconds
            end = min(final_timestamp, start + bucket_seconds)
            row = bucket(float(start))
            duration = end - cursor
            row["_occupancy_seconds"] += occupancy * duration
            row["_duration"] += duration
            row["peak_occupancy"] = max(row["peak_occupancy"], occupancy)
            cursor = end

    for track_id, (start_s, end_s) in track_intervals.items():
        first_bucket = int(start_s // bucket_seconds) * bucket_seconds
        last_bucket = int(end_s // bucket_seconds) * bucket_seconds
        current = first_bucket
        while current <= last_bucket:
            bucket(float(current))["_visitors"].add(track_id)
            current += bucket_seconds

    records: list[dict[str, Any]] = []
    for start in sorted(buckets):
        row = buckets[start]
        duration = row.pop("_duration")
        occupancy_seconds = row.pop("_occupancy_seconds")
        unique_ids = row.pop("_visitors")
        row["unique_visitors"] = len(unique_ids)
        row["mean_occupancy"] = occupancy_seconds / duration if duration > 0 else 0.0
        for zone_id in zone_dwells:
            events_in_zone = bucket_dwells[(start, zone_id)]
            row[f"zone_{zone_id}_mean_dwell_s"] = (
                sum(events_in_zone) / len(events_in_zone) if events_in_zone else 0.0
            )
        records.append(row)

    for totals in line_totals.values():
        totals["net"] = totals["in"] - totals["out"]
    dwell_values = [item["dwell_seconds"] for item in track_dwells]
    summary = {
        "total_unique_visitors": len(visitors),
        "total_entries": entries,
        "total_exits": exits,
        "line_totals": dict(line_totals),
        "per_zone_stats": {
            zone_id: {
                "current_occupancy": 0,
                "max_occupancy": zone_peak_occupancy[zone_id],
                "total_visits": zone_visits[zone_id],
                "average_dwell_seconds": sum(values) / len(values) if values else 0.0,
                "median_dwell_seconds": median(values) if values else 0.0,
                "completed_dwell_seconds": values,
            }
            for zone_id in set(zone_dwells) | set(zone_visits)
            for values in [zone_dwells.get(zone_id, [])]
        },
        "average_dwell_seconds": sum(dwell_values) / len(dwell_values)
        if dwell_values
        else 0.0,
        "top_5_longest_dwell_tracks": sorted(
            track_dwells,
            key=lambda item: item["dwell_seconds"],
            reverse=True,
        )[:5],
        "busiest_bucket": max(
            records,
            key=lambda record: (record["peak_occupancy"], record["entries"]),
            default=None,
        ),
    }
    return summary, records


def run(args: argparse.Namespace) -> int:
    """Replay events and write summary JSON and bucket CSV."""
    events = _read_events(args.events)
    summary, records = _replay(events, args.bucket_seconds)
    args.timeseries_out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(
        dict.fromkeys(
            [
                "bucket_start_s",
                "entries",
                "exits",
                "unique_visitors",
                "mean_occupancy",
                "peak_occupancy",
                *(key for row in records for key in row),
            ]
        )
    )
    with args.timeseries_out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Summary: {args.summary_out}")
    print(f"Time series: {args.timeseries_out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
