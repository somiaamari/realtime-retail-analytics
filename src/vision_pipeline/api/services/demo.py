"""Deterministic anonymous analytics data for demos and local API tests."""

import random
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from typing import Any


def synthetic_records(
    *,
    seed: int = 17,
    days: int = 14,
    now: datetime | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield seeded traffic and dwell aggregates across the requested period."""
    randomizer = random.Random(seed)
    end = now or datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    for day in range(days):
        day_start = start + timedelta(days=day)
        for hour in range(24):
            hour_start = day_start + timedelta(hours=hour)
            multiplier = 2 if 11 <= hour <= 14 or 17 <= hour <= 20 else 1
            visitors = randomizer.randint(3, 10) * multiplier
            yield {
                "kind": "bucket",
                "timestamp": hour_start,
                "entries": visitors,
                "exits": max(0, visitors - randomizer.randint(-2, 2)),
                "unique_visitors": visitors,
                "mean_occupancy": round(visitors * randomizer.uniform(0.2, 0.6), 2),
                "peak_occupancy": visitors,
                "zone_stats": {"main": visitors},
            }
            for _ in range(max(1, visitors // 4)):
                yield {
                    "kind": "dwell",
                    "timestamp": hour_start
                    + timedelta(minutes=randomizer.randrange(60)),
                    "zone_id": "main",
                    "dwell_seconds": round(randomizer.uniform(8.0, 180.0), 2),
                }


def synthetic_heatmap(seed: int = 17, width: int = 32, height: int = 24) -> list[list[float]]:
    """Return a seeded, coarse spatial density grid."""
    randomizer = random.Random(seed)
    return [
        [
            round(
                max(
                    0.0,
                    randomizer.random()
                    + 2.0 * (1.0 - abs(column - width * 0.55) / width)
                    * (1.0 - abs(row - height * 0.65) / height),
                ),
                3,
            )
            for column in range(width)
        ]
        for row in range(height)
    ]
