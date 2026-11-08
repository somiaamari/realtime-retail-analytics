"""Historical aggregate analytics endpoints."""

import io
from datetime import datetime
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from vision_pipeline.api.db.models import DwellRecord, EventRecord
from vision_pipeline.api.db.repositories import AnalyticsRepository
from vision_pipeline.api.deps import get_session

router = APIRouter(prefix="/streams/{stream_id}/analytics", tags=["analytics"])


def _require_stream(stream_id: str, session: Session) -> None:
    from vision_pipeline.api.db.models import Stream

    if session.get(Stream, stream_id) is None:
        raise HTTPException(status_code=404, detail="Stream not found.")


@router.get("/summary")
def summary(
    stream_id: str,
    session: Session = Depends(get_session),
    start: datetime | None = None,
    end: datetime | None = None,
) -> dict[str, Any]:
    """Return event totals and dwell aggregates for a stream."""
    _require_stream(stream_id, session)
    query = select(EventRecord).where(EventRecord.stream_id == stream_id)
    if start is not None:
        query = query.where(EventRecord.timestamp >= start)
    if end is not None:
        query = query.where(EventRecord.timestamp <= end)
    events = list(session.scalars(query).all())
    counts: dict[str, int] = {}
    zones: dict[str, int] = {}
    lines: dict[str, int] = {}
    visitors: set[tuple[int, int | None]] = set()
    for event in events:
        counts[event.event_type] = counts.get(event.event_type, 0) + 1
        visitors.add((event.track_id, event.run_id))
        if event.zone_id:
            zones[event.zone_id] = zones.get(event.zone_id, 0) + 1
        if event.line_id:
            lines[event.line_id] = lines.get(event.line_id, 0) + 1
    dwell_query = select(func.avg(DwellRecord.dwell_seconds)).where(
        DwellRecord.stream_id == stream_id
    )
    if start is not None:
        dwell_query = dwell_query.where(DwellRecord.timestamp >= start)
    if end is not None:
        dwell_query = dwell_query.where(DwellRecord.timestamp <= end)
    average_dwell = session.scalar(dwell_query)
    return {
        "stream_id": stream_id,
        "event_count": len(events),
        "event_counts": counts,
        "unique_track_count": len(visitors),
        "zone_events": zones,
        "line_events": lines,
        "average_dwell_seconds": float(average_dwell or 0.0),
    }


@router.get("/timeseries")
def timeseries(
    stream_id: str,
    session: Session = Depends(get_session),
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[dict[str, Any]]:
    """Return persisted traffic buckets."""
    _require_stream(stream_id, session)
    rows = AnalyticsRepository(session).buckets(stream_id, start, end)
    return [
        {
            "bucket_start": row.bucket_start.isoformat(),
            "entries": row.entries,
            "exits": row.exits,
            "unique_visitors": row.unique_visitors,
            "mean_occupancy": row.mean_occupancy,
            "peak_occupancy": row.peak_occupancy,
            "zones": row.zone_stats,
        }
        for row in rows
    ]


@router.get("/peak-hours")
def peak_hours(
    stream_id: str,
    session: Session = Depends(get_session),
    limit: int = Query(default=5, ge=1, le=24),
) -> list[dict[str, Any]]:
    """Return busiest persisted buckets ordered by peak occupancy."""
    _require_stream(stream_id, session)
    rows = AnalyticsRepository(session).buckets(stream_id, None, None)
    return [
        {
            "bucket_start": row.bucket_start.isoformat(),
            "peak_occupancy": row.peak_occupancy,
            "entries": row.entries,
        }
        for row in sorted(
            rows, key=lambda row: (row.peak_occupancy, row.entries), reverse=True
        )[:limit]
    ]


@router.get("/zones")
def zones(
    stream_id: str,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Summarize event counts grouped by zone."""
    _require_stream(stream_id, session)
    rows = session.execute(
        select(EventRecord.zone_id, EventRecord.event_type, func.count())
        .where(
            EventRecord.stream_id == stream_id,
            EventRecord.zone_id.is_not(None),
        )
        .group_by(EventRecord.zone_id, EventRecord.event_type)
    )
    result: dict[str, dict[str, int]] = {}
    for zone_id, event_type, count in rows:
        result.setdefault(str(zone_id), {})[str(event_type)] = int(count)
    return {"stream_id": stream_id, "zones": result}


@router.get("/dwell")
def dwell(
    stream_id: str,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Return aggregate dwell distribution by zone."""
    _require_stream(stream_id, session)
    rows = AnalyticsRepository(session).dwell(stream_id)
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row.zone_id or "unknown", []).append(row.dwell_seconds)
    return {
        "stream_id": stream_id,
        "zones": {
            zone: {
                "count": len(values),
                "average_seconds": sum(values) / len(values),
                "max_seconds": max(values),
            }
            for zone, values in grouped.items()
        },
    }


@router.get("/heatmap")
def heatmap(
    stream_id: str,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Return the latest coarse-grid heatmap aggregate."""
    _require_stream(stream_id, session)
    snapshot = AnalyticsRepository(session).heatmap(stream_id)
    if snapshot is None:
        return {"stream_id": stream_id, "timestamp": None, "values": []}
    return {
        "stream_id": stream_id,
        "timestamp": snapshot.timestamp.isoformat(),
        "width": snapshot.width,
        "height": snapshot.height,
        "values": snapshot.values,
    }


@router.get("/heatmap.png", response_class=Response)
def heatmap_png(
    stream_id: str,
    session: Session = Depends(get_session),
) -> Response:
    """Render the most recent heatmap aggregate as a PNG image."""
    _require_stream(stream_id, session)
    snapshot = AnalyticsRepository(session).heatmap(stream_id)
    if snapshot is None or not snapshot.values:
        raise HTTPException(status_code=404, detail="No heatmap snapshot is available.")
    array = np.asarray(snapshot.values, dtype=np.float32)
    normalized = cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    ok, encoded = cv2.imencode(".png", color)
    if not ok:
        raise HTTPException(status_code=500, detail="Unable to encode heatmap.")
    return Response(content=encoded.tobytes(), media_type="image/png")
