"""Cursor-paginated event history and CSV export."""

import csv
import io
import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from vision_pipeline.api.db.models import Stream
from vision_pipeline.api.db.repositories import EventRepository
from vision_pipeline.api.deps import get_session

router = APIRouter(prefix="/streams/{stream_id}/events", tags=["events"])


def _event_dict(row: Any) -> dict[str, Any]:
    return {
        "id": row.id,
        "stream_id": row.stream_id,
        "timestamp": row.timestamp.isoformat(),
        "timestamp_s": row.timestamp_s,
        "frame_id": row.frame_id,
        "track_id": row.track_id,
        "event_type": row.event_type,
        "zone_id": row.zone_id,
        "line_id": row.line_id,
        "extra": row.extra,
    }


@router.get("")
def list_events(
    stream_id: str,
    session: Session = Depends(get_session),
    cursor: int | None = Query(default=None, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
) -> dict[str, Any]:
    """Fetch event history using the last returned event ID as cursor."""
    if session.get(Stream, stream_id) is None:
        raise HTTPException(status_code=404, detail="Stream not found.")
    rows = EventRepository(session).page(stream_id, cursor=cursor, limit=limit)
    return {
        "items": [_event_dict(row) for row in rows],
        "next_cursor": rows[-1].id if rows else None,
    }


@router.get("/export.csv")
def export_events(
    stream_id: str,
    session: Session = Depends(get_session),
) -> StreamingResponse:
    """Download all events for a stream as CSV."""
    if session.get(Stream, stream_id) is None:
        raise HTTPException(status_code=404, detail="Stream not found.")
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(
        [
            "id",
            "timestamp",
            "frame_id",
            "track_id",
            "event_type",
            "zone_id",
            "line_id",
            "extra",
        ]
    )
    for row in EventRepository(session).page(stream_id, cursor=None, limit=100000):
        writer.writerow(
            [
                row.id,
                row.timestamp.isoformat(),
                row.frame_id,
                row.track_id,
                row.event_type,
                row.zone_id or "",
                row.line_id or "",
                json.dumps(row.extra, separators=(",", ":")),
            ]
        )
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{stream_id}-events.csv"'
        },
    )
