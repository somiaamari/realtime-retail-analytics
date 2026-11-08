"""Typed database query helpers used by services and routers."""

from datetime import datetime
from typing import Any

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from vision_pipeline.api.db.models import (
    BucketStat,
    DwellRecord,
    EventRecord,
    HeatmapSnapshot,
    Stream,
)


class EventRepository:
    """Read analytics event history with cursor-based pagination."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def page(
        self, stream_id: str, *, cursor: int | None, limit: int
    ) -> list[EventRecord]:
        """Return an ascending page after an optional event ID cursor."""
        query: Select[tuple[EventRecord]] = select(EventRecord).where(
            EventRecord.stream_id == stream_id
        )
        if cursor is not None:
            query = query.where(EventRecord.id > cursor)
        return list(
            self.session.scalars(
                query.order_by(EventRecord.id).limit(limit)
            ).all()
        )


class AnalyticsRepository:
    """Read bucket, dwell, heatmap, and event aggregates."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def buckets(
        self, stream_id: str, start: datetime | None, end: datetime | None
    ) -> list[BucketStat]:
        """Return buckets optionally restricted to a timestamp range."""
        query = select(BucketStat).where(BucketStat.stream_id == stream_id)
        if start is not None:
            query = query.where(BucketStat.bucket_start >= start)
        if end is not None:
            query = query.where(BucketStat.bucket_start <= end)
        return list(self.session.scalars(query.order_by(BucketStat.bucket_start)).all())

    def events(
        self, stream_id: str, start: datetime | None, end: datetime | None
    ) -> list[EventRecord]:
        """Return events optionally restricted to a timestamp range."""
        query = select(EventRecord).where(EventRecord.stream_id == stream_id)
        if start is not None:
            query = query.where(EventRecord.timestamp >= start)
        if end is not None:
            query = query.where(EventRecord.timestamp <= end)
        return list(self.session.scalars(query.order_by(EventRecord.timestamp)).all())

    def dwell(self, stream_id: str) -> list[DwellRecord]:
        """Return dwell aggregates for one stream."""
        return list(
            self.session.scalars(
                select(DwellRecord)
                .where(DwellRecord.stream_id == stream_id)
                .order_by(DwellRecord.timestamp)
            ).all()
        )

    def heatmap(self, stream_id: str) -> HeatmapSnapshot | None:
        """Return the most recent heatmap aggregate for one stream."""
        return self.session.scalar(
            select(HeatmapSnapshot)
            .where(HeatmapSnapshot.stream_id == stream_id)
            .order_by(HeatmapSnapshot.timestamp.desc())
            .limit(1)
        )


class StreamRepository:
    """Persist stream configuration records."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, stream_id: str) -> Stream | None:
        """Fetch one stream configuration."""
        return self.session.get(Stream, stream_id)

    def list(self) -> list[Stream]:
        """List configured streams in stable ID order."""
        return list(self.session.scalars(select(Stream).order_by(Stream.stream_id)).all())

    def upsert(self, stream: Stream) -> Stream:
        """Insert or update stream configuration."""
        self.session.merge(stream)
        self.session.commit()
        return stream
