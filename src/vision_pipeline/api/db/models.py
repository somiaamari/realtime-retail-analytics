"""SQLAlchemy models for anonymous events and aggregate analytics."""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base shared by all API tables."""


class Stream(Base):
    """Configured stream independent of any particular process run."""

    __tablename__ = "streams"

    stream_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    source: Mapped[str] = mapped_column(Text)
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    autostart: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )
    runs: Mapped[list["Run"]] = relationship(back_populates="stream")


class Run(Base):
    """One execution of a configured stream."""

    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stream_id: Mapped[str] = mapped_column(
        ForeignKey("streams.stream_id", ondelete="CASCADE"), index=True
    )
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    stopped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(32), default="starting")
    stream: Mapped[Stream] = relationship(back_populates="runs")


class EventRecord(Base):
    """Anonymous analytics event; track IDs are only meaningful within a run."""

    __tablename__ = "events"
    __table_args__ = (
        Index("ix_events_stream_ts", "stream_id", "timestamp"),
        Index("ix_events_event_type", "event_type"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stream_id: Mapped[str] = mapped_column(String(128), index=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("runs.id", ondelete="SET NULL"))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    timestamp_s: Mapped[float] = mapped_column(Float)
    frame_id: Mapped[int] = mapped_column(Integer)
    track_id: Mapped[int] = mapped_column(Integer)
    event_type: Mapped[str] = mapped_column(String(40))
    zone_id: Mapped[str | None] = mapped_column(String(128))
    line_id: Mapped[str | None] = mapped_column(String(128))
    extra: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class BucketStat(Base):
    """Persisted time bucket containing counts and aggregate occupancy."""

    __tablename__ = "bucket_stats"
    __table_args__ = (Index("ix_bucket_stream_start", "stream_id", "bucket_start"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stream_id: Mapped[str] = mapped_column(String(128), index=True)
    bucket_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    entries: Mapped[int] = mapped_column(Integer, default=0)
    exits: Mapped[int] = mapped_column(Integer, default=0)
    unique_visitors: Mapped[int] = mapped_column(Integer, default=0)
    mean_occupancy: Mapped[float] = mapped_column(Float, default=0.0)
    peak_occupancy: Mapped[int] = mapped_column(Integer, default=0)
    zone_stats: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class ZoneRecord(Base):
    """Persisted zone geometry and non-identifying labels."""

    __tablename__ = "zones"
    __table_args__ = (Index("ix_zones_stream_id", "stream_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stream_id: Mapped[str] = mapped_column(String(128), index=True)
    zone_id: Mapped[str] = mapped_column(String(128))
    name: Mapped[str] = mapped_column(String(200))
    geometry: Mapped[dict[str, Any]] = mapped_column(JSON)


class LineRecord(Base):
    """Persisted counting-line geometry and labels."""

    __tablename__ = "lines"
    __table_args__ = (Index("ix_lines_stream_id", "stream_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stream_id: Mapped[str] = mapped_column(String(128), index=True)
    line_id: Mapped[str] = mapped_column(String(128))
    name: Mapped[str] = mapped_column(String(200))
    geometry: Mapped[dict[str, Any]] = mapped_column(JSON)


class HeatmapSnapshot(Base):
    """Coarse spatial aggregate, never a stored video/image frame."""

    __tablename__ = "heatmap_snapshots"
    __table_args__ = (Index("ix_heatmap_stream_ts", "stream_id", "timestamp"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stream_id: Mapped[str] = mapped_column(String(128), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    values: Mapped[list[list[float]]] = mapped_column(JSON)


class DwellRecord(Base):
    """Aggregate dwell duration without retaining any visitor identity."""

    __tablename__ = "dwell_records"
    __table_args__ = (Index("ix_dwell_stream_ts", "stream_id", "timestamp"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stream_id: Mapped[str] = mapped_column(String(128), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    zone_id: Mapped[str | None] = mapped_column(String(128))
    dwell_seconds: Mapped[float] = mapped_column(Float)
