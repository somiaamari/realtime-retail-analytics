"""Initial anonymous analytics schema.

Revision ID: 0001
Revises:
"""

from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create stream, event, and aggregate tables with query indexes."""
    op.create_table(
        "streams",
        sa.Column("stream_id", sa.String(128), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("autostart", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "runs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "stream_id",
            sa.String(128),
            sa.ForeignKey("streams.stream_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("stopped_at", sa.DateTime(timezone=True)),
        sa.Column("status", sa.String(32), nullable=False),
    )
    op.create_index("ix_runs_stream_id", "runs", ["stream_id"])
    op.create_table(
        "events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("stream_id", sa.String(128), nullable=False),
        sa.Column(
            "run_id",
            sa.Integer(),
            sa.ForeignKey("runs.id", ondelete="SET NULL"),
        ),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("timestamp_s", sa.Float(), nullable=False),
        sa.Column("frame_id", sa.Integer(), nullable=False),
        sa.Column("track_id", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(40), nullable=False),
        sa.Column("zone_id", sa.String(128)),
        sa.Column("line_id", sa.String(128)),
        sa.Column("extra", sa.JSON(), nullable=False),
    )
    op.create_index("ix_events_stream_id", "events", ["stream_id"])
    op.create_index("ix_events_timestamp", "events", ["timestamp"])
    op.create_index("ix_events_stream_ts", "events", ["stream_id", "timestamp"])
    op.create_index("ix_events_event_type", "events", ["event_type"])
    op.create_table(
        "bucket_stats",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("stream_id", sa.String(128), nullable=False),
        sa.Column("bucket_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("entries", sa.Integer(), nullable=False),
        sa.Column("exits", sa.Integer(), nullable=False),
        sa.Column("unique_visitors", sa.Integer(), nullable=False),
        sa.Column("mean_occupancy", sa.Float(), nullable=False),
        sa.Column("peak_occupancy", sa.Integer(), nullable=False),
        sa.Column("zone_stats", sa.JSON(), nullable=False),
    )
    op.create_index("ix_bucket_stats_stream_id", "bucket_stats", ["stream_id"])
    op.create_index(
        "ix_bucket_stream_start", "bucket_stats", ["stream_id", "bucket_start"]
    )
    for table, columns in (
        (
            "zones",
            [
                ("zone_id", sa.String(128)),
                ("name", sa.String(200)),
                ("geometry", sa.JSON()),
            ],
        ),
        (
            "lines",
            [
                ("line_id", sa.String(128)),
                ("name", sa.String(200)),
                ("geometry", sa.JSON()),
            ],
        ),
    ):
        op.create_table(
            table,
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("stream_id", sa.String(128), nullable=False),
            *(sa.Column(name, kind, nullable=False) for name, kind in columns),  # type: ignore[arg-type]
        )
        op.create_index(f"ix_{table}_stream_id", table, ["stream_id"])
    op.create_table(
        "heatmap_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("stream_id", sa.String(128), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("values", sa.JSON(), nullable=False),
    )
    op.create_index(
        "ix_heatmap_snapshots_stream_id", "heatmap_snapshots", ["stream_id"]
    )
    op.create_index(
        "ix_heatmap_stream_ts", "heatmap_snapshots", ["stream_id", "timestamp"]
    )
    op.create_table(
        "dwell_records",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("stream_id", sa.String(128), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("zone_id", sa.String(128)),
        sa.Column("dwell_seconds", sa.Float(), nullable=False),
    )
    op.create_index("ix_dwell_records_stream_id", "dwell_records", ["stream_id"])
    op.create_index("ix_dwell_stream_ts", "dwell_records", ["stream_id", "timestamp"])


def downgrade() -> None:
    """Remove schema in reverse dependency order."""
    op.drop_table("dwell_records")
    op.drop_table("heatmap_snapshots")
    op.drop_table("lines")
    op.drop_table("zones")
    op.drop_table("bucket_stats")
    op.drop_table("events")
    op.drop_table("runs")
    op.drop_table("streams")
