"""Seed the configured API database with reproducible 14-day aggregates."""

from alembic import command
from alembic.config import Config
from sqlalchemy import select
from datetime import datetime, timezone

from vision_pipeline.api.db.models import (
    BucketStat,
    DwellRecord,
    HeatmapSnapshot,
    Stream,
)
from vision_pipeline.api.db.session import create_database_engine, create_session_factory
from vision_pipeline.api.services.demo import synthetic_heatmap, synthetic_records
from vision_pipeline.api.settings import APISettings


def main() -> None:
    """Migrate the configured database and insert demo aggregates once."""
    settings = APISettings()
    engine = create_database_engine(settings.database_url)
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", settings.database_url.replace("%", "%%"))
    command.upgrade(config, "head")
    factory = create_session_factory(engine)
    with factory() as session:
        if session.get(Stream, "demo") is None:
            session.add(
                Stream(
                    stream_id="demo",
                    name="Synthetic demo stream",
                    source="fake",
                    config={
                        "stream_id": "demo",
                        "name": "Synthetic demo stream",
                        "source": "fake",
                        "backend": "fake",
                        "autostart": True,
                    },
                    autostart=True,
                )
            )
        if session.scalar(select(BucketStat.id).limit(1)) is None:
            for item in synthetic_records():
                if item["kind"] == "bucket":
                    session.add(
                        BucketStat(
                            stream_id="demo",
                            bucket_start=item["timestamp"],
                            entries=item["entries"],
                            exits=item["exits"],
                            unique_visitors=item["unique_visitors"],
                            mean_occupancy=item["mean_occupancy"],
                            peak_occupancy=item["peak_occupancy"],
                            zone_stats=item["zone_stats"],
                        )
                    )
                else:
                    session.add(
                        DwellRecord(
                            stream_id="demo",
                            timestamp=item["timestamp"],
                            zone_id=item["zone_id"],
                            dwell_seconds=item["dwell_seconds"],
                        )
                    )
            grid = synthetic_heatmap()
            session.add(
                HeatmapSnapshot(
                    stream_id="demo",
                    timestamp=datetime.now(timezone.utc),
                    width=len(grid[0]),
                    height=len(grid),
                    values=grid,
                )
            )
        session.commit()
    engine.dispose()
    print("Seeded 14 days of deterministic synthetic aggregates for stream 'demo'.")


if __name__ == "__main__":
    main()
