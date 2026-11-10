"""Asynchronous-to-the-pipeline persistence worker for analytics events."""

import logging
import queue
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import delete
from sqlalchemy.orm import Session, sessionmaker

from vision_pipeline.analytics.events import Event
from vision_pipeline.api.db.models import (
    BucketStat,
    DwellRecord,
    EventRecord,
    HeatmapSnapshot,
)

logger = logging.getLogger(__name__)
_STOP = object()


class PersistenceWorker:
    """Buffer event writes on a dedicated thread with bounded memory usage."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        *,
        batch_size: int = 100,
        queue_size: int = 5000,
        retention_days: int = 30,
    ) -> None:
        self.session_factory = session_factory
        self.batch_size = batch_size
        self.retention_days = retention_days
        self._queue: queue.Queue[object] = queue.Queue(maxsize=queue_size)
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()
        self._lock = threading.Lock()
        self._metrics = {"enqueued": 0, "written": 0, "dropped": 0, "failures": 0}

    def start(self) -> None:
        """Start the single database writer thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stopping.clear()
        self._thread = threading.Thread(
            target=self._run, name="api-db-writer", daemon=True
        )
        self._thread.start()

    def enqueue_event(self, stream_id: str, event: Event) -> None:
        """Queue one event without blocking a streaming/pipeline thread."""
        if self._stopping.is_set():
            return
        item = (stream_id, event)
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                pass
            with self._lock:
                self._metrics["dropped"] += 1
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                with self._lock:
                    self._metrics["dropped"] += 1
                return
        with self._lock:
            self._metrics["enqueued"] += 1

    def _run(self) -> None:
        """Consume batches and retry transient database failures."""
        while not self._stopping.is_set() or not self._queue.empty():
            try:
                first = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if first is _STOP:
                self._queue.task_done()
                break
            batch = [first]
            while len(batch) < self.batch_size:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if item is _STOP:
                    self._stopping.set()
                    self._queue.task_done()
                    break
                batch.append(item)
            for attempt in range(4):
                try:
                    self._write_batch(batch)
                    with self._lock:
                        self._metrics["written"] += len(batch)
                    break
                except Exception:
                    with self._lock:
                        self._metrics["failures"] += 1
                    logger.exception(
                        "Persisting analytics batch failed (attempt %d)", attempt + 1
                    )
                    if attempt < 3:
                        time.sleep(0.1 * (2**attempt))
                    else:
                        with self._lock:
                            self._metrics["dropped"] += len(batch)
            for _ in batch:
                self._queue.task_done()

    def _write_batch(self, batch: list[object]) -> None:
        """Convert events into anonymous database records and commit atomically."""
        rows: list[EventRecord] = []
        for item in batch:
            stream_id, event = item  # type: ignore[misc]
            rows.append(
                EventRecord(
                    stream_id=stream_id,  # type: ignore[has-type]
                    timestamp=datetime.fromtimestamp(event.timestamp_s, timezone.utc),  # type: ignore[has-type]
                    timestamp_s=event.timestamp_s,  # type: ignore[has-type]
                    frame_id=event.frame_id,  # type: ignore[has-type]
                    track_id=event.track_id,  # type: ignore[has-type]
                    event_type=event.event_type.value,  # type: ignore[has-type]
                    zone_id=event.zone_id,  # type: ignore[has-type]
                    line_id=event.line_id,  # type: ignore[has-type]
                    extra=event.extra,  # type: ignore[has-type]
                )
            )
        with self.session_factory() as session:
            session.add_all(rows)
            session.commit()

    def persist_aggregates(
        self,
        stream_id: str,
        buckets: list[dict[str, Any]],
        dwell_records: list[dict[str, Any]],
        heatmap: list[list[float]] | None = None,
    ) -> None:
        """Persist a pipeline's aggregate snapshot outside its processing thread."""
        with self.session_factory() as session:
            for bucket in buckets:
                session.add(
                    BucketStat(
                        stream_id=stream_id,
                        bucket_start=datetime.fromtimestamp(
                            float(bucket["bucket_start_s"]), timezone.utc
                        ),
                        entries=int(bucket["entries"]),
                        exits=int(bucket["exits"]),
                        unique_visitors=int(bucket["unique_visitors"]),
                        mean_occupancy=float(bucket["mean_occupancy"]),
                        peak_occupancy=int(bucket["peak_occupancy"]),
                        zone_stats=bucket,
                    )
                )
            for dwell in dwell_records:
                session.add(
                    DwellRecord(
                        stream_id=stream_id,
                        timestamp=datetime.now(timezone.utc),
                        zone_id=dwell.get("zone_id"),
                        dwell_seconds=float(dwell["dwell_seconds"]),
                    )
                )
            if heatmap:
                session.add(
                    HeatmapSnapshot(
                        stream_id=stream_id,
                        timestamp=datetime.now(timezone.utc),
                        width=len(heatmap[0]) if heatmap else 0,
                        height=len(heatmap),
                        values=heatmap,
                    )
                )
            session.commit()

    def delete_expired(self) -> int:
        """Delete expired analytics rows according to configured retention."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        with self.session_factory() as session:
            count = 0
            for model, column in (
                (EventRecord, EventRecord.timestamp),
                (BucketStat, BucketStat.bucket_start),
                (HeatmapSnapshot, HeatmapSnapshot.timestamp),
                (DwellRecord, DwellRecord.timestamp),
            ):
                result = session.execute(delete(model).where(column < cutoff))
                count += int(result.rowcount or 0)
            session.commit()
        return count

    def stop(self, timeout: float = 10.0) -> None:
        """Drain queued writes and stop the worker thread."""
        self._stopping.set()
        thread = self._thread
        if thread is None:
            return
        try:
            self._queue.put(_STOP, timeout=timeout)
        except queue.Full:
            logger.error("Persistence queue did not drain before shutdown")
        thread.join(timeout)
        if thread.is_alive():
            raise TimeoutError("Database writer did not stop before timeout")

    def metrics(self) -> dict[str, int]:
        """Return worker counters and current queue depth."""
        with self._lock:
            return {**self._metrics, "queue_depth": self._queue.qsize()}
