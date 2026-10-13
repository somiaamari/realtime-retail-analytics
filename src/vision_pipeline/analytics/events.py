"""Typed analytics events and pluggable event sinks."""

import csv
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TextIO

from vision_pipeline.logging_utils import get_logger

logger = get_logger(__name__)


class EventType(str, Enum):
    """Discrete lifecycle and crossing events emitted by analytics."""

    ZONE_ENTER = "ZONE_ENTER"
    ZONE_EXIT = "ZONE_EXIT"
    LINE_CROSS_IN = "LINE_CROSS_IN"
    LINE_CROSS_OUT = "LINE_CROSS_OUT"
    DWELL_COMPLETE = "DWELL_COMPLETE"
    TRACK_STARTED = "TRACK_STARTED"
    TRACK_ENDED = "TRACK_ENDED"


@dataclass(frozen=True)
class Event:
    """A deterministic analytics event identified by source time and frame."""

    event_type: EventType
    timestamp_s: float
    frame_id: int
    track_id: int
    zone_id: str | None = None
    line_id: str | None = None
    extra: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialize the event to JSON-compatible scalar and mapping values."""
        return {
            "event_type": self.event_type.value,
            "timestamp_s": self.timestamp_s,
            "frame_id": self.frame_id,
            "track_id": self.track_id,
            "zone_id": self.zone_id,
            "line_id": self.line_id,
            "extra": self.extra,
        }


class EventSink(ABC):
    """Interface implemented by in-memory and durable event destinations."""

    @abstractmethod
    def write(self, event: Event) -> None:
        """Write one event."""

    @abstractmethod
    def flush(self) -> None:
        """Flush buffered event data to its destination."""

    def close(self) -> None:
        """Flush and close any resources owned by this sink."""
        self.flush()


class InMemorySink(EventSink):
    """Collect events in order for tests or short-lived consumers."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def write(self, event: Event) -> None:
        """Append an event to the in-memory collection."""
        self.events.append(event)

    def flush(self) -> None:
        """No-op because events are already available in memory."""


class _FileSink(EventSink):
    """Shared owned-file behavior for line-oriented event sinks."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream: TextIO = self.path.open("w", encoding="utf-8", newline="")

    def flush(self) -> None:
        """Flush buffered text to disk."""
        self._stream.flush()

    def close(self) -> None:
        """Flush and close the output file."""
        if not self._stream.closed:
            self.flush()
            self._stream.close()


class JsonlSink(_FileSink):
    """Append each event as one JSON object per output line."""

    def write(self, event: Event) -> None:
        """Write one compact JSON event record."""
        if self._stream.closed:
            raise ValueError(f"Event sink is closed: {self.path}")
        self._stream.write(json.dumps(event.to_dict(), separators=(",", ":")) + "\n")


class CsvSink(_FileSink):
    """Write events as tabular CSV with JSON-encoded extra data."""

    _fields = (
        "event_type",
        "timestamp_s",
        "frame_id",
        "track_id",
        "zone_id",
        "line_id",
        "extra",
    )

    def __init__(self, path: str | Path) -> None:
        super().__init__(path)
        self._writer = csv.DictWriter(self._stream, fieldnames=self._fields)
        self._writer.writeheader()

    def write(self, event: Event) -> None:
        """Write one CSV event row."""
        if self._stream.closed:
            raise ValueError(f"Event sink is closed: {self.path}")
        record = event.to_dict()
        record["extra"] = json.dumps(event.extra, separators=(",", ":"))
        self._writer.writerow(record)


class EventLogger:
    """Fan out events to configured sinks and expose explicit flush/close."""

    def __init__(self, sinks: list[EventSink] | None = None) -> None:
        self.sinks = list(sinks or [])

    def emit(self, event: Event) -> None:
        """Write an event to each sink and log the event type at debug level."""
        for sink in self.sinks:
            sink.write(event)
        logger.debug(
            "Analytics event %s for track %d at %.3fs",
            event.event_type.value,
            event.track_id,
            event.timestamp_s,
        )

    def flush(self) -> None:
        """Flush every configured sink."""
        for sink in self.sinks:
            sink.flush()

    def close(self) -> None:
        """Close every configured sink."""
        for sink in self.sinks:
            sink.close()
