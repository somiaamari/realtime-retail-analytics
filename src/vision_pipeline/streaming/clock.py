"""Clock abstractions for deterministic streaming and real-time timing."""

from __future__ import annotations

import threading
import time
from typing import Protocol


class Clock(Protocol):
    """Abstract monotonic clock used by the streaming pipeline."""

    def monotonic(self) -> float:
        """Return a monotonic timestamp in seconds."""

    def sleep(self, seconds: float) -> None:
        """Sleep for the provided duration in seconds."""


class SystemClock:
    """Use the process monotonic clock for live sources and timers."""

    def monotonic(self) -> float:
        """Return the current system monotonic timestamp."""
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        """Sleep for the requested number of seconds."""
        time.sleep(max(0.0, seconds))


class FakeClock:
    """Manual clock for deterministic tests and simulations."""

    def __init__(self, start: float = 0.0) -> None:
        self._lock = threading.Lock()
        self._now = float(start)

    def monotonic(self) -> float:
        """Return the current fake-clock time."""
        with self._lock:
            return self._now

    def sleep(self, seconds: float) -> None:
        """Advance the fake clock by the requested duration."""
        self.advance(max(0.0, float(seconds)))

    def advance(self, seconds: float) -> None:
        """Advance the fake clock by ``seconds`` without sleeping."""
        if seconds < 0:
            raise ValueError("seconds must be non-negative")
        with self._lock:
            self._now += float(seconds)

    def set_time(self, seconds: float) -> None:
        """Set the fake clock to an absolute timestamp."""
        with self._lock:
            self._now = float(seconds)
