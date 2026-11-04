"""Thread-safe bounded queue and latest-value primitives for streaming."""

from __future__ import annotations

import threading
import time as _time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

T = TypeVar("T")


class OverflowPolicy(str, Enum):
    """Queue behavior when a bounded queue is full."""

    BLOCK = "block"
    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"


class QueueClosedError(RuntimeError):
    """Raised when a queue is closed and cannot accept new items."""


class _Sentinel:
    """Internal sentinel object used to wake waiters on shutdown."""


SENTINEL = _Sentinel()


@dataclass
class QueueStats:
    """Snapshot of queue counters and occupancy."""

    put_count: int = 0
    got_count: int = 0
    dropped_count: int = 0
    high_water_mark: int = 0
    current_depth: int = 0


class BoundedQueue(Generic[T]):
    """Thread-safe bounded queue with configurable overflow semantics."""

    def __init__(
        self,
        maxsize: int,
        *,
        policy: OverflowPolicy = OverflowPolicy.DROP_OLDEST,
        clock: object | None = None,
    ) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        self.maxsize = maxsize
        self.policy = OverflowPolicy(policy)
        self.clock = clock
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._items: deque[T | _Sentinel] = deque()
        self._closed = False
        self._stats = QueueStats()

    @property
    def stats(self) -> QueueStats:
        """Return a copy of the current queue counters and occupancy."""
        with self._condition:
            return QueueStats(
                put_count=self._stats.put_count,
                got_count=self._stats.got_count,
                dropped_count=self._stats.dropped_count,
                high_water_mark=self._stats.high_water_mark,
                current_depth=len(self._items),
            )

    def put(
        self,
        item: T,
        *,
        block: bool | None = None,
        timeout: float | None = None,
    ) -> bool:
        """Enqueue an item, applying the configured overflow policy."""
        should_block = self.policy is OverflowPolicy.BLOCK if block is None else block
        with self._condition:
            if self._closed:
                raise QueueClosedError("queue is closed")
            deadline = _time.monotonic() + timeout if timeout is not None else 0.0
            while True:
                if self._closed:
                    raise QueueClosedError("queue is closed")
                if len(self._items) < self.maxsize:
                    self._items.append(item)
                    self._stats.put_count += 1
                    self._stats.high_water_mark = max(
                        self._stats.high_water_mark, len(self._items)
                    )
                    self._condition.notify_all()
                    return True
                if not should_block:
                    return self._apply_overflow(item)
                if timeout is not None:
                    remaining = deadline - _time.monotonic()
                    if remaining <= 0:
                        return False
                    self._condition.wait(timeout=remaining)
                else:
                    self._condition.wait()

    def put_nowait(self, item: T) -> None:
        """Enqueue an item without blocking."""
        self.put(item, block=False)

    def get(self, block: bool = True, timeout: float | None = None) -> T | _Sentinel:
        """Dequeue one item or wait until a new value becomes available."""
        with self._condition:
            deadline = _time.monotonic() + timeout if timeout is not None else 0.0
            while not self._items:
                if self._closed:
                    return SENTINEL
                if not block:
                    raise EmptyQueueError("queue is empty")
                if timeout is not None:
                    remaining = deadline - _time.monotonic()
                    if remaining <= 0:
                        raise EmptyQueueError("queue is empty")
                    self._condition.wait(timeout=remaining)
                else:
                    self._condition.wait()
            item = self._items.popleft()
            self._stats.got_count += 1
            self._condition.notify_all()
            return item

    def get_nowait(self) -> T | _Sentinel:
        """Return a queued item without blocking."""
        return self.get(block=False)

    def close(self) -> None:
        """Close the queue and wake any blocked consumers."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def __len__(self) -> int:
        """Return the current depth."""
        with self._condition:
            return len(self._items)

    def __bool__(self) -> bool:
        """Return whether the queue currently contains items."""
        return len(self) > 0

    def _apply_overflow(self, item: T) -> bool:
        """Drop or replace items according to the configured policy."""
        if self.policy is OverflowPolicy.BLOCK:
            raise FullQueueError("queue is full")
        if self.policy is OverflowPolicy.DROP_OLDEST:
            if self._items:
                self._items.popleft()
            self._items.append(item)
            self._stats.put_count += 1
            accepted = True
        elif self.policy is OverflowPolicy.DROP_NEWEST:
            accepted = False
        self._stats.dropped_count += 1
        self._stats.high_water_mark = max(self._stats.high_water_mark, len(self._items))
        return accepted


class EmptyQueueError(RuntimeError):
    """Raised when a non-blocking queue fetch is attempted while empty."""


class FullQueueError(RuntimeError):
    """Raised when a blocking queue cannot accept an item while full."""


class LatestValue(Generic[T]):
    """Single-slot, last-writer-wins holder with versioning and waits."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._value: T | None = None
        self._version = 0

    @property
    def version(self) -> int:
        """Return the current version counter."""
        with self._condition:
            return self._version

    def set(self, value: T) -> int:
        """Store a value and return the incremented version number."""
        with self._condition:
            self._value = value
            self._version += 1
            self._condition.notify_all()
            return self._version

    def get(self) -> tuple[int, T | None]:
        """Return the current version together with the stored value."""
        with self._condition:
            return self._version, self._value

    def wait_for_update(self, version: int, timeout: float | None = None) -> int:
        """Wait until a newer version is available than ``version``."""
        with self._condition:
            deadline = _time.monotonic() + timeout if timeout is not None else 0.0
            while self._version <= version:
                if timeout is not None:
                    remaining = deadline - _time.monotonic()
                    if remaining <= 0:
                        return self._version
                    self._condition.wait(timeout=remaining)
                else:
                    self._condition.wait()
            return self._version

    def clear(self) -> None:
        """Reset the stored value and version."""
        with self._condition:
            self._value = None
            self._version = 0
            self._condition.notify_all()


def time() -> float:
    """Compatibility helper for queue timing; not used directly."""
    return _time.monotonic()
