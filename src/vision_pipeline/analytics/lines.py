"""Finite-segment line crossing counters."""

from collections.abc import Sequence

from vision_pipeline.analytics.events import Event, EventType
from vision_pipeline.analytics.geometry import (
    AnalyticsConfig,
    CountingLine,
    segment_intersects,
    side_of_line,
)
from vision_pipeline.types import FrameData, Track


class LineCounter:
    """Count finite line crossings once per direction and cooldown period."""

    def __init__(
        self,
        lines: Sequence[CountingLine],
        *,
        cooldown_s: float = 1.0,
        reference_point: str = "bottom_center",
    ) -> None:
        if cooldown_s < 0:
            raise ValueError("cooldown_s must be non-negative.")
        self.lines = list(lines)
        self.cooldown_s = cooldown_s
        if reference_point not in ("bottom_center", "center"):
            raise ValueError("reference_point must be bottom_center or center.")
        self.reference_point = reference_point
        self.totals: dict[str, dict[str, int]] = {
            line.id: {"in": 0, "out": 0, "net": 0} for line in self.lines
        }
        self._last_nonzero: dict[tuple[int, str], tuple[int, tuple[float, float]]] = {}
        self._last_counted: dict[tuple[int, str, str], float] = {}

    @classmethod
    def from_config(cls, config: AnalyticsConfig) -> "LineCounter":
        """Create a line counter from validated app settings."""
        return cls(
            config.lines,
            cooldown_s=config.line_cooldown_s,
            reference_point=config.reference_point,
        )

    def update(self, tracks: Sequence[Track], frame_data: FrameData) -> list[Event]:
        """Detect side changes whose movement intersects a finite counting line."""
        events: list[Event] = []
        for track in tracks:
            track_id = track.track_id
            point = (
                track.center
                if self.reference_point == "center"
                else track.bottom_center
            )
            for line in self.lines:
                start, end = line.start, line.end
                current_side = side_of_line(point, start, end)
                key = (track_id, line.id)
                previous = self._last_nonzero.get(key)
                if current_side == 0:
                    continue
                self._last_nonzero[key] = (current_side, point)
                if previous is None:
                    continue
                previous_side, previous_point = previous
                if previous_side == current_side:
                    continue
                if not segment_intersects(previous_point, point, start, end):
                    continue
                direction = "in" if previous_side < current_side else "out"
                cooldown_key = (track_id, line.id, direction)
                last_counted = self._last_counted.get(cooldown_key)
                if (
                    last_counted is not None
                    and frame_data.timestamp_s - last_counted < self.cooldown_s
                ):
                    continue
                self._last_counted[cooldown_key] = frame_data.timestamp_s
                counts = self.totals[line.id]
                counts[direction] += 1
                counts["net"] = counts["in"] - counts["out"]
                events.append(
                    Event(
                        EventType.LINE_CROSS_IN
                        if direction == "in"
                        else EventType.LINE_CROSS_OUT,
                        frame_data.timestamp_s,
                        frame_data.frame_id,
                        track_id,
                        line_id=line.id,
                        extra={"direction": direction, "position": point},
                    )
                )
        return events

    def end_track(self, track_id: int) -> None:
        """Remove previous-position and cooldown state for a finished track."""
        for key in [key for key in self._last_nonzero if key[0] == track_id]:
            del self._last_nonzero[key]
        for cooldown_key in [key for key in self._last_counted if key[0] == track_id]:
            del self._last_counted[cooldown_key]

    def reset(self) -> None:
        """Clear all state and line totals."""
        self._last_nonzero.clear()
        self._last_counted.clear()
        for counts in self.totals.values():
            counts.update({"in": 0, "out": 0, "net": 0})
