"""Debounced zone occupancy and dwell-time analytics."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import mean, median

import cv2
import numpy as np

from vision_pipeline.analytics.events import Event, EventType
from vision_pipeline.analytics.geometry import AnalyticsConfig, PolygonZone
from vision_pipeline.types import FrameData, Track


@dataclass
class ZoneStats:
    """Historical visits, completed dwell times, and occupancy for one zone."""

    current_occupancy: int = 0
    max_occupancy: int = 0
    total_visits: int = 0
    completed_dwell_seconds: list[float] = field(default_factory=list)

    @property
    def average_dwell_seconds(self) -> float:
        """Return the mean of completed visit durations, or zero."""
        return (
            mean(self.completed_dwell_seconds) if self.completed_dwell_seconds else 0.0
        )

    @property
    def median_dwell_seconds(self) -> float:
        """Return the median of completed visit durations, or zero."""
        return (
            median(self.completed_dwell_seconds)
            if self.completed_dwell_seconds
            else 0.0
        )


@dataclass
class _ZoneState:
    """Per-track debounce and active-visit state for one zone."""

    inside: bool = False
    inside_count: int = 0
    outside_count: int = 0
    entered_at_s: float | None = None
    last_position: tuple[float, float] | None = None
    last_frame_id: int | None = None


class ZoneAnalyzer:
    """Measure per-zone occupancy and dwell using source-video timestamps."""

    def __init__(
        self,
        zones: Sequence[PolygonZone],
        *,
        dwell_min_seconds: float = 10.0,
        inside_debounce_frames: int = 1,
        outside_debounce_frames: int = 1,
        reference_point: str = "bottom_center",
    ) -> None:
        if dwell_min_seconds < 0:
            raise ValueError("dwell_min_seconds must be non-negative.")
        if inside_debounce_frames < 1 or outside_debounce_frames < 1:
            raise ValueError("Zone debounce frame counts must be at least one.")
        self.zones = list(zones)
        self.dwell_min_seconds = dwell_min_seconds
        self.inside_debounce_frames = inside_debounce_frames
        self.outside_debounce_frames = outside_debounce_frames
        if reference_point not in ("bottom_center", "center"):
            raise ValueError("reference_point must be bottom_center or center.")
        self.reference_point = reference_point
        self.stats: dict[str, ZoneStats] = {zone.id: ZoneStats() for zone in self.zones}
        self._contours = {
            zone.id: np.asarray(zone.points, dtype=np.float32).reshape(-1, 1, 2)
            for zone in self.zones
        }
        self._states: dict[tuple[int, str], _ZoneState] = {}

    @classmethod
    def from_config(cls, config: AnalyticsConfig) -> "ZoneAnalyzer":
        """Create a zone analyzer from validated app settings."""
        return cls(
            config.zones,
            dwell_min_seconds=config.dwell_min_seconds,
            inside_debounce_frames=config.zone_debounce_frames,
            outside_debounce_frames=config.outside_debounce_frames,
            reference_point=config.reference_point,
        )

    def update(self, tracks: Sequence[Track], frame_data: FrameData) -> list[Event]:
        """Update each track-zone state and return emitted transition events."""
        events: list[Event] = []
        for track in tracks:
            point = self._reference_position(track, self.reference_point)
            for zone in self.zones:
                key = (track.track_id, zone.id)
                state = self._states.setdefault(key, _ZoneState())
                if (
                    state.last_frame_id is not None
                    and frame_data.frame_id > state.last_frame_id + 1
                ):
                    state.inside_count = 0
                    state.outside_count = 0
                state.last_frame_id = frame_data.frame_id
                state.last_position = point
                inside_now = (
                    cv2.pointPolygonTest(
                        self._contours[zone.id],
                        (float(point[0]), float(point[1])),
                        False,
                    )
                    >= 0
                )
                if inside_now:
                    state.outside_count = 0
                    if state.inside:
                        continue
                    state.inside_count += 1
                    if state.inside_count >= self.inside_debounce_frames:
                        state.inside = True
                        state.entered_at_s = frame_data.timestamp_s
                        state.inside_count = 0
                        stats = self.stats[zone.id]
                        stats.total_visits += 1
                        events.append(
                            Event(
                                EventType.ZONE_ENTER,
                                frame_data.timestamp_s,
                                frame_data.frame_id,
                                track.track_id,
                                zone_id=zone.id,
                                extra={"position": point},
                            )
                        )
                else:
                    state.inside_count = 0
                    if not state.inside:
                        continue
                    state.outside_count += 1
                    if state.outside_count >= self.outside_debounce_frames:
                        events.extend(
                            self._close_visit(
                                track.track_id,
                                zone,
                                state,
                                frame_data.timestamp_s,
                                frame_data.frame_id,
                                terminated_by_track_loss=False,
                            )
                        )
        self._refresh_occupancy()
        return events

    def end_track(
        self,
        track_id: int,
        timestamp_s: float,
        frame_id: int,
        *,
        terminated_by_track_loss: bool = True,
    ) -> list[Event]:
        """Close open visits and remove all state for an ended track."""
        events: list[Event] = []
        for zone in self.zones:
            state = self._states.get((track_id, zone.id))
            if state is not None and state.inside:
                events.extend(
                    self._close_visit(
                        track_id,
                        zone,
                        state,
                        timestamp_s,
                        frame_id,
                        terminated_by_track_loss=terminated_by_track_loss,
                    )
                )
        for key in [key for key in self._states if key[0] == track_id]:
            del self._states[key]
        self._refresh_occupancy()
        return events

    def finalize(self, timestamp_s: float, frame_id: int) -> list[Event]:
        """Close every open visit at end-of-stream."""
        track_ids = {track_id for track_id, _zone_id in self._states}
        return [
            event
            for track_id in sorted(track_ids)
            for event in self.end_track(
                track_id,
                timestamp_s,
                frame_id,
                terminated_by_track_loss=False,
            )
        ]

    def occupancy(self) -> dict[str, int]:
        """Return current unique occupants for every configured zone."""
        counts = {zone_id: 0 for zone_id in self.stats}
        for (_track_id, zone_id), state in self._states.items():
            if state.inside:
                counts[zone_id] += 1
        return counts

    def _close_visit(
        self,
        track_id: int,
        zone: PolygonZone,
        state: _ZoneState,
        timestamp_s: float,
        frame_id: int,
        *,
        terminated_by_track_loss: bool,
    ) -> list[Event]:
        """Close an active visit, updating dwell history and producing events."""
        entered_at = state.entered_at_s
        dwell_seconds = (
            max(0.0, timestamp_s - entered_at) if entered_at is not None else 0.0
        )
        position = state.last_position
        state.inside = False
        state.inside_count = 0
        state.outside_count = 0
        state.entered_at_s = None
        self.stats[zone.id].completed_dwell_seconds.append(dwell_seconds)
        extra: dict[str, object] = {
            "dwell_seconds": dwell_seconds,
            "terminated_by_track_loss": terminated_by_track_loss,
        }
        if position is not None:
            extra["position"] = position
        events = [
            Event(
                EventType.ZONE_EXIT,
                timestamp_s,
                frame_id,
                track_id,
                zone_id=zone.id,
                extra=extra,
            )
        ]
        if dwell_seconds >= self.dwell_min_seconds:
            events.append(
                Event(
                    EventType.DWELL_COMPLETE,
                    timestamp_s,
                    frame_id,
                    track_id,
                    zone_id=zone.id,
                    extra=extra,
                )
            )
        return events

    def _refresh_occupancy(self) -> None:
        """Update current and historical peak occupancy for all zones."""
        for zone_id, count in self.occupancy().items():
            stats = self.stats[zone_id]
            stats.current_occupancy = count
            stats.max_occupancy = max(stats.max_occupancy, count)

    @staticmethod
    def _reference_position(
        track: Track,
        reference_point: str,
    ) -> tuple[float, float]:
        """Select the configured track reference position."""
        if reference_point == "center":
            return track.center
        return track.bottom_center
