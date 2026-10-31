"""Single-entry analytics facade for frame-by-frame tracked video."""

from dataclasses import dataclass
from typing import Any

from vision_pipeline.analytics.aggregator import AnalyticsSummary, TimeSeriesAggregator
from vision_pipeline.analytics.events import (
    Event,
    EventLogger,
    EventSink,
    EventType,
    InMemorySink,
)
from vision_pipeline.analytics.geometry import AnalyticsConfig
from vision_pipeline.analytics.heatmap import HeatmapAccumulator
from vision_pipeline.analytics.lines import LineCounter
from vision_pipeline.analytics.zones import ZoneAnalyzer
from vision_pipeline.types import FrameData, Track
from vision_pipeline.utils.profiling import LatencyStats, StageTimer


@dataclass(frozen=True)
class AnalyticsSnapshot:
    """Read-only live counts for a processed frame."""

    timestamp_s: float
    frame_id: int
    occupancy: dict[str, int]
    line_counts: dict[str, dict[str, int]]
    active_tracks: tuple[int, ...]
    unique_visitors: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of live state."""
        return {
            "timestamp_s": self.timestamp_s,
            "frame_id": self.frame_id,
            "occupancy": dict(self.occupancy),
            "line_counts": {
                key: dict(value) for key, value in self.line_counts.items()
            },
            "active_tracks": list(self.active_tracks),
            "unique_visitors": self.unique_visitors,
        }


class AnalyticsEngine:
    """Coordinate zone, line, heatmap, event, and bucket analytics."""

    def __init__(
        self,
        config: AnalyticsConfig | None = None,
        sinks: list[EventSink] | None = None,
    ) -> None:
        self.config = config or AnalyticsConfig()
        self._memory_sink = InMemorySink() if sinks is None else None
        configured_sinks: list[EventSink] = (
            [self._memory_sink] if self._memory_sink is not None else list(sinks or [])
        )
        self.event_logger = EventLogger(configured_sinks)
        self.latency_stats = LatencyStats()
        self.zones = ZoneAnalyzer.from_config(self.config)
        self.lines = LineCounter.from_config(self.config)
        self.aggregator = TimeSeriesAggregator(self.config.bucket_seconds)
        self.heatmap: HeatmapAccumulator | None = None
        self._active_last_seen: dict[int, tuple[int, float]] = {}
        self._unique_tracks: set[int] = set()
        self._last_frame_id: int | None = None
        self._last_timestamp_s = 0.0
        self._finalized = False
        self._geometry_initialized = False

    @property
    def events(self) -> list[Event]:
        """Return retained events when using the default in-memory sink."""
        return self._memory_sink.events if self._memory_sink is not None else []

    def update(self, tracks: list[Track], frame_data: FrameData) -> list[Event]:
        """Analyze one frame and return newly emitted events."""
        if self._finalized:
            raise RuntimeError("Analytics engine has been finalized; call reset first.")
        if (
            self._last_frame_id is not None
            and frame_data.frame_id <= self._last_frame_id
        ):
            raise ValueError("Analytics frame IDs must increase.")
        if frame_data.timestamp_s < self._last_timestamp_s:
            raise ValueError("Analytics timestamps must be non-decreasing.")
        self._last_frame_id = frame_data.frame_id
        self._last_timestamp_s = frame_data.timestamp_s
        self._initialize_geometry(frame_data)
        if self.heatmap is None:
            height, width = frame_data.frame.shape[:2]
            self.heatmap = HeatmapAccumulator(
                (width, height),
                self.config.heatmap_grid_scale,
                self.config.heatmap_sigma,
                self.config.heatmap_decay,
                mode=self.config.heatmap_mode,
                window_seconds=self.config.heatmap_window_seconds,
            )
        emitted: list[Event] = []
        seen_now: set[int] = set()
        with StageTimer("analytics", self.latency_stats):
            for track in tracks:
                track_id = track.track_id
                seen_now.add(track_id)
                self._unique_tracks.add(track_id)
                if track_id not in self._active_last_seen:
                    emitted.append(
                        Event(
                            EventType.TRACK_STARTED,
                            frame_data.timestamp_s,
                            frame_data.frame_id,
                            track_id,
                            extra={"position": self._position(track)},
                        )
                    )
                self._active_last_seen[track_id] = (
                    frame_data.frame_id,
                    frame_data.timestamp_s,
                )

            emitted.extend(self.zones.update(tracks, frame_data))
            emitted.extend(self.lines.update(tracks, frame_data))
            for track_id, (last_frame, _last_timestamp) in tuple(
                self._active_last_seen.items()
            ):
                if (
                    track_id not in seen_now
                    and frame_data.frame_id - last_frame
                    > self.config.track_end_grace_frames
                ):
                    emitted.extend(
                        self.zones.end_track(
                            track_id,
                            frame_data.timestamp_s,
                            frame_data.frame_id,
                        )
                    )
                    self.lines.end_track(track_id)
                    emitted.append(
                        Event(
                            EventType.TRACK_ENDED,
                            frame_data.timestamp_s,
                            frame_data.frame_id,
                            track_id,
                            extra={"terminated_by_track_loss": True},
                        )
                    )
                    del self._active_last_seen[track_id]

            self.heatmap.add(
                [self._position(track) for track in tracks],
                timestamp_s=frame_data.timestamp_s,
            )
            occupancy = self.zones.occupancy()
            self.aggregator.add_frame(
                frame_data.timestamp_s,
                occupancy,
                seen_now,
            )
            emitted.sort(key=lambda event: (event.timestamp_s, event.frame_id))
            for event in emitted:
                self.event_logger.emit(event)
                self.aggregator.add_event(event)
        return emitted

    def snapshot(self) -> AnalyticsSnapshot:
        """Return a point-in-time immutable snapshot of the live analytics state."""
        return AnalyticsSnapshot(
            timestamp_s=self._last_timestamp_s,
            frame_id=self._last_frame_id if self._last_frame_id is not None else -1,
            occupancy=self.zones.occupancy(),
            line_counts={key: dict(value) for key, value in self.lines.totals.items()},
            active_tracks=tuple(sorted(self._active_last_seen)),
            unique_visitors=len(self._unique_tracks),
        )

    def finalize(self) -> AnalyticsSummary:
        """Close active visits/tracks, flush sinks, and return summary statistics."""
        if not self._finalized:
            frame_id = self._last_frame_id if self._last_frame_id is not None else -1
            timestamp_s = self._last_timestamp_s
            for track_id in sorted(self._active_last_seen):
                for event in self.zones.end_track(
                    track_id,
                    timestamp_s,
                    frame_id,
                    terminated_by_track_loss=False,
                ):
                    self.event_logger.emit(event)
                    self.aggregator.add_event(event)
                self.lines.end_track(track_id)
                event = Event(
                    EventType.TRACK_ENDED,
                    timestamp_s,
                    frame_id,
                    track_id,
                    extra={"terminated_by_track_loss": False},
                )
                self.event_logger.emit(event)
                self.aggregator.add_event(event)
            self._active_last_seen.clear()
            self.event_logger.flush()
            self._finalized = True
        return self.aggregator.summary(self.zones.stats)

    def reset(self) -> None:
        """Clear all accumulated analytics and reopen event sinks if supported."""
        self.zones = ZoneAnalyzer.from_config(self.config)
        self.lines.reset()
        self.aggregator.reset()
        self.heatmap.reset() if self.heatmap is not None else None
        self.heatmap = None
        self._active_last_seen.clear()
        self._unique_tracks.clear()
        self._last_frame_id = None
        self._last_timestamp_s = 0.0
        self._finalized = False
        self._geometry_initialized = False
        if self._memory_sink is not None:
            self._memory_sink.events.clear()
        self.latency_stats = LatencyStats()

    def close(self) -> None:
        """Finalize analytics and close all configured sinks."""
        self.finalize()
        self.event_logger.close()

    def _position(self, track: Track) -> tuple[float, float]:
        """Return the configured reference point for a track."""
        if self.config.reference_point == "center":
            return track.center
        return track.bottom_center

    def _initialize_geometry(self, frame_data: FrameData) -> None:
        """Scale normalized geometry and validate pixel geometry for the source."""
        if self._geometry_initialized:
            return
        height, width = frame_data.frame.shape[:2]
        frame_size = (width, height)
        zones = [
            zone.model_copy(
                update={"points": zone.to_pixels(frame_size), "normalized": False}
            )
            for zone in self.config.zones
        ]
        lines = []
        for line in self.config.lines:
            start, end = line.to_pixels(frame_size)
            lines.append(
                line.model_copy(
                    update={
                        "start": start,
                        "end": end,
                        "normalized": False,
                    }
                )
            )
        self.zones = ZoneAnalyzer(
            zones,
            dwell_min_seconds=self.config.dwell_min_seconds,
            inside_debounce_frames=self.config.zone_debounce_frames,
            outside_debounce_frames=self.config.outside_debounce_frames,
            reference_point=self.config.reference_point,
        )
        self.lines = LineCounter(
            lines,
            cooldown_s=self.config.line_cooldown_s,
            reference_point=self.config.reference_point,
        )
        self._geometry_initialized = True
