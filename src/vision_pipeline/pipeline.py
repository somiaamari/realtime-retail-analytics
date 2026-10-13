"""Dependency-injected detection, tracking, analytics, and rendering pipeline."""

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import islice
from typing import Any

from numpy.typing import NDArray

from vision_pipeline.analytics.engine import AnalyticsEngine, AnalyticsSnapshot
from vision_pipeline.analytics.events import Event
from vision_pipeline.detection.yolo_detector import BaseDetector
from vision_pipeline.streaming.sources import VideoSource
from vision_pipeline.streaming.writer import VideoWriter
from vision_pipeline.tracking.base import BaseTracker
from vision_pipeline.types import Detection, FrameData, Track
from vision_pipeline.utils.profiling import FPSMeter, LatencyStats, StageTimer
from vision_pipeline.utils.visualization import (
    draw_lines,
    draw_overlay_stats,
    draw_tracks,
    draw_zones,
)


@dataclass(frozen=True)
class AnalyticsFrame:
    """All outputs produced for one source frame."""

    frame_data: FrameData
    detections: list[Detection]
    tracks: list[Track]
    events: list[Event]
    snapshot: AnalyticsSnapshot
    annotated: NDArray[Any]


class AnalyticsPipeline:
    """Compose source, detector, tracker, analytics, and optional output writer."""

    def __init__(
        self,
        source: VideoSource,
        detector: BaseDetector,
        tracker: BaseTracker,
        analytics: AnalyticsEngine,
        *,
        writer: VideoWriter | None = None,
        max_frames: int | None = None,
    ) -> None:
        if max_frames is not None and max_frames < 1:
            raise ValueError("max_frames must be positive when provided.")
        self.source = source
        self.detector = detector
        self.tracker = tracker
        self.analytics = analytics
        self.writer = writer
        self.max_frames = max_frames
        self.latency_stats = LatencyStats()
        self._fps_meter = FPSMeter()

    def run(self) -> Iterator[AnalyticsFrame]:
        """Yield tracked, analyzed, and annotated results for each source frame."""
        frames = (
            self.source
            if self.max_frames is None
            else islice(self.source, self.max_frames)
        )
        for frame_data in frames:
            detections = self.detector.detect(frame_data.frame)
            tracks = self.tracker.update(detections, frame_data.frame_id)
            timer = StageTimer("analytics", self.latency_stats)
            with timer:
                events = self.analytics.update(tracks, frame_data)
            snapshot = self.analytics.snapshot()
            annotated = draw_zones(
                frame_data.frame,
                self.analytics.config.zones,
                snapshot.occupancy,
            )
            annotated = draw_tracks(annotated, tracks)
            annotated = draw_lines(
                annotated,
                self.analytics.config.lines,
                snapshot.line_counts,
            )
            totals = {
                "in": sum(
                    value.get("in", 0) for value in snapshot.line_counts.values()
                ),
                "out": sum(
                    value.get("out", 0) for value in snapshot.line_counts.values()
                ),
            }
            totals["net"] = totals["in"] - totals["out"]
            busiest_zone = (
                max(
                    snapshot.occupancy,
                    key=lambda zone_id: snapshot.occupancy[zone_id],
                )
                if snapshot.occupancy
                else None
            )
            annotated = draw_overlay_stats(
                annotated,
                self._fps_meter.tick(),
                timer.duration_ms,
                frame_data.frame_id,
                total_unique=snapshot.unique_visitors,
                line_totals=totals,
                busiest_zone=busiest_zone,
            )
            if self.writer is not None:
                self.writer.write(annotated)
            yield AnalyticsFrame(
                frame_data=frame_data,
                detections=detections,
                tracks=tracks,
                events=events,
                snapshot=snapshot,
                annotated=annotated,
            )

    def close(self) -> None:
        """Finalize analytics and release an optional video writer."""
        try:
            self.analytics.finalize()
        finally:
            if self.writer is not None:
                self.writer.release()
