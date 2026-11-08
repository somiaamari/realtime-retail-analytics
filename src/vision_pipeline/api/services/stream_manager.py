"""Thread-safe ownership and lifecycle management for streaming pipelines."""

import asyncio
import logging
import threading
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from vision_pipeline.analytics.engine import AnalyticsEngine
from vision_pipeline.analytics.geometry import AnalyticsConfig
from vision_pipeline.api.realtime.hub import WebSocketHub
from vision_pipeline.api.schemas.common import StreamCreate
from vision_pipeline.api.services.persistence import PersistenceWorker
from vision_pipeline.config import AppConfig, VideoConfig, load_config
from vision_pipeline.detection.factory import create_detector
from vision_pipeline.detection.yolo_detector import FakeDetector
from vision_pipeline.streaming.engine import StreamingConfig, StreamingPipeline
from vision_pipeline.streaming.sources import FakeSource, VideoSource, create_source
from vision_pipeline.tracking.bytetrack import ByteTracker
from vision_pipeline.types import Detection

logger = logging.getLogger(__name__)
PipelineFactory = Callable[[StreamCreate], StreamingPipeline]


class StreamManagerError(ValueError):
    """A stream lifecycle request is invalid or cannot be fulfilled."""


class StreamManager:
    """Own pipelines, bridge their event threads to asyncio, and cap resources."""

    def __init__(
        self,
        *,
        config_path: Path,
        max_streams: int,
        demo_mode: bool,
        hub: WebSocketHub,
        persistence: PersistenceWorker,
        pipeline_factory: PipelineFactory | None = None,
    ) -> None:
        self.config_path = config_path
        self.max_streams = max_streams
        self.demo_mode = demo_mode
        self.hub = hub
        self.persistence = persistence
        self.pipeline_factory = pipeline_factory
        self._lock = threading.RLock()
        self._definitions: dict[str, StreamCreate] = {}
        self._pipelines: dict[str, StreamingPipeline] = {}
        self._unsubscribers: dict[str, Callable[[], None]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Store the API loop used to safely publish from pipeline threads."""
        self._loop = loop

    def register(self, config: StreamCreate) -> None:
        """Register a new stream definition after quick source validation."""
        with self._lock:
            if config.stream_id in self._definitions:
                raise StreamManagerError(f"Stream '{config.stream_id}' already exists.")
            if len(self._definitions) >= self.max_streams:
                raise StreamManagerError("Maximum concurrent stream count reached.")
            self._validate_source(config.source)
            self._definitions[config.stream_id] = config

    def update(self, config: StreamCreate) -> None:
        """Replace an existing stopped stream definition."""
        with self._lock:
            if config.stream_id not in self._definitions:
                raise KeyError(config.stream_id)
            if config.stream_id in self._pipelines:
                raise StreamManagerError("Stop the stream before updating its config.")
            self._validate_source(config.source)
            self._definitions[config.stream_id] = config

    def _validate_source(self, source: str | int) -> None:
        """Reject obviously invalid source identifiers without opening them."""
        if isinstance(source, int):
            if source < 0:
                raise StreamManagerError("Camera index must be non-negative.")
            return
        value = source.strip()
        if not value:
            raise StreamManagerError("Source must not be empty.")
        if value == "fake" or value.isdecimal():
            return
        if value.lower().startswith(("rtsp://", "rtsps://")):
            parts = urlsplit(value)
            if not parts.hostname:
                raise StreamManagerError("RTSP source must include a host.")
            return
        if not Path(value).is_file():
            raise StreamManagerError(f"Video source file does not exist: {value}")

    def create_pipeline(self, config: StreamCreate) -> StreamingPipeline:
        """Build a pipeline with injected fake or configured runtime backends."""
        if self.pipeline_factory is not None:
            return self.pipeline_factory(config)
        if self.demo_mode or config.source == "fake":
            source: VideoSource = FakeSource(
                infinite=True, fps=8.0, width=320, height=240
            )
            zones = config.zones or [
                {
                    "id": "demo-zone",
                    "name": "Demo zone",
                    "points": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
                    "normalized": True,
                }
            ]
            analytics_config = AnalyticsConfig.model_validate(
                {"zones": zones, "lines": config.lines}
            )
            detector = FakeDetector(
                [Detection((120, 80, 190, 210), 0.95, 0, "person")]
            )
            tracker_config = None
        else:
            app_config = load_config(self.config_path)
            app_config = app_config.model_copy(
                update={
                    "video": app_config.video.model_copy(
                        update={"source": config.source}
                    ),
                    "detector": app_config.detector.model_copy(
                        update={"backend": config.backend}
                    ),
                    "analytics": AnalyticsConfig.model_validate(
                        {"zones": config.zones, "lines": config.lines}
                    ),
                }
            )
            source = create_source(app_config.video)
            detector = create_detector(app_config)
            analytics_config = app_config.analytics
            tracker_config = app_config.tracker
        tracker = ByteTracker(tracker_config)
        analytics = AnalyticsEngine(analytics_config)
        return StreamingPipeline(
            StreamingConfig(stream_id=config.stream_id, live=True),
            source,
            detector,
            tracker,
            analytics,
        )

    def start(self, stream_id: str) -> StreamingPipeline:
        """Build and start a registered pipeline."""
        with self._lock:
            if stream_id not in self._definitions:
                raise KeyError(stream_id)
            if stream_id in self._pipelines:
                raise StreamManagerError(f"Stream '{stream_id}' is already running.")
            if len(self._pipelines) >= self.max_streams:
                raise StreamManagerError("Maximum concurrent stream count reached.")
            pipeline = self.create_pipeline(self._definitions[stream_id])
            unsubscribe = pipeline.subscribe_events(
                lambda event: self._on_event(stream_id, event), maxsize=512
            )
            self._pipelines[stream_id] = pipeline
            self._unsubscribers[stream_id] = unsubscribe
            try:
                pipeline.start()
            except Exception:
                self._pipelines.pop(stream_id, None)
                self._unsubscribers.pop(stream_id, None)
                unsubscribe()
                raise
            return pipeline

    def _on_event(self, stream_id: str, event: Any) -> None:
        """Queue persistence and schedule event delivery on the API event loop."""
        self.persistence.enqueue_event(stream_id, event)
        loop = self._loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(
                self.hub.publish,
                stream_id,
                {"type": "EVENT", "data": event.to_dict()},
            )

    def stop(self, stream_id: str) -> None:
        """Stop a running pipeline and detach its event subscription."""
        with self._lock:
            pipeline = self._pipelines.pop(stream_id, None)
            unsubscribe = self._unsubscribers.pop(stream_id, None)
        if pipeline is None:
            if stream_id not in self._definitions:
                raise KeyError(stream_id)
            return
        if unsubscribe is not None:
            unsubscribe()
        pipeline.stop()
        analytics = pipeline.analytics
        if analytics.aggregator is not None:
            heatmap = None
            if analytics.heatmap is not None:
                heatmap = analytics.heatmap.grid.tolist()
            self.persistence.persist_aggregates(
                stream_id,
                analytics.aggregator.to_records(),
                analytics.aggregator.dwell_records,
                heatmap,
            )

    def restart(self, stream_id: str) -> StreamingPipeline:
        """Restart a stream from its saved definition."""
        self.stop(stream_id)
        return self.start(stream_id)

    def delete(self, stream_id: str) -> None:
        """Stop and remove a stream definition."""
        self.stop(stream_id)
        with self._lock:
            self._definitions.pop(stream_id, None)

    def get_pipeline(self, stream_id: str) -> StreamingPipeline | None:
        """Return a live pipeline if this stream is running."""
        with self._lock:
            return self._pipelines.get(stream_id)

    def get_definition(self, stream_id: str) -> StreamCreate | None:
        """Return one in-memory stream definition."""
        with self._lock:
            return self._definitions.get(stream_id)

    def list_definitions(self) -> list[StreamCreate]:
        """Return registered stream definitions."""
        with self._lock:
            return list(self._definitions.values())

    def status(self, stream_id: str) -> str:
        """Return current manager-owned lifecycle status."""
        pipeline = self.get_pipeline(stream_id)
        if pipeline is None:
            return "STOPPED"
        return str(pipeline.get_health().get("state", "RUNNING"))

    def shutdown(self) -> None:
        """Stop all streams without allowing one failure to skip others."""
        for stream_id in list(self._pipelines):
            try:
                self.stop(stream_id)
            except Exception:
                logger.exception("Failed to stop stream %s during shutdown", stream_id)
