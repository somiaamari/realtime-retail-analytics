"""FastAPI application factory and service lifespan."""

import asyncio
import logging
import threading
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.engine import Engine
from starlette.exceptions import HTTPException as StarletteHTTPException

from vision_pipeline.api.db.models import Stream
from vision_pipeline.api.db.session import create_database_engine, create_session_factory
from vision_pipeline.api.realtime.hub import WebSocketHub
from vision_pipeline.api.routers import analytics, events, streams, system, video, ws
from vision_pipeline.api.schemas.common import ErrorResponse, StreamCreate
from vision_pipeline.api.services.persistence import PersistenceWorker
from vision_pipeline.api.services.stream_manager import PipelineFactory, StreamManager
from vision_pipeline.api.settings import APISettings

logger = logging.getLogger("vision_pipeline.api")


def _migration_config(database_url: str) -> Config:
    """Create Alembic config bound to this app's database connection."""
    config = Config()
    config.set_main_option(
        "script_location",
        str(Path(__file__).parent / "db" / "alembic"),
    )
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    return config


def create_app(
    settings: APISettings | None = None,
    *,
    pipeline_factory: PipelineFactory | None = None,
) -> FastAPI:
    """Create an independently configurable FastAPI application.

    Args:
        settings: API settings, or environment-backed defaults.
        pipeline_factory: Optional pipeline builder used by tests/integrators.
    """
    api_settings = settings or APISettings()
    engine = create_database_engine(api_settings.database_url)
    session_factory = create_session_factory(engine)
    hub = WebSocketHub()
    persistence = PersistenceWorker(
        session_factory, retention_days=api_settings.retention_days
    )
    manager = StreamManager(
        config_path=api_settings.default_config_path,
        max_streams=api_settings.max_streams,
        demo_mode=api_settings.demo_mode,
        hub=hub,
        persistence=persistence,
        pipeline_factory=pipeline_factory,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Migrate schema, launch workers/streams, then flush on shutdown."""
        migration = _migration_config(api_settings.database_url)
        with engine.begin() as connection:
            migration.attributes["connection"] = connection
            command.upgrade(migration, "head")
        persistence.start()
        manager.set_loop(asyncio.get_running_loop())
        with session_factory() as session:
            stored = list(session.query(Stream).order_by(Stream.stream_id).all())
        for item in stored:
            try:
                config = StreamCreate.model_validate(item.config)
                manager.register(config)
            except Exception:
                logger.exception("Unable to load stored stream %s", item.stream_id)
        if api_settings.demo_mode and not manager.list_definitions():
            demo_stream = StreamCreate(
                stream_id="demo",
                name="Synthetic demo stream",
                source="fake",
                backend="fake",
                autostart=True,
            )
            manager.register(demo_stream)
            with session_factory() as session:
                session.add(
                    Stream(
                        stream_id=demo_stream.stream_id,
                        name=demo_stream.name,
                        source="fake",
                        config=demo_stream.model_dump(mode="json"),
                        autostart=True,
                    )
                )
                session.commit()
        for definition in manager.list_definitions():
            if definition.autostart:
                try:
                    manager.start(definition.stream_id)
                except Exception:
                    logger.exception("Unable to autostart stream %s", definition.stream_id)

        async def maintenance() -> None:
            """Run retention and aggregate flush tasks outside pipeline threads."""
            while True:
                await asyncio.sleep(3600)
                try:
                    await asyncio.to_thread(persistence.delete_expired)
                except Exception:
                    logger.exception("API retention cleanup failed")

        maintenance_task = asyncio.create_task(maintenance(), name="api-maintenance")
        try:
            yield
        finally:
            maintenance_task.cancel()
            try:
                await maintenance_task
            except asyncio.CancelledError:
                pass
            manager.shutdown()
            persistence.stop()
            engine.dispose()

    app = FastAPI(
        title="Real-Time Retail Analytics API",
        description=(
            "Manage video analytics streams and query anonymous event and "
            "aggregate data. No source frames or biometric identifiers are persisted."
        ),
        version="0.9.0",
        openapi_tags=[
            {"name": "system", "description": "Health and service metadata."},
            {"name": "streams", "description": "Stream CRUD and lifecycle controls."},
            {"name": "analytics", "description": "Historical aggregate analytics."},
            {"name": "events", "description": "Anonymous event history and export."},
            {"name": "video", "description": "Live annotated video."},
            {"name": "realtime", "description": "Live WebSocket updates."},
        ],
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=api_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*", "X-API-Key", "X-Request-ID"],
    )
    app.state.settings = api_settings
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.hub = hub
    app.state.persistence = persistence
    app.state.stream_manager = manager
    app.state.mjpeg_guard = threading.Lock()
    app.state.mjpeg_clients = 0

    @app.middleware("http")
    async def request_context(request: Request, call_next: Any) -> Any:
        """Attach/echo a request ID and write a structured access log."""
        requested = request.headers.get("X-Request-ID", "")
        request_id = requested[:128] if requested and requested.isascii() else str(uuid.uuid4())
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "HTTP request failed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                },
            )
            raise
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "HTTP access",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round((time.perf_counter() - started) * 1000, 3),
            },
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
        payload = ErrorResponse(
            code="validation_error",
            message=str(error.errors()),
            request_id=getattr(request.state, "request_id", ""),
        )
        return JSONResponse(status_code=422, content=payload.model_dump())

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, error: Exception) -> JSONResponse:
        logger.exception("Unhandled API exception")
        payload = ErrorResponse(
            code="internal_error",
            message="An unexpected server error occurred.",
            request_id=getattr(request.state, "request_id", ""),
        )
        return JSONResponse(status_code=500, content=payload.model_dump())

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, error: StarletteHTTPException) -> JSONResponse:
        detail = error.detail
        code = "not_found" if error.status_code == 404 else "http_error"
        message = str(detail)
        if isinstance(detail, dict):
            code = str(detail.get("code", code))
            message = str(detail.get("message", message))
        payload = ErrorResponse(
            code=code,
            message=message,
            request_id=getattr(request.state, "request_id", ""),
        )
        return JSONResponse(status_code=error.status_code, content=payload.model_dump())

    app.include_router(system.router, prefix="/api/v1")
    app.include_router(streams.router, prefix="/api/v1")
    app.include_router(analytics.router, prefix="/api/v1")
    app.include_router(events.router, prefix="/api/v1")
    app.include_router(video.router, prefix="/api/v1")
    app.include_router(ws.router)
    return app
