"""System health, readiness, information, and metrics endpoints."""

import json
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse

router = APIRouter(tags=["system"])


@router.get("/health", summary="Service liveness")
def health(request: Request) -> dict[str, Any]:
    """Report process health without requiring database access."""
    manager = request.app.state.stream_manager
    return {
        "status": "ok",
        "streams": len(manager.list_definitions()),
        "running_streams": sum(
            manager.get_pipeline(item.stream_id) is not None
            for item in manager.list_definitions()
        ),
    }


@router.get("/ready", summary="Service readiness")
def ready(request: Request) -> dict[str, str]:
    """Check that the configured database is reachable."""
    from sqlalchemy import text

    with request.app.state.session_factory() as session:
        session.execute(text("SELECT 1"))
    return {"status": "ready"}


@router.get("/info", summary="API and service information")
def info(request: Request) -> dict[str, Any]:
    """Return service capabilities and safe runtime configuration."""
    return {
        "name": "vision-pipeline-api",
        "version": request.app.version,
        "api_prefix": "/api/v1",
        "demo_mode": request.app.state.settings.demo_mode,
    }


@router.get("/metrics", response_class=PlainTextResponse, summary="Prometheus metrics")
def metrics(request: Request) -> str:
    """Export a compact Prometheus text-format metrics snapshot."""
    manager = request.app.state.stream_manager
    persistence = request.app.state.persistence
    lines = [
        "# HELP api_streams_total Configured streams",
        "# TYPE api_streams_total gauge",
        f"api_streams_total {len(manager.list_definitions())}",
        "# HELP api_websocket_clients Active WebSocket clients",
        "# TYPE api_websocket_clients gauge",
        f"api_websocket_clients {request.app.state.hub.client_count}",
    ]
    for key, value in persistence.metrics().items():
        lines.extend(
            [
                f"# HELP api_persistence_{key} Persistence worker {key}",
                f"# TYPE api_persistence_{key} counter",
                f"api_persistence_{key} {value}",
            ]
        )
    return "\n".join(lines) + "\n"
