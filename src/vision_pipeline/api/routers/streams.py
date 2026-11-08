"""Stream configuration and lifecycle endpoints."""

from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from vision_pipeline.api.db.models import LineRecord, Stream, ZoneRecord
from vision_pipeline.api.deps import get_session, require_api_key
from vision_pipeline.api.schemas.common import GeometryUpdate, StreamCreate
from vision_pipeline.api.services.stream_manager import StreamManagerError

router = APIRouter(prefix="/streams", tags=["streams"])
mutating = [Depends(require_api_key)]


def _serialize(request: Request, config: StreamCreate) -> dict[str, Any]:
    manager = request.app.state.stream_manager
    pipeline = manager.get_pipeline(config.stream_id)
    return {
        "stream_id": config.stream_id,
        "name": config.name,
        "source": str(config.source),
        "autostart": config.autostart,
        "status": manager.status(config.stream_id),
        "health": pipeline.get_health() if pipeline else None,
        "metrics": pipeline.get_metrics() if pipeline else None,
    }


def _get_config(request: Request, stream_id: str) -> StreamCreate:
    definition = request.app.state.stream_manager.get_definition(stream_id)
    if definition is None:
        raise HTTPException(status_code=404, detail="Stream not found.")
    return cast(StreamCreate, definition)


@router.get("")
def list_streams(request: Request) -> list[dict[str, Any]]:
    """List all in-memory stream definitions and their current status."""
    return [
        _serialize(request, config)
        for config in request.app.state.stream_manager.list_definitions()
    ]


@router.post("", status_code=status.HTTP_201_CREATED, dependencies=mutating)
def create_stream(
    payload: StreamCreate,
    request: Request,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Create and persist a stopped stream configuration."""
    manager = request.app.state.stream_manager
    try:
        manager.register(payload)
    except StreamManagerError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    record = Stream(
        stream_id=payload.stream_id,
        name=payload.name,
        source=str(payload.source),
        config=payload.model_dump(mode="json"),
        autostart=payload.autostart,
    )
    session.add(record)
    session.commit()
    if payload.autostart:
        try:
            manager.start(payload.stream_id)
        except Exception as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
    return _serialize(request, payload)


@router.get("/{stream_id}")
def get_stream(stream_id: str, request: Request) -> dict[str, Any]:
    """Return a stream's status, health, and current metrics."""
    return _serialize(request, _get_config(request, stream_id))


@router.put("/{stream_id}", dependencies=mutating)
def update_stream(
    stream_id: str,
    payload: StreamCreate,
    request: Request,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Update an existing stopped stream configuration."""
    if stream_id != payload.stream_id:
        raise HTTPException(status_code=422, detail="Path and payload IDs must match.")
    manager = request.app.state.stream_manager
    try:
        manager.update(payload)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Stream not found.") from error
    except StreamManagerError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    session.merge(
        Stream(
            stream_id=payload.stream_id,
            name=payload.name,
            source=str(payload.source),
            config=payload.model_dump(mode="json"),
            autostart=payload.autostart,
        )
    )
    session.commit()
    return _serialize(request, payload)


@router.delete(
    "/{stream_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=mutating
)
def delete_stream(
    stream_id: str,
    request: Request,
    session: Session = Depends(get_session),
) -> None:
    """Stop and delete a stream configuration."""
    manager = request.app.state.stream_manager
    if manager.get_definition(stream_id) is None:
        raise HTTPException(status_code=404, detail="Stream not found.")
    manager.delete(stream_id)
    session.execute(delete(Stream).where(Stream.stream_id == stream_id))
    session.commit()


@router.post("/{stream_id}/start", dependencies=mutating)
def start_stream(stream_id: str, request: Request) -> dict[str, Any]:
    """Start a configured stream."""
    try:
        request.app.state.stream_manager.start(stream_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Stream not found.") from error
    except StreamManagerError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return _serialize(request, _get_config(request, stream_id))


@router.post("/{stream_id}/stop", dependencies=mutating)
def stop_stream(stream_id: str, request: Request) -> dict[str, str]:
    """Stop a configured stream and persist its aggregate snapshot."""
    try:
        request.app.state.stream_manager.stop(stream_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Stream not found.") from error
    return {"stream_id": stream_id, "status": "STOPPED"}


@router.post("/{stream_id}/restart", dependencies=mutating)
def restart_stream(stream_id: str, request: Request) -> dict[str, Any]:
    """Restart a configured stream."""
    try:
        request.app.state.stream_manager.restart(stream_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Stream not found.") from error
    except StreamManagerError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return _serialize(request, _get_config(request, stream_id))


@router.get("/{stream_id}/metrics")
def stream_metrics(stream_id: str, request: Request) -> dict[str, Any]:
    """Return current stage and queue metrics."""
    pipeline = request.app.state.stream_manager.get_pipeline(stream_id)
    if pipeline is None:
        _get_config(request, stream_id)
        return {"stream_id": stream_id, "status": "STOPPED"}
    return cast(dict[str, Any], pipeline.get_metrics())


@router.get("/{stream_id}/health")
def stream_health(stream_id: str, request: Request) -> dict[str, Any]:
    """Return pipeline liveness and per-stage health."""
    pipeline = request.app.state.stream_manager.get_pipeline(stream_id)
    if pipeline is None:
        _get_config(request, stream_id)
        return {"stream_id": stream_id, "state": "STOPPED"}
    return cast(dict[str, Any], pipeline.get_health())


@router.get("/{stream_id}/snapshot")
def stream_snapshot(stream_id: str, request: Request) -> dict[str, Any]:
    """Return the latest live analytics snapshot."""
    pipeline = request.app.state.stream_manager.get_pipeline(stream_id)
    if pipeline is None:
        _get_config(request, stream_id)
        return {"stream_id": stream_id, "snapshot": None}
    snapshot = pipeline.get_snapshot()
    return {
        "stream_id": stream_id,
        "snapshot": snapshot.to_dict() if snapshot else None,
    }


@router.get("/{stream_id}/zones")
def get_geometry(
    stream_id: str,
    request: Request,
    session: Session = Depends(get_session),
) -> dict[str, list[dict[str, Any]]]:
    """Read persisted zone and counting-line geometry."""
    _get_config(request, stream_id)
    zones = session.scalars(
        select(ZoneRecord).where(ZoneRecord.stream_id == stream_id)
    ).all()
    lines = session.scalars(
        select(LineRecord).where(LineRecord.stream_id == stream_id)
    ).all()
    return {
        "zones": [row.geometry for row in zones],
        "lines": [row.geometry for row in lines],
    }


@router.put("/{stream_id}/zones", dependencies=mutating)
def put_geometry(
    stream_id: str,
    payload: GeometryUpdate,
    request: Request,
    session: Session = Depends(get_session),
) -> dict[str, list[dict[str, Any]]]:
    """Validate, persist, and apply geometry to a stopped stream."""
    config = _get_config(request, stream_id)
    updated = config.model_copy(update={"zones": payload.zones, "lines": payload.lines})
    manager = request.app.state.stream_manager
    try:
        manager.update(updated)
    except StreamManagerError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    session.execute(delete(ZoneRecord).where(ZoneRecord.stream_id == stream_id))
    session.execute(delete(LineRecord).where(LineRecord.stream_id == stream_id))
    for zone in payload.zones:
        session.add(
            ZoneRecord(
                stream_id=stream_id,
                zone_id=str(zone["id"]),
                name=str(zone.get("name", zone["id"])),
                geometry=zone,
            )
        )
    for line in payload.lines:
        session.add(
            LineRecord(
                stream_id=stream_id,
                line_id=str(line["id"]),
                name=str(line.get("name", line["id"])),
                geometry=line,
            )
        )
    record = session.get(Stream, stream_id)
    if record is not None:
        record.config = updated.model_dump(mode="json")
    session.commit()
    return {"zones": payload.zones, "lines": payload.lines}
