"""Live snapshots, metrics, and events over WebSockets."""

import asyncio
import secrets
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["realtime"])


@router.websocket("/ws/streams/{stream_id}")
async def stream_websocket(websocket: WebSocket, stream_id: str) -> None:
    """Send STATE, 2 Hz snapshots, 1 Hz metrics, and immediate event messages."""
    settings = websocket.app.state.settings
    configured_key = settings.api_key
    supplied_key = websocket.query_params.get("api_key", "")
    if configured_key is not None and not secrets.compare_digest(
        configured_key, supplied_key
    ):
        await websocket.close(code=4401, reason="Invalid API key")
        return
    manager = websocket.app.state.stream_manager
    pipeline = manager.get_pipeline(stream_id)
    if pipeline is None:
        await websocket.close(code=4404, reason="Stream is not running")
        return
    try:
        client = websocket.app.state.hub.subscribe(stream_id, settings.ws_max_clients)
    except RuntimeError:
        await websocket.close(code=4429, reason="WebSocket client limit reached")
        return
    await websocket.accept()
    await websocket.send_json(
        {"type": "STATE", "data": pipeline.get_health(), "timestamp": time.time()}
    )
    snapshot_at = time.monotonic()
    metrics_at = snapshot_at
    heartbeat_at = snapshot_at
    try:
        while True:
            now = time.monotonic()
            timeout = max(
                0.0, min(snapshot_at + 0.5, metrics_at + 1.0, heartbeat_at + 20) - now
            )
            try:
                message: dict[str, Any] = await asyncio.wait_for(
                    client.queue.get(), timeout=timeout
                )
                await websocket.send_json(message)
            except asyncio.TimeoutError:
                pass
            now = time.monotonic()
            if now >= snapshot_at + 0.5:
                snapshot = pipeline.get_snapshot()
                await websocket.send_json(
                    {
                        "type": "SNAPSHOT",
                        "data": snapshot.to_dict() if snapshot else None,
                        "timestamp": time.time(),
                    }
                )
                snapshot_at = now
            if now >= metrics_at + 1.0:
                await websocket.send_json(
                    {
                        "type": "METRICS",
                        "data": pipeline.get_metrics(),
                        "timestamp": time.time(),
                    }
                )
                metrics_at = now
            if now >= heartbeat_at + 20:
                await websocket.send_json(
                    {"type": "HEARTBEAT", "timestamp": time.time()}
                )
                heartbeat_at = now
    except WebSocketDisconnect:
        pass
    finally:
        websocket.app.state.hub.unsubscribe(client)
