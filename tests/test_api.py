"""Fast deterministic API and migration tests."""

import json
import time
from pathlib import Path

from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient

from vision_pipeline.api.main import create_app
from vision_pipeline.api.settings import APISettings


def _settings(database: Path, **overrides: object) -> APISettings:
    return APISettings(
        _env_file=None,
        database_url=f"sqlite:///{database.as_posix()}",
        default_config_path=Path("configs/default.yaml"),
        **overrides,
    )


def test_app_factory_demo_event_persistence_and_summary(tmp_path: Path) -> None:
    """Demo pipeline emits events, persists them, and returns analytics totals."""
    app = create_app(_settings(tmp_path / "api.db", demo_mode=True))
    with TestClient(app) as client:
        response = client.get("/api/v1/health", headers={"X-Request-ID": "test-run"})
        assert response.status_code == 200
        assert response.headers["X-Request-ID"] == "test-run"
        assert response.json()["running_streams"] == 1

        deadline = time.monotonic() + 4
        events = []
        while time.monotonic() < deadline:
            response = client.get("/api/v1/streams/demo/events")
            assert response.status_code == 200
            events = response.json()["items"]
            if events:
                break
            time.sleep(0.05)
        assert events
        assert all("image" not in json.dumps(event).lower() for event in events)

        summary = client.get("/api/v1/streams/demo/analytics/summary").json()
        assert summary["event_count"] == len(events)
        assert summary["event_counts"].get("TRACK_STARTED", 0) >= 1
        assert (
            client.get("/api/v1/streams/demo/frame.jpg").headers["content-type"]
            == "image/jpeg"
        )
        assert client.get("/api/v1/metrics").text.startswith("# HELP")


def test_stream_crud_api_key_and_websocket(tmp_path: Path) -> None:
    """Mutations require configured credentials and WebSocket sends state."""
    app = create_app(
        _settings(
            tmp_path / "auth.db",
            demo_mode=True,
            api_key="test-key",
        )
    )
    with TestClient(app) as client:
        assert (
            client.post(
                "/api/v1/streams",
                json={"stream_id": "extra", "source": "fake"},
            ).status_code
            == 401
        )
        with client.websocket_connect("/ws/streams/demo?api_key=test-key") as socket:
            state = socket.receive_json()
            assert state["type"] == "STATE"
            message = socket.receive_json()
            assert message["type"] in {"SNAPSHOT", "METRICS", "EVENT"}

        created = client.post(
            "/api/v1/streams",
            headers={"X-API-Key": "test-key"},
            json={"stream_id": "extra", "source": "fake"},
        )
        assert created.status_code == 201
        assert client.get("/api/v1/streams/extra").json()["status"] == "STOPPED"


def test_alembic_upgrade_and_downgrade(tmp_path: Path) -> None:
    """The committed migration can upgrade and downgrade a temporary database."""
    database = tmp_path / "migration.db"
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database.as_posix()}")
    command.upgrade(config, "head")
    command.downgrade(config, "base")
