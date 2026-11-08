"""Environment-backed settings for the API service."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class APISettings(BaseSettings):
    """Runtime configuration loaded from ``API_`` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="API_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "sqlite:///./retail_analytics.db"
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:5173"]
    )
    api_key: str | None = None
    default_config_path: Path = Path("configs/default.yaml")
    max_mjpeg_clients: int = Field(default=5, ge=1)
    ws_max_clients: int = Field(default=50, ge=1)
    max_streams: int = Field(default=8, ge=1)
    retention_days: int = Field(default=30, ge=1)
    demo_mode: bool = False
    log_level: str = "INFO"
