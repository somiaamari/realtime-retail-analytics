"""Shared API response schemas."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class APIModel(BaseModel):
    """Base schema that rejects unexpected request fields."""

    model_config = ConfigDict(extra="forbid")


class ErrorResponse(BaseModel):
    """Consistent API error payload."""

    code: str
    message: str
    request_id: str


class StreamCreate(APIModel):
    """Stream configuration accepted by the management endpoints."""

    stream_id: str = Field(min_length=1, max_length=128, pattern=r"^[\w.-]+$")
    name: str = Field(default="Retail stream", min_length=1, max_length=200)
    source: str | int = "fake"
    backend: str = "pytorch"
    zones: list[dict[str, Any]] = Field(default_factory=list)
    lines: list[dict[str, Any]] = Field(default_factory=list)
    detect_every_n: int = Field(default=1, ge=1)
    autostart: bool = False


class StreamResponse(BaseModel):
    """Stream status and configuration summary."""

    stream_id: str
    name: str
    source: str
    autostart: bool
    status: str
    health: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None


class GeometryUpdate(APIModel):
    """Zone and line geometry update."""

    zones: list[dict[str, Any]] = Field(default_factory=list)
    lines: list[dict[str, Any]] = Field(default_factory=list)
