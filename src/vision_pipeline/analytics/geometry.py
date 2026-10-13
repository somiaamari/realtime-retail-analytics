"""Validated zone/line models and video-coordinate geometry helpers."""

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Point = tuple[float, float]
FrameSize = tuple[int, int]


class _AnalyticsModel(BaseModel):
    """Base model rejecting unknown analytics configuration fields."""

    model_config = ConfigDict(extra="forbid")


class PolygonZone(_AnalyticsModel):
    """Polygonal region of interest, optionally stored in normalized coordinates."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    points: list[Point] = Field(min_length=3)
    type: Literal["generic", "queue", "shelf", "entrance_area"] = "generic"
    color: tuple[int, int, int] = (0, 200, 255)
    normalized: bool = False
    occupancy_threshold: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_points(self) -> "PolygonZone":
        """Ensure points are valid for normalized or pixel-coordinate mode."""
        if len(set(self.points)) < 3:
            raise ValueError("A polygon must have at least three distinct points.")
        if any(not 0 <= channel <= 255 for channel in self.color):
            raise ValueError("Zone color channels must be between 0 and 255.")
        if self.normalized and any(
            coordinate < 0 or coordinate > 1
            for point in self.points
            for coordinate in point
        ):
            raise ValueError("Normalized zone coordinates must be within [0, 1].")
        return self

    def to_pixels(self, frame_size: FrameSize) -> list[tuple[int, int]]:
        """Return polygon vertices scaled to ``(width, height)`` pixels."""
        _validate_frame_size(frame_size)
        width, height = frame_size
        if not self.normalized:
            _validate_pixels([self.points], frame_size)
            return [(round(x), round(y)) for x, y in self.points]
        return [
            (round(x * (width - 1)), round(y * (height - 1))) for x, y in self.points
        ]


class CountingLine(_AnalyticsModel):
    """Finite counting segment; negative-to-positive side crossing means IN."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    start: Point
    end: Point
    normalized: bool = False

    @model_validator(mode="after")
    def validate_segment(self) -> "CountingLine":
        """Reject zero-length segments and invalid normalized coordinates."""
        if self.start == self.end:
            raise ValueError("A counting line must have distinct endpoints.")
        if self.normalized and any(
            coordinate < 0 or coordinate > 1 for coordinate in (*self.start, *self.end)
        ):
            raise ValueError("Normalized line coordinates must be within [0, 1].")
        return self

    def to_pixels(
        self, frame_size: FrameSize
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Return segment endpoints scaled to ``(width, height)`` pixels."""
        _validate_frame_size(frame_size)
        width, height = frame_size
        if not self.normalized:
            _validate_pixels([[self.start, self.end]], frame_size)
            return (
                (round(self.start[0]), round(self.start[1])),
                (round(self.end[0]), round(self.end[1])),
            )
        return (
            (round(self.start[0] * (width - 1)), round(self.start[1] * (height - 1))),
            (round(self.end[0] * (width - 1)), round(self.end[1] * (height - 1))),
        )


class AnalyticsConfig(_AnalyticsModel):
    """Validated settings and geometry for the analytics engine."""

    zones: list[PolygonZone] = Field(default_factory=list)
    lines: list[CountingLine] = Field(default_factory=list)
    dwell_min_seconds: float = Field(default=10.0, ge=0)
    bucket_seconds: float = Field(default=60.0, gt=0)
    reference_point: Literal["bottom_center", "center"] = "bottom_center"
    zone_debounce_frames: int = Field(default=1, ge=1)
    outside_debounce_frames: int = Field(default=1, ge=1)
    line_cooldown_s: float = Field(default=1.0, ge=0)
    track_end_grace_frames: int = Field(default=30, ge=0)
    heatmap_grid_scale: int = Field(default=8, ge=1)
    heatmap_sigma: float = Field(default=3.0, gt=0)
    heatmap_decay: float = Field(default=0.0, ge=0, le=1)
    heatmap_mode: Literal["cumulative", "rolling"] = "cumulative"
    heatmap_window_seconds: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "AnalyticsConfig":
        """Require IDs to be unique across zones and lines."""
        ids = [zone.id for zone in self.zones]
        ids.extend(line.id for line in self.lines)
        if len(ids) != len(set(ids)):
            raise ValueError("Zone and line IDs must be unique.")
        return self


def _validate_frame_size(frame_size: FrameSize) -> None:
    """Validate positive ``(width, height)`` dimensions."""
    if len(frame_size) != 2 or frame_size[0] <= 0 or frame_size[1] <= 0:
        raise ValueError("Frame size must contain positive width and height.")


def _validate_pixels(
    groups: list[list[Point]],
    frame_size: FrameSize,
) -> None:
    """Reject pixel coordinates outside a known frame."""
    width, height = frame_size
    if any(
        x < 0 or x > width - 1 or y < 0 or y > height - 1
        for group in groups
        for x, y in group
    ):
        raise ValueError("Pixel coordinates must lie within the frame dimensions.")


def point_in_polygon(point: Point, polygon: PolygonZone | list[Point]) -> bool:
    """Return whether a point lies inside or on the boundary of a polygon."""
    points = polygon.points if isinstance(polygon, PolygonZone) else polygon
    if len(points) < 3:
        raise ValueError("A polygon must contain at least three points.")
    contour = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0


def side_of_line(point: Point, start: Point, end: Point) -> int:
    """Return the sign of a directed line's 2D cross product for a point."""
    value = (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (
        point[0] - start[0]
    )
    return (value > 0) - (value < 0)


def segment_intersects(
    first_start: Point,
    first_end: Point,
    second_start: Point,
    second_end: Point,
) -> bool:
    """Return whether two closed line segments intersect, including endpoints."""

    def orientation(a: Point, b: Point, c: Point) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a: Point, b: Point, p: Point) -> bool:
        epsilon = 1e-9
        return (
            min(a[0], b[0]) - epsilon <= p[0] <= max(a[0], b[0]) + epsilon
            and min(a[1], b[1]) - epsilon <= p[1] <= max(a[1], b[1]) + epsilon
        )

    o1 = orientation(first_start, first_end, second_start)
    o2 = orientation(first_start, first_end, second_end)
    o3 = orientation(second_start, second_end, first_start)
    o4 = orientation(second_start, second_end, first_end)
    if ((o1 > 0 > o2) or (o1 < 0 < o2)) and ((o3 > 0 > o4) or (o3 < 0 < o4)):
        return True
    return (
        (o1 == 0 and on_segment(first_start, first_end, second_start))
        or (o2 == 0 and on_segment(first_start, first_end, second_end))
        or (o3 == 0 and on_segment(second_start, second_end, first_start))
        or (o4 == 0 and on_segment(second_start, second_end, first_end))
    )


def load_zones(path: str | Path) -> AnalyticsConfig:
    """Load and validate zones, lines, and optional analytics settings from YAML."""
    zones_path = Path(path)
    with zones_path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Zones configuration '{zones_path}' must be a YAML mapping.")
    allowed = {
        "zones",
        "lines",
        "dwell_min_seconds",
        "bucket_seconds",
        "reference_point",
        "zone_debounce_frames",
        "outside_debounce_frames",
        "line_cooldown_s",
        "track_end_grace_frames",
        "heatmap_grid_scale",
        "heatmap_sigma",
        "heatmap_decay",
        "heatmap_mode",
        "heatmap_window_seconds",
    }
    unknown = set(payload) - allowed - {"normalized"}
    if unknown:
        raise ValueError(f"Unknown zones configuration keys: {sorted(unknown)}")
    normalized_default = payload.get("normalized", False)
    if not isinstance(normalized_default, bool):
        raise ValueError("The normalized setting must be a boolean.")
    payload = dict(payload)
    payload.pop("normalized", None)
    for collection in ("zones", "lines"):
        values = payload.get(collection, [])
        if not isinstance(values, list):
            raise ValueError(f"The '{collection}' setting must be a list.")
        normalized_values: list[dict[str, object]] = []
        for value in values:
            if not isinstance(value, dict):
                raise ValueError(f"Every entry in '{collection}' must be a mapping.")
            normalized_values.append(
                {
                    **value,
                    "normalized": value.get("normalized", normalized_default),
                }
            )
        payload[collection] = normalized_values
    return AnalyticsConfig.model_validate(payload)


def validate_frame_coordinates(config: AnalyticsConfig, frame_size: FrameSize) -> None:
    """Validate all pixel-mode coordinates against the supplied video size."""
    for zone in config.zones:
        zone.to_pixels(frame_size)
    for line in config.lines:
        line.to_pixels(frame_size)
