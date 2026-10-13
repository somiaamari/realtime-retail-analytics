"""OpenCV visualization helpers for detections and per-frame HUD statistics."""

from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from vision_pipeline.analytics.geometry import CountingLine, PolygonZone
from vision_pipeline.types import Detection, Track


def _class_color(class_id: int) -> tuple[int, int, int]:
    """Return a stable, vivid BGR color for a class."""
    return (
        64 + (class_id * 67) % 192,
        64 + (class_id * 131) % 192,
        64 + (class_id * 193) % 192,
    )


def draw_detections(
    frame: NDArray[Any],
    detections: Sequence[Detection],
    *,
    color: tuple[int, int, int] | None = None,
    line_thickness: int = 2,
) -> NDArray[Any]:
    """Draw bounding boxes and confidence labels on a copied BGR frame.

    Args:
        frame: Input image in BGR order.
        detections: Detections to annotate.
        color: Optional fixed BGR color; otherwise each class has a stable color.
        line_thickness: Bounding-box line width in pixels.

    Returns:
        An annotated image with the same dimensions as ``frame``.
    """
    annotated = np.array(frame, copy=True)
    scale = max(0.4, min(frame.shape[1], frame.shape[0]) / 900.0)
    thickness = max(1, line_thickness)
    font_thickness = max(1, round(thickness * 0.65))
    for detection in detections:
        box_color = color or _class_color(detection.class_id)
        x1, y1, x2, y2 = (round(value) for value in detection.bbox_xyxy)
        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            box_color,
            thickness,
            lineType=cv2.LINE_AA,
        )
        label = f"{detection.class_name} {detection.confidence:.2f}"
        (label_width, label_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            font_thickness,
        )
        label_top = max(0, y1 - label_height - baseline - 6)
        cv2.rectangle(
            annotated,
            (x1, label_top),
            (x1 + label_width + 8, label_top + label_height + baseline + 6),
            box_color,
            cv2.FILLED,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 4, label_top + label_height + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    return annotated


def _track_color(track_id: int) -> tuple[int, int, int]:
    """Return a stable BGR color derived from a track ID."""
    value = (track_id * 0x9E3779B1) & 0xFFFFFFFF
    return (
        64 + (value & 0xFF) % 192,
        64 + ((value >> 8) & 0xFF) % 192,
        64 + ((value >> 16) & 0xFF) % 192,
    )


def draw_tracks(
    frame: NDArray[Any],
    tracks: Sequence[Track],
    *,
    show_trails: bool = True,
    line_thickness: int = 2,
) -> NDArray[Any]:
    """Draw persistent IDs, boxes, and optional bottom-center trails.

    Args:
        frame: Input image in BGR order.
        tracks: Tracks to annotate.
        show_trails: Whether to draw each track's bounded history.
        line_thickness: Bounding-box and trail width in pixels.

    Returns:
        An annotated image with the same dimensions as ``frame``.
    """
    annotated = np.array(frame, copy=True)
    scale = max(0.4, min(frame.shape[1], frame.shape[0]) / 900.0)
    thickness = max(1, line_thickness)
    for track in tracks:
        color = _track_color(track.track_id)
        x1, y1, x2, y2 = (round(value) for value in track.bbox_xyxy)
        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
        if show_trails and track.history is not None and len(track.history) > 1:
            points = np.asarray(
                [(round(x), round(y)) for x, y in track.history],
                dtype=np.int32,
            ).reshape(-1, 1, 2)
            cv2.polylines(
                annotated,
                [points],
                False,
                color,
                thickness,
                lineType=cv2.LINE_AA,
            )
        label = f"ID {track.track_id}"
        (label_width, label_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            max(1, round(thickness * 0.65)),
        )
        label_top = max(0, y1 - label_height - baseline - 6)
        cv2.rectangle(
            annotated,
            (x1, label_top),
            (x1 + label_width + 8, label_top + label_height + baseline + 6),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 4, label_top + label_height + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            max(1, round(thickness * 0.65)),
            lineType=cv2.LINE_AA,
        )
    return annotated


def draw_zones(
    frame: NDArray[Any],
    zones: Sequence[PolygonZone],
    occupancy: Mapping[str, int],
    *,
    thresholds: Mapping[str, int] | None = None,
    alpha: float = 0.22,
) -> NDArray[Any]:
    """Draw translucent zone boundaries and current occupancy labels.

    Args:
        frame: Input image in BGR order.
        zones: Zones to draw; normalized coordinates are scaled to this frame.
        occupancy: Current occupants indexed by zone ID.
        thresholds: Optional per-zone occupancy limits; over-limit zones turn red.
        alpha: Polygon fill opacity between zero and one.

    Returns:
        An annotated image with the same dimensions as ``frame``.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Zone fill alpha must be between zero and one.")
    annotated = np.array(frame, copy=True)
    height, width = frame.shape[:2]
    scale = max(0.4, min(width, height) / 900.0)
    for zone in zones:
        points = np.asarray(zone.to_pixels((width, height)), dtype=np.int32).reshape(
            -1, 1, 2
        )
        count = occupancy.get(zone.id, 0)
        color = tuple(zone.color)
        if thresholds is not None and zone.id in thresholds:
            if count > thresholds[zone.id]:
                color = (0, 0, 255)
        overlay = np.array(annotated, copy=True)
        cv2.fillPoly(overlay, [points], color, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, annotated, 1.0 - alpha, 0, annotated)
        cv2.polylines(
            annotated,
            [points],
            True,
            color,
            max(1, round(scale * 2)),
            lineType=cv2.LINE_AA,
        )
        anchor = tuple(int(value) for value in points[0, 0])
        cv2.putText(
            annotated,
            f"{zone.name}: {count}",
            (anchor[0], max(18, anchor[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            max(1, round(scale * 1.5)),
            lineType=cv2.LINE_AA,
        )
    return annotated


def draw_lines(
    frame: NDArray[Any],
    lines: Sequence[CountingLine],
    counts: Mapping[str, Mapping[str, int]],
) -> NDArray[Any]:
    """Draw directed counting lines and their live IN/OUT totals."""
    annotated = np.array(frame, copy=True)
    height, width = frame.shape[:2]
    scale = max(0.4, min(width, height) / 900.0)
    for line in lines:
        start, end = line.to_pixels((width, height))
        color = (255, 150, 0)
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = max(1.0, float(np.hypot(dx, dy)))
        midpoint = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        arrow_length = max(12, round(min(width, height) * 0.035))
        normal = (-dy / length, dx / length)
        arrow_start = (
            round(midpoint[0] - normal[0] * arrow_length),
            round(midpoint[1] - normal[1] * arrow_length),
        )
        arrow_end = (
            round(midpoint[0] + normal[0] * arrow_length),
            round(midpoint[1] + normal[1] * arrow_length),
        )
        cv2.arrowedLine(
            annotated,
            start,
            end,
            color,
            max(1, round(scale * 2)),
            line_type=cv2.LINE_AA,
        )
        cv2.arrowedLine(
            annotated,
            arrow_start,
            arrow_end,
            (0, 255, 0),
            max(1, round(scale * 2)),
            line_type=cv2.LINE_AA,
            tipLength=0.35,
        )
        totals = counts.get(line.id, {})
        label = f"{line.name} IN:{totals.get('in', 0)} OUT:{totals.get('out', 0)}"
        cv2.putText(
            annotated,
            label,
            (start[0], max(18, start[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            max(1, round(scale * 1.5)),
            lineType=cv2.LINE_AA,
        )
    return annotated


def draw_overlay_stats(
    frame: NDArray[Any],
    fps: float,
    latency_ms: float,
    frame_id: int,
    *,
    total_unique: int | None = None,
    line_totals: Mapping[str, int] | None = None,
    busiest_zone: str | None = None,
) -> NDArray[Any]:
    """Draw a translucent FPS/latency/frame HUD on a copied BGR frame."""
    overlay = np.array(frame, copy=True)
    height, width = overlay.shape[:2]
    padding = max(8, round(min(width, height) * 0.018))
    scale = max(0.4, min(width, height) / 900.0)
    lines = [
        f"FPS: {fps:.1f}",
        f"Latency: {latency_ms:.1f} ms",
        f"Frame: {frame_id}",
    ]
    if total_unique is not None:
        lines.append(f"Unique visitors: {total_unique}")
    if line_totals is not None:
        lines.append(
            "IN/OUT/NET: "
            f"{line_totals.get('in', 0)}/"
            f"{line_totals.get('out', 0)}/"
            f"{line_totals.get('net', 0)}"
        )
    if busiest_zone is not None:
        lines.append(f"Busiest zone: {busiest_zone}")
    text_sizes = [
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0] for line in lines
    ]
    panel_width = max(size[0] for size in text_sizes) + padding * 2
    line_height = max(size[1] for size in text_sizes) + padding
    panel_height = line_height * len(lines) + padding
    x2 = min(width, padding + panel_width)
    y2 = min(height, padding + panel_height)
    panel = np.array(overlay, copy=True)
    cv2.rectangle(panel, (padding, padding), (x2, y2), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(panel, 0.72, overlay, 0.28, 0, overlay)
    for index, line in enumerate(lines):
        y = padding * 2 + (index + 1) * line_height - padding // 2
        cv2.putText(
            overlay,
            line,
            (padding * 2, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (245, 245, 245),
            1,
            lineType=cv2.LINE_AA,
        )
    return overlay
