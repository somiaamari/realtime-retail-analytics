"""OpenCV visualization helpers for detections and per-frame HUD statistics."""

from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from vision_pipeline.types import Detection


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


def draw_overlay_stats(
    frame: NDArray[Any],
    fps: float,
    latency_ms: float,
    frame_id: int,
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
