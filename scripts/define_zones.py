"""Interactively define normalized analytics zones and counting lines."""

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from numpy.typing import NDArray

from vision_pipeline.analytics.geometry import load_zones
from vision_pipeline.logging_utils import setup_logging

ZONE_TYPES = ("generic", "queue", "shelf", "entrance_area")


def build_parser() -> argparse.ArgumentParser:
    """Build the zone-definition command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Reference image/video",
    )
    parser.add_argument(
        "--zones",
        type=Path,
        default=Path("configs/zones.yaml"),
        help="Zones file to create or preview",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview the existing zones file instead of defining zones",
    )
    return parser


def _reference_frame(path: Path) -> NDArray[Any]:
    """Read a reference image or the first frame of a video."""
    image = cv2.imread(str(path))
    if image is not None:
        return image
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"Unable to open reference image or video: {path}")
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise OSError(f"Unable to read a reference frame from: {path}")
    return frame


def _load_payload(path: Path) -> dict[str, Any]:
    """Read the source YAML mapping for rendering or updating."""
    with path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Zones file '{path}' must contain a YAML mapping.")
    return payload


def _pixel_point(point: list[float], width: int, height: int) -> tuple[int, int]:
    """Convert a normalized point to frame pixel coordinates."""
    return (
        round(point[0] * (width - 1)),
        round(point[1] * (height - 1)),
    )


def _preview(frame: NDArray[Any], path: Path) -> None:
    """Render configured polygons and lines over the reference frame."""
    payload = _load_payload(path)
    height, width = frame.shape[:2]
    for zone in payload.get("zones", []):
        points = zone["points"]
        if zone.get("normalized", payload.get("normalized", False)):
            pixels = [_pixel_point(point, width, height) for point in points]
        else:
            pixels = [(round(point[0]), round(point[1])) for point in points]
        color = tuple(zone.get("color", (0, 200, 255)))
        cv2.polylines(
            frame,
            [np.asarray(pixels, dtype=np.int32).reshape(-1, 1, 2)],
            True,
            color,
            2,
            lineType=cv2.LINE_AA,
        )
        x, y = pixels[0]
        cv2.putText(
            frame,
            zone.get("name", zone["id"]),
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            lineType=cv2.LINE_AA,
        )
    for line in payload.get("lines", []):
        start = line["start"]
        end = line["end"]
        if line.get("normalized", payload.get("normalized", False)):
            start_px = _pixel_point(start, width, height)
            end_px = _pixel_point(end, width, height)
        else:
            start_px = tuple(round(value) for value in start)
            end_px = tuple(round(value) for value in end)
        color = (255, 150, 0)
        cv2.arrowedLine(
            frame,
            start_px,
            end_px,
            color,
            2,
            line_type=cv2.LINE_AA,
            tipLength=0.12,
        )
        cv2.putText(
            frame,
            line.get("name", line["id"]),
            start_px,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            lineType=cv2.LINE_AA,
        )
    cv2.imshow("Analytics zones preview (q to close)", frame)
    while cv2.waitKey(0) & 0xFF != ord("q"):
        pass
    cv2.destroyAllWindows()


def _slug(value: str) -> str:
    """Create a stable YAML ID from a human-readable name."""
    result = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return result or "item"


def _unique_id(name: str, items: list[dict[str, Any]]) -> str:
    """Derive an ID not already used by another zone or line."""
    base = _slug(name)
    used = {item["id"] for item in items}
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


def _ask_zone_type() -> str:
    """Prompt for and validate the zone classification."""
    while True:
        value = input(f"Zone type {ZONE_TYPES} [generic]: ").strip() or "generic"
        if value in ZONE_TYPES:
            return value
        print(f"Choose one of: {', '.join(ZONE_TYPES)}")


def define(frame: NDArray[Any], destination: Path) -> None:
    """Collect polygon and line coordinates through an OpenCV window."""
    height, width = frame.shape[:2]
    zones: list[dict[str, Any]] = []
    lines: list[dict[str, Any]] = []
    points: list[tuple[int, int]] = []
    line_mode = False
    line_points: list[tuple[int, int]] = []

    def on_click(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if line_mode:
            if len(line_points) < 2:
                line_points.append((x, y))
        else:
            points.append((x, y))

    window = "Define zones: n polygon, l line, u undo, s save, q quit"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_click)
    try:
        while True:
            canvas = np.array(frame, copy=True)
            for zone in zones:
                pixels = [
                    _pixel_point(point, width, height) for point in zone["points"]
                ]
                cv2.polylines(
                    canvas,
                    [np.asarray(pixels, dtype=np.int32).reshape(-1, 1, 2)],
                    True,
                    tuple(zone["color"]),
                    2,
                )
                cv2.putText(
                    canvas,
                    zone["name"],
                    pixels[0],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    tuple(zone["color"]),
                    2,
                )
            for line in lines:
                start = _pixel_point(line["start"], width, height)
                end = _pixel_point(line["end"], width, height)
                cv2.arrowedLine(canvas, start, end, (255, 150, 0), 2)
                cv2.putText(
                    canvas,
                    line["name"],
                    start,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 150, 0),
                    2,
                )
            visible_points = line_points if line_mode else points
            for point in visible_points:
                cv2.circle(canvas, point, 5, (0, 0, 255), cv2.FILLED)
            if len(visible_points) > 1:
                cv2.polylines(
                    canvas,
                    [np.asarray(visible_points, dtype=np.int32).reshape(-1, 1, 2)],
                    False,
                    (0, 0, 255),
                    2,
                )
            cv2.imshow(window, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                break
            if key == ord("l"):
                line_mode = True
                line_points.clear()
            elif key == ord("n"):
                line_mode = False
                if len(points) < 3:
                    print("A polygon needs at least three points.")
                    continue
                name = input("Zone name: ").strip()
                if not name:
                    print("Zone name cannot be empty.")
                    continue
                zones.append(
                    {
                        "id": _unique_id(name, zones + lines),
                        "name": name,
                        "type": _ask_zone_type(),
                        "normalized": True,
                        "color": [0, 200, 255],
                        "points": [
                            [x / (width - 1), y / (height - 1)] for x, y in points
                        ],
                    }
                )
                points.clear()
            elif line_mode and len(line_points) == 2:
                name = input("Counting line name: ").strip()
                if not name:
                    print("Line name cannot be empty.")
                    line_points.clear()
                    line_mode = False
                    continue
                lines.append(
                    {
                        "id": _unique_id(name, zones + lines),
                        "name": name,
                        "normalized": True,
                        "start": [
                            line_points[0][0] / (width - 1),
                            line_points[0][1] / (height - 1),
                        ],
                        "end": [
                            line_points[1][0] / (width - 1),
                            line_points[1][1] / (height - 1),
                        ],
                    }
                )
                line_points.clear()
                line_mode = False
            elif key == ord("u"):
                if line_mode and line_points:
                    line_points.pop()
                elif points:
                    points.pop()
                elif lines:
                    lines.pop()
                elif zones:
                    zones.pop()
            elif key == ord("s"):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(
                    yaml.safe_dump(
                        {"zones": zones, "lines": lines},
                        sort_keys=False,
                    ),
                    encoding="utf-8",
                )
                load_zones(destination)
                print(
                    f"Saved {len(zones)} zones and {len(lines)} lines to {destination}"
                )
    finally:
        cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> int:
    """Run the zone definition or preview tool."""
    setup_logging()
    args = build_parser().parse_args(argv)
    try:
        frame = _reference_frame(args.source)
        if args.preview:
            load_zones(args.zones)
            _preview(frame, args.zones)
        else:
            define(frame, args.zones)
        return 0
    except (OSError, ValueError, cv2.error) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
