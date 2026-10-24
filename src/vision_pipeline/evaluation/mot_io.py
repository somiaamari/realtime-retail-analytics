"""MOTChallenge I/O and sequence metadata helpers."""

from __future__ import annotations

import configparser
import csv
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2

from vision_pipeline.types import FrameData


@dataclass(frozen=True)
class MOTRecord:
    """One MOTChallenge box row using one-based frame numbers."""

    frame: int
    track_id: int
    x: float
    y: float
    width: float
    height: float
    confidence: float = 1.0
    class_id: int = 1
    visibility: float = -1.0

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        """Return the box in ``(x1, y1, x2, y2)`` coordinates."""
        return self.x, self.y, self.x + self.width, self.y + self.height


@dataclass(frozen=True)
class SequenceInfo:
    """Metadata and local paths for an image-sequence or video benchmark."""

    name: str
    frame_rate: float
    width: int
    height: int
    sequence_length: int
    directory: Path
    image_directory: Path | None
    video_path: Path | None


def read_mot(
    path: str | Path,
    *,
    ground_truth: bool = False,
    pedestrian_class: int = 1,
) -> list[MOTRecord]:
    """Read comma-separated MOT rows.

    Ground-truth rows follow MOTChallenge conventions: only marked pedestrian
    rows (class 1 and mark/confidence greater than zero) are returned. This
    removes unmarked rows and distractor classes from MOT17-style annotations.
    """
    source = Path(path)
    records: list[MOTRecord] = []
    with source.open(encoding="utf-8", newline="") as stream:
        for line_number, row in enumerate(csv.reader(stream), start=1):
            if not row or row[0].strip().startswith("#"):
                continue
            if len(row) < 6 or len(row) > 9:
                raise ValueError(
                    f"{source}:{line_number}: expected 6 to 9 MOT columns, "
                    f"got {len(row)}"
                )
            try:
                values = [float(value.strip()) for value in row]
                record = MOTRecord(
                    frame=int(values[0]),
                    track_id=int(values[1]),
                    x=values[2],
                    y=values[3],
                    width=values[4],
                    height=values[5],
                    confidence=values[6] if len(values) > 6 else 1.0,
                    class_id=int(values[7]) if len(values) > 7 else pedestrian_class,
                    visibility=values[8] if len(values) > 8 else -1.0,
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{source}:{line_number}: invalid numeric MOT row"
                ) from error
            if record.frame < 1 or record.width < 0 or record.height < 0:
                raise ValueError(
                    f"{source}:{line_number}: frame must be positive and "
                    "box dimensions non-negative"
                )
            if ground_truth and (
                record.class_id != pedestrian_class or record.confidence <= 0
            ):
                continue
            records.append(record)
    return records


def write_mot(records: Iterable[MOTRecord], path: str | Path) -> None:
    """Write MOT records with the standard nine-column representation."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        for item in records:
            if item.frame < 1:
                raise ValueError("MOT frame numbers must be one-based and positive")
            writer.writerow(
                (
                    item.frame,
                    item.track_id,
                    item.x,
                    item.y,
                    item.width,
                    item.height,
                    item.confidence,
                    item.class_id,
                    item.visibility,
                )
            )


def tracks_csv_to_mot(
    source: str | Path,
    destination: str | Path,
    *,
    frame_column: str = "frame_id",
) -> list[MOTRecord]:
    """Convert tracker CSV rows with xyxy boxes into MOT format.

    The accepted CSV box columns are ``x1,y1,x2,y2`` or ``left,top,width,height``.
    Frame IDs in the project CSV are zero-based and are converted to MOT's
    one-based frame numbering.
    """
    records: list[MOTRecord] = []
    with Path(source).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"Track CSV has no header: {source}")
        fields = set(reader.fieldnames)
        required = {frame_column, "track_id"}
        if not required.issubset(fields):
            raise ValueError(f"Track CSV must include columns {sorted(required)}")
        xyxy = {"x1", "y1", "x2", "y2"}
        ltrb = {"left", "top", "width", "height"}
        if not (xyxy.issubset(fields) or ltrb.issubset(fields)):
            raise ValueError("Track CSV needs x1,y1,x2,y2 or left,top,width,height")
        for line_number, row in enumerate(reader, start=2):
            try:
                frame = int(row[frame_column]) + 1
                track_id = int(row["track_id"])
                if xyxy.issubset(fields):
                    x1, y1, x2, y2 = (
                        float(row[key]) for key in ("x1", "y1", "x2", "y2")
                    )
                    x, y, width, height = x1, y1, x2 - x1, y2 - y1
                else:
                    x, y, width, height = (
                        float(row[key]) for key in ("left", "top", "width", "height")
                    )
                records.append(
                    MOTRecord(
                        frame,
                        track_id,
                        x,
                        y,
                        width,
                        height,
                        float(row.get("confidence") or 1.0),
                        int(row.get("class_id") or 1),
                    )
                )
            except (TypeError, ValueError, KeyError) as error:
                raise ValueError(
                    f"{source}:{line_number}: invalid tracker CSV row"
                ) from error
    write_mot(records, destination)
    return records


def mot_to_tracks_csv(source: str | Path, destination: str | Path) -> None:
    """Convert MOT rows to the project's zero-based xyxy tracker CSV."""
    records = read_mot(source)
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            ("frame_id", "track_id", "x1", "y1", "x2", "y2", "confidence", "class_id")
        )
        for item in records:
            writer.writerow(
                (
                    item.frame - 1,
                    item.track_id,
                    *item.xyxy,
                    item.confidence,
                    item.class_id,
                )
            )


def load_sequence(sequence_dir: str | Path) -> SequenceInfo:
    """Read ``seqinfo.ini`` and locate ``img1`` or a sibling sequence video."""
    directory = Path(sequence_dir)
    ini_path = directory / "seqinfo.ini"
    parser = configparser.ConfigParser()
    if not parser.read(ini_path, encoding="utf-8"):
        raise FileNotFoundError(f"Sequence metadata not found: {ini_path}")
    section = parser["Sequence"]
    name = section.get("name", directory.name)
    frame_rate = section.getfloat("frameRate", fallback=0.0)
    width = section.getint("imWidth", fallback=0)
    height = section.getint("imHeight", fallback=0)
    sequence_length = section.getint("seqLength", fallback=0)
    image_dir_name = section.get("imDir", "img1")
    candidate_image_directory = directory / image_dir_name
    image_directory: Path | None = candidate_image_directory
    video_name = section.get("video", "")
    candidates = [directory / video_name] if video_name else []
    candidates.extend((directory / f"{name}.mp4", directory / f"{name}.avi"))
    video_path = next((item for item in candidates if item.is_file()), None)
    if not candidate_image_directory.is_dir():
        image_directory = None
    if image_directory is None and video_path is None:
        raise FileNotFoundError(
            f"Neither image directory '{image_dir_name}' nor sequence video "
            f"was found in {directory}"
        )
    if frame_rate <= 0 or width <= 0 or height <= 0 or sequence_length <= 0:
        raise ValueError(f"Invalid sequence metadata in {ini_path}")
    return SequenceInfo(
        name,
        frame_rate,
        width,
        height,
        sequence_length,
        directory,
        image_directory,
        video_path,
    )


def iter_sequence_frames(sequence: SequenceInfo) -> Iterator[FrameData]:
    """Yield decoded frames from an image directory or video file."""
    if sequence.image_directory is not None:
        for frame_id in range(sequence.sequence_length):
            candidates = (
                sequence.image_directory / f"{frame_id + 1:06d}.jpg",
                sequence.image_directory / f"{frame_id + 1:06d}.png",
            )
            image_path = next((item for item in candidates if item.is_file()), None)
            if image_path is None:
                raise FileNotFoundError(
                    f"Missing image for sequence frame {frame_id + 1}: "
                    f"{sequence.image_directory}"
                )
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise OSError(f"Unable to decode image: {image_path}")
            yield FrameData(
                frame, frame_id, frame_id / sequence.frame_rate, sequence.frame_rate
            )
        return
    assert sequence.video_path is not None
    capture = cv2.VideoCapture(str(sequence.video_path))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"Unable to open sequence video: {sequence.video_path}")
    try:
        frame_id = 0
        while frame_id < sequence.sequence_length:
            ok, frame = capture.read()
            if not ok:
                raise OSError(
                    f"Sequence video ended at frame {frame_id + 1}; "
                    f"seqinfo.ini declares {sequence.sequence_length} frames"
                )
            yield FrameData(
                frame, frame_id, frame_id / sequence.frame_rate, sequence.frame_rate
            )
            frame_id += 1
    finally:
        capture.release()
