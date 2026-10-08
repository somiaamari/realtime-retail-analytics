"""Application configuration models and YAML loading."""

from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field


def _default_classes() -> list[str | int]:
    """Select the COCO person class by default."""
    return [0]


class StrictConfigModel(BaseModel):
    """Base model that rejects unknown configuration keys."""

    model_config = ConfigDict(extra="forbid")


class VideoConfig(StrictConfigModel):
    """Video source and frame dimensions."""

    source: str | int = "0"
    width: int = Field(default=1920, gt=0)
    height: int = Field(default=1080, gt=0)
    fps: Annotated[float, Field(gt=0)] = 30.0
    resize: bool = False
    max_retries: int = Field(default=5, ge=0)
    reconnect_delay_s: Annotated[float, Field(ge=0.0)] = 1.0


class ModelConfig(StrictConfigModel):
    """Model artifact location and inference thresholds."""

    weights_path: Path = Path("models/yolov8n.pt")
    conf_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.25
    iou_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.45
    device: str = "auto"
    imgsz: int = Field(default=640, gt=0)
    classes: list[str | int] = Field(default_factory=_default_classes)


class TrackerConfig(StrictConfigModel):
    """Tracking thresholds and track retention settings."""

    track_thresh: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    match_thresh: Annotated[float, Field(ge=0.0, le=1.0)] = 0.8
    track_buffer: int = Field(default=30, gt=0)


class PathsConfig(StrictConfigModel):
    """Filesystem locations used by the application."""

    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")


class AppConfig(StrictConfigModel):
    """Validated configuration for the whole application."""

    video: VideoConfig = Field(default_factory=VideoConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)


def load_config(path: str | Path) -> AppConfig:
    """Load and validate application settings from a YAML file.

    Args:
        path: Path to a YAML configuration file.

    Returns:
        A validated application configuration.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        OSError: If the configuration file cannot be read.
        yaml.YAMLError: If the YAML document is malformed.
        ValueError: If the YAML root is not a mapping.
        pydantic.ValidationError: If a setting is invalid.
    """
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Configuration in '{config_path}' must contain a YAML mapping."
        )

    return AppConfig.model_validate(data)
