"""Tests for application configuration."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from vision_pipeline.config import AppConfig, load_config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def test_load_default_config() -> None:
    """The repository's default YAML produces a validated application config."""
    config = load_config(DEFAULT_CONFIG)

    assert config.video.source == "0"
    assert config.video.width == 1920
    assert config.model.conf_threshold == 0.25
    assert config.tracker.track_buffer == 30
    assert config.paths.data_dir == Path("data")


def test_invalid_model_threshold_is_rejected() -> None:
    """Thresholds outside the supported range raise Pydantic validation errors."""
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"model": {"conf_threshold": 1.5}})


def test_missing_config_file_has_clear_error(tmp_path: Path) -> None:
    """A missing configuration file reports its path."""
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="missing.yaml"):
        load_config(missing_path)


def test_non_mapping_yaml_is_rejected(tmp_path: Path) -> None:
    """A scalar YAML document is not accepted as application configuration."""
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("- not-a-mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML mapping"):
        load_config(config_path)


def test_malformed_yaml_preserves_parser_error(tmp_path: Path) -> None:
    """Malformed YAML exposes its parser error."""
    config_path = tmp_path / "malformed.yaml"
    config_path.write_text("video: [", encoding="utf-8")

    with pytest.raises(yaml.YAMLError):
        load_config(config_path)
