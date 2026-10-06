"""Tests for logging utilities."""

import logging
from pathlib import Path

from vision_pipeline.logging_utils import get_logger, setup_logging


def test_get_logger_returns_named_logger() -> None:
    """The logger helper returns a standard named logger."""
    logger = get_logger("vision_pipeline.test")

    assert logger is logging.getLogger("vision_pipeline.test")


def test_setup_logging_writes_formatted_log(tmp_path: Path) -> None:
    """Configured logging writes the message and standard metadata to a file."""
    log_path = tmp_path / "application.log"
    setup_logging(level="INFO", log_file=log_path)

    get_logger("vision_pipeline.test").info("logging-ready")

    content = log_path.read_text(encoding="utf-8")
    assert "INFO" in content
    assert "vision_pipeline.test" in content
    assert "logging-ready" in content


def test_setup_logging_accepts_numeric_level() -> None:
    """Logging can be configured with a numeric standard-library level."""
    setup_logging(level=logging.WARNING)

    assert logging.getLogger().level == logging.WARNING
