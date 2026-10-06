"""Consistent application logging configuration."""

import logging
from pathlib import Path

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
) -> None:
    """Configure root logging with consistent console and optional file output.

    Args:
        level: Logging threshold, as a standard logging level or its name.
        log_file: Optional path to a UTF-8 log file.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(logging.FileHandler(Path(log_file), encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        handlers=handlers,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger for the requested module or component.

    Args:
        name: Logger name, typically ``__name__``.

    Returns:
        The named standard-library logger.
    """
    return logging.getLogger(name)
