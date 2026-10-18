"""Project logging helpers."""

from __future__ import annotations

import logging
import sys


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure structured console logging for scripts and CLI commands."""

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        stream=sys.stderr,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a logger using the project logging format."""

    if not logging.getLogger().handlers:
        configure_logging()
    return logging.getLogger(name)
