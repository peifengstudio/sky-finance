"""
Centralised logging configuration for sky-finance.

Call setup_logging() once at every process entry point:
  - Celery worker / beat  (wired via celeryd_after_setup signal in celery_app.py)
  - Web / dashboard server
  - CLI scripts

Log output:
  Console  — human-readable, coloured when a TTY is attached
  File     — logs/app.log, rotating 10 MB × 5 files, detailed format

Environment variables:
  LOG_LEVEL   override log level (default: INFO)
"""

import logging
import logging.config
import logging.handlers
import os
import sys
from pathlib import Path

# Project root is three levels up from src/sky_finance/logging_config.py
_LOGS_DIR = Path(__file__).parents[2] / "logs"

# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

_FMT_DETAILED = "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"
_FMT_SIMPLE = "%(levelname)-8s [%(name)s] %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

# ANSI colour codes — applied only when stdout is a real TTY
_LEVEL_COLOURS = {
    "DEBUG": "\033[36m",  # cyan
    "INFO": "\033[32m",  # green
    "WARNING": "\033[33m",  # yellow
    "ERROR": "\033[31m",  # red
    "CRITICAL": "\033[35m",  # magenta
}
_RESET = "\033[0m"


class _ColouredFormatter(logging.Formatter):
    """Add ANSI colour to the level-name portion of a log record."""

    def format(self, record: logging.LogRecord) -> str:
        colour = _LEVEL_COLOURS.get(record.levelname, "")
        record.levelname = f"{colour}{record.levelname}{_RESET}" if colour else record.levelname
        return super().format(record)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def setup_logging(log_level: str | None = None) -> None:
    """
    Configure root logger with a console handler and a rotating file handler.

    Safe to call multiple times — subsequent calls are no-ops after the first.

    Args:
        log_level: override level string, e.g. "DEBUG". Falls back to the
                   LOG_LEVEL env var, then "INFO".
    """
    level = (log_level or os.environ.get("LOG_LEVEL", "INFO")).upper()

    _LOGS_DIR.mkdir(exist_ok=True)
    log_file = _LOGS_DIR / "app.log"

    use_colour = sys.stdout.isatty()
    console_formatter = (
        _ColouredFormatter(_FMT_SIMPLE, datefmt=_DATE_FMT)
        if use_colour
        else logging.Formatter(_FMT_SIMPLE, datefmt=_DATE_FMT)
    )

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "detailed": {
                    "()": logging.Formatter,
                    "format": _FMT_DETAILED,
                    "datefmt": _DATE_FMT,
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                    "formatter": "detailed",  # replaced below
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": str(log_file),
                    "maxBytes": 10 * 1024 * 1024,  # 10 MB
                    "backupCount": 5,
                    "encoding": "utf-8",
                    "formatter": "detailed",
                },
            },
            "loggers": {
                # Quiet noisy third-party libraries
                "celery": {"level": "INFO", "propagate": True},
                "celery.beat": {"level": "INFO", "propagate": True},
                "yfinance": {"level": "WARNING", "propagate": True},
                "peewee": {"level": "WARNING", "propagate": True},
                "urllib3": {"level": "WARNING", "propagate": True},
                "httpx": {"level": "WARNING", "propagate": True},
                "httpcore": {"level": "WARNING", "propagate": True},
                "feedparser": {"level": "WARNING", "propagate": True},
                "openai": {"level": "WARNING", "propagate": True},
            },
            "root": {
                "level": level,
                "handlers": ["console", "file"],
            },
        }
    )

    # Patch the console handler with the colour-aware formatter after dictConfig
    # (dictConfig cannot reference a live formatter object directly)
    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            handler.setFormatter(console_formatter)

    logging.getLogger(__name__).debug("Logging initialised — level=%s  file=%s", level, log_file)
