"""Structured logging for RL_agents handlers.

Call ``configure_logging()`` once at process startup (RL server / API entry
point). Handlers emit JSON log lines with alert levels ``red``, ``yellow``, or
``green`` for downstream monitoring.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

RED = "red"
YELLOW = "yellow"
GREEN = "green"

_ALERT_LEVELS = {RED, YELLOW, GREEN}
_ALERT_TO_LOGGING = {
    RED: logging.ERROR,
    YELLOW: logging.WARNING,
    GREEN: logging.INFO,
}

_CONFIGURED = False
DEFAULT_LOG_DIR = Path("logs")
DEFAULT_LOG_FILE = "rl_agents.log"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event_context = getattr(record, "event_context", None) or {}
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": event_context.get("alert_level", GREEN),
            "logger": record.name,
            "message": record.getMessage(),
        }
        if event_context:
            payload["context"] = event_context
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(
    log_dir: str | Path = DEFAULT_LOG_DIR,
    log_file: str = DEFAULT_LOG_FILE,
    max_bytes: int = 10_000_000,
    backup_count: int = 10,
) -> Path:
    """Set up the shared RL_agents log file once when the server starts.

    ``configure_logging`` does not log trial events itself. It registers where
    handler logs should be written (path, rotation, JSON format). After this
    runs, ``log_alert(...)`` calls from ``Query.py`` / ``RL.py`` append lines
    to that file.
    """
    global _CONFIGURED

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_file

    handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger("RL_agents")
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)
    root.propagate = False

    _CONFIGURED = True
    log_alert(
        root,
        GREEN,
        "RL_agents logging configured",
        log_path=str(log_path),
    )
    return log_path


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_alert(
    logger: logging.Logger,
    alert_level: str,
    message: str,
    **context: Any,
) -> None:
    if alert_level not in _ALERT_LEVELS:
        raise ValueError(f"alert_level must be one of {_ALERT_LEVELS}")

    event_context = {"alert_level": alert_level, **context}
    logger.log(
        _ALERT_TO_LOGGING[alert_level],
        message,
        extra={"event_context": event_context},
    )
