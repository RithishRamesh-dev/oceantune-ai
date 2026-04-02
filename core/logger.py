"""
core/logger.py
--------------
Centralised logging for OceanTune AI.

Every module imports get_logger() from here. Logs are written to:
  - Console (plain, human-readable)
  - File    (newline-delimited JSON for later analysis)

Usage
-----
    from core.logger import get_logger
    log = get_logger(__name__)
    log.info("vLLM server started", port=8000, model="mistral")
"""

import logging
import sys
import json
from pathlib import Path
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).resolve().parent.parent / "storage" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

_SESSION_TS = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"oceantune_{_SESSION_TS}.jsonl"


# ---------------------------------------------------------------------------
# JSON file handler  (one JSON object per line)
# ---------------------------------------------------------------------------
class _JSONLineHandler(logging.Handler):
    """Writes each log record as a single JSON line to LOG_FILE."""

    def __init__(self, path: Path):
        super().__init__()
        self._path = path

    def emit(self, record: logging.LogRecord) -> None:
        entry = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Attach any extra kwargs passed by the caller
        for k, v in record.__dict__.items():
            if k not in (
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
            ):
                entry[k] = v
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception:  # noqa: BLE001
            pass  # never crash the main program due to logging failure


# ---------------------------------------------------------------------------
# Console handler
# ---------------------------------------------------------------------------
_CONSOLE_FMT = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Return a logger with the given name.

    Handlers are attached only once per name (safe to call multiple times).
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # prevent double-logging if root has handlers

    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(_CONSOLE_FMT)
        logger.addHandler(ch)

        # JSON file handler
        fh = _JSONLineHandler(LOG_FILE)
        fh.setLevel(level)
        logger.addHandler(fh)

    _LOGGERS[name] = logger
    return logger


def log_dict(logger: logging.Logger, level: str, msg: str, **kwargs) -> None:
    """
    Helper to attach structured key-value pairs to a log record.

    Example:
        log_dict(log, "info", "benchmark done", throughput=1200, p95=188)
    """
    method = getattr(logger, level.lower(), logger.info)
    method(msg, extra=kwargs)


# ---------------------------------------------------------------------------
# Module-level logger for core itself
# ---------------------------------------------------------------------------
_log = get_logger("core.logger")
log_dict(_log, "info", "Logging initialised", log_file=str(LOG_FILE))
