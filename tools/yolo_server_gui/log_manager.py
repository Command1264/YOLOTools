from __future__ import annotations

import logging
import queue
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Optional

LOGGER_NAME = "yolo_server_gui"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
WERKZEUG_TIME_PATTERN = re.compile(r"\s\[[0-9]{2}/[A-Za-z]{3}/[0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2}\]")

_LOG_CONTEXT: Optional["LogContext"] = None


@dataclass
class LogContext:
    """Hold log configuration and runtime state."""

    start_time: datetime
    log_dir: Path
    log_file: Path
    log_queue: queue.Queue[str]
    logger: logging.Logger


class GuiLogHandler(logging.Handler):
    """Push formatted log messages into a queue for GUI consumption."""

    def __init__(self, log_queue: queue.Queue[str]) -> None:
        super().__init__()
        self._queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:
            return
        try:
            self._queue.put_nowait(message)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(message)
            except queue.Full:
                pass


class WerkzeugCleanFilter(logging.Filter):
    """Remove werkzeug's embedded time from message."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if record.name.startswith("werkzeug") and isinstance(record.msg, str):
                record.msg = WERKZEUG_TIME_PATTERN.sub("", record.msg)
        except Exception:
            return True
        return True


class MillisecondFormatter(logging.Formatter):
    """Formatter with millisecond precision."""

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        datefmt = datefmt or LOG_DATE_FORMAT
        dt = datetime.fromtimestamp(record.created)
        base = dt.strftime(datefmt)
        millis = int(record.msecs)
        return f"{base}.{millis:03d}"


def setup_logging(app_dir: Path) -> LogContext:
    """
    Set up logging for console, file, and GUI.

    Args:
        app_dir (Path): Base directory for log folder creation.

    Returns:
        LogContext: Initialized logging context.
    """
    global _LOG_CONTEXT
    if _LOG_CONTEXT is not None:
        return _LOG_CONTEXT

    start_time = datetime.now()
    date_dir = start_time.strftime("%Y-%m-%d")
    log_dir = app_dir / "log" / date_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"yolo_server_log_{start_time.strftime('%Y%m%d_%H%M%S')}.log"

    log_queue: queue.Queue[str] = queue.Queue(maxsize=5000)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = MillisecondFormatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=10,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    gui_handler = GuiLogHandler(log_queue)
    gui_handler.setLevel(logging.INFO)
    gui_handler.setFormatter(formatter)

    if logger.handlers:
        logger.handlers.clear()

    clean_filter = WerkzeugCleanFilter()
    file_handler.addFilter(clean_filter)
    console_handler.addFilter(clean_filter)
    gui_handler.addFilter(clean_filter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.addHandler(gui_handler)

    _attach_handlers("flask.app", [file_handler, console_handler, gui_handler], logging.INFO)
    _attach_handlers("werkzeug", [file_handler, console_handler, gui_handler], logging.INFO)

    _LOG_CONTEXT = LogContext(
        start_time=start_time,
        log_dir=log_dir,
        log_file=log_file,
        log_queue=log_queue,
        logger=logger,
    )
    return _LOG_CONTEXT


def get_log_context() -> Optional[LogContext]:
    """
    Retrieve the active log context.

    Returns:
        Optional[LogContext]: Current log context if initialized.
    """
    return _LOG_CONTEXT


def _attach_handlers(name: str, handlers: list[logging.Handler], level: int) -> None:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        logger.handlers.clear()
    for handler in handlers:
        logger.addHandler(handler)


def get_logger() -> logging.Logger:
    """
    Get the configured logger or a safe fallback.

    Returns:
        logging.Logger: Logger instance.
    """
    if _LOG_CONTEXT is not None:
        return _LOG_CONTEXT.logger
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def get_log_queue() -> queue.Queue[str]:
    """
    Get the GUI log queue.

    Returns:
        queue.Queue[str]: Queue for GUI log streaming.
    """
    if _LOG_CONTEXT is not None:
        return _LOG_CONTEXT.log_queue
    return queue.Queue()


def get_active_log_file() -> Optional[Path]:
    """
    Get the active log file path.

    Returns:
        Optional[Path]: Active log file if logging is initialized.
    """
    if _LOG_CONTEXT is None:
        return None
    return _LOG_CONTEXT.log_file


def list_log_files() -> List[Path]:
    """
    List log files for the current log session date.

    Returns:
        List[Path]: Sorted log file paths.
    """
    if _LOG_CONTEXT is None:
        return []
    log_dir = _LOG_CONTEXT.log_dir
    if not log_dir.exists():
        return []
    files = [p for p in log_dir.iterdir() if p.is_file()]
    return sorted(files, key=lambda p: p.stat().st_mtime)
