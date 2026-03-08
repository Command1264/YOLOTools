from __future__ import annotations

import logging
import logging.handlers
import queue
import re
import sys
import threading
from dataclasses import dataclass
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, TextIO

LOGGER_NAME: str = "yolo_server_gui"
LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
WERKZEUG_TIME_PATTERN = re.compile(r"\s\[[0-9]{2}/[A-Za-z]{3}/[0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2}\]")
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

_LOG_CONTEXT: Optional["LogContext"] = None


@dataclass
class LogContext:
    """Hold log configuration and runtime state."""

    start_time: datetime
    log_root: Path
    event_queue: queue.SimpleQueue[logging.LogRecord]
    gui_broadcaster: "GuiLogBroadcaster"
    logger: logging.Logger
    file_handler: "SessionRollingFileHandler"
    queue_listener: logging.handlers.QueueListener


class GuiLogBroadcaster:
    """Broadcast sanitized log lines to GUI subscribers."""

    def __init__(self, history_limit: int = 500) -> None:
        self._lock = threading.RLock()
        self._subscribers: set[queue.SimpleQueue[tuple[int, str]]] = set()
        self._recent_entries: deque[tuple[int, str]] = deque(maxlen=history_limit)
        self._sequence: int = 0

    def publish(self, message: str) -> None:
        """Publish one sanitized log line to subscribers."""
        with self._lock:
            self._sequence += 1
            entry = (self._sequence, message)
            self._recent_entries.append(entry)
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            subscriber.put(entry)

    def register(self) -> queue.SimpleQueue[tuple[int, str]]:
        """Register a subscriber queue for future log lines."""
        subscriber: queue.SimpleQueue[tuple[int, str]] = queue.SimpleQueue()
        with self._lock:
            self._subscribers.add(subscriber)
        return subscriber

    def unregister(self, subscriber: queue.SimpleQueue[tuple[int, str]]) -> None:
        """Unregister a subscriber queue."""
        with self._lock:
            self._subscribers.discard(subscriber)

    def recent_entries_since(self, sequence: int) -> list[tuple[int, str]]:
        """Return entries published after the specified sequence."""
        with self._lock:
            return [entry for entry in self._recent_entries if entry[0] > sequence]

    def current_sequence(self) -> int:
        """Return the latest published sequence."""
        with self._lock:
            return self._sequence


class SessionRollingFileHandler(logging.Handler):
    """Rotate log files by size and date boundary."""

    def __init__(self, log_root: Path, session_start: datetime, max_bytes: int, encoding: str = "utf-8") -> None:
        super().__init__()
        self._log_root: Path = log_root
        self._session_start: datetime = session_start
        self._max_bytes: int = max_bytes
        self._encoding: str = encoding
        self._stream: Optional[TextIO] = None
        self._current_file: Optional[Path] = None
        self._current_day: str = ""
        self._lock = threading.RLock()
        self._open_new_file(datetime.now())

    @property
    def current_file(self) -> Optional[Path]:
        """Get current active log file path."""
        return self._current_file

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            with self._lock:
                self._write(message)
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        with self._lock:
            try:
                if self._stream is not None:
                    self._write_file_footer(datetime.now(), None)
                    self._stream.flush()
                    self._stream.close()
            finally:
                self._stream = None
                self._current_file = None
                super().close()

    def _write(self, message: str) -> None:
        if self._stream is None:
            self._open_new_file(datetime.now())
        encoded = f"{message}\n".encode(self._encoding, errors="replace")
        now = datetime.now()
        if self._should_rollover(now, len(encoded)):
            self._rollover(now)
        if self._stream is None:
            return
        self._stream.write(message)
        self._stream.write("\n")
        self._stream.flush()

    def _should_rollover(self, now: datetime, incoming_size: int) -> bool:
        if self._current_file is None:
            return True
        day_key = now.strftime("%Y-%m-%d")
        if day_key != self._current_day:
            return True
        try:
            current_size = self._current_file.stat().st_size
        except Exception:
            current_size = 0
        return current_size + incoming_size > self._max_bytes

    def _rollover(self, now: datetime) -> None:
        next_file_path = self._resolve_file_path(now)
        self._write_file_footer(now, next_file_path.name)
        if self._stream is not None:
            self._stream.flush()
            self._stream.close()
            self._stream = None
        self._open_file(next_file_path, now)

    def _open_new_file(self, now: datetime) -> None:
        self._open_file(self._resolve_file_path(now), now)

    def _open_file(self, file_path: Path, now: datetime) -> None:
        if self._stream is not None:
            self._stream.flush()
            self._stream.close()
            self._stream = None
        self._current_day = now.strftime("%Y-%m-%d")
        self._stream = file_path.open("a", encoding=self._encoding)
        self._current_file = file_path
        if file_path.stat().st_size == 0:
            session_text = self._session_start.strftime("%Y-%m-%d %H:%M:%S")
            file_text = now.strftime("%Y-%m-%d %H:%M:%S")
            self._stream.write(f"session_start={session_text}\n")
            self._stream.write(f"file_start={file_text}\n")
            self._stream.flush()

    def _resolve_file_path(self, now: datetime) -> Path:
        day_dir = self._log_root / now.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        return self._build_log_file_path(day_dir, now)

    def _write_file_footer(self, now: datetime, next_file_name: Optional[str]) -> None:
        if self._stream is None:
            return
        file_text = now.strftime("%Y-%m-%d %H:%M:%S")
        next_name = next_file_name or "none"
        self._stream.write(f"file_end={file_text}\n")
        self._stream.write(f"next_file={next_name}\n")

    @staticmethod
    def _build_log_file_path(day_dir: Path, now: datetime) -> Path:
        base = f"yolo_server_log_{now.strftime('%Y%m%d_%H%M%S')}"
        candidate = day_dir / f"{base}.log"
        if not candidate.exists():
            return candidate
        index = 1
        while True:
            numbered = day_dir / f"{base}_{index:02d}.log"
            if not numbered.exists():
                return numbered
            index += 1


class GuiLogHandler(logging.Handler):
    """Push formatted log messages into a queue for GUI consumption."""

    def __init__(self, gui_broadcaster: GuiLogBroadcaster) -> None:
        super().__init__()
        self._gui_broadcaster = gui_broadcaster

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:
            return
        self._gui_broadcaster.publish(message)


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


class PlainTextFormatter(MillisecondFormatter):
    """Formatter that removes terminal-only control sequences."""

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        return sanitize_log_text(message)


def sanitize_log_text(text: str) -> str:
    """
    Remove terminal-only escape sequences and control characters.

    Args:
        text (str): Raw formatted log text.

    Returns:
        str: Sanitized plain text safe for file and GUI logs.
    """
    sanitized = ANSI_ESCAPE_PATTERN.sub("", text)
    sanitized = sanitized.replace("\r\n", "\n").replace("\r", "\n")
    sanitized = sanitized.replace("\u2028", "\n").replace("\u2029", "\n")
    sanitized = CONTROL_CHAR_PATTERN.sub("", sanitized)
    return sanitized


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

    start_time: datetime = datetime.now()
    log_root: Path = app_dir / "log"
    log_root.mkdir(parents=True, exist_ok=True)

    event_queue: queue.SimpleQueue[logging.LogRecord] = queue.SimpleQueue()
    gui_broadcaster = GuiLogBroadcaster()

    logger: logging.Logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter: MillisecondFormatter = MillisecondFormatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    plain_formatter: PlainTextFormatter = PlainTextFormatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    file_handler: SessionRollingFileHandler = SessionRollingFileHandler(
        log_root=log_root,
        session_start=start_time,
        max_bytes=5 * 1024 * 1024,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(plain_formatter)

    console_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    gui_handler: GuiLogHandler = GuiLogHandler(gui_broadcaster)
    gui_handler.setLevel(logging.INFO)
    gui_handler.setFormatter(plain_formatter)

    clean_filter: WerkzeugCleanFilter = WerkzeugCleanFilter()
    file_handler.addFilter(clean_filter)
    console_handler.addFilter(clean_filter)
    gui_handler.addFilter(clean_filter)

    if logger.handlers:
        logger.handlers.clear()
    queue_handler = logging.handlers.QueueHandler(event_queue)
    queue_handler.setLevel(logging.INFO)
    logger.addHandler(queue_handler)

    queue_listener = logging.handlers.QueueListener(
        event_queue,
        file_handler,
        console_handler,
        gui_handler,
        respect_handler_level=True,
    )
    queue_listener.start()

    _attach_handlers("flask.app", [queue_handler], logging.INFO)
    _attach_handlers("werkzeug", [queue_handler], logging.INFO)

    _LOG_CONTEXT = LogContext(
        start_time=start_time,
        log_root=log_root,
        event_queue=event_queue,
        gui_broadcaster=gui_broadcaster,
        logger=logger,
        file_handler=file_handler,
        queue_listener=queue_listener,
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


class LogController:
    """Provide safe logging APIs to avoid scattered try/except blocks."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self._logger: logging.Logger = logger or get_logger()

    @staticmethod
    def _handle_error(method: str, exc: Exception) -> None:
        try:
            print(f"[LogController] {method} failed: {exc}", file=sys.stderr)
        except Exception:
            return

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        try:
            self._logger.info(msg, *args, **kwargs)
        except Exception as exc:
            self._handle_error("info", exc)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        try:
            self._logger.warning(msg, *args, **kwargs)
        except Exception as exc:
            self._handle_error("warning", exc)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        try:
            self._logger.error(msg, *args, **kwargs)
        except Exception as exc:
            self._handle_error("error", exc)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        try:
            self._logger.exception(msg, *args, **kwargs)
        except Exception as exc:
            self._handle_error("exception", exc)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        try:
            self._logger.debug(msg, *args, **kwargs)
        except Exception as exc:
            self._handle_error("debug", exc)

    def log(self, level: int, msg: str, *args: Any, **kwargs: Any) -> None:
        try:
            self._logger.log(level, msg, *args, **kwargs)
        except Exception as exc:
            self._handle_error("log", exc)

    def get_logger(self) -> logging.Logger:
        return self._logger


def get_log_queue() -> queue.Queue[str]:
    """
    Get the GUI log queue.

    Returns:
        queue.Queue[str]: Queue for GUI log streaming.
    """
    return queue.Queue()


def register_log_subscriber() -> queue.SimpleQueue[tuple[int, str]]:
    """
    Register a GUI subscriber for live log lines.

    Returns:
        queue.SimpleQueue[tuple[int, str]]: Subscriber queue receiving future log lines.
    """
    if _LOG_CONTEXT is None:
        return queue.SimpleQueue()
    return _LOG_CONTEXT.gui_broadcaster.register()


def unregister_log_subscriber(subscriber: queue.SimpleQueue[tuple[int, str]]) -> None:
    """
    Unregister a GUI subscriber queue.

    Args:
        subscriber (queue.SimpleQueue[tuple[int, str]]): Subscriber queue to remove.
    """
    if _LOG_CONTEXT is None:
        return
    _LOG_CONTEXT.gui_broadcaster.unregister(subscriber)


def get_recent_gui_logs_since(sequence: int) -> list[str]:
    """
    Get recent sanitized GUI log lines after the specified sequence.

    Returns:
        list[str]: Recent log lines.
    """
    if _LOG_CONTEXT is None:
        return []
    return [entry[1] for entry in _LOG_CONTEXT.gui_broadcaster.recent_entries_since(sequence)]


def get_current_gui_log_sequence() -> int:
    """
    Get the latest GUI log sequence number.

    Returns:
        int: Latest published sequence.
    """
    if _LOG_CONTEXT is None:
        return 0
    return _LOG_CONTEXT.gui_broadcaster.current_sequence()


def get_active_log_file() -> Optional[Path]:
    """
    Get the active log file path.

    Returns:
        Optional[Path]: Active log file if logging is initialized.
    """
    if _LOG_CONTEXT is None:
        return None
    return _LOG_CONTEXT.file_handler.current_file


def list_log_files() -> List[Path]:
    """
    List log files for the current app session.

    Returns:
        List[Path]: Sorted log file paths.
    """
    if _LOG_CONTEXT is None:
        return []
    log_root = _LOG_CONTEXT.log_root
    if not log_root.exists():
        return []
    session_marker = f"session_start={_LOG_CONTEXT.start_time.strftime('%Y-%m-%d %H:%M:%S')}"
    files = [
        p
        for p in log_root.rglob("*.log")
        if p.is_file() and _has_session_marker(p, session_marker)
    ]
    return sorted(files, key=lambda p: p.stat().st_mtime)


def _has_session_marker(path: Path, marker: str) -> bool:
    """Check whether the first line matches the target session marker."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            first_line = f.readline().strip()
        return first_line == marker
    except Exception:
        return False
