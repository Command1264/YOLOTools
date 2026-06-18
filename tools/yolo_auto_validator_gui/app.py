from __future__ import annotations

import signal
import sys
import traceback

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

if __package__ in {None, ""}:
    from logging_utils import configure_logging, get_logger
    from main_window import MainWindow
else:
    from .logging_utils import configure_logging, get_logger
    from .main_window import MainWindow

LOGGER = get_logger(__name__)


def _install_exception_hooks() -> None:
    def handle_exception(exc_type, exc_value, exc_traceback) -> None:
        traceback_text = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        LOGGER.error("Unhandled exception reached application hook.\n%s", traceback_text.rstrip())
        app = QApplication.instance()
        if app is None:
            return
        QMessageBox.critical(None, "未處理例外", f"發生未處理錯誤，但程式不會自動關閉。\n\n{exc_value}")

    def handle_thread_exception(args) -> None:
        handle_exception(args.exc_type, args.exc_value, args.exc_traceback)

    sys.excepthook = handle_exception
    if hasattr(__import__("threading"), "excepthook"):
        __import__("threading").excepthook = handle_thread_exception


def _has_terminal_session() -> bool:
    """Return whether the process appears to have an attached terminal."""
    streams = (sys.stdin, sys.stdout, sys.stderr)
    for stream in streams:
        isatty = getattr(stream, "isatty", None)
        if callable(isatty):
            try:
                if isatty():
                    return True
            except Exception:
                continue
    return False


def _request_graceful_exit(app: QApplication | None, window: MainWindow | None, reason: str) -> None:
    """Request a graceful Qt shutdown.

    Args:
        app (QApplication | None): Qt application.
        window (MainWindow | None): Main window.
        reason (str): Shutdown reason for logging.
    """
    LOGGER.info("Graceful exit requested. reason=%s", reason)
    if window is not None:
        window._allow_exit = True
    if app is not None:
        app.quit()


def _install_ctrl_c_handler(app: QApplication, window: MainWindow) -> QTimer | None:
    """Install Ctrl+C handling when launched from a terminal.

    Args:
        app (QApplication): Qt application.
        window (MainWindow): Main window.

    Returns:
        QTimer | None: Keepalive timer required for Python signal dispatch, or None if skipped.
    """
    if not _has_terminal_session():
        LOGGER.debug("Skipping Ctrl+C handler installation because no terminal session was detected.")
        return None

    def handle_sigint(_signum, _frame) -> None:
        LOGGER.info("SIGINT received from terminal.")
        _request_graceful_exit(app, window, "SIGINT")

    signal.signal(signal.SIGINT, handle_sigint)
    keepalive_timer = QTimer()
    keepalive_timer.setInterval(200)
    keepalive_timer.timeout.connect(lambda: None)
    keepalive_timer.start()
    LOGGER.info("Ctrl+C handler installed for terminal session.")
    return keepalive_timer


def main() -> int:
    """啟動 YOLO 自動驗證器。"""
    configure_logging("yolo_auto_validator.app")
    _install_exception_hooks()
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    LOGGER.info("YOLO auto validator application starting.")
    window = MainWindow()
    window.show()
    ctrl_c_timer = _install_ctrl_c_handler(app, window)
    exit_code = app.exec()
    if ctrl_c_timer is not None:
        ctrl_c_timer.stop()
    window.request_shutdown()
    LOGGER.info("YOLO auto validator application stopped. exit_code=%s", exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
