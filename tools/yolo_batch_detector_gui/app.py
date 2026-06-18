from __future__ import annotations

import signal
import sys
import traceback

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

if __package__ in {None, ""}:
    from main_window import MainWindow
else:
    from .main_window import MainWindow


def _install_exception_hooks() -> None:
    def handle_exception(exc_type, exc_value, exc_traceback) -> None:
        traceback_text = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        print(traceback_text, file=sys.stderr)
        if QApplication.instance() is not None:
            QMessageBox.critical(None, "未處理例外", f"發生未處理錯誤：\n\n{exc_value}")

    sys.excepthook = handle_exception
    if hasattr(__import__("threading"), "excepthook"):
        __import__("threading").excepthook = (
            lambda args: handle_exception(args.exc_type, args.exc_value, args.exc_traceback)
        )


def _has_terminal_session() -> bool:
    """Return whether the process appears to have an attached terminal.

    Returns:
        bool: True when stdin/stdout/stderr has a TTY.
    """
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        isatty = getattr(stream, "isatty", None)
        if callable(isatty) and isatty():
            return True
    return False


def _install_ctrl_c_handler(app: QApplication) -> QTimer | None:
    """Install Ctrl+C support when the app is launched from a terminal.

    Args:
        app (QApplication): Qt application.

    Returns:
        QTimer | None: Keepalive timer for signal dispatch, if installed.
    """
    if not _has_terminal_session():
        return None

    def handle_sigint(_signum, _frame) -> None:
        app.quit()

    signal.signal(signal.SIGINT, handle_sigint)
    timer = QTimer()
    timer.setInterval(200)
    timer.timeout.connect(lambda: None)
    timer.start()
    return timer


def main() -> int:
    """Start the YOLO batch detector GUI.

    Returns:
        int: Qt application exit code.
    """
    _install_exception_hooks()
    app = QApplication(sys.argv)
    app.setApplicationName("YOLO 批次辨識檢視器")
    window = MainWindow()
    window.show()
    timer = _install_ctrl_c_handler(app)
    exit_code = app.exec()
    if timer is not None:
        timer.stop()
    window.request_shutdown()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
