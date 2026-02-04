from __future__ import annotations

import sys
from pathlib import Path

from PySide6 import QtWidgets

from config_store import ConfigStore
from log_store import LogStore
from ui_main import MainWindow


def resolve_exec_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    exec_dir = resolve_exec_dir()
    log_store = LogStore(exec_dir)
    config_store = ConfigStore(exec_dir)
    window = MainWindow(exec_dir, log_store, config_store)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
