from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from tools.yolo_auto_validator_gui import main_window as main_window_module

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class UiResponsivenessTests(unittest.TestCase):
    """確認背景工作期間 UI 事件仍能被處理。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_qt_timer_still_fires_while_worker_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            runs_dir = Path(temp_dir) / "runs"
            with (
                patch.object(main_window_module, "CONFIG_PATH", config_path),
                patch.object(main_window_module, "RUNS_DIR", runs_dir),
            ):
                window = main_window_module.MainWindow()
                state = {"timer_fired": False, "done": False}

                def task(log_cb, progress_known_cb, progress_unknown_cb, cancel_cb):
                    progress_unknown_cb("測試中...")
                    time.sleep(0.4)
                    return {"ok": True}

                def on_finished(_payload):
                    state["done"] = True

                QTimer.singleShot(50, lambda: state.__setitem__("timer_fired", True))
                window._start_worker(task, on_finished)

                deadline = time.time() + 3
                while time.time() < deadline and not state["done"]:
                    self._app.processEvents()
                    time.sleep(0.01)

                window.request_shutdown()

        self.assertTrue(state["timer_fired"])
        self.assertTrue(state["done"])

    def test_worker_cleanup_runs_on_ui_thread(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            runs_dir = Path(temp_dir) / "runs"
            with (
                patch.object(main_window_module, "CONFIG_PATH", config_path),
                patch.object(main_window_module, "RUNS_DIR", runs_dir),
            ):
                window = main_window_module.MainWindow()
                state = {"done": False, "cleanup_thread_name": None}
                original_cleanup = window._cleanup_worker

                def wrapped_cleanup():
                    state["cleanup_thread_name"] = threading.current_thread().name
                    return original_cleanup()

                window._cleanup_worker = wrapped_cleanup

                def task(log_cb, progress_known_cb, progress_unknown_cb, cancel_cb):
                    time.sleep(0.2)
                    return {"ok": True}

                def on_finished(_payload):
                    state["done"] = True

                window._start_worker(task, on_finished)

                deadline = time.time() + 3
                while time.time() < deadline and not state["done"]:
                    self._app.processEvents()
                    time.sleep(0.01)

                self._app.processEvents()
                window.request_shutdown()

        self.assertTrue(state["done"])
        self.assertEqual(state["cleanup_thread_name"], "MainThread")


if __name__ == "__main__":
    unittest.main()
