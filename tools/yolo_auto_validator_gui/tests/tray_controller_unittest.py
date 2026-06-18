from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox

from tools.yolo_auto_validator_gui.tray_controller import TrayController

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class TrayControllerTests(unittest.TestCase):
    """驗證 system tray 關閉行為。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_close_event_minimizes_instead_of_exit(self) -> None:
        window = QMainWindow()
        tray = TrayController(window)
        tray.setup()
        event = QCloseEvent()
        handled = tray.handle_close_event(event)
        self.assertTrue(handled)
        self.assertFalse(window.isVisible())

    def test_request_exit_sets_allow_exit(self) -> None:
        window = QMainWindow()
        tray = TrayController(window)
        tray.setup()
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
            tray.request_exit()
        self.assertTrue(window._allow_exit)


if __name__ == "__main__":
    unittest.main()
