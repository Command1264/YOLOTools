from __future__ import annotations

import sys
import unittest
from pathlib import Path

from PySide6.QtWidgets import QApplication

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GUI_DIR = PROJECT_ROOT / "tools" / "yolo_batch_detector_gui"
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))

from error_dialog import CopyableErrorMessageBox


QT_APP = QApplication.instance() or QApplication([])


class CopyableErrorMessageBoxTests(unittest.TestCase):
    """Cover copyable error message box behavior."""

    def test_copy_button_writes_error_text_to_clipboard(self) -> None:
        """The custom error dialog should expose one-click copy behavior."""
        box = CopyableErrorMessageBox(parent=None, title="辨識錯誤", message="line 1\nline 2")

        box.copy_error_text()

        self.assertEqual(QApplication.clipboard().text(), "line 1\nline 2")
        self.assertEqual(box.windowTitle(), "辨識錯誤")
        self.assertIn("line 1", box.detailedText())


if __name__ == "__main__":
    unittest.main()
