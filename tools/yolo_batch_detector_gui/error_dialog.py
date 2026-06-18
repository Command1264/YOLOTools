from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget


class CopyableErrorMessageBox(QMessageBox):
    """Message box that shows an error and provides a copy button."""

    def __init__(self, parent: QWidget | None, title: str, message: str) -> None:
        """Initialize a copyable error message box.

        Args:
            parent (QWidget | None): Parent widget.
            title (str): Dialog title.
            message (str): Full error text.
        """
        super().__init__(parent)
        self._message = message
        self.setIcon(QMessageBox.Critical)
        self.setWindowTitle(title)
        self.setText("發生錯誤，詳細內容如下：")
        self.setInformativeText(message)
        self.setDetailedText(message)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        self.setStandardButtons(QMessageBox.Ok)
        copy_button = self.addButton("複製錯誤", QMessageBox.ActionRole)
        copy_button.clicked.connect(self.copy_error_text)

    def copy_error_text(self) -> None:
        """Copy the full error message to the system clipboard."""
        QApplication.clipboard().setText(self._message)
