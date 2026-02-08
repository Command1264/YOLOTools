from __future__ import annotations

import json
from typing import Any, Dict

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class DeleteHistoryDialog(QDialog):
    """Confirm history deletion and optional zip removal."""

    def __init__(self, count: int, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("刪除確認")
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"即將刪除 {count} 筆歷史紀錄。\n此操作無法復原，是否繼續？", self))
        self.chk_delete_zip = QCheckBox("同時刪除對應的輸出 zip", self)
        layout.addWidget(self.chk_delete_zip)

        row = QHBoxLayout()
        row.addStretch(1)
        btn_cancel = QPushButton("取消", self)
        btn_ok = QPushButton("刪除", self)
        btn_cancel.clicked.connect(self.reject)
        btn_ok.clicked.connect(self.accept)
        row.addWidget(btn_cancel)
        row.addWidget(btn_ok)
        layout.addLayout(row)


def show_json_record_dialog(parent: QWidget, title: str, record: Dict[str, Any]) -> None:
    """Show record details in a read-only JSON dialog."""
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    dialog.resize(920, 580)
    layout = QVBoxLayout(dialog)
    text = QPlainTextEdit(dialog)
    text.setReadOnly(True)
    text.setPlainText(json.dumps(record, ensure_ascii=False, indent=2))
    layout.addWidget(text)
    dialog.exec()
