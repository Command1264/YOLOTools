from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QTimer
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from log_manager import LogController, get_active_log_file, get_log_queue, get_logger, list_log_files


class LogViewer(QDialog):
    """提供 GUI 方式瀏覽與即時顯示運行日誌（PySide6 版本）。"""

    def __init__(self, parent: QWidget, on_close: Optional[Callable[[], None]] = None) -> None:
        super().__init__(parent)
        self._on_close: Optional[Callable[[], None]] = on_close
        self._log_ctrl: LogController = LogController(get_logger())
        self._log_queue: queue.Queue[str] = get_log_queue()
        self._files: list[Path] = []
        self._current_file: Optional[Path] = None
        self._live_file: Optional[Path] = get_active_log_file()
        self._loading: bool = False
        self._load_seq: int = 0
        self._pending_live_lines: list[str] = []
        self._load_result_queue: queue.Queue[tuple[int, str, list[str]]] = queue.Queue()

        self.setWindowTitle("運行日誌")
        self.resize(820, 520)
        self.setMinimumSize(720, 420)
        self.setModal(True)

        self._build_ui()

        self._poll_timer: QTimer = QTimer(self)
        self._poll_timer.setInterval(200)
        self._poll_timer.timeout.connect(self._poll_queues)
        self._poll_timer.start()

        QTimer.singleShot(0, self._deferred_load)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        top_bar = QHBoxLayout()
        root.addLayout(top_bar)

        top_bar.addWidget(QLabel("檔案", self))
        self.cmb_files = QComboBox(self)
        self.cmb_files.currentIndexChanged.connect(self._on_file_selected)
        top_bar.addWidget(self.cmb_files, 1)

        self.btn_prev = QPushButton("上一頁", self)
        self.btn_prev.clicked.connect(self._prev_file)
        top_bar.addWidget(self.btn_prev)

        self.btn_next = QPushButton("下一頁", self)
        self.btn_next.clicked.connect(self._next_file)
        top_bar.addWidget(self.btn_next)

        btn_refresh = QPushButton("重新整理", self)
        btn_refresh.clicked.connect(lambda: self._refresh_files(select_latest=False))
        top_bar.addWidget(btn_refresh)

        btn_copy_sel = QPushButton("複製選取", self)
        btn_copy_sel.clicked.connect(self._copy_selection)
        top_bar.addWidget(btn_copy_sel)

        btn_copy_all = QPushButton("複製全部", self)
        btn_copy_all.clicked.connect(self._copy_all)
        top_bar.addWidget(btn_copy_all)

        self.txt_log = QPlainTextEdit(self)
        self.txt_log.setReadOnly(True)
        root.addWidget(self.txt_log, 1)

    def _deferred_load(self) -> None:
        self._set_loading(True)
        self._refresh_files(select_latest=True)

    def _refresh_files(self, select_latest: bool) -> None:
        self._files = list_log_files()
        self.cmb_files.blockSignals(True)
        self.cmb_files.clear()
        self.cmb_files.addItems([p.name for p in self._files])
        self.cmb_files.blockSignals(False)

        if not self._files:
            self._set_text("尚無日誌檔案。")
            self._update_nav_buttons()
            return

        target: Optional[Path] = None
        if select_latest:
            target = self._files[-1]
        else:
            current = self._current_file
            if current and current in self._files:
                target = current
            else:
                target = self._files[-1]

        if target is not None:
            self._set_current_file(target)
        self._update_nav_buttons()

    def _set_current_file(self, path: Path) -> None:
        self._current_file = path
        self._live_file = get_active_log_file()
        self.cmb_files.blockSignals(True)
        self.cmb_files.setCurrentText(path.name)
        self.cmb_files.blockSignals(False)
        self._load_file_async(path)
        self._update_nav_buttons()

    def _load_file_async(self, path: Path) -> None:
        self._load_seq += 1
        load_id: int = self._load_seq
        self._pending_live_lines = []
        self._set_loading(True)

        def _worker() -> None:
            if not path.exists():
                self._load_result_queue.put((load_id, "日誌檔案不存在。", []))
                return
            lines: list[str] = []
            try:
                with path.open("r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        if load_id != self._load_seq:
                            return
                        lines.append(line.rstrip("\n"))
            except Exception:
                self._load_result_queue.put((load_id, "無法讀取日誌檔案。", []))
                return
            self._load_result_queue.put((load_id, "", lines))

        threading.Thread(target=_worker, daemon=True).start()

    def _finish_loading(self, load_id: int, error_text: str, lines: list[str]) -> None:
        if load_id != self._load_seq:
            return
        if error_text:
            self._set_text(error_text)
            self._set_loading(False)
            return

        full_lines = lines
        if self._pending_live_lines:
            full_lines.extend(self._pending_live_lines)
            self._pending_live_lines = []

        self._set_text("\n".join(full_lines))
        self._scroll_to_end()
        self._set_loading(False)

    def _set_text(self, text: str) -> None:
        self.txt_log.setPlainText(text or "")

    def _append_line(self, line: str) -> None:
        cursor = self.txt_log.textCursor()
        cursor.movePosition(QTextCursor.End)
        if self.txt_log.toPlainText():
            cursor.insertText("\n")
        cursor.insertText(line)
        self.txt_log.setTextCursor(cursor)

    def _set_loading(self, loading: bool) -> None:
        self._loading = loading
        if loading:
            self._set_text("正在載入中")
            self.btn_prev.setEnabled(False)
            self.btn_next.setEnabled(False)
            self.cmb_files.setEnabled(False)
            return
        self.cmb_files.setEnabled(True)
        self._update_nav_buttons()

    def _scroll_to_end(self) -> None:
        cursor = self.txt_log.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.txt_log.setTextCursor(cursor)
        self.txt_log.ensureCursorVisible()

    def _poll_queues(self) -> None:
        try:
            while True:
                load_id, error_text, lines = self._load_result_queue.get_nowait()
                self._finish_loading(load_id, error_text, lines)
        except queue.Empty:
            pass

        try:
            while True:
                line: str = self._log_queue.get_nowait()
                if self._loading and self._current_file and self._live_file and self._current_file == self._live_file:
                    self._pending_live_lines.append(line)
                elif self._current_file and self._live_file and self._current_file == self._live_file:
                    self._append_line(line)
                    self._scroll_to_end()
        except queue.Empty:
            pass

    def _on_file_selected(self, index: int) -> None:
        if index < 0 or index >= len(self._files):
            return
        self._set_current_file(self._files[index])

    def _prev_file(self) -> None:
        if not self._files:
            return
        if self._current_file not in self._files:
            self._set_current_file(self._files[-1])
            return
        idx = self._files.index(self._current_file)
        if idx <= 0:
            return
        self._set_current_file(self._files[idx - 1])

    def _next_file(self) -> None:
        if not self._files:
            return
        if self._current_file not in self._files:
            self._set_current_file(self._files[-1])
            return
        idx = self._files.index(self._current_file)
        if idx >= len(self._files) - 1:
            return
        self._set_current_file(self._files[idx + 1])

    def _update_nav_buttons(self) -> None:
        if not self._files or self._current_file not in self._files:
            self.btn_prev.setEnabled(False)
            self.btn_next.setEnabled(False)
            return
        idx: int = self._files.index(self._current_file)
        self.btn_prev.setEnabled(idx > 0)
        self.btn_next.setEnabled(idx < len(self._files) - 1)

    def _copy_all(self) -> None:
        text: str = self.txt_log.toPlainText()
        self._copy_to_clipboard(text)

    def _copy_selection(self) -> None:
        cursor = self.txt_log.textCursor()
        text = cursor.selectedText()
        if not text:
            QMessageBox.information(self, "複製", "請先選取要複製的文字。")
            return
        self._copy_to_clipboard(text)

    def _copy_to_clipboard(self, text: str) -> None:
        clipboard = QApplication.clipboard()
        if clipboard is None:
            return
        clipboard.setText(text)

    def closeEvent(self, event) -> None:
        if self._on_close:
            try:
                self._on_close()
            except Exception:
                self._log_ctrl.exception("日誌視窗關閉回呼失敗。")
        super().closeEvent(event)
