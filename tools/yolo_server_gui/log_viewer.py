from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from log_manager import (
    get_current_gui_log_sequence,
    LogController,
    get_active_log_file,
    get_recent_gui_logs_since,
    get_logger,
    list_log_files,
    register_log_subscriber,
    sanitize_log_text,
    unregister_log_subscriber,
)

LIVE_APPEND_BATCH_SIZE = 200
MAX_VISIBLE_LOG_LINES = 3000
AUTO_REFRESH_INTERVAL_TICKS = 10


class LogViewer(QDialog):
    """提供 GUI 方式瀏覽與即時顯示運行日誌（PySide6 版本）。"""

    def __init__(self, parent: QWidget, on_close: Optional[Callable[[], None]] = None) -> None:
        super().__init__(parent)
        self._on_close: Optional[Callable[[], None]] = on_close
        self._log_ctrl: LogController = LogController(get_logger())
        self._live_queue: Optional[queue.SimpleQueue[tuple[int, str]]] = None
        self._files: list[Path] = []
        self._current_file: Optional[Path] = None
        self._live_file: Optional[Path] = get_active_log_file()
        self._loading: bool = False
        self._load_seq: int = 0
        self._load_result_queue: queue.Queue[tuple[int, str, list[str]]] = queue.Queue()
        self._auto_refresh_ticks: int = 0
        self._is_disposed: bool = False
        self._pending_live_lines: list[str] = []
        self._visible_line_count: int = 0
        self._loaded_lines: list[str] = []
        self._load_start_sequence: int = 0

        self.setWindowTitle("運行日誌")
        self.resize(820, 520)
        self.setMinimumSize(720, 420)
        self.setModal(True)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        self._build_ui()

        self._poll_timer: QTimer = QTimer(self)
        self._poll_timer.setInterval(500)
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
        self.txt_log.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.txt_log.setStyleSheet(
            "QPlainTextEdit { font-family: Consolas, 'Courier New', monospace; font-size: 10pt; }"
        )
        self.txt_log.document().setMaximumBlockCount(MAX_VISIBLE_LOG_LINES)
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
        if self._is_disposed:
            return
        if self._live_queue is not None:
            unregister_log_subscriber(self._live_queue)
            self._live_queue = None
        self._current_file = path
        self._live_file = get_active_log_file()
        self._pending_live_lines.clear()
        self._loaded_lines = []
        self._load_start_sequence = get_current_gui_log_sequence()
        self.cmb_files.blockSignals(True)
        self.cmb_files.setCurrentText(path.name)
        self.cmb_files.blockSignals(False)
        self._load_file_async(path)
        self._update_nav_buttons()

    def _load_file_async(self, path: Path) -> None:
        if self._is_disposed:
            return
        self._load_seq += 1
        load_id: int = self._load_seq
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
                        lines.append(sanitize_log_text(line.rstrip("\n")))
            except Exception:
                self._load_result_queue.put((load_id, "無法讀取日誌檔案。", []))
                return
            self._load_result_queue.put((load_id, "", lines))

        threading.Thread(target=_worker, daemon=True).start()

    def _finish_loading(self, load_id: int, error_text: str, lines: list[str]) -> None:
        if self._is_disposed:
            return
        if load_id != self._load_seq:
            return
        if error_text:
            self._set_text(error_text)
            self._set_loading(False)
            return

        self._set_lines(lines)
        self._loaded_lines = list(lines)
        if self._is_following_live_file():
            self._attach_live_subscription()
            self._pending_live_lines.extend(self._get_recent_live_lines_since_load())
        if self._is_following_live_file() and self._pending_live_lines:
            deduped_lines = self._drop_overlapping_prefix(self._dedupe_adjacent_lines(self._pending_live_lines), self._loaded_lines)
            if deduped_lines:
                self._append_lines(deduped_lines)
            self._pending_live_lines.clear()
        self._scroll_to_end()
        self._set_loading(False)

    def _set_text(self, text: str) -> None:
        self._set_lines([text] if text else [])

    def _set_lines(self, lines: list[str]) -> None:
        sanitized_lines = [sanitize_log_text(line) for line in lines]
        self.txt_log.setPlainText("\n".join(sanitized_lines))
        self._visible_line_count = len(sanitized_lines)
        self._loaded_lines = list(sanitized_lines)

    def _append_line(self, line: str) -> None:
        self._append_lines([line])

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
        if self._is_disposed or not self.isVisible():
            return
        self._auto_refresh_ticks += 1
        if self._auto_refresh_ticks >= AUTO_REFRESH_INTERVAL_TICKS:
            self._auto_refresh_ticks = 0
            self._auto_refresh_files()

        try:
            while True:
                load_id, error_text, lines = self._load_result_queue.get_nowait()
                self._finish_loading(load_id, error_text, lines)
        except queue.Empty:
            pass

        self._consume_live_queue()

    def _auto_refresh_files(self) -> None:
        if self._is_disposed:
            return
        prev_live = self._live_file
        was_following_live = self._current_file is not None and self._current_file == prev_live
        self._live_file = get_active_log_file()

        latest_files = list_log_files()
        if latest_files == self._files:
            if was_following_live and self._live_file and self._current_file != self._live_file:
                self._set_current_file(self._live_file)
            return

        self._files = latest_files
        self.cmb_files.blockSignals(True)
        self.cmb_files.clear()
        self.cmb_files.addItems([p.name for p in self._files])
        self.cmb_files.blockSignals(False)

        if not self._files:
            self._set_text("尚無日誌檔案。")
            self._current_file = None
            self._update_nav_buttons()
            return

        target: Optional[Path] = None
        if was_following_live and self._live_file and self._live_file in self._files:
            target = self._live_file
        elif self._current_file in self._files:
            target = self._current_file
        else:
            target = self._files[-1]

        if target is not None and target != self._current_file:
            self._set_current_file(target)
            return

        if target is not None:
            self.cmb_files.blockSignals(True)
            self.cmb_files.setCurrentText(target.name)
            self.cmb_files.blockSignals(False)
        self._update_nav_buttons()

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
        text: str = self._normalize_clipboard_text(self.txt_log.toPlainText())
        self._copy_to_clipboard(text)

    def _copy_selection(self) -> None:
        cursor = self.txt_log.textCursor()
        text = self._normalize_clipboard_text(cursor.selectedText())
        if not text:
            QMessageBox.information(self, "複製", "請先選取要複製的文字。")
            return
        self._copy_to_clipboard(text)

    def _copy_to_clipboard(self, text: str) -> None:
        clipboard = QApplication.clipboard()
        if clipboard is None:
            return
        clipboard.setText(text)

    @staticmethod
    def _normalize_clipboard_text(text: str) -> str:
        normalized = text.replace("\u2029", "\n").replace("\u2028", "\n")
        return sanitize_log_text(normalized)

    def closeEvent(self, event) -> None:
        self._dispose_viewer()
        if self._on_close:
            try:
                self._on_close()
            except Exception:
                self._log_ctrl.exception("日誌視窗關閉回呼失敗。")
        super().closeEvent(event)

    def _dispose_viewer(self) -> None:
        if self._is_disposed:
            return
        self._is_disposed = True
        self._load_seq += 1
        self._current_file = None
        self._live_file = None
        self._files = []
        if self._live_queue is not None:
            unregister_log_subscriber(self._live_queue)
            self._live_queue = None
        if self._poll_timer.isActive():
            self._poll_timer.stop()
        try:
            while True:
                self._load_result_queue.get_nowait()
        except queue.Empty:
            pass

    def _consume_live_queue(self) -> None:
        if self._live_queue is None:
            return
        lines: list[str] = []
        try:
            while len(lines) < LIVE_APPEND_BATCH_SIZE:
                _, line = self._live_queue.get_nowait()
                lines.append(sanitize_log_text(line))
        except queue.Empty:
            pass

        if not lines:
            return

        if not self._is_following_live_file():
            return

        if self._loading:
            self._pending_live_lines.extend(lines)
            return

        self._append_lines(lines)
        self._scroll_to_end()

    def _append_lines(self, lines: list[str]) -> None:
        if not lines:
            return
        sanitized_lines = [sanitize_log_text(line) for line in lines]
        cursor = self.txt_log.textCursor()
        cursor.movePosition(QTextCursor.End)
        if self._visible_line_count > 0:
            cursor.insertText("\n")
        cursor.insertText("\n".join(sanitized_lines))
        self.txt_log.setTextCursor(cursor)
        self._visible_line_count = self.txt_log.blockCount()
        self._loaded_lines.extend(sanitized_lines)
        if len(self._loaded_lines) > MAX_VISIBLE_LOG_LINES:
            self._loaded_lines = self._loaded_lines[-MAX_VISIBLE_LOG_LINES:]

    def _is_following_live_file(self) -> bool:
        return self._current_file is not None and self._live_file is not None and self._current_file == self._live_file

    def _attach_live_subscription(self) -> None:
        if self._live_queue is None:
            self._live_queue = register_log_subscriber()

    def _get_recent_live_lines_since_load(self) -> list[str]:
        recent_lines = [sanitize_log_text(line) for line in get_recent_gui_logs_since(self._load_start_sequence)]
        return self._drop_overlapping_prefix(recent_lines, self._loaded_lines)

    @staticmethod
    def _drop_overlapping_prefix(pending_lines: list[str], loaded_lines: list[str]) -> list[str]:
        if not pending_lines:
            return []
        max_overlap = min(len(pending_lines), len(loaded_lines))
        for overlap in range(max_overlap, 0, -1):
            if loaded_lines[-overlap:] == pending_lines[:overlap]:
                return pending_lines[overlap:]
        return pending_lines

    @staticmethod
    def _dedupe_adjacent_lines(lines: list[str]) -> list[str]:
        if not lines:
            return []
        deduped = [lines[0]]
        for line in lines[1:]:
            if line != deduped[-1]:
                deduped.append(line)
        return deduped
