from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from build_worker import BuildSettings, BuildWorker
from config_store import ConfigStore
from log_store import LogStore
from ui_advanced import AdvancedDialog


class LogLineNumberArea(QtWidgets.QWidget):
    def __init__(self, parent: "BuildLogTextEdit") -> None:
        super().__init__(parent)
        self._editor = parent

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(self._editor.line_number_area_width(), 0)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        self._editor.paint_line_numbers(event)


class BuildLogTextEdit(QtWidgets.QPlainTextEdit):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._line_area = LogLineNumberArea(self)
        self.blockCountChanged.connect(self._update_line_number_area_width)
        self.updateRequest.connect(self._update_line_number_area)
        self._update_line_number_area_width(0)

    def line_number_area_width(self) -> int:
        digits = max(4, len(str(max(1, self.blockCount()))))
        metrics = QtGui.QFontMetricsF(self.font())
        return int(metrics.horizontalAdvance("9" * digits) + metrics.horizontalAdvance(" | ") + 8)

    def paint_line_numbers(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self._line_area)
        painter.fillRect(event.rect(), self.palette().window())
        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        metrics = QtGui.QFontMetricsF(self.font())
        top = self.blockBoundingGeometry(block).translated(self.contentOffset()).top()
        bottom = top + self.blockBoundingRect(block).height()
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                number = str(block_number + 1).rjust(4, "0")
                text = f"{number} | "
                painter.setPen(self.palette().text().color())
                painter.drawText(
                    0,
                    int(top),
                    self._line_area.width() - 4,
                    int(metrics.height()),
                    QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter,
                    text,
                )
            block = block.next()
            top = bottom
            bottom = top + self.blockBoundingRect(block).height()
            block_number += 1

    def _update_line_number_area_width(self, _new_block_count: int) -> None:
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)

    def _update_line_number_area(self, rect: QtCore.QRect, dy: int) -> None:
        if dy:
            self._line_area.scroll(0, dy)
        else:
            self._line_area.update(0, rect.y(), self._line_area.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self._update_line_number_area_width(0)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        cr = self.contentsRect()
        self._line_area.setGeometry(QtCore.QRect(cr.left(), cr.top(), self.line_number_area_width(), cr.height()))


class BuildLogDialog(QtWidgets.QDialog):
    stop_requested = QtCore.Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("打包輸出")
        self.setModal(False)
        self.resize(700, 420)
        self.text = BuildLogTextEdit()
        self.text.setReadOnly(True)
        self.text.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        font = self.text.font()
        font.setFamily("Consolas")
        self.text.setFont(font)
        self._line_no = 0
        self._line_prefix_width = 0
        self.chk_wrap = QtWidgets.QCheckBox("自動換行")
        self.chk_wrap.toggled.connect(self._toggle_wrap)
        self.input = QtWidgets.QLineEdit()
        self.input.setPlaceholderText("輸入指令（例如: y）")
        self.btn_send = QtWidgets.QPushButton("送出")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.chk_wrap)
        row.addStretch(1)
        row.addWidget(self.input, 1)
        row.addWidget(self.btn_send)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.text, 1)
        layout.addLayout(row)
        self.btn_send.clicked.connect(self._emit_send)
        self.input.returnPressed.connect(self._emit_send)
        self.on_send = None

    def append_line(self, line: str) -> None:
        self._append_ansi(line)

    def clear(self) -> None:
        self.text.clear()
        self._line_no = 0

    def _toggle_wrap(self, enabled: bool) -> None:
        mode = QtWidgets.QPlainTextEdit.WidgetWidth if enabled else QtWidgets.QPlainTextEdit.NoWrap
        self.text.setLineWrapMode(mode)
        self._apply_wrap_indent()

    def _apply_wrap_indent(self) -> None:
        option = self.text.document().defaultTextOption()
        option.setWrapMode(
            QtGui.QTextOption.WrapAtWordBoundaryOrAnywhere
            if self.text.lineWrapMode() != QtWidgets.QPlainTextEdit.NoWrap
            else QtGui.QTextOption.NoWrap
        )
        self.text.document().setDefaultTextOption(option)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.stop_requested.emit()
        event.accept()

    def _emit_send(self) -> None:
        if self.on_send is None:
            return
        text = self.input.text().strip()
        if not text:
            return
        self.input.clear()
        self.on_send(text)

    def _append_ansi(self, text: str) -> None:
        h_scroll = self.text.horizontalScrollBar().value()
        cursor = self.text.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        fmt = QtGui.QTextCharFormat()
        fmt.setForeground(self.text.palette().text().color())
        self._line_no += 1
        self._apply_wrap_indent()
        block_fmt = cursor.blockFormat()
        parts = self._split_ansi(text)
        for seg, style in parts:
            if style is not None:
                fmt = style
            cursor.insertText(seg, fmt)
        cursor.insertText("\n")
        self.text.setTextCursor(cursor)
        self.text.ensureCursorVisible()
        self.text.horizontalScrollBar().setValue(h_scroll)

    def _split_ansi(self, text: str):
        import re

        ansi_re = re.compile(r"\x1b\[([0-9;]*)m")
        pos = 0
        current = None
        for m in ansi_re.finditer(text):
            if m.start() > pos:
                yield text[pos : m.start()], current
            codes = m.group(1).split(";") if m.group(1) else ["0"]
            current = self._ansi_to_format(codes)
            pos = m.end()
        if pos < len(text):
            yield text[pos:], current

    def _ansi_to_format(self, codes):
        fmt = QtGui.QTextCharFormat()
        fmt.setForeground(self.text.palette().text().color())
        bold = False
        for c in codes:
            try:
                code = int(c)
            except Exception:
                continue
            if code == 0:
                return fmt
            if code == 1:
                bold = True
            if 30 <= code <= 37:
                fmt.setForeground(self._ansi_color(code - 30))
            if 90 <= code <= 97:
                fmt.setForeground(self._ansi_color(code - 90, bright=True))
        if bold:
            fmt.setFontWeight(QtGui.QFont.Bold)
        return fmt

    def _ansi_color(self, idx: int, bright: bool = False):
        base = [
            (0, 0, 0),
            (170, 0, 0),
            (0, 170, 0),
            (170, 85, 0),
            (0, 0, 170),
            (170, 0, 170),
            (0, 170, 170),
            (170, 170, 170),
        ]
        bright_base = [
            (85, 85, 85),
            (255, 85, 85),
            (85, 255, 85),
            (255, 255, 85),
            (85, 85, 255),
            (255, 85, 255),
            (85, 255, 255),
            (255, 255, 255),
        ]
        r, g, b = (bright_base if bright else base)[max(0, min(idx, 7))]
        return QtGui.QColor(r, g, b)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, exec_dir: Path, log_store: LogStore, config_store: ConfigStore) -> None:
        super().__init__()
        self.exec_dir = exec_dir
        self.log_store = log_store
        self.config_store = config_store
        self.worker: Optional[BuildWorker] = None
        self.build_log = BuildLogDialog(self)
        self.advanced = AdvancedDialog(self)

        self.setWindowTitle("PyInstaller 打包工具")
        self.resize(840, 620)

        self._build_ui()
        self._refresh_config_list()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.cmb_config = QtWidgets.QComboBox()
        self.cmb_config.currentIndexChanged.connect(self._load_selected_config)
        btn_save_cfg = QtWidgets.QPushButton("儲存設定")
        btn_save_cfg.clicked.connect(self._save_config)
        btn_refresh_cfg = QtWidgets.QPushButton("重新整理")
        btn_refresh_cfg.clicked.connect(self._refresh_config_list)

        cfg_row = QtWidgets.QHBoxLayout()
        cfg_row.addWidget(QtWidgets.QLabel("設定檔"))
        cfg_row.addWidget(self.cmb_config, 1)
        cfg_row.addWidget(btn_save_cfg)
        cfg_row.addWidget(btn_refresh_cfg)
        layout.addLayout(cfg_row)

        form = QtWidgets.QFormLayout()

        self.cmb_packager = QtWidgets.QComboBox()
        self.cmb_packager.addItems(["pyinstaller", "nuitka"])
        self.cmb_packager.currentIndexChanged.connect(self._on_packager_changed)
        form.addRow("打包工具", self.cmb_packager)

        self.txt_script = QtWidgets.QLineEdit()
        btn_script = QtWidgets.QPushButton("選擇")
        btn_script.clicked.connect(self._pick_script)
        form.addRow("入口腳本", self._with_btn(self.txt_script, btn_script))

        self.txt_name = QtWidgets.QLineEdit()
        form.addRow("程式名稱", self.txt_name)

        self.cmb_bundle = QtWidgets.QComboBox()
        self.cmb_bundle.addItem("資料夾 (onedir)", "onedir")
        self.cmb_bundle.addItem("單一檔案 (onefile)", "onefile")
        self.chk_noconsole = QtWidgets.QCheckBox("不顯示 Console")
        self.chk_clean = QtWidgets.QCheckBox("清理 (--clean)")
        opts_row = QtWidgets.QHBoxLayout()
        opts_row.addWidget(QtWidgets.QLabel("打包模式"))
        opts_row.addWidget(self.cmb_bundle)
        opts_row.addWidget(self.chk_noconsole)
        opts_row.addWidget(self.chk_clean)
        opts_row.addStretch(1)
        form.addRow("常用選項", opts_row)

        self.txt_icon = QtWidgets.QLineEdit()
        btn_icon = QtWidgets.QPushButton("選擇")
        btn_icon.clicked.connect(self._pick_icon)
        self.lbl_icon = QtWidgets.QLabel("Icon")
        form.addRow(self.lbl_icon, self._with_btn(self.txt_icon, btn_icon))

        self.txt_add_data = QtWidgets.QPlainTextEdit()
        self.txt_add_data.setPlaceholderText("每行一個，格式：來源|目標")
        self.lbl_add_data = QtWidgets.QLabel("附加資料檔")
        self.lbl_add_data_hint = QtWidgets.QLabel("")
        self.lbl_add_data_hint.setAlignment(QtCore.Qt.AlignLeft)
        add_label = self._stack_label(self.lbl_add_data, self.lbl_add_data_hint)
        form.addRow(add_label, self.txt_add_data)

        self.txt_hidden = QtWidgets.QPlainTextEdit()
        self.txt_hidden.setPlaceholderText("每行一個")
        self.lbl_hidden = QtWidgets.QLabel("隱藏匯入")
        self.lbl_hidden_hint = QtWidgets.QLabel("")
        self.lbl_hidden_hint.setAlignment(QtCore.Qt.AlignLeft)
        hidden_label = self._stack_label(self.lbl_hidden, self.lbl_hidden_hint)
        form.addRow(hidden_label, self.txt_hidden)

        self.txt_build = QtWidgets.QLineEdit(str(self.exec_dir / "build"))
        btn_build = QtWidgets.QPushButton("選擇")
        btn_build.clicked.connect(lambda: self._pick_dir(self.txt_build))
        form.addRow("Build 路徑", self._with_btn(self.txt_build, btn_build))

        self.txt_dist = QtWidgets.QLineEdit(str(self.exec_dir / "dist"))
        btn_dist = QtWidgets.QPushButton("選擇")
        btn_dist.clicked.connect(lambda: self._pick_dir(self.txt_dist))
        form.addRow("Dist 路徑", self._with_btn(self.txt_dist, btn_dist))

        self.txt_spec = QtWidgets.QLineEdit(str(self.exec_dir / "spec"))
        btn_spec = QtWidgets.QPushButton("選擇")
        btn_spec.clicked.connect(lambda: self._pick_dir(self.txt_spec))
        form.addRow("Spec 路徑", self._with_btn(self.txt_spec, btn_spec))

        layout.addLayout(form)

        btns = QtWidgets.QHBoxLayout()
        self.btn_build = QtWidgets.QPushButton("開始打包")
        self.btn_build.clicked.connect(self._on_build_clicked)
        self.btn_advanced = QtWidgets.QPushButton("進階參數")
        self.btn_advanced.clicked.connect(self._open_advanced)
        btns.addWidget(self.btn_build)
        btns.addWidget(self.btn_advanced)
        btns.addStretch(1)
        layout.addLayout(btns)

        self.status = QtWidgets.QLabel("")
        layout.addWidget(self.status)

        self._on_packager_changed()

    def _with_btn(self, widget: QtWidgets.QWidget, btn: QtWidgets.QPushButton) -> QtWidgets.QWidget:
        wrap = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(wrap)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(widget, 1)
        row.addWidget(btn)
        return wrap

    def _stack_label(self, title: QtWidgets.QLabel, hint: QtWidgets.QLabel) -> QtWidgets.QWidget:
        wrap = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(wrap)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)
        col.addWidget(title)
        col.addWidget(hint)
        return wrap

    def _pick_script(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "選擇入口腳本", str(self.exec_dir), "Python (*.py)")
        if path:
            self.txt_script.setText(path)

    def _pick_icon(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "選擇 Icon", str(self.exec_dir), "Icon (*.ico)")
        if path:
            self.txt_icon.setText(path)

    def _pick_dir(self, target: QtWidgets.QLineEdit) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "選擇資料夾", target.text() or str(self.exec_dir))
        if path:
            target.setText(path)

    def _open_advanced(self) -> None:
        self.advanced.set_packager(self.cmb_packager.currentText())
        if self.advanced.exec() == QtWidgets.QDialog.Accepted:
            pass

    def _on_build_clicked(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.btn_build.setEnabled(False)
            return

        settings = self._collect_settings()
        if not settings.script_path:
            QtWidgets.QMessageBox.warning(self, "缺少腳本", "請選擇入口腳本")
            return

        self.build_log.clear()
        self.build_log.show()
        self.btn_build.setText("停止")
        self.btn_build.setEnabled(True)
        self.btn_advanced.setEnabled(False)
        self.status.setText("打包中...")

        self.worker = BuildWorker(settings)
        self.worker.output_line.connect(self._on_build_line)
        self.worker.finished_ok.connect(self._on_build_finished)
        self.worker.start()
        self.build_log.on_send = self._send_build_input
        self.build_log.stop_requested.connect(self._stop_build_from_dialog)

    def _on_build_line(self, line: str) -> None:
        self.build_log.append_line(line)
        self.log_store.logger.info(line)

    def _on_build_finished(self, ok: bool) -> None:
        self.btn_build.setText("開始打包")
        self.btn_build.setEnabled(True)
        self.btn_advanced.setEnabled(True)
        self.status.setText("完成" if ok else "失敗或停止")
        self.build_log.on_send = None
        try:
            self.build_log.stop_requested.disconnect(self._stop_build_from_dialog)
        except Exception:
            pass

    def _send_build_input(self, text: str) -> None:
        if self.worker:
            self.worker.send_input(text)

    def _stop_build_from_dialog(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.stop()

    def _collect_settings(self) -> BuildSettings:
        add_data = self.txt_add_data.toPlainText().splitlines()
        hidden = self.txt_hidden.toPlainText().splitlines()
        return BuildSettings(
            packager=self.cmb_packager.currentText(),
            script_path=self.txt_script.text().strip(),
            name=self.txt_name.text().strip(),
            onefile=self.cmb_bundle.currentData() == "onefile",
            no_console=self.chk_noconsole.isChecked(),
            icon_path=self.txt_icon.text().strip(),
            add_data=add_data,
            hidden_imports=hidden,
            work_path=self.txt_build.text().strip(),
            dist_path=self.txt_dist.text().strip(),
            spec_path=self.txt_spec.text().strip(),
            clean=self.chk_clean.isChecked(),
            extra_args=self.advanced.get_extra_args(),
            strip=bool(self.advanced.to_dict().get("strip", False)),
            upx=bool(self.advanced.to_dict().get("upx", False)),
            upx_dir=str(self.advanced.to_dict().get("upx_dir", "")),
            debug=bool(self.advanced.to_dict().get("debug", False)),
            runtime_tmp=str(self.advanced.to_dict().get("runtime_tmp", "")),
            lto=bool(self.advanced.to_dict().get("lto", False)),
            plugins=str(self.advanced.to_dict().get("plugins", "")),
        )

    def _refresh_config_list(self) -> None:
        self.cmb_config.blockSignals(True)
        self.cmb_config.clear()
        self.cmb_config.addItem("")
        for name in self.config_store.list_names():
            self.cmb_config.addItem(name)
        self.cmb_config.blockSignals(False)

    def _save_config(self) -> None:
        name, ok = QtWidgets.QInputDialog.getText(self, "儲存設定", "名稱")
        if not ok or not name.strip():
            return
        data = self._settings_to_dict()
        self.config_store.save(name.strip(), data)
        self._refresh_config_list()
        idx = self.cmb_config.findText(name.strip())
        if idx >= 0:
            self.cmb_config.setCurrentIndex(idx)

    def _load_selected_config(self) -> None:
        name = self.cmb_config.currentText().strip()
        if not name:
            return
        data = self.config_store.load(name)
        if data:
            self._apply_settings(data)
        if "bundle_mode" not in data and "onefile" in data:
            self.cmb_bundle.setCurrentIndex(1 if data.get("onefile") else 0)

    def _settings_to_dict(self) -> Dict[str, object]:
        return {
            "packager": self.cmb_packager.currentText(),
            "script_path": self.txt_script.text().strip(),
            "name": self.txt_name.text().strip(),
            "bundle_mode": self.cmb_bundle.currentData(),
            "noconsole": self.chk_noconsole.isChecked(),
            "clean": self.chk_clean.isChecked(),
            "icon_path": self.txt_icon.text().strip(),
            "add_data": self.txt_add_data.toPlainText(),
            "hidden_imports": self.txt_hidden.toPlainText(),
            "work_path": self.txt_build.text().strip(),
            "dist_path": self.txt_dist.text().strip(),
            "spec_path": self.txt_spec.text().strip(),
            **self.advanced.to_dict(),
        }

    def _apply_settings(self, data: Dict[str, object]) -> None:
        packager = str(data.get("packager", "pyinstaller")) or "pyinstaller"
        idx = self.cmb_packager.findText(packager)
        if idx >= 0:
            self.cmb_packager.setCurrentIndex(idx)
        self.txt_script.setText(str(data.get("script_path", "")))
        self.txt_name.setText(str(data.get("name", "")))
        bundle = str(data.get("bundle_mode", "onedir"))
        idx = self.cmb_bundle.findData(bundle)
        if idx >= 0:
            self.cmb_bundle.setCurrentIndex(idx)
        self.chk_noconsole.setChecked(bool(data.get("noconsole", False)))
        self.chk_clean.setChecked(bool(data.get("clean", False)))
        self.txt_icon.setText(str(data.get("icon_path", "")))
        self.txt_add_data.setPlainText(str(data.get("add_data", "")))
        self.txt_hidden.setPlainText(str(data.get("hidden_imports", "")))
        self.txt_build.setText(str(data.get("work_path", "")) or str(self.exec_dir / "build"))
        self.txt_dist.setText(str(data.get("dist_path", "")) or str(self.exec_dir / "dist"))
        self.txt_spec.setText(str(data.get("spec_path", "")) or str(self.exec_dir / "spec"))
        self.advanced.apply_dict(data)

    def _on_packager_changed(self) -> None:
        is_py = self.cmb_packager.currentText() == "pyinstaller"
        self.txt_build.setEnabled(is_py)
        self.txt_spec.setEnabled(is_py)
        self.chk_clean.setEnabled(is_py)
        self._update_packager_hints(is_py)

    def _update_packager_hints(self, is_py: bool) -> None:
        if is_py:
            self.lbl_add_data_hint.setText("--add-data")
            self.txt_add_data.setPlaceholderText("每行一個，格式：來源|目標（PyInstaller 會轉成 ;）")
            self.lbl_hidden_hint.setText("--hidden-import")
            self.txt_hidden.setPlaceholderText("每行一個 module")
            self.lbl_icon.setText("Icon")
            self.chk_noconsole.setText("不顯示 Console")
        else:
            self.lbl_add_data_hint.setText("--include-data-file")
            self.txt_add_data.setPlaceholderText("每行一個，格式：來源|目標（Nuitka 會轉成 ;）")
            self.lbl_hidden_hint.setText("--include-module")
            self.txt_hidden.setPlaceholderText("每行一個 module/package")
            self.lbl_icon.setText("Icon")
            self.chk_noconsole.setText("不顯示 Console")
