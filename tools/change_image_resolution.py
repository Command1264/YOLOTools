from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QComboBox,
    QVBoxLayout,
    QWidget,
)


ANCHORS = ["左上", "上中", "右上", "左中", "置中", "右中", "左下", "下中", "右下"]


def get_app_dir() -> str:
    """Resolve application directory safely."""
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except Exception:
        return os.getcwd()


def initial_dir_from(path_str: str) -> str:
    """Resolve best initial directory for file dialogs."""
    app_dir = get_app_dir()
    p = (path_str or "").strip()
    if not p:
        return app_dir
    if os.path.isdir(p):
        return p
    if os.path.isfile(p):
        return os.path.dirname(p) or app_dir
    return app_dir


def cv_imread_unicode(path: str):
    """Read image from unicode path with OpenCV."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def cv_imwrite_unicode(path: str, img_bgr, ext: str, params=None) -> bool:
    """Write image to unicode path with OpenCV."""
    if not ext.startswith("."):
        ext = "." + ext
    ok, buf = cv2.imencode(ext, img_bgr, params if params else [])
    if not ok:
        return False
    try:
        buf.tofile(path)
        return True
    except Exception:
        return False


def is_image_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


def hex_to_bgr(hex_str: str):
    s = hex_str.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) != 6:
        raise ValueError("Hex must be 6 digits")
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return b, g, r


def bgr_to_hex(bgr) -> str:
    b, g, r = bgr
    return f"#{r:02X}{g:02X}{b:02X}"


def ensure_unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 1
    while True:
        cand = f"{base}({i}){ext}"
        if not os.path.exists(cand):
            return cand
        i += 1


def compute_offset(canvas_w: int, canvas_h: int, img_w: int, img_h: int, anchor: str):
    if anchor in ("左上", "左中", "左下"):
        x = 0
    elif anchor in ("上中", "置中", "下中"):
        x = (canvas_w - img_w) // 2
    else:
        x = canvas_w - img_w

    if anchor in ("左上", "上中", "右上"):
        y = 0
    elif anchor in ("左中", "置中", "右中"):
        y = (canvas_h - img_h) // 2
    else:
        y = canvas_h - img_h
    return max(0, x), max(0, y)


def resize_letterbox(img_bgr, target_w: int, target_h: int, fill_bgr=(0, 0, 0), anchor="置中"):
    if img_bgr is None:
        return None
    h, w = img_bgr.shape[:2]
    if w <= 0 or h <= 0 or target_w <= 0 or target_h <= 0:
        return None

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(
        img_bgr,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
    )
    canvas = np.full((target_h, target_w, 3), fill_bgr, dtype=np.uint8)
    x, y = compute_offset(target_w, target_h, new_w, new_h, anchor)
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


class ImageResizerWindow(QMainWindow):
    """Image resizer GUI using PySide6."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Image Resizer (OpenCV) - 等比例縮放 + 補色 + 批次")
        self.resize(1300, 800)
        self.setMinimumSize(1120, 700)

        self.fill_bgr = (0, 0, 0)
        self.worker_thread: Optional[threading.Thread] = None
        self.stop_flag = False

        self._state_lock = threading.Lock()
        self._progress_done = 0
        self._progress_total = 0
        self._current_name = "（尚未開始）"
        self._finish_dialog: Optional[tuple[str, str, str]] = None
        self._preview_source = None

        self._build_ui()
        self._bind_events()

        self._ui_timer = QTimer(self)
        self._ui_timer.setInterval(80)
        self._ui_timer.timeout.connect(self._flush_worker_updates)
        self._ui_timer.start()

        QTimer.singleShot(120, self.update_preview)

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)

        left_wrap = QWidget(root)
        left = QVBoxLayout(left_wrap)
        outer.addWidget(left_wrap, 0)

        in_box = QGroupBox("輸入（圖片或資料夾）", left_wrap)
        in_form = QFormLayout(in_box)
        row_in = QWidget(in_box)
        row_in_l = QHBoxLayout(row_in)
        row_in_l.setContentsMargins(0, 0, 0, 0)
        self.ent_input = QLineEdit(row_in)
        btn_pick_file = QPushButton("選圖片", row_in)
        btn_pick_dir = QPushButton("選資料夾", row_in)
        btn_pick_file.clicked.connect(self.pick_input_file)
        btn_pick_dir.clicked.connect(self.pick_input_dir)
        row_in_l.addWidget(self.ent_input, 1)
        row_in_l.addWidget(btn_pick_file)
        row_in_l.addWidget(btn_pick_dir)
        in_form.addRow(row_in)
        left.addWidget(in_box)

        out_box = QGroupBox("輸出資料夾", left_wrap)
        out_form = QFormLayout(out_box)
        row_out = QWidget(out_box)
        row_out_l = QHBoxLayout(row_out)
        row_out_l.setContentsMargins(0, 0, 0, 0)
        self.ent_output = QLineEdit(row_out)
        btn_pick_out = QPushButton("瀏覽", row_out)
        btn_pick_out.clicked.connect(self.pick_output_dir)
        row_out_l.addWidget(self.ent_output, 1)
        row_out_l.addWidget(btn_pick_out)
        out_form.addRow(row_out)
        left.addWidget(out_box)

        res_box = QGroupBox("目標解析度（寬 x 高）", left_wrap)
        res_form = QFormLayout(res_box)
        row_res = QWidget(res_box)
        row_res_l = QHBoxLayout(row_res)
        row_res_l.setContentsMargins(0, 0, 0, 0)
        self.ent_w = QLineEdit("640", row_res)
        self.ent_h = QLineEdit("640", row_res)
        self.ent_w.setMaximumWidth(100)
        self.ent_h.setMaximumWidth(100)
        row_res_l.addWidget(QLabel("寬", row_res))
        row_res_l.addWidget(self.ent_w)
        row_res_l.addWidget(QLabel("高", row_res))
        row_res_l.addWidget(self.ent_h)
        row_res_l.addStretch(1)
        res_form.addRow(row_res)
        left.addWidget(res_box)

        anchor_box = QGroupBox("定位（貼齊方式）", left_wrap)
        anchor_form = QFormLayout(anchor_box)
        self.cmb_anchor = QComboBox(anchor_box)
        self.cmb_anchor.addItems(ANCHORS)
        self.cmb_anchor.setCurrentText("置中")
        anchor_form.addRow("縮放後圖片貼到畫布的：", self.cmb_anchor)
        left.addWidget(anchor_box)

        color_box = QGroupBox("填滿顏色（剩餘空白）", left_wrap)
        color_form = QFormLayout(color_box)
        row_color = QWidget(color_box)
        row_color_l = QHBoxLayout(row_color)
        row_color_l.setContentsMargins(0, 0, 0, 0)
        self.lbl_color = QLabel(row_color)
        self.lbl_color.setFixedSize(34, 18)
        self.lbl_color.setFrameShape(QLabel.Box)
        self.ent_color_hex = QLineEdit(bgr_to_hex(self.fill_bgr), row_color)
        self.ent_color_hex.setMaximumWidth(100)
        btn_color = QPushButton("調色盤...", row_color)
        btn_color.clicked.connect(self.choose_color_builtin)
        row_color_l.addWidget(self.lbl_color)
        row_color_l.addWidget(self.ent_color_hex)
        row_color_l.addWidget(btn_color)
        row_color_l.addStretch(1)
        color_form.addRow(row_color)
        color_form.addRow(QLabel("色碼格式：#RRGGBB", color_box))
        left.addWidget(color_box)
        self._update_color_swatch_from_hex()

        suffix_box = QGroupBox("檔名後綴（空=不加）", left_wrap)
        suffix_form = QFormLayout(suffix_box)
        self.ent_suffix = QLineEdit("", suffix_box)
        suffix_form.addRow(self.ent_suffix)
        left.addWidget(suffix_box)

        fmt_box = QGroupBox("輸出格式與品質", left_wrap)
        fmt_form = QFormLayout(fmt_box)
        self.cmb_outfmt = QComboBox(fmt_box)
        self.cmb_outfmt.addItems(["保持原格式", "PNG", "JPG", "WEBP"])
        fmt_form.addRow("輸出格式：", self.cmb_outfmt)

        self.ent_jpg_quality = QLineEdit("95", fmt_box)
        self.ent_webp_quality = QLineEdit("90", fmt_box)
        self.ent_png_compress = QLineEdit("3", fmt_box)
        fmt_form.addRow("JPG 品質（0-100）：", self.ent_jpg_quality)
        fmt_form.addRow("WEBP 品質（0-100）：", self.ent_webp_quality)
        fmt_form.addRow("PNG 壓縮（0-9）：", self.ent_png_compress)
        self.chk_overwrite = QCheckBox("同名檔案直接覆蓋", fmt_box)
        fmt_form.addRow(self.chk_overwrite)
        left.addWidget(fmt_box)

        prog_box = QGroupBox("進度", left_wrap)
        prog_form = QFormLayout(prog_box)
        self.lbl_count = QLabel("0 / 0", prog_box)
        self.lbl_current = QLabel("（尚未開始）", prog_box)
        self.progress = QProgressBar(prog_box)
        self.progress.setRange(0, 1)
        prog_form.addRow("已處理/總件數：", self.lbl_count)
        prog_form.addRow("目前檔案：", self.lbl_current)
        prog_form.addRow(self.progress)
        left.addWidget(prog_box)

        row_btn = QWidget(left_wrap)
        row_btn_l = QHBoxLayout(row_btn)
        row_btn_l.setContentsMargins(0, 0, 0, 0)
        self.btn_start = QPushButton("開始處理", row_btn)
        self.btn_start.clicked.connect(self.start)
        self.btn_stop = QPushButton("停止", row_btn)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop)
        row_btn_l.addWidget(self.btn_start, 1)
        row_btn_l.addWidget(self.btn_stop, 1)
        left.addWidget(row_btn)
        left.addStretch(1)

        right_wrap = QWidget(root)
        right = QVBoxLayout(right_wrap)
        outer.addWidget(right_wrap, 1)

        prev_box = QGroupBox("預覽（左：原圖 / 右：處理後）", right_wrap)
        prev_l = QHBoxLayout(prev_box)
        self.lbl_prev_in = QLabel("（尚未選擇有效的圖片或資料夾）", prev_box)
        self.lbl_prev_in.setAlignment(Qt.AlignCenter)
        self.lbl_prev_in.setStyleSheet("background:#222;color:#ddd;")
        self.lbl_prev_out = QLabel("", prev_box)
        self.lbl_prev_out.setAlignment(Qt.AlignCenter)
        self.lbl_prev_out.setStyleSheet("background:#222;color:#ddd;")
        prev_l.addWidget(self.lbl_prev_in, 1)
        prev_l.addWidget(self.lbl_prev_out, 1)
        right.addWidget(prev_box, 1)

        self.lbl_preview_info = QLabel("提示：改參數會更新預覽（取第一張圖）。", right_wrap)
        right.addWidget(self.lbl_preview_info)

    def _bind_events(self) -> None:
        for w in [self.ent_input, self.ent_output, self.ent_w, self.ent_h, self.ent_suffix]:
            w.textChanged.connect(self.schedule_preview_update)
        self.ent_color_hex.textChanged.connect(self.on_color_hex_change)
        self.cmb_anchor.currentTextChanged.connect(self.schedule_preview_update)
        self.cmb_outfmt.currentTextChanged.connect(self.schedule_preview_update)
        self.ent_jpg_quality.textChanged.connect(self.schedule_preview_update)
        self.ent_webp_quality.textChanged.connect(self.schedule_preview_update)
        self.ent_png_compress.textChanged.connect(self.schedule_preview_update)

    def schedule_preview_update(self) -> None:
        QTimer.singleShot(120, self.update_preview)

    def pick_input_file(self) -> None:
        initdir = initial_dir_from(self.ent_input.text())
        path, _ = QFileDialog.getOpenFileName(
            self,
            "選擇圖片",
            initdir,
            "Image (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp);;All (*)",
        )
        if path:
            self.ent_input.setText(path)

    def pick_input_dir(self) -> None:
        initdir = initial_dir_from(self.ent_input.text())
        path = QFileDialog.getExistingDirectory(self, "選擇圖片資料夾", initdir)
        if path:
            self.ent_input.setText(path)

    def pick_output_dir(self) -> None:
        initdir = initial_dir_from(self.ent_output.text())
        path = QFileDialog.getExistingDirectory(self, "選擇輸出資料夾", initdir)
        if path:
            self.ent_output.setText(path)

    def choose_color_builtin(self) -> None:
        color = QColorDialog.getColor()
        if color.isValid():
            self.ent_color_hex.setText(color.name().upper())

    def _update_color_swatch_from_hex(self) -> None:
        hx = self.ent_color_hex.text().strip().upper()
        try:
            bgr = hex_to_bgr(hx)
        except Exception:
            self.lbl_color.setStyleSheet("background:#FFFFFF;")
            return
        self.fill_bgr = bgr
        self.lbl_color.setStyleSheet(f"background:{bgr_to_hex(bgr)};")

    def on_color_hex_change(self) -> None:
        self._update_color_swatch_from_hex()
        self.schedule_preview_update()

    def _get_first_image_path(self, input_path: str) -> Optional[str]:
        if not input_path:
            return None
        if os.path.isfile(input_path) and is_image_file(input_path):
            return input_path
        if os.path.isdir(input_path):
            files = [os.path.join(input_path, fn) for fn in os.listdir(input_path) if is_image_file(fn)]
            files.sort()
            return files[0] if files else None
        return None

    def _set_label_pixmap(self, label: QLabel, pil_img: Image.Image) -> None:
        max_w = max(260, label.width() - 10)
        max_h = max(260, label.height() - 10)
        out = pil_img.copy()
        out.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        arr = np.array(out.convert("RGB"), copy=False)
        h, w = arr.shape[:2]
        qimg = QImage(arr.data, w, h, w * 3, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg))
        self._preview_source = arr  # keep buffer referenced

    def update_preview(self) -> None:
        first = self._get_first_image_path(self.ent_input.text().strip())
        if not first:
            self.lbl_prev_in.setText("（尚未選擇有效的圖片或資料夾）")
            self.lbl_prev_in.setPixmap(QPixmap())
            self.lbl_prev_out.setText("")
            self.lbl_prev_out.setPixmap(QPixmap())
            return

        try:
            tw = int(self.ent_w.text())
            th = int(self.ent_h.text())
            if tw <= 0 or th <= 0:
                return
        except Exception:
            return

        img = cv_imread_unicode(first)
        if img is None:
            self.lbl_prev_in.setText("（無法讀取圖片）")
            self.lbl_prev_out.setPixmap(QPixmap())
            return

        rgb_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self._set_label_pixmap(self.lbl_prev_in, Image.fromarray(rgb_in))
        self.lbl_prev_in.setText("")

        out = resize_letterbox(img, tw, th, fill_bgr=self.fill_bgr, anchor=self.cmb_anchor.currentText())
        rgb_out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        self._set_label_pixmap(self.lbl_prev_out, Image.fromarray(rgb_out))
        self.lbl_prev_out.setText("")
        self.lbl_preview_info.setText(
            f"預覽：{os.path.basename(first)} | 目標 {tw}x{th} | 貼齊 {self.cmb_anchor.currentText()} | 補色 {bgr_to_hex(self.fill_bgr)}"
        )

    def collect_input_paths(self, input_path: str) -> list[str]:
        if os.path.isfile(input_path):
            return [input_path] if is_image_file(input_path) else []
        if os.path.isdir(input_path):
            paths = [os.path.join(input_path, fn) for fn in os.listdir(input_path) if is_image_file(fn)]
            paths.sort()
            return paths
        return []

    def get_output_ext_and_params(self, src_path: str):
        fmt = self.cmb_outfmt.currentText()
        if fmt == "保持原格式":
            ext = os.path.splitext(src_path)[1].lower()
            if ext == ".jpeg":
                ext = ".jpg"
            if ext not in [".jpg", ".png", ".webp", ".bmp", ".tif", ".tiff"]:
                ext = ".png"
            if ext == ".jpg":
                return ext, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.ent_jpg_quality.text() or "95")]
            if ext == ".webp":
                return ext, [int(cv2.IMWRITE_WEBP_QUALITY), int(self.ent_webp_quality.text() or "90")]
            if ext == ".png":
                return ext, [int(cv2.IMWRITE_PNG_COMPRESSION), int(self.ent_png_compress.text() or "3")]
            return ext, []
        if fmt == "PNG":
            return ".png", [int(cv2.IMWRITE_PNG_COMPRESSION), int(self.ent_png_compress.text() or "3")]
        if fmt == "JPG":
            return ".jpg", [int(cv2.IMWRITE_JPEG_QUALITY), int(self.ent_jpg_quality.text() or "95")]
        return ".webp", [int(cv2.IMWRITE_WEBP_QUALITY), int(self.ent_webp_quality.text() or "90")]

    def _set_worker_progress(self, done: int, total: int, current_name: str) -> None:
        with self._state_lock:
            self._progress_done = done
            self._progress_total = total
            self._current_name = current_name

    def _flush_worker_updates(self) -> None:
        with self._state_lock:
            done = self._progress_done
            total = self._progress_total
            current = self._current_name
            finish_dialog = self._finish_dialog
            if finish_dialog is not None:
                self._finish_dialog = None
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)
        self.lbl_count.setText(f"{done} / {total}")
        self.lbl_current.setText(current)

        if finish_dialog is not None:
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            level, title, msg = finish_dialog
            if level == "error":
                QMessageBox.critical(self, title, msg)
            elif level == "warn":
                QMessageBox.warning(self, title, msg)
            else:
                QMessageBox.information(self, title, msg)

    def start(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            QMessageBox.information(self, "進行中", "目前正在處理中。")
            return

        input_path = self.ent_input.text().strip()
        output_dir = self.ent_output.text().strip()
        if not input_path:
            QMessageBox.warning(self, "缺少輸入", "請選擇輸入圖片或資料夾。")
            return
        if not output_dir:
            QMessageBox.warning(self, "缺少輸出", "請選擇輸出資料夾。")
            return
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            QMessageBox.critical(self, "建立輸出資料夾失敗", str(exc))
            return

        try:
            tw = int(self.ent_w.text())
            th = int(self.ent_h.text())
            if tw <= 0 or th <= 0:
                raise ValueError
        except Exception:
            QMessageBox.warning(self, "解析度錯誤", "請輸入正整數的寬與高。")
            return

        try:
            _ = hex_to_bgr(self.ent_color_hex.text().strip())
        except Exception:
            QMessageBox.warning(self, "顏色錯誤", "填滿顏色請輸入正確的 #RRGGBB。")
            return

        paths = self.collect_input_paths(input_path)
        if not paths:
            QMessageBox.warning(self, "沒有圖片", "找不到支援的圖片檔。")
            return

        suffix = self.ent_suffix.text()
        anchor = self.cmb_anchor.currentText()
        overwrite = self.chk_overwrite.isChecked()
        fill_bgr = self.fill_bgr

        self.stop_flag = False
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._set_worker_progress(0, len(paths), "（準備開始...）")

        self.worker_thread = threading.Thread(
            target=self._worker,
            args=(paths, output_dir, tw, th, suffix, fill_bgr, anchor, overwrite),
            daemon=True,
        )
        self.worker_thread.start()

    def stop(self) -> None:
        self.stop_flag = True

    def _worker(
        self,
        paths: list[str],
        output_dir: str,
        tw: int,
        th: int,
        suffix: str,
        fill_bgr,
        anchor: str,
        overwrite: bool,
    ) -> None:
        total = len(paths)
        done = 0
        failed = 0
        for src_path in paths:
            if self.stop_flag:
                break

            base = os.path.basename(src_path)
            self._set_worker_progress(done, total, base)

            img = cv_imread_unicode(src_path)
            if img is None:
                failed += 1
                done += 1
                self._set_worker_progress(done, total, f"{base}（讀取失敗）")
                continue

            out = resize_letterbox(img, tw, th, fill_bgr=fill_bgr, anchor=anchor)
            if out is None:
                failed += 1
                done += 1
                self._set_worker_progress(done, total, f"{base}（處理失敗）")
                continue

            ext, params = self.get_output_ext_and_params(src_path)
            name_no_ext = os.path.splitext(base)[0]
            new_name = f"{name_no_ext}{suffix}{ext}" if suffix else f"{name_no_ext}{ext}"
            save_path = os.path.join(output_dir, new_name)
            if not overwrite:
                save_path = ensure_unique_path(save_path)

            ok = cv_imwrite_unicode(save_path, out, ext=ext, params=params)
            if not ok:
                failed += 1

            done += 1
            self._set_worker_progress(done, total, base)

        if self.stop_flag:
            msg = f"已停止處理：{done}/{total}，失敗 {failed}。"
            self._set_worker_progress(done, total, "（已停止）")
            with self._state_lock:
                self._finish_dialog = ("info", "已停止", msg)
            return

        msg = f"處理完成：{done}/{total}，失敗 {failed}。\n輸出：{output_dir}"
        self._set_worker_progress(done, total, "（完成）")
        with self._state_lock:
            self._finish_dialog = ("info", "完成", msg)


def main() -> None:
    app = QApplication([])
    win = ImageResizerWindow()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
