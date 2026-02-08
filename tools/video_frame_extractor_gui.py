from __future__ import annotations

import os
import re
import threading
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
from PIL import Image, ImageOps
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)


def _get_lanczos():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    return Image.LANCZOS


LANCZOS = _get_lanczos()


def sanitize_filename_part(text: str, fallback: str = "LINE") -> str:
    """Sanitize text for safe file-name usage on Windows."""
    if text is None:
        return fallback
    text = unicodedata.normalize("NFKC", str(text))
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    text = re.sub(r'[\\/:*?"<>|]', "_", text)
    text = text.strip().rstrip(" .")
    return text if text else fallback


class VideoFrameExtractorWindow(QMainWindow):
    """Video frame extractor GUI based on PySide6."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Video Frame Extractor (OpenCV)")
        self.resize(960, 680)
        self.setMinimumSize(820, 560)

        self.worker_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.is_running = False
        self.ended_naturally = False

        self.total_frames = 0
        self.current_frame_idx = 0
        self.saved_count = 0
        self.overlay_until_ts = 0.0

        self._latest_frame_bgr = None
        self._last_frame_for_redraw = None
        self._latest_saved_flag = False
        self._latest_frame_lock = threading.Lock()

        self._build_ui()

        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(30)
        self._tick_timer.timeout.connect(self._ui_tick)
        self._tick_timer.start()

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        form = QFormLayout()
        layout.addLayout(form)

        self.ent_video = QLineEdit(self)
        row_video = QWidget(self)
        row_video_l = QHBoxLayout(row_video)
        row_video_l.setContentsMargins(0, 0, 0, 0)
        row_video_l.addWidget(self.ent_video, 1)
        btn_pick_video = QPushButton("選擇...", row_video)
        btn_pick_video.clicked.connect(self.pick_video)
        row_video_l.addWidget(btn_pick_video)
        form.addRow("影片檔：", row_video)

        self.ent_output = QLineEdit(str(Path("./output_frames").resolve()), self)
        row_output = QWidget(self)
        row_output_l = QHBoxLayout(row_output)
        row_output_l.setContentsMargins(0, 0, 0, 0)
        row_output_l.addWidget(self.ent_output, 1)
        btn_pick_output = QPushButton("選擇...", row_output)
        btn_pick_output.clicked.connect(self.pick_output_dir)
        row_output_l.addWidget(btn_pick_output)
        form.addRow("輸出資料夾：", row_output)

        self.ent_interval = QLineEdit("1.0", self)
        form.addRow("每隔 n 秒擷取（0=每幀）：", self.ent_interval)

        self.ent_line_id = QLineEdit("LINE01", self)
        form.addRow("流水線編號：", self.ent_line_id)

        row_action = QWidget(self)
        row_action_l = QHBoxLayout(row_action)
        row_action_l.setContentsMargins(0, 0, 0, 0)
        self.btn_toggle = QPushButton("開始", row_action)
        self.btn_toggle.clicked.connect(self.toggle_start_stop)
        row_action_l.addWidget(self.btn_toggle)
        self.lbl_status = QLabel("狀態：未開始", row_action)
        self.lbl_saved_count = QLabel("已擷取：0 張", row_action)
        row_action_l.addWidget(self.lbl_status, 1)
        row_action_l.addWidget(self.lbl_saved_count)
        layout.addWidget(row_action)

        row_progress = QWidget(self)
        row_progress_l = QHBoxLayout(row_progress)
        row_progress_l.setContentsMargins(0, 0, 0, 0)
        row_progress_l.addWidget(QLabel("進度：", row_progress))
        self.progress_bar = QProgressBar(row_progress)
        self.progress_bar.setRange(0, 100)
        row_progress_l.addWidget(self.progress_bar, 1)
        layout.addWidget(row_progress)

        self.preview_label = QLabel(self)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("background: black;")
        self.preview_label.setMinimumHeight(320)
        layout.addWidget(self.preview_label, 1)

    def pick_video(self) -> None:
        current = self.ent_video.text().strip()
        initial = str(Path(current).parent) if current and Path(current).exists() else str(Path.cwd())
        path, _ = QFileDialog.getOpenFileName(
            self,
            "選擇影片檔",
            initial,
            "Video Files (*.mp4 *.avi *.mov *.mkv *.m4v *.wmv);;All Files (*.*)",
        )
        if path:
            self.ent_video.setText(path)

    def pick_output_dir(self) -> None:
        current = self.ent_output.text().strip()
        initial = current if current and Path(current).exists() else str(Path.cwd())
        path = QFileDialog.getExistingDirectory(self, "選擇輸出資料夾", initial)
        if path:
            self.ent_output.setText(path)
            Path(path).mkdir(parents=True, exist_ok=True)

    def toggle_start_stop(self) -> None:
        if self.is_running:
            self.stop()
            return
        self.start()

    def start(self) -> None:
        video_path = self.ent_video.text().strip()
        if not video_path or not Path(video_path).is_file():
            QMessageBox.critical(self, "錯誤", "請先選擇有效的影片檔。")
            return

        out_dir = self.ent_output.text().strip()
        if not out_dir:
            QMessageBox.critical(self, "錯誤", "請設定輸出資料夾。")
            return
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        try:
            interval = float(self.ent_interval.text().strip())
            if interval < 0:
                raise ValueError
        except Exception:
            QMessageBox.critical(self, "錯誤", "n 秒必須是 >= 0 的數字（可小於 1）。")
            return

        line_id_raw = self.ent_line_id.text().strip()
        if not line_id_raw:
            QMessageBox.critical(self, "錯誤", "請填寫流水線編號。")
            return
        safe_line_id = sanitize_filename_part(line_id_raw, fallback="LINE01")
        if safe_line_id != line_id_raw:
            self.ent_line_id.setText(safe_line_id)
            QMessageBox.warning(
                self,
                "流水線編號已自動修正",
                "流水線編號含有檔名不允許字元，已自動修正。",
            )

        self.saved_count = 0
        self.current_frame_idx = 0
        self.total_frames = 0
        self.ended_naturally = False
        self.lbl_saved_count.setText("已擷取：0 張")
        self.progress_bar.setValue(0)
        self.progress_bar.setRange(0, 100)

        with self._latest_frame_lock:
            self._latest_frame_bgr = None
            self._last_frame_for_redraw = None
            self._latest_saved_flag = False

        self.stop_event.clear()
        self.is_running = True
        self.btn_toggle.setText("停止")
        self.lbl_status.setText("狀態：擷取中...")

        self.worker_thread = threading.Thread(
            target=self._worker_extract,
            args=(video_path, out_dir, interval, safe_line_id),
            daemon=True,
        )
        self.worker_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.lbl_status.setText("狀態：停止中...")
        self.btn_toggle.setEnabled(False)

    def _on_worker_finished(self) -> None:
        def success():
            self.lbl_status.setText("狀態：已完成")
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(100)
            QMessageBox.information(self, "擷取完成", "已完成擷取")

        self.is_running = False
        self.btn_toggle.setEnabled(True)
        self.btn_toggle.setText("開始")
        if self.stop_event.is_set():
            self.lbl_status.setText("狀態：已停止")
        else:
            success()

    def _ui_tick(self) -> None:
        if self.worker_thread and (not self.worker_thread.is_alive()) and self.is_running:
            self._on_worker_finished()

        frame = None
        saved_flag = False
        with self._latest_frame_lock:
            if self._latest_frame_bgr is not None:
                frame = self._latest_frame_bgr.copy()
                self._latest_frame_bgr = None
            saved_flag = self._latest_saved_flag
            self._latest_saved_flag = False

        if frame is not None:
            with self._latest_frame_lock:
                self._last_frame_for_redraw = frame.copy()
            self._render_preview(frame, saved_flag)
        elif self._last_frame_for_redraw is not None and self.preview_label.width() > 0:
            self._render_preview(self._last_frame_for_redraw, False)

        if self.is_running:
            if self.total_frames > 0:
                pct = int(max(0.0, min(100.0, (self.current_frame_idx / self.total_frames) * 100.0)))
                self.progress_bar.setRange(0, 100)
                self.progress_bar.setValue(pct)
            else:
                self.progress_bar.setRange(0, 0)
        else:
            if self.progress_bar.maximum() == 0:
                self.progress_bar.setRange(0, 100)

    def _render_preview(self, frame_bgr, saved_flag: bool) -> None:
        label_w = max(1, self.preview_label.width())
        label_h = max(1, self.preview_label.height())

        box_w = label_w
        box_h = int(round(box_w * 9 / 16))
        if box_h > label_h:
            box_h = label_h
            box_w = int(round(box_h * 16 / 9))
        box_w = max(1, box_w)
        box_h = max(1, box_h)

        box_x = (label_w - box_w) // 2
        box_y = (label_h - box_h) // 2

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame_rgb)
        contained = ImageOps.contain(pil, (box_w, box_h), method=LANCZOS)

        bg = Image.new("RGB", (label_w, label_h), (0, 0, 0))
        inner_x = box_x + (box_w - contained.size[0]) // 2
        inner_y = box_y + (box_h - contained.size[1]) // 2
        bg.paste(contained, (inner_x, inner_y))

        now = time.time()
        if saved_flag:
            self.overlay_until_ts = now + 0.35
        if now < self.overlay_until_ts:
            # draw simple block text with PIL
            from PIL import ImageDraw

            draw = ImageDraw.Draw(bg)
            draw.rectangle((10, 10, 125, 52), fill=(0, 0, 0))
            draw.text((22, 24), "Saved!", fill=(255, 255, 255))

        arr = bg.tobytes("raw", "RGB")
        qimg = QImage(arr, bg.width, bg.height, bg.width * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.preview_label.setPixmap(pix)

    def _worker_extract(self, video_path: str, out_dir: str, interval_sec: float, line_id: str) -> None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.lbl_status.setText("狀態：錯誤（無法開啟影片）")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 30.0
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_step = 1 if interval_sec == 0 else max(1, int(round(interval_sec * fps)))

        frame_idx = 0
        saved_idx = 0
        ended_naturally = False

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                ended_naturally = True
                break

            with self._latest_frame_lock:
                self._latest_frame_bgr = frame

            saved_flag = False
            if frame_idx % frame_step == 0:
                now = datetime.now()
                ts = now.strftime("%Y%m%d_%H%M%S") + f"_{now.microsecond // 1000:03d}"
                saved_idx += 1
                filename = f"{line_id}_{saved_idx:06d}_{ts}.jpg"
                out_path = str(Path(out_dir) / filename)
                ok = cv2.imwrite(out_path, frame)
                if ok:
                    self.saved_count += 1
                    saved_flag = True
                    self.lbl_saved_count.setText(f"已擷取：{self.saved_count} 張")

            if saved_flag:
                with self._latest_frame_lock:
                    self._latest_saved_flag = True

            frame_idx += 1
            self.current_frame_idx = frame_idx
            time.sleep(0.001)

        cap.release()
        self.ended_naturally = ended_naturally and (not self.stop_event.is_set())


def main() -> None:
    app = QApplication([])
    win = VideoFrameExtractorWindow()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
