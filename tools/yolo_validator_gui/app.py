from __future__ import annotations

import base64
import json
import queue
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

import cv2
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

import sys
from pathlib import Path as _Path

APP_DIR = _Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
SERVER_DIR = ROOT_DIR / "yolo_server_gui"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from yolo_engine import YoloEngine


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
DEFAULT_HTTP_URL = "http://127.0.0.1:60922/detect"


class HttpInferError(Exception):
    """HTTP 推論錯誤。"""


class YoloHttpClient:
    """YOLO HTTP 推論用戶端。"""

    def __init__(self, url: str, timeout_sec: float = 8.0) -> None:
        self._url = url
        self._timeout_sec = timeout_sec

    def infer(self, frame_bgr) -> list[dict]:
        try:
            ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                raise HttpInferError("圖片編碼失敗。")
            image_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            payload = {"threadName": "yolo_validator_gui", "image": image_b64}
            raw = json.dumps(payload, ensure_ascii=True).encode("utf-8")
            req = urllib.request.Request(
                self._url,
                data=raw,
                headers={"Content-Type": "application/json; charset=utf-8"},
            )
            with urllib.request.urlopen(req, timeout=self._timeout_sec) as resp:
                data = resp.read()
            decoded = json.loads(data.decode("utf-8"))
            if not isinstance(decoded, dict):
                raise HttpInferError("回傳格式錯誤。")
            result = decoded.get("result")
            if isinstance(result, list):
                result = result[0] if result else {}
            if not isinstance(result, dict):
                raise HttpInferError("回傳內容錯誤。")
            dets = result.get("detections", [])
            if not isinstance(dets, list):
                return []
            return [d for d in dets if isinstance(d, dict)]
        except HttpInferError:
            raise
        except urllib.error.HTTPError as exc:
            raise HttpInferError(f"HTTP 錯誤: {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise HttpInferError(f"連線失敗: {exc.reason}") from exc
        except Exception as exc:
            raise HttpInferError(f"HTTP 推論失敗: {exc}") from exc


class App(QMainWindow):
    """YOLOv26 模型驗證器（PySide6）。"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("YOLOv26 模型驗證器")
        self.resize(1100, 720)
        self.setMinimumSize(980, 640)

        self._config_path = Path(__file__).resolve().parent / "yolo_validator_config.json"
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._ui_token = 0
        self._events: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self._last_frame = None
        self._frame_buf = None
        self._loading_config = False
        self._last_model_dir: Optional[Path] = None
        self._last_input_dir: Optional[Path] = None
        self._order_map = {"圖片優先": "images_first", "影片優先": "videos_first"}
        self._order_map_rev = {v: k for k, v in self._order_map.items()}

        self._build_ui()
        self._load_config()
        self._log("就緒。請選擇模型與圖片/影片。")
        self._try_preview_on_start()

        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._poll_events)
        self._timer.start()

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        path_box = QGroupBox("模型與檔案", root)
        path_form = QFormLayout(path_box)

        row_model = QWidget(path_box)
        row_model_l = QHBoxLayout(row_model)
        row_model_l.setContentsMargins(0, 0, 0, 0)
        self.ent_model = QLineEdit(row_model)
        self.ent_model.editingFinished.connect(self._save_config)
        btn_model = QPushButton("瀏覽...", row_model)
        btn_model.clicked.connect(self.browse_model)
        row_model_l.addWidget(self.ent_model, 1)
        row_model_l.addWidget(btn_model)
        path_form.addRow("YOLO 模型 (.pt):", row_model)

        row_input = QWidget(path_box)
        row_input_l = QHBoxLayout(row_input)
        row_input_l.setContentsMargins(0, 0, 0, 0)
        self.ent_input = QLineEdit(row_input)
        self.ent_input.editingFinished.connect(self._on_input_changed)
        btn_input = QPushButton("瀏覽檔案...", row_input)
        btn_input.clicked.connect(self.browse_input)
        btn_input_dir = QPushButton("瀏覽資料夾...", row_input)
        btn_input_dir.clicked.connect(self.browse_input_dir)
        row_input_l.addWidget(self.ent_input, 1)
        row_input_l.addWidget(btn_input)
        row_input_l.addWidget(btn_input_dir)
        path_form.addRow("圖片或影片:", row_input)
        layout.addWidget(path_box)

        opt_box = QGroupBox("推論設定", root)
        opt_form = QFormLayout(opt_box)

        row_conf = QWidget(opt_box)
        row_conf_l = QHBoxLayout(row_conf)
        row_conf_l.setContentsMargins(0, 0, 0, 0)
        self.sld_conf = QSlider(Qt.Horizontal, row_conf)
        self.sld_conf.setRange(1, 100)
        self.sld_conf.setValue(70)
        self.sld_conf.valueChanged.connect(lambda v: self.ent_conf.setText(f"{v/100:.2f}"))
        self.ent_conf = QLineEdit("0.70", row_conf)
        self.ent_conf.setMaximumWidth(80)
        row_conf_l.addWidget(self.sld_conf, 1)
        row_conf_l.addWidget(self.ent_conf)
        opt_form.addRow("conf:", row_conf)

        row_iou = QWidget(opt_box)
        row_iou_l = QHBoxLayout(row_iou)
        row_iou_l.setContentsMargins(0, 0, 0, 0)
        self.sld_iou = QSlider(Qt.Horizontal, row_iou)
        self.sld_iou.setRange(0, 100)
        self.sld_iou.setValue(45)
        self.sld_iou.valueChanged.connect(lambda v: self.ent_iou.setText(f"{v/100:.2f}"))
        self.ent_iou = QLineEdit("0.45", row_iou)
        self.ent_iou.setMaximumWidth(80)
        row_iou_l.addWidget(self.sld_iou, 1)
        row_iou_l.addWidget(self.ent_iou)
        opt_form.addRow("iou:", row_iou)

        row_more = QWidget(opt_box)
        row_more_l = QHBoxLayout(row_more)
        row_more_l.setContentsMargins(0, 0, 0, 0)
        self.ent_device = QLineEdit("", row_more)
        self.ent_device.setPlaceholderText("空白=auto")
        self.chk_show = QCheckBox("顯示 YOLO 判斷框與標籤", row_more)
        self.chk_show.setChecked(True)
        self.chk_ignore_conf = QCheckBox("忽略 conf", row_more)
        self.chk_ignore_iou = QCheckBox("忽略 iou", row_more)
        self.ent_interval = QLineEdit("1.0", row_more)
        self.ent_interval.setMaximumWidth(80)
        self.cmb_order = QComboBox(row_more)
        self.cmb_order.addItems(["圖片優先", "影片優先"])
        row_more_l.addWidget(QLabel("device:", row_more))
        row_more_l.addWidget(self.ent_device)
        row_more_l.addWidget(self.chk_show)
        row_more_l.addWidget(self.chk_ignore_conf)
        row_more_l.addWidget(self.chk_ignore_iou)
        row_more_l.addWidget(QLabel("間隔(秒):", row_more))
        row_more_l.addWidget(self.ent_interval)
        row_more_l.addWidget(QLabel("順序:", row_more))
        row_more_l.addWidget(self.cmb_order)
        row_more_l.addStretch(1)
        opt_form.addRow(row_more)

        row_http = QWidget(opt_box)
        row_http_l = QHBoxLayout(row_http)
        row_http_l.setContentsMargins(0, 0, 0, 0)
        self.chk_use_http = QCheckBox("使用 HTTP 推論", row_http)
        self.ent_http_url = QLineEdit(DEFAULT_HTTP_URL, row_http)
        row_http_l.addWidget(self.chk_use_http)
        row_http_l.addWidget(QLabel("HTTP URL:", row_http))
        row_http_l.addWidget(self.ent_http_url, 1)
        opt_form.addRow(row_http)
        layout.addWidget(opt_box)

        act = QWidget(root)
        act_l = QHBoxLayout(act)
        act_l.setContentsMargins(0, 0, 0, 0)
        self.btn_start = QPushButton("開始", act)
        self.btn_start.clicked.connect(self.start)
        self.btn_stop = QPushButton("停止", act)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_stop.setEnabled(False)
        self.lbl_status = QLabel("就緒", act)
        self.lbl_device = QLabel("device: -", act)
        act_l.addWidget(self.btn_start)
        act_l.addWidget(self.btn_stop)
        act_l.addWidget(self.lbl_status, 1)
        act_l.addWidget(self.lbl_device)
        layout.addWidget(act)

        splitter = QSplitter(Qt.Vertical, root)
        self.lbl_preview = QLabel("", splitter)
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        self.lbl_preview.setStyleSheet("background:#111;")
        self.log_box = QPlainTextEdit(splitter)
        self.log_box.setReadOnly(True)
        splitter.addWidget(self.lbl_preview)
        splitter.addWidget(self.log_box)
        splitter.setSizes([560, 130])
        layout.addWidget(splitter, 1)

    def _log(self, msg: str) -> None:
        self.log_box.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def _emit(self, kind: str, payload: object) -> None:
        self._events.put((kind, payload))

    def _poll_events(self) -> None:
        while True:
            try:
                kind, payload = self._events.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self._log(str(payload))
            elif kind == "status":
                self.lbl_status.setText(str(payload))
            elif kind == "device":
                self.lbl_device.setText(f"device: {payload}")
            elif kind == "error":
                msg = str(payload)
                self._log(msg)
                QMessageBox.critical(self, "錯誤", msg)
            elif kind == "frame":
                frame, info = payload
                self._last_frame = frame
                self._render_frame(frame)
                self.lbl_status.setText(str(info))
            elif kind == "done":
                stopped = bool(payload)
                self.btn_start.setEnabled(True)
                self.btn_stop.setEnabled(False)
                self.lbl_status.setText("已停止" if stopped else "完成")

    def _render_frame(self, frame_bgr) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        pix = pix.scaled(self.lbl_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._frame_buf = rgb
        self.lbl_preview.setPixmap(pix)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._last_frame is not None:
            self._render_frame(self._last_frame)

    def browse_model(self) -> None:
        initial = self._resolve_initial_dir(self.ent_model.text(), self._last_model_dir)
        p, _ = QFileDialog.getOpenFileName(self, "選擇 YOLO 模型", initial, "YOLO Model (*.pt);;All (*.*)")
        if p:
            self.ent_model.setText(p)
            self._last_model_dir = Path(p).parent
            self._stop_if_running()
            self._save_config()

    def browse_input(self) -> None:
        initial = self._resolve_initial_dir(self.ent_input.text(), self._last_input_dir)
        p, _ = QFileDialog.getOpenFileName(
            self,
            "選擇圖片或影片",
            initial,
            "Images/Video (*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.avi *.mov *.mkv *.wmv);;All (*.*)",
        )
        if p:
            self.ent_input.setText(p)
            self._last_input_dir = Path(p).parent
            self._preview_input(Path(p))
            self._stop_if_running()
            self._save_config()

    def browse_input_dir(self) -> None:
        initial = self._resolve_initial_dir(self.ent_input.text(), self._last_input_dir)
        p = QFileDialog.getExistingDirectory(self, "選擇資料夾", initial)
        if p:
            self.ent_input.setText(p)
            self._last_input_dir = Path(p)
            self._preview_input(Path(p))
            self._stop_if_running()
            self._save_config()

    def stop(self) -> None:
        self._stop_event.set()
        self.lbl_status.setText("停止中...")

    def _run(self, model_path: Optional[Path], input_path: Path, token: int, use_http: bool, http_url: str) -> None:
        stopped = False
        try:
            engine = None
            http_client = None
            if use_http:
                http_client = YoloHttpClient(http_url)
                self._emit("device", "http")
            else:
                if model_path is None:
                    self._emit("error", "模型路徑無效。")
                    return
                engine = YoloEngine(str(model_path), device=self.ent_device.text().strip())
                engine.load()
                self._emit("device", engine.device_name)

            if input_path.is_dir():
                items = self._collect_items(input_path)
            else:
                items = [input_path]

            for item in items:
                if self._stop_event.is_set() or token != self._ui_token:
                    stopped = True
                    break
                suffix = item.suffix.lower()
                if suffix in IMG_EXTS:
                    frame = cv2.imread(str(item))
                    if frame is None:
                        self._emit("log", f"讀取失敗: {item.name}")
                        continue
                    self._infer_and_emit(engine, http_client, frame, token)
                    self._sleep_interval(token)
                elif suffix in VID_EXTS:
                    cap = cv2.VideoCapture(str(item))
                    if not cap.isOpened():
                        self._emit("log", f"讀取失敗: {item.name}")
                        continue
                    while not self._stop_event.is_set() and token == self._ui_token:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        self._infer_and_emit(engine, http_client, frame, token)
                    cap.release()
                    self._sleep_interval(token)
        except Exception as exc:
            self._emit("error", f"執行失敗: {exc}")
        finally:
            self._emit("done", stopped or self._stop_event.is_set())

    def _infer_and_emit(self, engine, http_client, frame_bgr, token: int) -> None:
        if token != self._ui_token:
            return
        if not self.chk_show.isChecked():
            self._emit("frame", (frame_bgr, "顯示原始影像（未顯示判斷）"))
            return
        if http_client is not None:
            dets = http_client.infer(frame_bgr)
            annotated = frame_bgr.copy()
            for d in dets:
                xyxy = d.get("xyxy", [])
                if not isinstance(xyxy, list) or len(xyxy) != 4:
                    continue
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                name = str(d.get("className", ""))
                conf = float(d.get("conf", 0.0))
                label = f"{name} {conf:.2f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            info = "未偵測到物件" if not dets else f"偵測 {len(dets)} 個"
            self._emit("frame", (annotated, info))
            return
        conf = self._conf_value()
        iou = self._iou_value()
        if self.chk_ignore_conf.isChecked():
            conf = 0.01
        if self.chk_ignore_iou.isChecked():
            iou = 1.0
        result, dets = engine.infer(frame_bgr, conf=conf, iou=iou)
        annotated = result.plot() if result is not None else frame_bgr
        info = "未偵測到物件" if not dets else f"偵測 {len(dets)} 個"
        self._emit("frame", (annotated, info))

    def _sleep_interval(self, token: int) -> None:
        try:
            interval = max(0.0, float(self.ent_interval.text()))
        except Exception:
            interval = 0.0
        end = time.time() + interval
        while time.time() < end:
            if self._stop_event.is_set() or token != self._ui_token:
                return
            time.sleep(0.05)

    def _collect_items(self, folder: Path) -> list[Path]:
        images = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
        videos = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VID_EXTS])
        if self._order_map.get(self.cmb_order.currentText(), "images_first") == "videos_first":
            return videos + images
        return images + videos

    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            QMessageBox.warning(self, "執行中", "目前正在執行。")
            return
        use_http = self.chk_use_http.isChecked()
        model_path: Optional[Path] = None
        if not use_http:
            model_path = Path(self.ent_model.text().strip())
            if not model_path.exists():
                QMessageBox.critical(self, "錯誤", "模型檔不存在。")
                return
        http_url = self._normalize_http_url(self.ent_http_url.text()) if use_http else ""
        if use_http and not http_url:
            QMessageBox.critical(self, "錯誤", "HTTP URL 無效。")
            return
        input_path = Path(self.ent_input.text().strip())
        if not input_path.exists():
            QMessageBox.critical(self, "錯誤", "輸入路徑不存在。")
            return
        if input_path.is_dir() and not self._collect_items(input_path):
            QMessageBox.critical(self, "錯誤", "資料夾內找不到可用的圖片或影片。")
            return

        self._stop_event.clear()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_status.setText("執行中...")
        self._ui_token += 1
        token = self._ui_token
        self._worker = threading.Thread(
            target=self._run,
            args=(model_path, input_path, token, use_http, http_url),
            daemon=True,
        )
        self._worker.start()
        self._save_config()

    def _on_input_changed(self) -> None:
        raw = self.ent_input.text().strip()
        if raw:
            p = Path(raw)
            if p.exists():
                self._last_input_dir = p if p.is_dir() else p.parent
                self._preview_input(p)
            else:
                self._log("圖片或影片路徑不存在。")
        self._save_config()

    def _preview_input(self, path: Path) -> None:
        try:
            if path.is_dir():
                items = self._collect_items(path)
                if not items:
                    return
                self._preview_input(items[0])
                return
            if path.suffix.lower() in IMG_EXTS:
                frame = cv2.imread(str(path))
                if frame is None:
                    return
                self._last_frame = frame
                self._render_frame(frame)
                self.lbl_status.setText("預覽圖片")
                return
            if path.suffix.lower() in VID_EXTS:
                cap = cv2.VideoCapture(str(path))
                if not cap.isOpened():
                    return
                ok, frame = cap.read()
                cap.release()
                if ok:
                    self._last_frame = frame
                    self._render_frame(frame)
                    self.lbl_status.setText("預覽影片第一幀")
        except Exception:
            return

    def _stop_if_running(self) -> None:
        if self._worker and self._worker.is_alive():
            self._ui_token += 1
            self._stop_event.set()
            self.lbl_status.setText("已停止（重新選擇）")

    def _save_config(self) -> None:
        if self._loading_config:
            return
        try:
            data = {
                "model_path": self.ent_model.text(),
                "input_path": self.ent_input.text(),
                "conf": self._conf_value(),
                "iou": self._iou_value(),
                "device": self.ent_device.text(),
                "show": self.chk_show.isChecked(),
                "ignore_conf": self.chk_ignore_conf.isChecked(),
                "ignore_iou": self.chk_ignore_iou.isChecked(),
                "interval_sec": self.ent_interval.text(),
                "order": self._order_map.get(self.cmb_order.currentText(), "images_first"),
                "use_http": self.chk_use_http.isChecked(),
                "http_url": self.ent_http_url.text(),
            }
            self._config_path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")
        except Exception:
            return

    def _load_config(self) -> None:
        if not self._config_path.exists():
            return
        try:
            self._loading_config = True
            data = json.loads(self._config_path.read_text(encoding="utf-8"))
            self.ent_model.setText(data.get("model_path", ""))
            self.ent_input.setText(data.get("input_path", ""))
            conf = max(0.01, min(1.0, float(data.get("conf", 0.70))))
            iou = max(0.0, min(1.0, float(data.get("iou", 0.45))))
            self.ent_conf.setText(f"{conf:.2f}")
            self.ent_iou.setText(f"{iou:.2f}")
            self.sld_conf.setValue(int(round(conf * 100)))
            self.sld_iou.setValue(int(round(iou * 100)))
            self.ent_device.setText(data.get("device", ""))
            self.chk_show.setChecked(bool(data.get("show", True)))
            self.chk_ignore_conf.setChecked(bool(data.get("ignore_conf", False)))
            self.chk_ignore_iou.setChecked(bool(data.get("ignore_iou", False)))
            self.ent_interval.setText(str(data.get("interval_sec", "1.0")))
            self.cmb_order.setCurrentText(self._order_map_rev.get(data.get("order", "images_first"), "圖片優先"))
            self.chk_use_http.setChecked(bool(data.get("use_http", False)))
            self.ent_http_url.setText(data.get("http_url", DEFAULT_HTTP_URL))
        except Exception:
            return
        finally:
            self._loading_config = False

    def _conf_value(self) -> float:
        try:
            v = float(self.ent_conf.text())
        except Exception:
            v = 0.70
        return max(0.01, min(1.0, v))

    def _iou_value(self) -> float:
        try:
            v = float(self.ent_iou.text())
        except Exception:
            v = 0.45
        return max(0.0, min(1.0, v))

    @staticmethod
    def _resolve_initial_dir(path_value: str, last_dir: Optional[Path]) -> str:
        try:
            raw = (path_value or "").strip()
            if raw:
                p = Path(raw)
                if p.exists():
                    return str(p if p.is_dir() else p.parent)
            if last_dir and last_dir.exists():
                return str(last_dir)
        except Exception:
            pass
        return str(Path.cwd())

    @staticmethod
    def _normalize_http_url(raw: str) -> str:
        text = (raw or "").strip()
        if not text:
            return ""
        if not text.startswith("http://") and not text.startswith("https://"):
            text = f"http://{text}"
        text = text.rstrip("/")
        if not text.endswith("/detect"):
            text = f"{text}/detect"
        return text

    def _try_preview_on_start(self) -> None:
        raw = self.ent_input.text().strip()
        if not raw:
            return
        p = Path(raw)
        if p.exists():
            self._preview_input(p)


def main() -> None:
    app = QApplication([])
    win = App()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
