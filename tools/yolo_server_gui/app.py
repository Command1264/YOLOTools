from __future__ import annotations

import ipaddress
import os
import queue
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Any, Optional

import portalocker
import yaml
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QAction, QCloseEvent, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from log_manager import LogContext, LogController, setup_logging
from log_viewer import LogViewer
from server import YoloServer
from tray import create_tray_icon
from tray_base import TrayBase
from yolo_engine import YoloEngine


def _resolve_exec_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _resolve_resource_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent)).resolve()
    return Path(__file__).resolve().parent


EXEC_DIR: Path = _resolve_exec_dir()
RESOURCE_DIR: Path = _resolve_resource_dir()
CONFIG_PATH: Path = EXEC_DIR / "yolo_server_gui_config.yml"
STARTUP_FILE: str = "yolo_server_gui_startup.cmd"
ICON_ICO_PATH: Path = RESOURCE_DIR / "yolo_server_icon.ico"
ICON_PNG_PATH: Path = RESOURCE_DIR / "yolo_server_icon.png"

CLOSE_LABELS: dict[str, str] = {
    "ask": "詢問",
    "minimize": "縮到工具列",
    "exit": "直接關閉",
}

APP_LOG_CTRL: LogController = LogController()
LOCK_FILE_NAME: str = "yolo_server_gui.lock"


def _select_icon_path() -> Optional[Path]:
    if os.name == "nt":
        primary, secondary = ICON_ICO_PATH, ICON_PNG_PATH
    else:
        primary, secondary = ICON_PNG_PATH, ICON_ICO_PATH
    if primary.exists():
        return primary
    if secondary.exists():
        APP_LOG_CTRL.warning(
            "找不到主要圖示，改用備援圖示。primary=%s fallback=%s",
            str(primary),
            str(secondary),
        )
        return secondary
    return None


def normalize_path(path_str: str) -> str:
    if not path_str:
        return ""
    try:
        return Path(path_str).as_posix()
    except Exception:
        return path_str.replace("\\", "/")


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


@dataclass
class AppConfig:
    """儲存 GUI 與啟動相關設定。"""

    model_path: str = ""
    host: str = "127.0.0.1"
    port: int = 60922
    auto_start_server: bool = False
    launch_on_startup: bool = False
    close_behavior: str = "ask"

    @classmethod
    def from_dict(cls, data: Any) -> "AppConfig":
        cfg = cls()
        if not isinstance(data, dict):
            return cfg
        cfg.model_path = normalize_path(str(data.get("model_path", cfg.model_path)))
        cfg.host = str(data.get("host", cfg.host)).strip() or cfg.host
        cfg.port = _coerce_int(data.get("port", cfg.port), cfg.port)
        cfg.auto_start_server = _coerce_bool(
            data.get("auto_start_server", cfg.auto_start_server), cfg.auto_start_server
        )
        cfg.launch_on_startup = _coerce_bool(
            data.get("launch_on_startup", cfg.launch_on_startup), cfg.launch_on_startup
        )
        behavior = str(data.get("close_behavior", cfg.close_behavior)).strip()
        cfg.close_behavior = behavior if behavior in CLOSE_LABELS else cfg.close_behavior
        return cfg

    @classmethod
    def load(cls, path: Path) -> "AppConfig":
        if not path.exists():
            return cls()
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            return cls()
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": normalize_path(self.model_path),
            "host": self.host,
            "port": int(self.port),
            "auto_start_server": bool(self.auto_start_server),
            "launch_on_startup": bool(self.launch_on_startup),
            "close_behavior": self.close_behavior,
        }

    def save(self, path: Path) -> None:
        try:
            path.write_text(
                yaml.safe_dump(self.to_dict(), allow_unicode=False, sort_keys=False),
                encoding="utf-8",
            )
        except Exception:
            APP_LOG_CTRL.exception("儲存設定失敗。path=%s", str(path))


def validate_host(value: str) -> tuple[bool, str]:
    val = (value or "").strip()
    if not val:
        return False, "IP 不能空白"
    if val.lower() == "localhost":
        return True, "localhost"
    try:
        ipaddress.ip_address(val)
        return True, val
    except Exception:
        return False, f"無效 IP：{val}"


def validate_port(value: str) -> tuple[bool, Optional[int], str]:
    try:
        port = int(str(value).strip())
    except Exception:
        return False, None, "Port 必須是整數"
    if port < 1 or port > 65535:
        return False, None, "Port 必須在 1~65535 之間"
    return True, port, ""


def startup_cmd_path() -> Optional[Path]:
    if os.name != "nt":
        return None
    appdata = os.environ.get("APPDATA", "")
    if not appdata:
        return None
    return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / STARTUP_FILE


def build_startup_command() -> str:
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}"'
    return f'"{sys.executable}" "{Path(__file__).resolve()}"'


class SingleInstanceLock:
    """以 lock file 確保單一實例執行。"""

    def __init__(self, lock_path: Path, log_ctrl: LogController) -> None:
        self._lock_path: Path = lock_path
        self._log_ctrl: LogController = log_ctrl
        self._lock = threading.Lock()
        self._file: Optional[object] = None
        self._stop_event = threading.Event()
        self._retainer: Optional[threading.Thread] = None

    def try_acquire(self) -> bool:
        with self._lock:
            if self._file is not None:
                return True
            handle: Optional[object] = None
            try:
                handle = open(self._lock_path, "a+")
                portalocker.lock(handle, portalocker.LOCK_EX | portalocker.LOCK_NB)
                self._file = handle
                return True
            except portalocker.exceptions.LockException:
                if handle is not None:
                    try:
                        handle.close()
                    except Exception:
                        self._log_ctrl.exception("關閉鎖檔失敗。path=%s", str(self._lock_path))
                return False
            except Exception:
                self._log_ctrl.exception("取得鎖失敗。path=%s", str(self._lock_path))
                if handle is not None:
                    try:
                        handle.close()
                    except Exception:
                        self._log_ctrl.exception("關閉鎖檔失敗。path=%s", str(self._lock_path))
                return False

    def start_retainer(self, interval_sec: float = 3.0) -> None:
        if self._retainer and self._retainer.is_alive():
            return

        def _worker() -> None:
            while not self._stop_event.is_set():
                if self._file is None:
                    self.try_acquire()
                time.sleep(interval_sec)

        self._retainer = threading.Thread(target=_worker, daemon=True)
        self._retainer.start()

    def release(self) -> None:
        with self._lock:
            if self._file is None:
                return
            try:
                portalocker.unlock(self._file)
            except Exception:
                self._log_ctrl.exception("釋放鎖失敗。path=%s", str(self._lock_path))
            try:
                self._file.close()
            except Exception:
                self._log_ctrl.exception("關閉鎖檔失敗。path=%s", str(self._lock_path))
            self._file = None

    def stop(self) -> None:
        self._stop_event.set()
        self.release()


class App(QMainWindow):
    """YOLO 伺服器主視窗（PySide6 版本）。"""

    def __init__(self, instance_lock: Optional[SingleInstanceLock] = None) -> None:
        super().__init__()
        self.setWindowTitle("YOLO Server")
        self.resize(720, 360)
        self.setMinimumSize(640, 320)

        self.log_context: LogContext = setup_logging(EXEC_DIR)
        self.logger: Logger = self.log_context.logger
        self.log_ctrl: LogController = LogController(self.logger)
        self.log_ctrl.info("GUI 啟動")

        icon_path = _select_icon_path()
        if icon_path is not None:
            self.setWindowIcon(QIcon(str(icon_path)))
        else:
            self.log_ctrl.warning("找不到應用程式圖示檔案。")

        self.cfg: AppConfig = AppConfig.load(CONFIG_PATH)
        self.server: Optional[YoloServer] = None
        self._deps_ready: bool = False
        self._deps_loading: bool = False
        self._deps_token: int = 0
        self._dep_result_queue: queue.Queue[tuple[int, Optional[str]]] = queue.Queue()
        self._device_queue: queue.Queue[str] = queue.Queue()
        self._tray_queue: queue.Queue[str] = queue.Queue()
        self._log_viewer: Optional[LogViewer] = None
        self._instance_lock: Optional[SingleInstanceLock] = instance_lock
        self._quitting: bool = False

        tray_icon_path = _select_icon_path()
        self.tray: TrayBase = create_tray_icon(
            tooltip="YOLO Server",
            on_exit=self._enqueue_tray_exit,
            on_show=self._enqueue_tray_show,
            icon_path=str(tray_icon_path) if tray_icon_path else None,
        )

        self._build_ui()
        self._apply_config_to_ui()
        self._init_dep_status()
        self._apply_startup_setting()
        self.tray.start()

        self._poll_timer: QTimer = QTimer(self)
        self._poll_timer.setInterval(200)
        self._poll_timer.timeout.connect(self._poll_background_queues)
        self._poll_timer.start()

        if self.cfg.auto_start_server:
            QTimer.singleShot(200, self._try_auto_start)

    def _build_ui(self) -> None:
        menu_bar: QMenuBar = self.menuBar()
        menu_more: QMenu = menu_bar.addMenu("更多功能")
        action_log: QAction = QAction("瀏覽運行日誌", self)
        action_log.triggered.connect(self._open_log_viewer)
        menu_more.addAction(action_log)
        action_settings: QAction = QAction("設定...", self)
        action_settings.triggered.connect(self._open_settings)
        menu_more.addAction(action_settings)

        root: QWidget = QWidget(self)
        self.setCentralWidget(root)
        main_layout: QVBoxLayout = QVBoxLayout(root)

        form: QFormLayout = QFormLayout()
        main_layout.addLayout(form)

        model_row: QWidget = QWidget(root)
        model_layout: QHBoxLayout = QHBoxLayout(model_row)
        model_layout.setContentsMargins(0, 0, 0, 0)
        self.ent_model: QLineEdit = QLineEdit(model_row)
        self.btn_pick_model: QPushButton = QPushButton("選擇...", model_row)
        self.btn_pick_model.clicked.connect(self._pick_model)
        model_layout.addWidget(self.ent_model, 1)
        model_layout.addWidget(self.btn_pick_model)
        form.addRow("模型路徑", model_row)

        ip_row: QWidget = QWidget(root)
        ip_layout: QHBoxLayout = QHBoxLayout(ip_row)
        ip_layout.setContentsMargins(0, 0, 0, 0)
        self.ent_host: QLineEdit = QLineEdit(ip_row)
        self.ent_port: QLineEdit = QLineEdit(ip_row)
        self.ent_port.setMaximumWidth(120)
        ip_layout.addWidget(self.ent_host, 1)
        ip_layout.addWidget(QLabel("Port", ip_row))
        ip_layout.addWidget(self.ent_port)
        form.addRow("IP", ip_row)

        status_row: QWidget = QWidget(root)
        status_layout: QHBoxLayout = QHBoxLayout(status_row)
        status_layout.setContentsMargins(0, 0, 0, 0)
        self.btn_toggle: QPushButton = QPushButton("啟動伺服器", status_row)
        self.btn_toggle.clicked.connect(self._toggle_server)
        status_text_wrap: QWidget = QWidget(status_row)
        status_text_layout: QVBoxLayout = QVBoxLayout(status_text_wrap)
        status_text_layout.setContentsMargins(0, 0, 0, 0)
        status_text_layout.setSpacing(4)
        self.lbl_status: QLabel = QLabel("狀態：未啟動", status_text_wrap)
        self.lbl_device: QLabel = QLabel("裝置：未啟動", status_text_wrap)
        status_text_layout.addWidget(self.lbl_status)
        status_text_layout.addWidget(self.lbl_device)
        status_layout.addWidget(self.btn_toggle)
        status_layout.addWidget(status_text_wrap, 1)
        main_layout.addWidget(status_row)

        main_layout.addStretch(1)
        self.lbl_dep_status: QLabel = QLabel("核心套件：尚未載入", root)
        main_layout.addWidget(self.lbl_dep_status)

        self.ent_host.editingFinished.connect(lambda: self._apply_quick_settings(False))
        self.ent_port.editingFinished.connect(lambda: self._apply_quick_settings(False))

    def _apply_config_to_ui(self) -> None:
        self.ent_model.setText(self.cfg.model_path)
        self.ent_host.setText(self.cfg.host)
        self.ent_port.setText(str(self.cfg.port))

    def _pick_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "選擇模型",
            str(EXEC_DIR),
            "YOLO weights (*.pt);;All files (*.*)",
        )
        if path:
            self.ent_model.setText(normalize_path(path))
            self._apply_quick_settings(False)

    def _show_error(self, title: str, text: str) -> None:
        QMessageBox.critical(self, title, text)

    def _show_warn(self, title: str, text: str) -> None:
        QMessageBox.warning(self, title, text)

    def _apply_quick_settings(self, require_model: bool) -> bool:
        model_path = normalize_path(self.ent_model.text().strip())
        if require_model:
            if not model_path:
                self._show_error("設定錯誤", "請選擇模型路徑")
                self.ent_model.setText(self.cfg.model_path)
                return False
            if not Path(model_path).exists():
                self._show_error("設定錯誤", "模型路徑不存在")
                self.ent_model.setText(self.cfg.model_path)
                return False
        elif model_path and not Path(model_path).exists():
            self._show_error("設定錯誤", "模型路徑不存在")
            self.ent_model.setText(self.cfg.model_path)
            return False

        host_ok, host_val = validate_host(self.ent_host.text())
        if not host_ok:
            self._show_error("設定錯誤", host_val)
            self.ent_host.setText(self.cfg.host)
            return False

        port_ok, port_val, port_msg = validate_port(self.ent_port.text())
        if not port_ok:
            self._show_error("設定錯誤", port_msg)
            self.ent_port.setText(str(self.cfg.port))
            return False

        self.cfg.model_path = model_path
        self.cfg.host = host_val
        self.cfg.port = int(port_val)
        self.cfg.save(CONFIG_PATH)
        self.ent_model.setText(model_path)
        self.ent_host.setText(host_val)
        self.ent_port.setText(str(port_val))
        return True

    def _toggle_server(self) -> None:
        if self.server and self.server.is_running():
            self._stop_server()
        else:
            self._start_server()

    def _start_server(self) -> None:
        if not self._apply_quick_settings(True):
            return
        host: str = self.cfg.host
        port: int = int(self.cfg.port)
        model_path: str = self.cfg.model_path
        try:
            server_icon_path = _select_icon_path()
            icon_path = str(server_icon_path) if server_icon_path else None
            if self.server is None:
                self.server = YoloServer(
                    model_path,
                    host,
                    port,
                    logger=self.logger,
                    icon_path=icon_path,
                )
            else:
                self.server.update_settings(
                    model_path=model_path,
                    host=host,
                    port=port,
                    icon_path=icon_path,
                )
            self.server.start()
        except Exception as exc:
            self.server = None
            self.log_ctrl.exception("啟動伺服器失敗。")
            self._show_error("啟動失敗", f"無法啟動伺服器：\n{exc}")
            return
        self._set_running_state(True)
        self.lbl_status.setText(f"狀態：執行中 http://{host}:{port}")
        self.lbl_device.setText("裝置：載入中...")
        self._load_device_async()
        self.log_ctrl.info("伺服器已啟動。host=%s port=%s", host, port)

    def _stop_server(self) -> None:
        try:
            if self.server:
                self.server.stop()
        finally:
            self.server = None
        self._set_running_state(False)
        self.lbl_status.setText("狀態：未啟動")
        self.lbl_device.setText("裝置：未啟動")
        self.log_ctrl.info("伺服器已停止。")

    def _set_running_state(self, running: bool) -> None:
        self.ent_model.setEnabled(not running)
        self.ent_host.setEnabled(not running)
        self.ent_port.setEnabled(not running)
        self.btn_pick_model.setEnabled(not running)
        self.btn_toggle.setText("停止伺服器" if running else "啟動伺服器")
        self._update_toggle_state()

    def _open_log_viewer(self) -> None:
        if self._log_viewer is not None and self._log_viewer.isVisible():
            self._log_viewer.raise_()
            self._log_viewer.activateWindow()
            return

        def _on_close() -> None:
            self._log_viewer = None

        self.log_ctrl.info("開啟運行日誌視窗。")
        self._log_viewer = LogViewer(self, on_close=_on_close)
        self._center_dialog(self._log_viewer)
        self._log_viewer.show()

    def _open_settings(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("設定")
        dialog.setModal(True)

        layout: QVBoxLayout = QVBoxLayout(dialog)
        chk_auto_start = QCheckBox("啟動時自動開啟伺服器", dialog)
        chk_auto_start.setChecked(bool(self.cfg.auto_start_server))
        chk_launch_startup = QCheckBox("開機自動啟動應用程式", dialog)
        chk_launch_startup.setChecked(bool(self.cfg.launch_on_startup))
        layout.addWidget(chk_auto_start)
        layout.addWidget(chk_launch_startup)

        close_row: QWidget = QWidget(dialog)
        close_layout: QHBoxLayout = QHBoxLayout(close_row)
        close_layout.setContentsMargins(0, 0, 0, 0)
        close_layout.addWidget(QLabel("關閉按鈕行為", close_row))
        cmb_close = QComboBox(close_row)
        cmb_close.addItems(list(CLOSE_LABELS.values()))
        cmb_close.setCurrentText(CLOSE_LABELS.get(self.cfg.close_behavior, "詢問"))
        close_layout.addWidget(cmb_close, 1)
        layout.addWidget(close_row)

        btn_row: QWidget = QWidget(dialog)
        btn_layout: QHBoxLayout = QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.addStretch(1)
        btn_cancel = QPushButton("取消", btn_row)
        btn_save = QPushButton("儲存", btn_row)
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_save)
        layout.addWidget(btn_row)

        btn_cancel.clicked.connect(dialog.reject)

        def _save_and_close() -> None:
            self.cfg.auto_start_server = chk_auto_start.isChecked()
            self.cfg.launch_on_startup = chk_launch_startup.isChecked()
            label_to_key = {v: k for k, v in CLOSE_LABELS.items()}
            self.cfg.close_behavior = label_to_key.get(cmb_close.currentText(), "ask")
            self.cfg.save(CONFIG_PATH)
            self._apply_startup_setting()
            self.tray.start()
            dialog.accept()

        btn_save.clicked.connect(_save_and_close)

        self._center_dialog(dialog)
        dialog.exec()

    def _center_dialog(self, dialog: QWidget) -> None:
        parent_geo = self.frameGeometry()
        dialog.adjustSize()
        dialog_geo = dialog.frameGeometry()
        dialog_geo.moveCenter(parent_geo.center())
        dialog.move(dialog_geo.topLeft())

    def _enqueue_tray_show(self) -> None:
        try:
            self._tray_queue.put_nowait("show")
        except Exception:
            self.log_ctrl.exception("加入顯示事件到 tray 佇列失敗。")

    def _enqueue_tray_exit(self) -> None:
        try:
            self._tray_queue.put_nowait("exit")
        except Exception:
            self.log_ctrl.exception("加入退出事件到 tray 佇列失敗。")

    def _poll_background_queues(self) -> None:
        self._poll_tray_queue()
        self._poll_dep_queue()
        self._poll_device_queue()

    def _poll_tray_queue(self) -> None:
        try:
            while True:
                action: str = self._tray_queue.get_nowait()
                if action == "show":
                    self._restore_window()
                elif action == "exit":
                    self._exit_app()
        except queue.Empty:
            return

    def _poll_dep_queue(self) -> None:
        try:
            while True:
                token, error_msg = self._dep_result_queue.get_nowait()
                self._on_dep_preload_done(token, error_msg)
        except queue.Empty:
            return

    def _poll_device_queue(self) -> None:
        try:
            while True:
                device_name = self._device_queue.get_nowait()
                self._apply_device_name(device_name)
        except queue.Empty:
            return

    def _apply_startup_setting(self) -> None:
        if os.name != "nt":
            return
        enabled: bool = bool(self.cfg.launch_on_startup)
        path: Optional[Path] = startup_cmd_path()
        if not path:
            return
        try:
            if enabled:
                cmd: str = build_startup_command()
                path.write_text(f"@echo off\n{cmd}\n", encoding="utf-8")
            elif path.exists():
                path.unlink()
        except Exception:
            self._show_warn("設定提醒", "無法更新開機啟動設定")

    def _try_auto_start(self) -> None:
        if self.server and self.server.is_running():
            return
        if self._deps_ready:
            self._start_server()
            return
        if self._deps_loading:
            QTimer.singleShot(300, self._try_auto_start)
            return
        self.log_ctrl.warning("自動啟動失敗：核心套件未就緒。")

    def _init_dep_status(self) -> None:
        self._start_dep_preload()

    def _set_dep_status(self, text: str) -> None:
        self.lbl_dep_status.setText(text)
        self._update_toggle_state()

    def _update_toggle_state(self) -> None:
        if self.server and self.server.is_running():
            self.btn_toggle.setEnabled(True)
            return
        self.btn_toggle.setEnabled(self._deps_ready)

    def _start_dep_preload(self) -> None:
        if self._deps_loading:
            return
        self._deps_token += 1
        token = self._deps_token
        self._deps_loading = True
        self._deps_ready = False
        self._set_dep_status("核心套件：載入中...")

        def _worker() -> None:
            error_msg: Optional[str] = None
            try:
                YoloEngine.preload_dependencies()
            except Exception as exc:
                error_msg = str(exc) or "unknown error"
            self._dep_result_queue.put((token, error_msg))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_dep_preload_done(self, token: int, error_msg: Optional[str]) -> None:
        if token != self._deps_token:
            return
        self._deps_loading = False
        if error_msg:
            self._deps_ready = False
            self._set_dep_status("核心套件：載入失敗")
            self.log_ctrl.warning("核心套件載入失敗：%s", error_msg)
            return
        self._deps_ready = True
        self._set_dep_status("核心套件：就緒")

    def _minimize_to_tray(self) -> None:
        if os.name != "nt":
            self._exit_app()
            return
        self.hide()
        self.tray.start()

    def _restore_window(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _exit_app(self) -> None:
        if self._quitting:
            return
        self._quitting = True
        try:
            self._stop_server()
        except Exception:
            self.log_ctrl.exception("停止伺服器時發生錯誤。")
        try:
            self.tray.stop()
        except Exception:
            self.log_ctrl.exception("停止 tray 時發生錯誤。")
        try:
            if self._instance_lock is not None:
                self._instance_lock.stop()
        except Exception:
            self.log_ctrl.exception("釋放執行鎖失敗。")
        self.log_ctrl.info("GUI 結束。")
        app = QApplication.instance()
        if app is not None:
            app.quit()
            return
        self.close()

    def _load_device_async(self) -> None:
        def _worker() -> None:
            device_name: str = "unknown"
            try:
                if self.server:
                    device_name = self.server.get_device_name()
            except Exception:
                device_name = "unknown"
            self._device_queue.put(device_name)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

    def _apply_device_name(self, device_name: str) -> None:
        if not self.server:
            return
        self.lbl_device.setText(f"裝置：{device_name}")
        self.log_ctrl.info("使用裝置：%s", device_name)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._quitting:
            event.accept()
            return
        behavior: str = self.cfg.close_behavior
        if behavior == "ask":
            box = QMessageBox(self)
            box.setWindowTitle("關閉")
            box.setText("要縮到工具列嗎？\n是：縮到工具列\n否：直接關閉\n取消：不動作")
            box.setIcon(QMessageBox.Question)
            yes_btn = box.addButton("是", QMessageBox.YesRole)
            no_btn = box.addButton("否", QMessageBox.NoRole)
            cancel_btn = box.addButton("取消", QMessageBox.RejectRole)
            box.exec()
            clicked = box.clickedButton()
            if clicked is cancel_btn:
                event.ignore()
                return
            if clicked is yes_btn:
                self.cfg.close_behavior = "minimize"
                self.cfg.save(CONFIG_PATH)
                event.ignore()
                self._minimize_to_tray()
                return
            if clicked is no_btn:
                self.cfg.close_behavior = "exit"
                self.cfg.save(CONFIG_PATH)
                event.ignore()
                self._exit_app()
                return
            event.ignore()
            return
        if behavior == "minimize":
            event.ignore()
            self._minimize_to_tray()
            return
        event.ignore()
        self._exit_app()


if __name__ == "__main__":
    qt_app = QApplication(sys.argv)
    icon_path = _select_icon_path()
    if icon_path is not None:
        qt_app.setWindowIcon(QIcon(str(icon_path)))

    lock_path = Path(tempfile.gettempdir()) / LOCK_FILE_NAME
    instance_lock = SingleInstanceLock(lock_path, APP_LOG_CTRL)
    allow_start = True
    if not instance_lock.try_acquire():
        result = QMessageBox.question(
            None,
            "已在執行",
            "偵測到已有程式在執行。\n仍要開啟新的視窗嗎？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        allow_start = result == QMessageBox.Yes
    if not allow_start:
        sys.exit(0)

    instance_lock.start_retainer()
    app = App(instance_lock=instance_lock)
    app.show()
    sys.exit(qt_app.exec())
