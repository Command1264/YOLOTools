from __future__ import annotations

import os
import queue
import sys
import tempfile
from logging import Logger
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QTimer
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
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from config_model import AppConfig, CLOSE_LABELS, HTTP_PROFILE_LABELS
from config_service import (
    normalize_path,
)
from log_manager import LogContext, LogController, reconfigure_logging, resolve_log_root, setup_logging
from log_viewer import LogViewer
from server_controller import ServerController
from server import YoloServer
from single_instance import SingleInstanceLock
from tray_controller import TrayController


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
ICON_ICO_PATH: Path = RESOURCE_DIR / "yolo_server_icon.ico"
ICON_PNG_PATH: Path = RESOURCE_DIR / "yolo_server_icon.png"

APP_LOG_CTRL: LogController = LogController()
LOCK_FILE_NAME: str = "yolo_server_gui.lock"


def _resolve_log_base_dir(configured_path: str) -> Path:
    normalized = normalize_path(str(configured_path).strip())
    if not normalized:
        return EXEC_DIR
    return Path(normalized)


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


class App(QMainWindow):
    """YOLO 伺服器主視窗（PySide6 版本）。"""

    def __init__(self, instance_lock: Optional[SingleInstanceLock] = None) -> None:
        super().__init__()
        self.setWindowTitle("YOLO Server")
        self.resize(760, 360)
        self.setMinimumSize(700, 320)

        self.cfg: AppConfig = AppConfig.load(CONFIG_PATH)
        self.log_context: LogContext = setup_logging(_resolve_log_base_dir(self.cfg.log_base_dir), self.cfg.log_level)
        self.logger: Logger = self.log_context.logger
        self.log_ctrl: LogController = LogController(self.logger)
        self.log_ctrl.info("GUI 啟動")

        icon_path = _select_icon_path()
        if icon_path is not None:
            self.setWindowIcon(QIcon(str(icon_path)))
        else:
            self.log_ctrl.warning("找不到應用程式圖示檔案。")

        self.config_path: Path = CONFIG_PATH
        self.server: Optional[YoloServer] = None
        self._deps_ready: bool = False
        self._deps_loading: bool = False
        self._deps_token: int = 0
        self._dep_result_queue: queue.Queue[tuple[int, Optional[str]]] = queue.Queue()
        self._device_queue: queue.Queue[str] = queue.Queue()
        self._log_viewer: Optional[LogViewer] = None
        self._instance_lock: Optional[SingleInstanceLock] = instance_lock
        self._quitting: bool = False
        self.tray_icon: Optional[QSystemTrayIcon] = None

        self.server_controller: ServerController = ServerController(
            app=self,
            config_path=CONFIG_PATH,
            exec_dir=EXEC_DIR,
            select_icon_path=_select_icon_path,
        )
        self.tray_controller: TrayController = TrayController(self)

        self._build_ui()
        self.tray_controller.setup_tray(_select_icon_path)
        self._apply_config_to_ui()
        self._init_dep_status()
        self._apply_startup_setting()

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
        ip_layout.setSpacing(6)
        self.ent_host: QLineEdit = QLineEdit(ip_row)
        host_field_width = self.ent_host.fontMetrics().horizontalAdvance("0" * 14) + 24
        self.ent_host.setMinimumWidth(host_field_width)
        self.ent_port: QLineEdit = QLineEdit(ip_row)
        compact_field_width = self.ent_port.fontMetrics().horizontalAdvance("0" * 7) + 24
        self.ent_port.setFixedWidth(compact_field_width)
        self.ent_worker: QLineEdit = QLineEdit(ip_row)
        self.ent_worker.setFixedWidth(compact_field_width)
        self.ent_decode: QLineEdit = QLineEdit(ip_row)
        self.ent_decode.setFixedWidth(compact_field_width)
        self.cmb_http_profile: QComboBox = QComboBox(ip_row)
        self.cmb_http_profile.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        for key, label in HTTP_PROFILE_LABELS.items():
            self.cmb_http_profile.addItem(label, key)
        http_profile_width = max(
            96,
            max(
                self.cmb_http_profile.fontMetrics().horizontalAdvance(label)
                for label in HTTP_PROFILE_LABELS.values()
            )
            + 48,
        )
        self.cmb_http_profile.setFixedWidth(http_profile_width)
        ip_layout.addWidget(self.ent_host, 1)
        ip_layout.addWidget(QLabel("Port", ip_row))
        ip_layout.addWidget(self.ent_port)
        ip_layout.addWidget(QLabel("GPU", ip_row))
        ip_layout.addWidget(self.ent_worker)
        ip_layout.addWidget(QLabel("Decode", ip_row))
        ip_layout.addWidget(self.ent_decode)
        ip_layout.addWidget(QLabel("HTTP", ip_row))
        ip_layout.addWidget(self.cmb_http_profile)
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
        self.ent_worker.editingFinished.connect(lambda: self._apply_quick_settings(False))
        self.ent_decode.editingFinished.connect(lambda: self._apply_quick_settings(False))
        self.cmb_http_profile.currentIndexChanged.connect(lambda: self._apply_quick_settings(False))
        self._apply_window_width_constraints()

    def _apply_window_width_constraints(self) -> None:
        """Ensure the main window cannot be resized narrower than its current controls support."""
        required_width = max(700, self.minimumSizeHint().width())
        self.setMinimumSize(required_width, 320)
        preferred_width = max(760, required_width)
        if self.width() < preferred_width:
            self.resize(preferred_width, self.height())

    def _apply_config_to_ui(self) -> None:
        self.ent_model.setText(self.cfg.model_path)
        self.ent_host.setText(self.cfg.host)
        self.ent_port.setText(str(self.cfg.port))
        self.ent_worker.setText(str(self.cfg.gpu_replica_count))
        self.ent_decode.setText(str(self.cfg.decode_worker_count))
        idx = self.cmb_http_profile.findData(self.cfg.http_profile)
        self.cmb_http_profile.blockSignals(True)
        self.cmb_http_profile.setCurrentIndex(idx if idx >= 0 else 0)
        self.cmb_http_profile.blockSignals(False)

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
        return self.server_controller.apply_quick_settings(require_model)

    def _toggle_server(self) -> None:
        self.server_controller.toggle_server()

    def _start_server(self) -> None:
        self.server_controller.start_server()

    def _stop_server(self) -> None:
        self.server_controller.stop_server()

    def _set_running_state(self, running: bool) -> None:
        self.server_controller.set_running_state(running)

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
        dialog.setMinimumWidth(760)

        layout: QVBoxLayout = QVBoxLayout(dialog)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        chk_auto_start = QCheckBox("啟動時自動開啟伺服器", dialog)
        chk_auto_start.setChecked(bool(self.cfg.auto_start_server))
        chk_launch_startup = QCheckBox("開機自動啟動應用程式", dialog)
        chk_launch_startup.setChecked(bool(self.cfg.launch_on_startup))
        layout.addWidget(chk_auto_start)
        layout.addWidget(chk_launch_startup)

        log_row: QWidget = QWidget(dialog)
        log_layout: QHBoxLayout = QHBoxLayout(log_row)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(8)
        log_layout.addWidget(QLabel("Log 位置", log_row))
        ent_log_base = QLineEdit(log_row)
        ent_log_base.setText(normalize_path(str(_resolve_log_base_dir(self.cfg.log_base_dir))))
        ent_log_base.setMinimumWidth(460)
        btn_pick_log_base = QPushButton("瀏覽...", log_row)
        btn_pick_log_base.setFixedWidth(96)
        log_layout.addWidget(ent_log_base, 1)
        log_layout.addWidget(btn_pick_log_base)
        layout.addWidget(log_row)

        lbl_log_target = QLabel(dialog)
        lbl_log_target.setWordWrap(True)
        layout.addWidget(lbl_log_target)

        close_row: QWidget = QWidget(dialog)
        close_layout: QHBoxLayout = QHBoxLayout(close_row)
        close_layout.setContentsMargins(0, 0, 0, 0)
        close_layout.setSpacing(8)
        close_layout.addWidget(QLabel("關閉按鈕行為", close_row))
        cmb_close = QComboBox(close_row)
        cmb_close.addItems(list(CLOSE_LABELS.values()))
        cmb_close.setCurrentText(CLOSE_LABELS.get(self.cfg.close_behavior, "詢問"))
        cmb_close.setMinimumWidth(180)
        close_layout.addWidget(cmb_close, 1)
        layout.addWidget(close_row)

        btn_row: QWidget = QWidget(dialog)
        btn_layout: QHBoxLayout = QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(8)
        btn_layout.addStretch(1)
        btn_cancel = QPushButton("取消", btn_row)
        btn_save = QPushButton("儲存", btn_row)
        btn_cancel.setFixedWidth(96)
        btn_save.setFixedWidth(96)
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_save)
        layout.addWidget(btn_row)

        initial_auto_start = bool(self.cfg.auto_start_server)
        initial_launch_on_startup = bool(self.cfg.launch_on_startup)
        initial_close_behavior = self.cfg.close_behavior
        initial_log_base_dir = normalize_path(str(_resolve_log_base_dir(self.cfg.log_base_dir)))

        def _selected_log_base_dir() -> Path:
            return _resolve_log_base_dir(ent_log_base.text())

        def _update_log_target_label() -> None:
            log_root = resolve_log_root(_selected_log_base_dir())
            lbl_log_target.setText(f"實際儲存位置：{normalize_path(str(log_root))}")

        def _pick_log_base_dir() -> None:
            selected_dir = QFileDialog.getExistingDirectory(
                dialog,
                "選擇 Log 資料夾",
                str(_selected_log_base_dir()),
            )
            if selected_dir:
                ent_log_base.setText(normalize_path(selected_dir))

        ent_log_base.textChanged.connect(_update_log_target_label)
        btn_pick_log_base.clicked.connect(_pick_log_base_dir)
        _update_log_target_label()

        def _has_unsaved_changes() -> bool:
            label_to_key = {v: k for k, v in CLOSE_LABELS.items()}
            current_close_behavior = label_to_key.get(cmb_close.currentText(), "ask")
            return any(
                (
                    chk_auto_start.isChecked() != initial_auto_start,
                    chk_launch_startup.isChecked() != initial_launch_on_startup,
                    current_close_behavior != initial_close_behavior,
                    normalize_path(str(_selected_log_base_dir())) != initial_log_base_dir,
                )
            )

        def _confirm_discard_changes() -> bool:
            if not _has_unsaved_changes():
                return True
            result = QMessageBox.question(
                dialog,
                "取消設定",
                "目前有尚未儲存的變更，確定要取消嗎？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            return result == QMessageBox.Yes

        def _cancel_and_close() -> None:
            if _confirm_discard_changes():
                dialog.reject()

        btn_cancel.clicked.connect(_cancel_and_close)

        def _save_and_close() -> None:
            log_base_dir = normalize_path(str(_selected_log_base_dir()))
            log_root = resolve_log_root(Path(log_base_dir))
            try:
                log_root.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                self._show_error("設定錯誤", f"無法建立 Log 資料夾：\n{exc}")
                return
            self.cfg.auto_start_server = chk_auto_start.isChecked()
            self.cfg.launch_on_startup = chk_launch_startup.isChecked()
            self.cfg.log_base_dir = log_base_dir
            label_to_key = {v: k for k, v in CLOSE_LABELS.items()}
            self.cfg.close_behavior = label_to_key.get(cmb_close.currentText(), "ask")
            self.cfg.save(self.config_path)
            self.log_context = reconfigure_logging(Path(log_base_dir), self.cfg.log_level)
            self.logger = self.log_context.logger
            self.log_ctrl = LogController(self.logger)
            if self._log_viewer is not None and self._log_viewer.isVisible():
                self._log_viewer._refresh_files(select_latest=True)
            self.log_ctrl.info("Log 位置已更新。log_root=%s", str(self.log_context.log_root))
            self._apply_startup_setting()
            self.tray_controller.ensure_tray_visible()
            dialog.accept()

        btn_save.clicked.connect(_save_and_close)

        original_close_event = dialog.closeEvent

        def _dialog_close_event(event: QCloseEvent) -> None:
            if _confirm_discard_changes():
                original_close_event(event)
                return
            event.ignore()

        dialog.closeEvent = _dialog_close_event  # type: ignore[method-assign]

        self._center_dialog(dialog)
        dialog.exec()

    def _center_dialog(self, dialog: QWidget) -> None:
        parent_geo = self.frameGeometry()
        dialog.adjustSize()
        dialog_geo = dialog.frameGeometry()
        dialog_geo.moveCenter(parent_geo.center())
        dialog.move(dialog_geo.topLeft())

    def _poll_background_queues(self) -> None:
        self._poll_dep_queue()
        self._poll_device_queue()

    def _poll_dep_queue(self) -> None:
        self.server_controller.poll_dep_queue()

    def _poll_device_queue(self) -> None:
        self.server_controller.poll_device_queue()

    def _apply_startup_setting(self) -> None:
        self.tray_controller.apply_startup_setting()

    def _try_auto_start(self) -> None:
        self.server_controller.try_auto_start()

    def _init_dep_status(self) -> None:
        self.server_controller.init_dep_status()

    def _set_dep_status(self, text: str) -> None:
        self.server_controller.set_dep_status(text)

    def _update_toggle_state(self) -> None:
        self.server_controller.update_toggle_state()

    def _start_dep_preload(self) -> None:
        self.server_controller.start_dep_preload()

    def _on_dep_preload_done(self, token: int, error_msg: Optional[str]) -> None:
        self.server_controller.on_dep_preload_done(token, error_msg)

    def _load_device_async(self) -> None:
        self.server_controller.load_device_async()

    def _apply_device_name(self, device_name: str) -> None:
        self.server_controller.apply_device_name(device_name)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.tray_controller.handle_close_event(event):
            return
        super().closeEvent(event)

    def _timer_single_shot(self, msec: int, callback) -> None:
        """Small indirection for controller timer calls."""
        QTimer.singleShot(msec, callback)


if __name__ == "__main__":
    qt_app = QApplication(sys.argv)
    qt_app.setQuitOnLastWindowClosed(False)
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
