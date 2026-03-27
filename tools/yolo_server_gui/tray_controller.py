from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Optional

from PySide6.QtGui import QAction, QCloseEvent, QCursor, QIcon
from PySide6.QtWidgets import QApplication, QMenu, QMessageBox, QSystemTrayIcon

from config_service import build_startup_command, startup_cmd_path


class TrayController:
    """Handle tray events, close behavior, and app exit flow."""

    def __init__(self, app: Any) -> None:
        self.app = app
        self._tray_menu: Optional[QMenu] = None
        self._tray_action_show: Optional[QAction] = None
        self._tray_action_exit: Optional[QAction] = None

    def setup_tray(self, select_icon_path: Callable[[], Optional[Path]]) -> None:
        """Initialize Qt native tray icon and context menu."""
        self.app.tray_icon = QSystemTrayIcon(self.app)
        icon_path = select_icon_path()
        if icon_path is not None:
            self.app.tray_icon.setIcon(QIcon(str(icon_path)))
        self.app.tray_icon.setToolTip("YOLO Server")
        self._tray_menu = QMenu(self.app)
        self._tray_action_show = QAction("顯示", self.app)
        self._tray_action_exit = QAction("關閉", self.app)
        self._tray_action_show.triggered.connect(self.restore_window)
        self._tray_action_exit.triggered.connect(self.exit_app)
        self._tray_menu.addAction(self._tray_action_show)
        self._tray_menu.addAction(self._tray_action_exit)
        self.app.tray_icon.setContextMenu(self._tray_menu)
        self.app.tray_icon.activated.connect(self.on_tray_activated)
        self.app.tray_icon.show()

    def ensure_tray_visible(self) -> None:
        """Ensure tray icon is visible."""
        if self.app.tray_icon is not None:
            self.app.tray_icon.show()

    def apply_startup_setting(self) -> None:
        """Apply launch-on-startup setting on Windows."""
        if os.name != "nt":
            return
        enabled: bool = bool(self.app.cfg.launch_on_startup)
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
            self.app._show_warn("設定提醒", "無法更新開機啟動設定")

    def minimize_to_tray(self) -> None:
        """Hide app window to tray on Windows, otherwise exit."""
        if os.name != "nt":
            self.exit_app()
            return
        self.app.hide()
        if self.app.tray_icon is not None:
            self.app.tray_icon.show()

    def restore_window(self) -> None:
        """Restore app window from tray."""
        self.app.showNormal()
        self.app.raise_()
        self.app.activateWindow()

    def exit_app(self) -> None:
        """Exit app and release runtime resources."""
        if self.app._quitting:
            return
        self.app._quitting = True
        try:
            self.app.server_controller.stop_server()
        except Exception:
            self.app.log_ctrl.exception("停止伺服器時發生錯誤。")
        try:
            if self.app.tray_icon is not None:
                self.app.tray_icon.hide()
        except Exception:
            self.app.log_ctrl.exception("停止 tray 時發生錯誤。")
        try:
            if self.app._instance_lock is not None:
                self.app._instance_lock.stop()
        except Exception:
            self.app.log_ctrl.exception("釋放執行鎖失敗。")
        self.app.log_ctrl.info("GUI 結束。")
        qt_app = QApplication.instance()
        if qt_app is not None:
            qt_app.quit()
            return
        self.app.close()

    def on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        """Handle tray activation events."""
        self.app.log_ctrl.debug(
            "Tray activated. reason=%s(%s)",
            getattr(reason, "name", str(reason)),
            getattr(reason, "value", "unknown"),
        )
        if reason == QSystemTrayIcon.Trigger:
            self.restore_window()
            return
        if reason == QSystemTrayIcon.DoubleClick:
            self.restore_window()
            return
        if reason == QSystemTrayIcon.Context:
            self.app.log_ctrl.debug("Tray context menu requested.")

    def handle_close_event(self, event: QCloseEvent) -> bool:
        """Handle close event based on configured behavior.

        Returns:
            bool: True if handled.
        """
        if self.app._quitting:
            event.accept()
            return True
        behavior: str = self.app.cfg.close_behavior
        if behavior == "ask":
            box = QMessageBox(self.app)
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
                return True
            if clicked is yes_btn:
                self.app.cfg.close_behavior = "minimize"
                self.app.cfg.save(self.app.config_path)
                event.ignore()
                self.minimize_to_tray()
                return True
            if clicked is no_btn:
                self.app.cfg.close_behavior = "exit"
                self.app.cfg.save(self.app.config_path)
                event.ignore()
                self.exit_app()
                return True
            event.ignore()
            return True
        if behavior == "minimize":
            event.ignore()
            self.minimize_to_tray()
            return True
        event.ignore()
        self.exit_app()
        return True
