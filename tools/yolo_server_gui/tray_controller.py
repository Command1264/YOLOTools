from __future__ import annotations

import os
import queue
from pathlib import Path
from typing import Any, Optional

from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QMessageBox

from config_model import CLOSE_LABELS
from config_service import build_startup_command, startup_cmd_path


class TrayController:
    """Handle tray events, close behavior, and app exit flow."""

    def __init__(self, app: Any) -> None:
        self.app = app

    def enqueue_tray_show(self) -> None:
        """Queue tray show action."""
        try:
            self.app._tray_queue.put_nowait("show")
        except Exception:
            self.app.log_ctrl.exception("加入顯示事件到 tray 佇列失敗。")

    def enqueue_tray_exit(self) -> None:
        """Queue tray exit action."""
        try:
            self.app._tray_queue.put_nowait("exit")
        except Exception:
            self.app.log_ctrl.exception("加入退出事件到 tray 佇列失敗。")

    def poll_tray_queue(self) -> None:
        """Drain tray queue and execute actions."""
        try:
            while True:
                action: str = self.app._tray_queue.get_nowait()
                if action == "show":
                    self.restore_window()
                elif action == "exit":
                    self.exit_app()
        except queue.Empty:
            return

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
        self.app.tray.start()

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
            self.app.tray.stop()
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
