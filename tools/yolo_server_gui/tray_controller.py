from __future__ import annotations

import os
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QAction, QCloseEvent, QCursor, QIcon, QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QApplication, QMenu, QMessageBox, QSystemTrayIcon

from config_service import build_startup_command, startup_cmd_path


TRAY_MENU_TRIGGER_KEYS: frozenset[int] = frozenset(
    {
        int(Qt.Key.Key_Return),
        int(Qt.Key.Key_Enter),
        int(Qt.Key.Key_Space),
        int(Qt.Key.Key_Select),
    }
)


class LeftClickOnlyMenu(QMenu):
    """Allow tray actions to run only when the user left-clicks a menu item."""

    def __init__(self, parent: Optional[Any] = None) -> None:
        super().__init__(parent)
        self._pending_left_click_action: Optional[QAction] = None

    def consume_left_click_trigger(self, action: Optional[QAction]) -> bool:
        """Return whether the action was triggered by a left-click release."""
        allowed = action is not None and action is self._pending_left_click_action
        self._pending_left_click_action = None
        return allowed

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            self._pending_left_click_action = None
            event.ignore()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            self._pending_left_click_action = None
            event.ignore()
            return
        self._pending_left_click_action = self.actionAt(event.pos())
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if int(event.key()) in TRAY_MENU_TRIGGER_KEYS:
            self._pending_left_click_action = None
            event.ignore()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if int(event.key()) in TRAY_MENU_TRIGGER_KEYS:
            self._pending_left_click_action = None
            event.ignore()
            return
        super().keyReleaseEvent(event)


class TrayController:
    """Handle tray events, close behavior, and app exit flow."""

    def __init__(self, app: Any) -> None:
        self.app = app
        self._tray_menu: Optional[LeftClickOnlyMenu] = None
        self._tray_action_show: Optional[QAction] = None
        self._tray_action_exit: Optional[QAction] = None
        self._tray_available: bool = False

    def setup_tray(self, select_icon_path: Callable[[], Optional[Path]]) -> None:
        """Initialize Qt native tray icon and context menu."""
        if not QSystemTrayIcon.isSystemTrayAvailable():
            self._tray_available = False
            self.app.tray_icon = None
            self.app.log_ctrl.warning("系統工具列不可用，停用 tray 功能。")
            return
        self._tray_available = True
        self.app.tray_icon = QSystemTrayIcon(self.app)
        icon_path = select_icon_path()
        if icon_path is not None:
            self.app.tray_icon.setIcon(QIcon(str(icon_path)))
        self.app.tray_icon.setToolTip("YOLO Server")
        self._tray_menu = LeftClickOnlyMenu(self.app)
        self._tray_action_show = QAction("顯示", self.app)
        self._tray_action_exit = QAction("關閉", self.app)
        self._tray_menu.addAction(self._tray_action_show)
        self._tray_menu.addAction(self._tray_action_exit)
        self._tray_menu.triggered.connect(self._handle_tray_menu_triggered)
        self.app.tray_icon.activated.connect(self.on_tray_activated)
        self.app.tray_icon.show()

    def ensure_tray_visible(self) -> None:
        """Ensure tray icon is visible."""
        if self._tray_available and self.app.tray_icon is not None:
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
        if os.name != "nt" or not self._tray_available or self.app.tray_icon is None:
            if os.name == "nt" and not self._tray_available:
                self.app.log_ctrl.warning("系統工具列不可用，改為直接關閉程式。")
            self.exit_app(origin="minimize_to_tray_without_tray")
            return
        self.app.hide()
        self.app.tray_icon.show()

    def restore_window(self) -> None:
        """Restore app window from tray."""
        self.app.showNormal()
        self.app.raise_()
        self.app.activateWindow()

    def show_tray_menu(self, source: str) -> None:
        """Show the tray menu at the current cursor position."""
        if self._tray_menu is None:
            self.app.log_ctrl.warning("Tray menu 尚未建立，無法顯示。source=%s", source)
            return
        cursor_pos = QCursor.pos()
        self.app.log_ctrl.debug(
            "Tray menu popup requested. source=%s x=%s y=%s",
            source,
            cursor_pos.x(),
            cursor_pos.y(),
        )
        self._tray_menu.popup(cursor_pos)

    def request_exit_from_tray(self, origin: str = "tray_menu_exit") -> None:
        """Close the tray menu first, then start the application shutdown flow."""
        if self.app._quitting:
            return
        if self._tray_action_exit is not None:
            self._tray_action_exit.setEnabled(False)
        self.app.log_ctrl.info("Tray 結束請求。origin=%s", origin)
        QTimer.singleShot(0, partial(self.exit_app, origin=origin))

    def _handle_tray_menu_triggered(self, action: QAction) -> None:
        """Dispatch tray actions only when they come from a left-click."""
        if self._tray_menu is None or not self._tray_menu.consume_left_click_trigger(action):
            self.app.log_ctrl.debug(
                "忽略非左鍵觸發的 tray menu 動作。action=%s",
                action.text() if action is not None else "unknown",
            )
            return
        self.app.log_ctrl.debug("Tray menu 左鍵觸發。action=%s", action.text())
        if action is self._tray_action_show:
            self.restore_window()
            return
        if action is self._tray_action_exit:
            self.request_exit_from_tray(origin="tray_menu_exit")

    def exit_app(self, origin: str = "unknown") -> None:
        """Exit app and release runtime resources."""
        if self.app._quitting:
            return
        self.app.log_ctrl.info("GUI 結束流程開始。origin=%s", origin)
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
            self.show_tray_menu("context")

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
                self.exit_app(origin="close_dialog_exit")
                return True
            event.ignore()
            return True
        if behavior == "minimize":
            event.ignore()
            self.minimize_to_tray()
            return True
        event.ignore()
        self.exit_app(origin="close_behavior_exit")
        return True
