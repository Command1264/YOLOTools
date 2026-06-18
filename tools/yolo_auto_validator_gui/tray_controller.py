from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QIcon
from PySide6.QtWidgets import QApplication, QMenu, QMessageBox, QSystemTrayIcon


class TrayController(QObject):
    """System tray 控制器。"""

    def __init__(self, window) -> None:
        super().__init__(window)
        self._window = window
        self._tray_icon: QSystemTrayIcon | None = None
        self._tray_menu: QMenu | None = None
        self._action_show: QAction | None = None
        self._action_exit: QAction | None = None
        self._health_timer = QTimer(self)
        self._health_timer.setInterval(30_000)
        self._health_timer.timeout.connect(self.ensure_tray_visible)

    def setup(self, icon_path: Path | None = None) -> None:
        """建立 tray icon 與 menu。"""
        self._tray_icon = QSystemTrayIcon(self._window)
        if icon_path is not None and icon_path.exists():
            self._tray_icon.setIcon(QIcon(str(icon_path)))
        self._tray_icon.setToolTip("YOLO 自動驗證器")
        self._tray_menu = QMenu(self._window)
        self._action_show = QAction("顯示", self._window)
        self._action_exit = QAction("關閉應用程式", self._window)
        self._action_show.triggered.connect(self.restore_window)
        self._action_exit.triggered.connect(self.request_exit)
        self._tray_menu.addAction(self._action_show)
        self._tray_menu.addAction(self._action_exit)
        self._tray_icon.setContextMenu(self._tray_menu)
        self._tray_icon.activated.connect(self._on_activated)
        self._tray_icon.show()
        self._health_timer.start()

    def ensure_tray_visible(self) -> None:
        """確保 tray icon 可見。"""
        if self._tray_icon is None:
            return
        if not self._tray_icon.isVisible():
            self._tray_icon.show()
        if self._tray_icon.contextMenu() is None and self._tray_menu is not None:
            self._tray_icon.setContextMenu(self._tray_menu)

    def handle_close_event(self, event: QCloseEvent) -> bool:
        """攔截關閉事件。"""
        if getattr(self._window, "_allow_exit", False):
            event.accept()
            return True
        event.ignore()
        self.minimize_to_tray()
        return True

    def minimize_to_tray(self) -> None:
        """縮到 tray。"""
        self._window.hide()
        self.ensure_tray_visible()

    def restore_window(self) -> None:
        """從 tray 還原視窗。"""
        self._window.showNormal()
        self._window.raise_()
        self._window.activateWindow()

    def request_exit(self) -> None:
        """顯式要求結束應用程式。"""
        if QMessageBox.question(self._window, "關閉", "確定要關閉 YOLO 自動驗證器嗎？") != QMessageBox.Yes:
            return
        self._window._allow_exit = True
        if self._tray_icon is not None:
            self._tray_icon.hide()
        app = QApplication.instance()
        if app is not None:
            app.quit()

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in {QSystemTrayIcon.Trigger, QSystemTrayIcon.DoubleClick}:
            self.restore_window()
