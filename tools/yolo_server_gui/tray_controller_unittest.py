from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QAction, QMouseEvent
from PySide6.QtWidgets import QApplication

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GUI_DIR = PROJECT_ROOT / "tools" / "yolo_server_gui"
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))

import tray_controller as tray_controller_module
from tray_controller import LeftClickOnlyMenu, TrayController


QT_APP = QApplication.instance() or QApplication([])


class DummyLogController:
    """Collect warning logs without depending on the real logger."""

    def __init__(self) -> None:
        self.warning_messages: list[str] = []
        self.debug_messages: list[str] = []
        self.info_messages: list[str] = []

    def warning(self, msg: str, *args) -> None:
        self.warning_messages.append(msg % args if args else msg)

    def debug(self, msg: str, *args) -> None:
        self.debug_messages.append(msg % args if args else msg)

    def info(self, msg: str, *args) -> None:
        self.info_messages.append(msg % args if args else msg)


class DummyAction:
    """Minimal action stub used to assert enabled state changes."""

    def __init__(self, text: str = "") -> None:
        self._text = text
        self.enabled_values: list[bool] = []

    def setEnabled(self, enabled: bool) -> None:
        self.enabled_values.append(bool(enabled))

    def text(self) -> str:
        return self._text


class DummyMenu:
    """Control whether a tray action should be treated as a left-click trigger."""

    def __init__(self, allow_trigger: bool) -> None:
        self.allow_trigger = allow_trigger
        self.seen_actions: list[object] = []
        self.popup_positions: list[object] = []
        self.reset_calls: int = 0

    def consume_left_click_trigger(self, action: object) -> bool:
        self.seen_actions.append(action)
        allowed = self.allow_trigger
        self.allow_trigger = False
        return allowed

    def popup(self, position: object) -> None:
        self.popup_positions.append(position)

    def reset_pending_trigger(self) -> None:
        self.reset_calls += 1
        self.allow_trigger = False


class TrayControllerTests(unittest.TestCase):
    """Cover tray setup and shutdown entry points."""

    def test_setup_tray_skips_native_tray_when_unavailable(self) -> None:
        """System tray should be optional instead of crashing setup."""
        app = SimpleNamespace(
            tray_icon="unexpected",
            log_ctrl=DummyLogController(),
        )
        controller = TrayController(app)

        with patch.object(tray_controller_module.QSystemTrayIcon, "isSystemTrayAvailable", return_value=False):
            controller.setup_tray(lambda: None)

        self.assertIsNone(app.tray_icon)
        self.assertTrue(any("系統工具列不可用" in msg for msg in app.log_ctrl.warning_messages))

    def test_request_exit_from_tray_defers_shutdown_until_menu_closes(self) -> None:
        """Tray exit should schedule shutdown after the native menu closes."""
        app = SimpleNamespace(_quitting=False, log_ctrl=DummyLogController())
        controller = TrayController(app)
        controller._tray_action_exit = DummyAction()
        scheduled_calls: list[tuple[int, object]] = []

        with patch.object(
            tray_controller_module.QTimer,
            "singleShot",
            side_effect=lambda msec, callback: scheduled_calls.append((msec, callback)),
        ):
            controller.request_exit_from_tray()

        self.assertEqual(controller._tray_action_exit.enabled_values, [False])
        self.assertEqual(len(scheduled_calls), 1)
        self.assertEqual(scheduled_calls[0][0], 0)
        self.assertEqual(scheduled_calls[0][1].func.__self__, controller)
        self.assertEqual(scheduled_calls[0][1].func.__name__, "exit_app")
        self.assertEqual(scheduled_calls[0][1].keywords, {"origin": "tray_menu_exit"})
        self.assertTrue(any("Tray 結束請求" in msg for msg in app.log_ctrl.info_messages))

    def test_handle_tray_menu_triggered_ignores_non_left_click_triggers(self) -> None:
        """Tray action callbacks should not run unless the menu reports a left-click trigger."""
        app = SimpleNamespace(_quitting=False, log_ctrl=DummyLogController())
        controller = TrayController(app)
        controller._tray_menu = DummyMenu(allow_trigger=False)
        controller._tray_action_show = DummyAction("顯示")
        restore_calls: list[str] = []
        controller.restore_window = lambda: restore_calls.append("restore")  # type: ignore[method-assign]

        controller._handle_tray_menu_triggered(controller._tray_action_show)  # type: ignore[arg-type]

        self.assertEqual(restore_calls, [])
        self.assertEqual(controller._tray_menu.seen_actions, [controller._tray_action_show])
        self.assertTrue(any("忽略非左鍵觸發" in msg for msg in app.log_ctrl.debug_messages))

    def test_handle_tray_menu_triggered_runs_show_action_for_left_click(self) -> None:
        """Tray show action should still run when triggered by a left-click."""
        app = SimpleNamespace(_quitting=False, log_ctrl=DummyLogController())
        controller = TrayController(app)
        controller._tray_menu = DummyMenu(allow_trigger=True)
        controller._tray_action_show = DummyAction("顯示")
        restore_calls: list[str] = []
        controller.restore_window = lambda: restore_calls.append("restore")  # type: ignore[method-assign]

        with patch.object(tray_controller_module.time, "monotonic", return_value=10.0):
            controller._handle_tray_menu_triggered(controller._tray_action_show)  # type: ignore[arg-type]

        self.assertEqual(restore_calls, ["restore"])
        self.assertEqual(controller._tray_menu.seen_actions, [controller._tray_action_show])

    def test_on_tray_activated_context_shows_custom_popup_menu(self) -> None:
        """Right-click tray activation should use the custom Qt popup path."""
        app = SimpleNamespace(_quitting=False, log_ctrl=DummyLogController())
        controller = TrayController(app)
        controller._tray_menu = DummyMenu(allow_trigger=False)
        fake_point = QPoint(100, 200)

        with (
            patch.object(tray_controller_module.QCursor, "pos", return_value=fake_point),
            patch.object(tray_controller_module.time, "monotonic", return_value=1.0),
        ):
            controller.on_tray_activated(tray_controller_module.QSystemTrayIcon.Context)

        self.assertEqual(
            controller._tray_menu.popup_positions,
            [fake_point + QPoint(0, tray_controller_module.TRAY_MENU_CURSOR_OFFSET_PX)],
        )
        self.assertTrue(any("Tray context menu requested." in msg for msg in app.log_ctrl.debug_messages))
        self.assertTrue(any("Tray menu popup requested." in msg for msg in app.log_ctrl.debug_messages))
        self.assertEqual(controller._tray_menu.reset_calls, 1)

    def test_show_tray_menu_debounces_rapid_popup_requests(self) -> None:
        """Rapid repeated popup requests should be ignored while the debounce window is active."""
        app = SimpleNamespace(_quitting=False, log_ctrl=DummyLogController())
        controller = TrayController(app)
        controller._tray_menu = DummyMenu(allow_trigger=False)
        fake_point = QPoint(50, 60)

        with (
            patch.object(tray_controller_module.QCursor, "pos", return_value=fake_point),
            patch.object(tray_controller_module.time, "monotonic", side_effect=[1.0, 1.1]),
        ):
            controller.show_tray_menu("context")
            controller.show_tray_menu("context")

        self.assertEqual(len(controller._tray_menu.popup_positions), 1)
        self.assertTrue(any("忽略過快的 tray menu 顯示要求" in msg for msg in app.log_ctrl.debug_messages))

    def test_handle_tray_menu_triggered_runs_left_click_action_inside_guard_window(self) -> None:
        """A valid left-click should still work even if it happens immediately after popup."""
        app = SimpleNamespace(_quitting=False, log_ctrl=DummyLogController())
        controller = TrayController(app)
        controller._tray_menu = DummyMenu(allow_trigger=True)
        controller._tray_action_show = DummyAction("顯示")
        controller._tray_menu_guard_until = 5.5
        restore_calls: list[str] = []
        controller.restore_window = lambda: restore_calls.append("restore")  # type: ignore[method-assign]

        with patch.object(tray_controller_module.time, "monotonic", return_value=5.2):
            controller._handle_tray_menu_triggered(controller._tray_action_show)  # type: ignore[arg-type]

        self.assertEqual(restore_calls, ["restore"])
        self.assertTrue(any("Tray menu 左鍵觸發" in msg for msg in app.log_ctrl.debug_messages))

    def test_about_to_hide_defers_reset_until_after_left_click_trigger_dispatch(self) -> None:
        """Menu hide should not clear a valid pending left-click before the action is dispatched."""
        app = SimpleNamespace(_quitting=False, log_ctrl=DummyLogController())
        controller = TrayController(app)
        controller._tray_menu = DummyMenu(allow_trigger=True)
        controller._tray_action_show = DummyAction("顯示")
        restore_calls: list[str] = []
        scheduled_calls: list[tuple[int, object]] = []
        controller.restore_window = lambda: restore_calls.append("restore")  # type: ignore[method-assign]

        with patch.object(
            tray_controller_module.QTimer,
            "singleShot",
            side_effect=lambda msec, callback: scheduled_calls.append((msec, callback)),
        ):
            controller._on_tray_menu_about_to_hide()

        self.assertEqual(scheduled_calls[0][0], 0)

        with patch.object(tray_controller_module.time, "monotonic", return_value=10.0):
            controller._handle_tray_menu_triggered(controller._tray_action_show)  # type: ignore[arg-type]

        self.assertEqual(restore_calls, ["restore"])
        scheduled_calls[0][1]()
        self.assertEqual(controller._tray_menu.reset_calls, 1)

    def test_on_tray_activated_ignores_reentry_while_menu_visible(self) -> None:
        """Tray activation should not reopen or reroute while the menu is already visible."""
        app = SimpleNamespace(_quitting=False, log_ctrl=DummyLogController())
        controller = TrayController(app)
        controller._tray_menu = DummyMenu(allow_trigger=False)
        controller._tray_menu_visible = True

        controller.on_tray_activated(tray_controller_module.QSystemTrayIcon.Context)

        self.assertEqual(controller._tray_menu.popup_positions, [])
        self.assertTrue(any("menu 已顯示" in msg for msg in app.log_ctrl.debug_messages))


class LeftClickOnlyMenuTests(unittest.TestCase):
    """Cover the actual tray menu click gesture rules."""

    def _build_mouse_event(
        self,
        event_type: QEvent.Type,
        button: Qt.MouseButton,
        buttons: Qt.MouseButton,
    ) -> QMouseEvent:
        return QMouseEvent(
            event_type,
            QPointF(1.0, 1.0),
            button,
            buttons,
            Qt.KeyboardModifier.NoModifier,
        )

    def test_release_without_matching_left_press_does_not_trigger(self) -> None:
        """A release-only sequence must not be treated as a valid left-click action."""
        menu = LeftClickOnlyMenu()
        action = QAction("關閉", menu)
        release_event = self._build_mouse_event(
            QEvent.Type.MouseButtonRelease,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
        )

        with (
            patch.object(tray_controller_module.QMenu, "mouseReleaseEvent", autospec=True),
            patch.object(menu, "actionAt", return_value=action),
        ):
            menu.mouseReleaseEvent(release_event)

        self.assertFalse(menu.consume_left_click_trigger(action))

    def test_left_press_and_release_on_same_action_triggers(self) -> None:
        """A complete left-click on the same action should still be accepted."""
        menu = LeftClickOnlyMenu()
        action = QAction("顯示", menu)
        press_event = self._build_mouse_event(
            QEvent.Type.MouseButtonPress,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
        )
        release_event = self._build_mouse_event(
            QEvent.Type.MouseButtonRelease,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
        )

        with (
            patch.object(tray_controller_module.QMenu, "mousePressEvent", autospec=True),
            patch.object(tray_controller_module.QMenu, "mouseReleaseEvent", autospec=True),
            patch.object(menu, "actionAt", side_effect=[action, action]),
        ):
            menu.mousePressEvent(press_event)
            menu.mouseReleaseEvent(release_event)

        self.assertTrue(menu.consume_left_click_trigger(action))

    def test_left_press_and_release_on_different_actions_does_not_trigger(self) -> None:
        """The click should be rejected when press and release land on different actions."""
        menu = LeftClickOnlyMenu()
        press_action = QAction("顯示", menu)
        release_action = QAction("關閉", menu)
        press_event = self._build_mouse_event(
            QEvent.Type.MouseButtonPress,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
        )
        release_event = self._build_mouse_event(
            QEvent.Type.MouseButtonRelease,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
        )

        with (
            patch.object(tray_controller_module.QMenu, "mousePressEvent", autospec=True),
            patch.object(tray_controller_module.QMenu, "mouseReleaseEvent", autospec=True),
            patch.object(menu, "actionAt", side_effect=[press_action, release_action]),
        ):
            menu.mousePressEvent(press_event)
            menu.mouseReleaseEvent(release_event)

        self.assertFalse(menu.consume_left_click_trigger(release_action))


if __name__ == "__main__":
    unittest.main()
