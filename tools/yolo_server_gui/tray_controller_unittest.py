from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GUI_DIR = PROJECT_ROOT / "tools" / "yolo_server_gui"
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))

import tray_controller as tray_controller_module
from tray_controller import TrayController


class DummyLogController:
    """Collect warning logs without depending on the real logger."""

    def __init__(self) -> None:
        self.warning_messages: list[str] = []

    def warning(self, msg: str, *args) -> None:
        self.warning_messages.append(msg % args if args else msg)


class DummyAction:
    """Minimal action stub used to assert enabled state changes."""

    def __init__(self) -> None:
        self.enabled_values: list[bool] = []

    def setEnabled(self, enabled: bool) -> None:
        self.enabled_values.append(bool(enabled))


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
        app = SimpleNamespace(_quitting=False)
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
        self.assertIs(scheduled_calls[0][1].__self__, controller)
        self.assertEqual(scheduled_calls[0][1].__name__, "exit_app")


if __name__ == "__main__":
    unittest.main()
