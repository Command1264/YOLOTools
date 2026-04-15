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

import server as server_module
import server_controller as server_controller_module
from server import ServerStartupError, YoloServer
from server_controller import ServerController


class DummyDispatcher:
    """Minimal dispatcher stub for startup tests."""

    def __init__(self) -> None:
        self.start_calls = 0
        self.stop_calls = 0
        self.is_ready = False
        self.is_warming_up = False
        self.warmup_error = None
        self.device_name = "unknown"
        self.queue_size = 0
        self.gpu_replica_count = 1

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1

    def wait_until_ready(self, timeout_sec: float | None = None) -> bool:
        return True


class DummyLabel:
    """Minimal label stub used by controller tests."""

    def __init__(self, text: str = "") -> None:
        self.text = text

    def setText(self, text: str) -> None:
        self.text = text


class DummyLogController:
    """Collect log calls without depending on the real logger."""

    def __init__(self) -> None:
        self.error_messages: list[str] = []
        self.exception_messages: list[str] = []

    def error(self, msg: str, *args) -> None:
        self.error_messages.append(msg % args if args else msg)

    def exception(self, msg: str, *args) -> None:
        self.exception_messages.append(msg % args if args else msg)

    def info(self, msg: str, *args) -> None:
        return


class ServerStartupTests(unittest.TestCase):
    """Cover recoverable GUI server startup failures."""

    def test_start_converts_werkzeug_system_exit_to_startup_error(self) -> None:
        """SystemExit from Werkzeug should become a recoverable startup error."""
        dispatcher = DummyDispatcher()

        def raise_system_exit(*args, **kwargs):
            print("嘗試存取通訊端被拒絕，因為存取權限不足。", file=sys.stderr)
            raise SystemExit(1)

        with (
            patch.object(YoloServer, "_create_dispatcher", lambda self, model_path: dispatcher),
            patch.object(YoloServer, "_validate_bind_target", lambda self: None),
            patch.object(server_module, "make_server", side_effect=raise_system_exit),
        ):
            server = YoloServer(model_path="dummy.pt", host="127.0.0.1", port=3000)

            with self.assertRaises(ServerStartupError) as ctx:
                server.start()

        self.assertIn("127.0.0.1:3000", ctx.exception.user_message)
        self.assertIn("系統拒絕綁定", ctx.exception.user_message)
        self.assertEqual(dispatcher.start_calls, 0)
        self.assertEqual(dispatcher.stop_calls, 1)
        self.assertEqual(server.server_state, "stopped")
        self.assertFalse(server.is_running())

    def test_controller_shows_friendly_bind_error_message(self) -> None:
        """Controller should keep the GUI alive and show a readable bind error."""
        expected_message = (
            "無法啟動伺服器，因為系統拒絕綁定 127.0.0.1:3000。\n"
            "可能是該 Port 需要較高權限、已被系統保留，或被安全性軟體封鎖。\n"
            "建議改用其他 Port（例如 60922）後再試一次。\n\n"
            "系統訊息：嘗試存取通訊端被拒絕，因為存取權限不足。"
        )
        shown_errors: list[tuple[str, str]] = []
        running_states: list[bool] = []

        class FailingServer:
            """Server stub that fails during startup."""

            def __init__(self, *args, **kwargs) -> None:
                return

            def start(self) -> None:
                raise ServerStartupError(
                    user_message=expected_message,
                    detail="嘗試存取通訊端被拒絕，因為存取權限不足。",
                )

        app = SimpleNamespace(
            cfg=SimpleNamespace(
                host="127.0.0.1",
                port=3000,
                model_path="dummy.pt",
                gpu_replica_count=4,
                http_profile="default",
            ),
            server=None,
            logger=None,
            log_ctrl=DummyLogController(),
            lbl_status=DummyLabel("狀態：未啟動"),
            lbl_device=DummyLabel("裝置：未啟動"),
            _show_error=lambda title, text: shown_errors.append((title, text)),
        )

        with (
            patch.object(ServerController, "apply_quick_settings", lambda self, require_model: True),
            patch.object(ServerController, "set_running_state", lambda self, running: running_states.append(running)),
            patch.object(server_controller_module, "YoloServer", FailingServer),
        ):
            controller = ServerController(
                app=app,
                config_path=Path("dummy.yml"),
                exec_dir=Path("."),
                select_icon_path=lambda: None,
            )
            controller.start_server()

        self.assertIsNone(app.server)
        self.assertEqual(running_states, [False])
        self.assertEqual(app.lbl_status.text, "狀態：啟動失敗")
        self.assertEqual(app.lbl_device.text, "裝置：未啟動")
        self.assertEqual(shown_errors, [("啟動失敗", expected_message)])
        self.assertTrue(any("host=127.0.0.1 port=3000" in message for message in app.log_ctrl.error_messages))


if __name__ == "__main__":
    unittest.main()
