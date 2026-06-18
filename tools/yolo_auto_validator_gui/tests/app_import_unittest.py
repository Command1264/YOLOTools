from __future__ import annotations

import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tools.yolo_auto_validator_gui import app as app_module


class AppImportTests(unittest.TestCase):
    """確認匯入 app 不會連帶載入 ultralytics。"""

    def test_import_app_does_not_import_ultralytics(self) -> None:
        script = (
            "import sys;"
            "mod=__import__('tools.yolo_auto_validator_gui.app', fromlist=['main']);"
            "print(mod.__name__);"
            "print(any(name.startswith('ultralytics') for name in sys.modules))"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
        )
        output_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        self.assertEqual(output_lines[0], "tools.yolo_auto_validator_gui.app")
        self.assertEqual(output_lines[1], "False")

    def test_has_terminal_session_detects_tty_stream(self) -> None:
        streams = (
            SimpleNamespace(isatty=lambda: False),
            SimpleNamespace(isatty=lambda: True),
            SimpleNamespace(isatty=lambda: False),
        )
        with (
            patch.object(app_module.sys, "stdin", streams[0]),
            patch.object(app_module.sys, "stdout", streams[1]),
            patch.object(app_module.sys, "stderr", streams[2]),
        ):
            self.assertTrue(app_module._has_terminal_session())

    def test_request_graceful_exit_marks_window_and_quits_app(self) -> None:
        state = {"quit_called": False}

        class FakeApp:
            def quit(self) -> None:
                state["quit_called"] = True

        window = SimpleNamespace(_allow_exit=False)
        app_module._request_graceful_exit(FakeApp(), window, "unit-test")

        self.assertTrue(window._allow_exit)
        self.assertTrue(state["quit_called"])


if __name__ == "__main__":
    unittest.main()
