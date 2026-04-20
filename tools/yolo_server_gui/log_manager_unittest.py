from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GUI_DIR = PROJECT_ROOT / "tools" / "yolo_server_gui"
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))

from log_manager import (
    LOG_FOLDER_NAME,
    get_active_log_file,
    reconfigure_logging,
    resolve_log_root,
    setup_logging,
    shutdown_logging,
)


class LogManagerTests(unittest.TestCase):
    """Cover configurable log root behavior."""

    def tearDown(self) -> None:
        shutdown_logging()

    def test_resolve_log_root_appends_fixed_folder_name(self) -> None:
        """Configured base directory should always end in YOLOServerLogs."""
        self.assertEqual(
            resolve_log_root(Path("D:/CustomLogs")),
            Path("D:/CustomLogs") / LOG_FOLDER_NAME,
        )

    def test_setup_logging_uses_fixed_log_folder_under_base_dir(self) -> None:
        """Initial logging setup should create the log root under the selected base folder."""
        with tempfile.TemporaryDirectory() as temp_dir:
            context = setup_logging(Path(temp_dir), "info")
            self.assertEqual(context.log_root, Path(temp_dir) / LOG_FOLDER_NAME)
            self.assertTrue(context.log_root.exists())
            self.assertIsNotNone(get_active_log_file())
            self.assertTrue(str(get_active_log_file()).startswith(str(context.log_root)))
            shutdown_logging()

    def test_reconfigure_logging_switches_to_new_log_root(self) -> None:
        """Reconfiguring logging should move future log files to the new base folder."""
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first_context = setup_logging(Path(first_dir), "info")
            second_context = reconfigure_logging(Path(second_dir), "info")
            self.assertEqual(first_context.log_root, Path(first_dir) / LOG_FOLDER_NAME)
            self.assertEqual(second_context.log_root, Path(second_dir) / LOG_FOLDER_NAME)
            self.assertNotEqual(first_context.log_root, second_context.log_root)
            self.assertIsNotNone(get_active_log_file())
            self.assertTrue(str(get_active_log_file()).startswith(str(second_context.log_root)))
            shutdown_logging()


if __name__ == "__main__":
    unittest.main()
