from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtWidgets import QHBoxLayout

from PySide6.QtWidgets import QApplication

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GUI_DIR = PROJECT_ROOT / "tools" / "yolo_batch_detector_gui"
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))

from main_window import MainWindow
from models import ImagePrediction


QT_APP = QApplication.instance() or QApplication([])


class _DeletedThreadStub:
    def isRunning(self) -> bool:  # noqa: N802
        raise RuntimeError("Internal C++ object (PySide6.QtCore.QThread) already deleted.")


class _FakeErrorBox:
    instances: list["_FakeErrorBox"] = []

    def __init__(self, parent, title: str, message: str) -> None:
        self.parent = parent
        self.title = title
        self.message = message
        self.executed = False
        self.instances.append(self)

    def exec(self) -> int:
        self.executed = True
        return 0


class MainWindowTests(unittest.TestCase):
    """Cover GUI lifecycle edge cases."""

    def test_request_shutdown_ignores_already_deleted_thread_wrappers(self) -> None:
        """Closing the window should not show errors from stale QThread wrappers."""
        window = MainWindow()
        window._prediction_thread = _DeletedThreadStub()

        try:
            window.request_shutdown()
        finally:
            window.close()

    def test_error_result_double_click_slot_opens_copyable_error_dialog(self) -> None:
        """Double-clicking a failed result row should open a copyable error dialog."""
        window = MainWindow()
        _FakeErrorBox.instances.clear()
        prediction = ImagePrediction(
            image_path=Path("bad.jpg"),
            detections=(),
            labels=(),
            error_message="YOLO 推論失敗：bad bbox",
        )

        try:
            window._add_prediction(prediction)
            with patch("main_window_results.CopyableErrorMessageBox", _FakeErrorBox):
                window._show_error_for_result_item(window.tbl_results.item(0, 4))
        finally:
            window.close()

        self.assertEqual(len(_FakeErrorBox.instances), 1)
        self.assertEqual(_FakeErrorBox.instances[0].message, "YOLO 推論失敗：bad bbox")
        self.assertTrue(_FakeErrorBox.instances[0].executed)

    def test_model_input_and_output_dialogs_remember_independent_last_dirs(self) -> None:
        """Each browse category should open from its own last selected directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            window = MainWindow(config_path=root / "detector_config.json")
            try:
                model_dir = root / "models"
                image_dir = root / "images"
                output_dir = root / "exports"
                next_model_dir = root / "next_models"
                next_input_dir = root / "next_images"
                next_output_dir = root / "next_exports"
                for folder in (model_dir, image_dir, output_dir, next_model_dir, next_input_dir, next_output_dir):
                    folder.mkdir()
                first_model = model_dir / "first.pt"
                second_model = next_model_dir / "second.pt"
                first_image = image_dir / "first.jpg"
                first_model.write_bytes(b"fake")
                second_model.write_bytes(b"fake")
                first_image.write_bytes(b"fake")
                open_file_calls: list[str] = []
                existing_dir_calls: list[str] = []

                def fake_open_file(_parent, _title, initial_dir, _filter):
                    open_file_calls.append(initial_dir)
                    if len(open_file_calls) == 1:
                        return str(first_model), ""
                    if len(open_file_calls) == 2:
                        return str(first_image), ""
                    return str(second_model), ""

                def fake_existing_dir(_parent, _title, initial_dir):
                    existing_dir_calls.append(initial_dir)
                    if len(existing_dir_calls) == 1:
                        return str(next_input_dir)
                    return str(next_output_dir)

                with (
                    patch("main_window_browse.QFileDialog.getOpenFileName", side_effect=fake_open_file),
                    patch("main_window_browse.QFileDialog.getExistingDirectory", side_effect=fake_existing_dir),
                    patch("main_window_browse.Path.cwd", return_value=root),
                ):
                    window._pick_model()
                    window._pick_image()
                    window._pick_folder()
                    window._pick_output_dir()
                    window._pick_model()

                self.assertEqual(open_file_calls, [str(root), str(root), str(model_dir)])
                self.assertEqual(existing_dir_calls, [str(image_dir), str(root)])
                self.assertEqual(window.ent_input.text(), str(next_input_dir))
                self.assertEqual(window.ent_output.text(), str(next_output_dir))

            finally:
                window.close()

    def test_window_applies_and_saves_config_values(self) -> None:
        """Window should load persisted settings and save current UI state."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "detector_config.json"
            config_path.write_text(
                (
                    '{"model_path":"model.pt","input_path":"images","output_dir":"exports",'
                    '"export_csv":false,"export_json":true,"export_annotated_images":true,'
                    '"conf":0.33,"iou":0.66,"device":"cuda:0"}'
                ),
                encoding="utf-8",
            )
            window = MainWindow(config_path=config_path)

            try:
                self.assertEqual(window.ent_model.text(), "model.pt")
                self.assertEqual(window.ent_input.text(), "images")
                self.assertEqual(window.ent_output.text(), "exports")
                self.assertFalse(window.chk_csv.isChecked())
                self.assertTrue(window.chk_json.isChecked())
                self.assertTrue(window.chk_annotated.isChecked())
                self.assertEqual(window.sp_conf.value(), 0.33)
                self.assertEqual(window.sp_iou.value(), 0.66)
                self.assertEqual(window.ent_device.text(), "cuda:0")

                window.ent_model.setText("new.pt")
                window.chk_csv.setChecked(True)
                window.sp_conf.setValue(0.44)
                window._save_config()
            finally:
                window.close()

            payload = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["model_path"], "new.pt")
        self.assertTrue(payload["export_csv"])
        self.assertEqual(payload["conf"], 0.44)

    def test_export_format_row_uses_compact_checkbox_layout(self) -> None:
        """Export checkboxes should sit next to each other without first-item stretch."""
        window = MainWindow()

        try:
            layout = window.chk_csv.parentWidget().layout()
            self.assertIsInstance(layout, QHBoxLayout)
            self.assertEqual(layout.stretch(0), 0)
            self.assertEqual(layout.stretch(1), 0)
            self.assertEqual(layout.stretch(2), 0)
            self.assertGreaterEqual(layout.count(), 4)
            self.assertIsNone(layout.itemAt(3).widget())
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
