from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.yolo_batch_detector_gui.config_store import DetectorConfig, load_config, save_config


class ConfigStoreTests(unittest.TestCase):
    """Cover persistent GUI configuration storage."""

    def test_save_and_load_config_round_trip(self) -> None:
        """Config should persist paths, export formats, thresholds, and device."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "detector_config.json"
            config = DetectorConfig(
                model_path="D:/models/fire.pt",
                input_path="D:/images",
                output_dir="D:/exports",
                export_csv=False,
                export_json=True,
                export_annotated_images=True,
                conf=0.35,
                iou=0.55,
                device="cuda:0",
            )

            save_config(config_path, config)
            loaded = load_config(config_path)

        self.assertEqual(loaded, config)

    def test_load_config_uses_defaults_for_missing_fields(self) -> None:
        """Older or partial config files should keep safe defaults."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "detector_config.json"
            config_path.write_text(json.dumps({"model_path": "model.pt"}), encoding="utf-8")

            loaded = load_config(config_path)

        self.assertEqual(loaded.model_path, "model.pt")
        self.assertEqual(loaded.input_path, "")
        self.assertEqual(loaded.output_dir, "")
        self.assertTrue(loaded.export_csv)
        self.assertTrue(loaded.export_json)
        self.assertFalse(loaded.export_annotated_images)
        self.assertEqual(loaded.conf, 0.25)
        self.assertEqual(loaded.iou, 0.45)
        self.assertEqual(loaded.device, "")


if __name__ == "__main__":
    unittest.main()
