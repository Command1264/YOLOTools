from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from tools.yolo_auto_validator_gui.source_preview_service import preview_source


class SourcePreviewTests(unittest.TestCase):
    """驗證資料集來源預檢。"""

    def test_preview_folder_reads_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "images" / "val").mkdir(parents=True)
            (root / "labels" / "val").mkdir(parents=True)
            (root / "data.yaml").write_text(
                "path: .\nval: images/val\nnames: [Fire, Smoke]\n",
                encoding="utf-8",
            )
            preview = preview_source(root)
        self.assertEqual(preview.source_kind, "folder")
        self.assertEqual(preview.names, ["Fire", "Smoke"])
        self.assertEqual(preview.split_keys_present, ("val",))

    def test_preview_zip_reads_yaml_member(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive_path = root / "dataset.zip"
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(
                    "nested/data.yaml",
                    "path: .\nval: images/val\nnames:\n  0: Fire\n  1: Smoke\n",
                )
            preview = preview_source(archive_path)
        self.assertEqual(preview.source_kind, "zip")
        self.assertEqual(preview.yaml_locator, "nested/data.yaml")
        self.assertEqual(preview.names, ["Fire", "Smoke"])


if __name__ == "__main__":
    unittest.main()

