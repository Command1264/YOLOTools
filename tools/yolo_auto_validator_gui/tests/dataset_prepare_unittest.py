from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.yolo_auto_validator_gui.class_mapping_model import build_default_mapping
from tools.yolo_auto_validator_gui.dataset_prepare_service import prepare_dataset
from tools.yolo_auto_validator_gui.source_preview_service import preview_source


class DatasetPrepareTests(unittest.TestCase):
    """驗證 prepared dataset 生成。"""

    def test_prepare_dataset_keeps_unlabeled_images(self) -> None:
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as temp_dir:
            root = Path(source_dir)
            image_dir = root / "images" / "val"
            label_dir = root / "labels" / "val"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            Image.new("RGB", (16, 16), color="white").save(image_dir / "a.jpg")
            Image.new("RGB", (16, 16), color="black").save(image_dir / "b.jpg")
            (label_dir / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            (root / "data.yaml").write_text(
                "path: .\nval: images/val\nnames: [Fire, Smoke]\n",
                encoding="utf-8",
            )

            preview = preview_source(root)
            spec = prepare_dataset(
                preview=preview,
                model_names=["Fire", "Smoke"],
                mapping_rules=build_default_mapping(preview.names, ["Fire", "Smoke"]),
                temp_root=Path(temp_dir),
            )

            prepared_images = sorted((spec.prepared_root / "images" / "val").glob("*.jpg"))
            prepared_labels = sorted((spec.prepared_root / "labels" / "val").glob("*.txt"))

            self.assertEqual(len(prepared_images), 2)
            self.assertEqual(len(prepared_labels), 2)
            self.assertTrue(spec.contains_unlabeled_images)
            self.assertEqual(spec.empty_label_count, 1)
            self.assertTrue(any(path.read_text(encoding="utf-8") == "" for path in prepared_labels))

    def test_prepare_dataset_supports_absolute_yaml_path_root(self) -> None:
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as temp_dir:
            root = Path(source_dir)
            config_dir = root / "config"
            image_dir = root / "images" / "val"
            label_dir = root / "labels" / "val"
            config_dir.mkdir(parents=True)
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            Image.new("RGB", (16, 16), color="red").save(image_dir / "a.jpg")
            (label_dir / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            (config_dir / "data.yaml").write_text(
                f"path: {root.resolve().as_posix()}\nval: images/val\nnames: [Fire]\n",
                encoding="utf-8",
            )

            preview = preview_source(root)
            spec = prepare_dataset(
                preview=preview,
                model_names=["Fire"],
                mapping_rules=build_default_mapping(preview.names, ["Fire"]),
                temp_root=Path(temp_dir),
            )

            prepared_images = sorted((spec.prepared_root / "images" / "val").glob("*.jpg"))
            prepared_labels = sorted((spec.prepared_root / "labels" / "val").glob("*.txt"))

            self.assertEqual(len(prepared_images), 1)
            self.assertEqual(len(prepared_labels), 1)
            self.assertFalse(spec.contains_unlabeled_images)
            self.assertEqual(spec.empty_label_count, 0)

    def test_prepare_dataset_falls_back_to_raw_root_when_absolute_yaml_path_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as temp_dir:
            root = Path(source_dir)
            config_dir = root / "config"
            image_dir = root / "images" / "val"
            label_dir = root / "labels" / "val"
            config_dir.mkdir(parents=True)
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            Image.new("RGB", (16, 16), color="blue").save(image_dir / "a.jpg")
            (label_dir / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            stale_root = (root / "missing_original_root").resolve()
            (config_dir / "data.yaml").write_text(
                f"path: {stale_root.as_posix()}\nval: images/val\nnames: [Fire]\n",
                encoding="utf-8",
            )

            preview = preview_source(root)
            spec = prepare_dataset(
                preview=preview,
                model_names=["Fire"],
                mapping_rules=build_default_mapping(preview.names, ["Fire"]),
                temp_root=Path(temp_dir),
            )

            prepared_images = sorted((spec.prepared_root / "images" / "val").glob("*.jpg"))
            prepared_labels = sorted((spec.prepared_root / "labels" / "val").glob("*.txt"))

            self.assertEqual(len(prepared_images), 1)
            self.assertEqual(len(prepared_labels), 1)
            self.assertFalse(spec.contains_unlabeled_images)
            self.assertEqual(spec.empty_label_count, 0)


if __name__ == "__main__":
    unittest.main()
