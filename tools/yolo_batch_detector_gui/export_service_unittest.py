from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.yolo_batch_detector_gui.export_service import write_detection_exports
from tools.yolo_batch_detector_gui.models import DetectionBox, ExportOptions, ImagePrediction
from tools.yolo_batch_detector_gui.statistics_service import build_detection_summary


class ExportServiceTests(unittest.TestCase):
    """Cover CSV, JSON, and annotated-image export behavior."""

    def test_write_detection_exports_creates_selected_outputs(self) -> None:
        """Selected export formats should be written with stable result content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            source_image = image_dir / "sample.jpg"
            source_image.write_bytes(b"source")
            annotated_image = root / "annotated-source.jpg"
            annotated_image.write_bytes(b"annotated")
            output_dir = root / "out"
            prediction = ImagePrediction(
                image_path=source_image,
                detections=(
                    DetectionBox(
                        class_id=1,
                        label="smoke",
                        confidence=0.876,
                        x1=10,
                        y1=20,
                        x2=30,
                        y2=40,
                    ),
                ),
                labels=("smoke",),
                annotated_image_path=annotated_image,
            )
            empty_prediction = ImagePrediction(image_path=image_dir / "empty.jpg", detections=(), labels=())
            summary = build_detection_summary([prediction, empty_prediction])

            result = write_detection_exports(
                predictions=[prediction, empty_prediction],
                summary=summary,
                model_labels={0: "fire", 1: "smoke"},
                inference_settings={"conf": 0.25, "iou": 0.45, "device": ""},
                input_root=image_dir,
                options=ExportOptions(
                    output_dir=output_dir,
                    write_csv=True,
                    write_json=True,
                    write_annotated_images=True,
                ),
            )

            csv_path = result.files["csv"]
            json_path = result.files["json"]
            copied_image = output_dir / "annotated" / "sample.jpg"
            with csv_path.open("r", encoding="utf-8-sig", newline="") as file_obj:
                rows = list(csv.DictReader(file_obj))
            with json_path.open("r", encoding="utf-8") as file_obj:
                payload = json.load(file_obj)
            copied_exists = copied_image.exists()
            copied_bytes = copied_image.read_bytes() if copied_exists else b""

        self.assertEqual(rows[0]["status"], "smoke")
        self.assertEqual(rows[0]["label"], "smoke")
        self.assertEqual(rows[1]["status"], "no_detection")
        self.assertEqual(payload["summary"]["per_label_image_counts"], {"smoke": 1})
        self.assertEqual(payload["model_labels"], ["fire", "smoke"])
        self.assertTrue(copied_exists)
        self.assertEqual(copied_bytes, b"annotated")

    def test_write_detection_exports_skips_unselected_outputs(self) -> None:
        """Export options should control which files are produced."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "out"
            prediction = ImagePrediction(image_path=root / "empty.jpg", detections=(), labels=())
            summary = build_detection_summary([prediction])

            result = write_detection_exports(
                predictions=[prediction],
                summary=summary,
                model_labels={},
                inference_settings={},
                input_root=root,
                options=ExportOptions(
                    output_dir=output_dir,
                    write_csv=False,
                    write_json=True,
                    write_annotated_images=False,
                ),
            )

        self.assertEqual(set(result.files), {"json"})
        self.assertFalse((output_dir / "detections.csv").exists())
        self.assertFalse((output_dir / "annotated").exists())


if __name__ == "__main__":
    unittest.main()
