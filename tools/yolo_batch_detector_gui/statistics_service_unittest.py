from __future__ import annotations

import unittest
from pathlib import Path

from tools.yolo_batch_detector_gui.models import DetectionBox, ImagePrediction
from tools.yolo_batch_detector_gui.statistics_service import build_detection_summary, classify_prediction


def _box(label: str, class_id: int = 0) -> DetectionBox:
    return DetectionBox(class_id=class_id, label=label, confidence=0.9, x1=1, y1=2, x2=3, y2=4)


class StatisticsServiceTests(unittest.TestCase):
    """Cover image-level and label-level summary rules."""

    def test_no_detection_prediction_is_counted_as_none(self) -> None:
        """Images without boxes should be counted in the no-detection bucket."""
        prediction = ImagePrediction(image_path=Path("empty.jpg"), detections=(), labels=())

        summary = build_detection_summary([prediction])

        self.assertEqual(classify_prediction(prediction), "沒有")
        self.assertEqual(summary.total_images, 1)
        self.assertEqual(summary.no_detection_count, 1)
        self.assertEqual(summary.multi_label_count, 0)

    def test_duplicate_boxes_for_same_label_count_as_single_label_image(self) -> None:
        """Multiple boxes with the same label should count once for that image."""
        prediction = ImagePrediction(
            image_path=Path("fire.jpg"),
            detections=(_box("fire"), _box("fire")),
            labels=("fire",),
        )

        summary = build_detection_summary([prediction])

        self.assertEqual(classify_prediction(prediction), "fire")
        self.assertEqual(summary.single_label_counts, {"fire": 1})
        self.assertEqual(summary.per_label_image_counts, {"fire": 1})
        self.assertEqual(summary.multi_label_count, 0)

    def test_multiple_unique_labels_count_as_multi_and_each_label_incremented(self) -> None:
        """Images with different labels should increment each per-label image count."""
        prediction = ImagePrediction(
            image_path=Path("mixed.jpg"),
            detections=(_box("fire", 0), _box("smoke", 1), _box("smoke", 1)),
            labels=("fire", "smoke"),
        )

        summary = build_detection_summary([prediction])

        self.assertEqual(classify_prediction(prediction), "多個 label")
        self.assertEqual(summary.single_label_counts, {})
        self.assertEqual(summary.multi_label_count, 1)
        self.assertEqual(summary.per_label_image_counts, {"fire": 1, "smoke": 1})


if __name__ == "__main__":
    unittest.main()
