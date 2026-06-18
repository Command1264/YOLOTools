from __future__ import annotations

import unittest
from pathlib import Path

from tools.yolo_batch_detector_gui.yolo_model_service import YoloModelService, normalize_model_labels


class _FakeScalar:
    def __init__(self, value: float) -> None:
        self._value = value

    def item(self) -> float:
        return self._value


class _FakeCoord:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def cpu(self) -> "_FakeCoord":
        return self

    def numpy(self) -> "_FakeArray":
        return _FakeArray(self._values)


class _FakeArray:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def astype(self, _dtype: object) -> "_FakeArray":
        return self

    def tolist(self) -> list[int]:
        return [int(value) for value in self._values]


class _FakeNestedArray(_FakeArray):
    def tolist(self) -> list[list[float]]:
        return [self._values]


class _FakeNestedCoord(_FakeCoord):
    def numpy(self) -> "_FakeNestedArray":
        return _FakeNestedArray(self._values)


class _FakeBox:
    def __init__(self, class_id: int, confidence: float, xyxy: list[float]) -> None:
        self.cls = _FakeScalar(class_id)
        self.conf = _FakeScalar(confidence)
        self.xyxy = [_FakeCoord(xyxy)]


class _FakeNestedBox(_FakeBox):
    def __init__(self, class_id: int, confidence: float, xyxy: list[float]) -> None:
        self.cls = _FakeScalar(class_id)
        self.conf = _FakeScalar(confidence)
        self.xyxy = _FakeNestedCoord(xyxy)


class _FakeResult:
    def __init__(self, boxes: list[_FakeBox]) -> None:
        self.boxes = boxes


class _FakeYolo:
    def __init__(self, _model_path: str) -> None:
        self.names = {"1": "smoke", "0": "fire"}

    def predict(self, **_kwargs: object) -> list[_FakeResult]:
        return [_FakeResult([_FakeBox(1, 0.91, [10.2, 20.8, 30.1, 40.9])])]


class YoloModelServiceTests(unittest.TestCase):
    """Cover YOLO label normalization and result parsing."""

    def test_normalize_model_labels_accepts_dict_and_list_names(self) -> None:
        """Model names should become integer-keyed labels ordered by class id."""
        self.assertEqual(normalize_model_labels({"1": "smoke", "0": "fire"}), {0: "fire", 1: "smoke"})
        self.assertEqual(normalize_model_labels(["fire", "smoke"]), {0: "fire", 1: "smoke"})

    def test_load_model_labels_uses_factory_and_normalizes_names(self) -> None:
        """Loading a model should expose normalized labels."""
        service = YoloModelService(yolo_factory=_FakeYolo)

        labels = service.load_model(Path("fake.pt"))

        self.assertEqual(labels, {0: "fire", 1: "smoke"})

    def test_predict_image_parses_detection_boxes_from_yolo_result(self) -> None:
        """YOLO result boxes should be converted into project detection models."""
        service = YoloModelService(yolo_factory=_FakeYolo)
        service.load_model(Path("fake.pt"))

        prediction = service.parse_result(Path("image.jpg"), _FakeResult([_FakeBox(1, 0.91, [10.2, 20.8, 30.1, 40.9])]))

        self.assertEqual(prediction.labels, ("smoke",))
        self.assertEqual(len(prediction.detections), 1)
        self.assertEqual(prediction.detections[0].label, "smoke")
        self.assertEqual(prediction.detections[0].confidence, 0.91)
        self.assertEqual(prediction.detections[0].xyxy, (10, 20, 30, 40))

    def test_parse_result_flattens_nested_xyxy_arrays(self) -> None:
        """YOLO tensor coordinates may become [[x1, y1, x2, y2]] after tolist()."""
        service = YoloModelService(yolo_factory=_FakeYolo)
        service.load_model(Path("fake.pt"))

        prediction = service.parse_result(
            Path("image.jpg"),
            _FakeResult([_FakeNestedBox(1, 0.91, [10.2, 20.8, 30.1, 40.9])]),
        )

        self.assertEqual(prediction.detections[0].xyxy, (10, 20, 30, 40))


if __name__ == "__main__":
    unittest.main()
