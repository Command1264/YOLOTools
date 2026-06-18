from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from pathlib import Path

if __package__ in {None, ""}:
    from models import DetectionBox, ImagePrediction, YoloModelError
else:
    from .models import DetectionBox, ImagePrediction, YoloModelError


def normalize_model_labels(raw_names: object) -> dict[int, str]:
    """Normalize Ultralytics model names into an integer-keyed mapping.

    Args:
        raw_names (object): Value from `model.names` or `model.model.names`.

    Returns:
        dict[int, str]: Class id to label mapping.
    """
    if isinstance(raw_names, dict):
        normalized: dict[int, str] = {}
        for key, value in raw_names.items():
            normalized[int(key)] = str(value)
        return dict(sorted(normalized.items()))
    if isinstance(raw_names, Sequence) and not isinstance(raw_names, (str, bytes, bytearray)):
        return {index: str(value) for index, value in enumerate(raw_names)}
    return {}


class YoloModelService:
    """Load YOLO models and convert inference results into project models."""

    def __init__(self, yolo_factory: Callable[[str], object] | None = None) -> None:
        """Initialize the service.

        Args:
            yolo_factory (Callable[[str], object] | None): Optional YOLO constructor for tests.
        """
        self._yolo_factory = yolo_factory
        self._model: object | None = None
        self._model_path: Path | None = None
        self.model_labels: dict[int, str] = {}

    def load_model(self, model_path: Path | str) -> dict[int, str]:
        """Load a YOLO model and return its labels.

        Args:
            model_path (Path | str): YOLO `.pt` model path.

        Returns:
            dict[int, str]: Model labels keyed by class id.

        Raises:
            YoloModelError: If model loading fails.
        """
        path = Path(model_path)
        try:
            factory = self._yolo_factory or self._import_yolo_factory()
            self._model = factory(str(path))
            self._model_path = path
            raw_names = getattr(self._model, "names", None)
            if raw_names is None and getattr(self._model, "model", None) is not None:
                raw_names = getattr(getattr(self._model, "model"), "names", None)
            self.model_labels = normalize_model_labels(raw_names)
            return dict(self.model_labels)
        except Exception as exc:
            raise YoloModelError(f"載入 YOLO 模型失敗：{path}") from exc

    def predict_image(
        self,
        image_path: Path,
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = "",
        annotated_dir: Path | None = None,
        relative_root: Path | None = None,
    ) -> ImagePrediction:
        """Run YOLO prediction for one image path.

        Args:
            image_path (Path): Source image path.
            conf (float): Confidence threshold.
            iou (float): IoU threshold.
            device (str): Optional execution device.
            annotated_dir (Path | None): Optional directory for rendered result.
            relative_root (Path | None): Root used to preserve annotated image subpaths.

        Returns:
            ImagePrediction: Parsed prediction, including per-image errors.

        Raises:
            YoloModelError: If the model has not been loaded.
        """
        if self._model is None:
            raise YoloModelError("YOLO 模型尚未載入。")
        try:
            kwargs: dict[str, object] = {
                "source": str(image_path),
                "conf": conf,
                "iou": iou,
                "verbose": False,
            }
            if device.strip():
                kwargs["device"] = device.strip()
            results = list(getattr(self._model, "predict")(**kwargs))
            result_item = results[0] if results else None
            prediction = self.parse_result(image_path, result_item)
            if result_item is not None and annotated_dir is not None:
                annotated_path = self._write_annotated_image(result_item, image_path, annotated_dir, relative_root)
                prediction = ImagePrediction(
                    image_path=prediction.image_path,
                    detections=prediction.detections,
                    labels=prediction.labels,
                    annotated_image_path=annotated_path,
                    error_message=prediction.error_message,
                )
            return prediction
        except Exception as exc:
            return ImagePrediction(
                image_path=image_path,
                detections=(),
                labels=(),
                error_message=f"YOLO 推論失敗：{exc}",
            )

    def parse_result(self, image_path: Path, result_item: object | None) -> ImagePrediction:
        """Convert one Ultralytics result object into an image prediction.

        Args:
            image_path (Path): Source image path.
            result_item (object | None): One Ultralytics result.

        Returns:
            ImagePrediction: Parsed prediction.
        """
        if result_item is None or getattr(result_item, "boxes", None) is None:
            return ImagePrediction(image_path=image_path, detections=(), labels=())
        detections: list[DetectionBox] = []
        for box in getattr(result_item, "boxes"):
            class_id = int(_scalar_value(getattr(box, "cls", 0)))
            confidence = float(_scalar_value(getattr(box, "conf", 0.0)))
            x1, y1, x2, y2 = _xyxy_values(getattr(box, "xyxy"))
            detections.append(
                DetectionBox(
                    class_id=class_id,
                    label=self.model_labels.get(class_id, str(class_id)),
                    confidence=confidence,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )
        labels = tuple(dict.fromkeys(detection.label for detection in detections))
        return ImagePrediction(image_path=image_path, detections=tuple(detections), labels=labels)

    @staticmethod
    def _import_yolo_factory() -> Callable[[str], object]:
        ultralytics_module = importlib.import_module("ultralytics")
        return getattr(ultralytics_module, "YOLO")

    def _write_annotated_image(
        self,
        result_item: object,
        image_path: Path,
        annotated_dir: Path,
        relative_root: Path | None,
    ) -> Path | None:
        if not hasattr(result_item, "plot"):
            return None
        rendered = getattr(result_item, "plot")()
        output_path = annotated_dir / _relative_output_path(image_path, relative_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2_module = importlib.import_module("cv2")
        if not cv2_module.imwrite(str(output_path), rendered):
            raise YoloModelError(f"標註圖片寫入失敗：{output_path}")
        return output_path


def _scalar_value(value: object) -> float:
    item = getattr(value, "item", None)
    if callable(item):
        return float(item())
    return float(value)


def _xyxy_values(value: object) -> tuple[int, int, int, int]:
    first_value = value[0] if isinstance(value, Sequence) else value
    cpu = getattr(first_value, "cpu", None)
    if callable(cpu):
        first_value = cpu()
    numpy = getattr(first_value, "numpy", None)
    if callable(numpy):
        first_value = numpy()
    astype = getattr(first_value, "astype", None)
    if callable(astype):
        first_value = astype(int)
    tolist = getattr(first_value, "tolist", None)
    raw_values = tolist() if callable(tolist) else list(first_value)
    flattened_values = _flatten_coordinate_values(raw_values)
    if len(flattened_values) < 4:
        raise ValueError(f"YOLO bbox 座標數量不足：{raw_values}")
    x1, y1, x2, y2 = [int(value_item) for value_item in flattened_values[:4]]
    return x1, y1, x2, y2


def _flatten_coordinate_values(raw_values: object) -> list[object]:
    if isinstance(raw_values, Sequence) and not isinstance(raw_values, (str, bytes, bytearray)):
        values: list[object] = []
        for item in raw_values:
            values.extend(_flatten_coordinate_values(item))
        return values
    return [raw_values]


def _relative_output_path(image_path: Path, relative_root: Path | None) -> Path:
    if relative_root is None:
        return Path(image_path.name)
    root = relative_root if relative_root.is_dir() else relative_root.parent
    try:
        return image_path.relative_to(root)
    except ValueError:
        return Path(image_path.name)
