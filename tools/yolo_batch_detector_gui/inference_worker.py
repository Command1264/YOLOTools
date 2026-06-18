from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

if __package__ in {None, ""}:
    from models import DetectionSummary, ImagePrediction
    from statistics_service import build_detection_summary
    from yolo_model_service import YoloModelService
else:
    from .models import DetectionSummary, ImagePrediction
    from .statistics_service import build_detection_summary
    from .yolo_model_service import YoloModelService


class ModelLoadWorker(QObject):
    """Load YOLO model labels away from the GUI thread."""

    labels_loaded = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, model_path: Path) -> None:
        """Initialize the model loading worker.

        Args:
            model_path (Path): YOLO model path.
        """
        super().__init__()
        self._model_path = model_path

    @Slot()
    def run(self) -> None:
        """Load the model and emit normalized labels."""
        try:
            service = YoloModelService()
            labels = service.load_model(self._model_path)
            self.labels_loaded.emit(labels)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class PredictionWorker(QObject):
    """Run batch YOLO prediction away from the GUI thread."""

    model_labels_loaded = Signal(object)
    prediction_ready = Signal(object)
    progress_changed = Signal(int, int)
    status_changed = Signal(str)
    failed = Signal(str)
    finished = Signal(object, object)

    def __init__(
        self,
        model_path: Path,
        image_paths: list[Path],
        input_root: Path,
        conf: float,
        iou: float,
        device: str,
        annotated_dir: Path,
    ) -> None:
        """Initialize the prediction worker.

        Args:
            model_path (Path): YOLO model path.
            image_paths (list[Path]): Images to process.
            input_root (Path): Original input root used for relative outputs.
            conf (float): Confidence threshold.
            iou (float): IoU threshold.
            device (str): Optional execution device.
            annotated_dir (Path): Temporary annotated image directory.
        """
        super().__init__()
        self._model_path = model_path
        self._image_paths = list(image_paths)
        self._input_root = input_root
        self._conf = conf
        self._iou = iou
        self._device = device
        self._annotated_dir = annotated_dir
        self._stop_requested = False

    @Slot()
    def run(self) -> None:
        """Load the model, run predictions, and emit progress."""
        predictions: list[ImagePrediction] = []
        try:
            service = YoloModelService()
            self.status_changed.emit("載入模型中...")
            labels = service.load_model(self._model_path)
            self.model_labels_loaded.emit(labels)
            total = len(self._image_paths)
            self.progress_changed.emit(0, total)
            for index, image_path in enumerate(self._image_paths, start=1):
                if self._stop_requested:
                    self.status_changed.emit("已停止，保留已完成結果。")
                    break
                self.status_changed.emit(f"辨識中：{image_path.name}")
                prediction = service.predict_image(
                    image_path=image_path,
                    conf=self._conf,
                    iou=self._iou,
                    device=self._device,
                    annotated_dir=self._annotated_dir,
                    relative_root=self._input_root,
                )
                predictions.append(prediction)
                self.prediction_ready.emit(prediction)
                self.progress_changed.emit(index, total)
            summary = build_detection_summary(predictions)
            self.finished.emit(predictions, summary)
        except Exception as exc:
            self.failed.emit(str(exc))
            summary = build_detection_summary(predictions)
            self.finished.emit(predictions, summary)

    @Slot()
    def stop(self) -> None:
        """Request a graceful stop after the current image finishes."""
        self._stop_requested = True
