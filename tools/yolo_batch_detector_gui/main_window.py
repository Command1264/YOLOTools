from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox

if __package__ in {None, ""}:
    from config_store import load_config
    from export_service import write_detection_exports
    from image_source_service import collect_image_paths
    from inference_worker import ModelLoadWorker, PredictionWorker
    from main_window_browse import MainWindowBrowseMixin
    from main_window_config import MainWindowConfigMixin
    from main_window_results import MainWindowResultsMixin
    from main_window_ui import MainWindowUiMixin
    from models import DetectionSummary, ExportOptions, ImagePrediction
    from statistics_service import build_detection_summary
else:
    from .config_store import load_config
    from .export_service import write_detection_exports
    from .image_source_service import collect_image_paths
    from .inference_worker import ModelLoadWorker, PredictionWorker
    from .main_window_browse import MainWindowBrowseMixin
    from .main_window_config import MainWindowConfigMixin
    from .main_window_results import MainWindowResultsMixin
    from .main_window_ui import MainWindowUiMixin
    from .models import DetectionSummary, ExportOptions, ImagePrediction
    from .statistics_service import build_detection_summary


class MainWindow(MainWindowBrowseMixin, MainWindowConfigMixin, MainWindowResultsMixin, MainWindowUiMixin, QMainWindow):
    """YOLO 批次辨識檢視器主視窗。"""

    def __init__(self, config_path: Path | None = None) -> None:
        """Initialize the main window."""
        super().__init__()
        self.setWindowTitle("YOLO 批次辨識檢視器")
        self.resize(1280, 820)
        self.setMinimumSize(1040, 680)

        self._model_labels: dict[int, str] = {}
        self._predictions: list[ImagePrediction] = []
        self._summary: DetectionSummary = build_detection_summary([])
        self._input_root: Path | None = None
        self._model_thread: QThread | None = None
        self._model_worker: ModelLoadWorker | None = None
        self._prediction_thread: QThread | None = None
        self._prediction_worker: PredictionWorker | None = None
        self._preview_dir: Path | None = None
        self._image_dialogs: list[object] = []
        self._last_model_dir: Path | None = None
        self._last_input_dir: Path | None = None
        self._last_output_dir: Path | None = None
        self._config_path = config_path or Path(__file__).resolve().parent / "yolo_batch_detector_config.json"
        self._loading_config = False

        self._build_ui()
        self._apply_config(load_config(self._config_path))
        self._connect_config_signals()
        self._set_running(False)

    def request_shutdown(self) -> None:
        """Request graceful worker shutdown before app exit."""
        if self._prediction_worker is not None:
            self._prediction_worker.stop()
        self._shutdown_thread(self._prediction_thread)
        self._shutdown_thread(self._model_thread)

    def _shutdown_thread(self, thread: QThread | None) -> None:
        """Stop a thread if its Qt object still exists."""
        if thread is None:
            return
        try:
            if thread.isRunning():
                thread.quit()
                thread.wait(2000)
        except RuntimeError as exc:
            if "already deleted" not in str(exc):
                raise

    def _load_model_labels(self) -> None:
        model_path = self._validated_model_path()
        if model_path is None:
            return
        self.btn_load_labels.setEnabled(False)
        self.lbl_status.setText("載入 labels 中...")
        self._model_thread = QThread(self)
        self._model_worker = ModelLoadWorker(model_path)
        self._model_worker.moveToThread(self._model_thread)
        self._model_thread.started.connect(self._model_worker.run)
        self._model_worker.labels_loaded.connect(self._set_model_labels)
        self._model_worker.failed.connect(lambda message: QMessageBox.critical(self, "模型載入失敗", message))
        self._model_worker.finished.connect(self._model_thread.quit)
        self._model_worker.finished.connect(lambda: self.btn_load_labels.setEnabled(True))
        self._model_thread.finished.connect(self._model_worker.deleteLater)
        self._model_thread.finished.connect(self._clear_model_thread_refs)
        self._model_thread.finished.connect(self._model_thread.deleteLater)
        self._model_thread.start()

    def _start_prediction(self) -> None:
        model_path = self._validated_model_path()
        if model_path is None:
            return
        input_path = Path(self.ent_input.text().strip())
        try:
            image_paths = collect_image_paths(input_path)
        except Exception as exc:
            QMessageBox.critical(self, "輸入錯誤", str(exc))
            return
        self._input_root = input_path
        self._preview_dir = Path(tempfile.mkdtemp(prefix="yolo_batch_detector_"))
        self._clear_results()
        self._set_running(True)
        self.progress.setRange(0, len(image_paths))
        self._prediction_thread = QThread(self)
        self._prediction_worker = PredictionWorker(
            model_path=model_path,
            image_paths=image_paths,
            input_root=input_path,
            conf=float(self.sp_conf.value()),
            iou=float(self.sp_iou.value()),
            device=self.ent_device.text().strip(),
            annotated_dir=self._preview_dir,
        )
        self._prediction_worker.moveToThread(self._prediction_thread)
        self._prediction_thread.started.connect(self._prediction_worker.run)
        self._prediction_worker.model_labels_loaded.connect(self._set_model_labels)
        self._prediction_worker.prediction_ready.connect(self._add_prediction)
        self._prediction_worker.progress_changed.connect(self._set_progress)
        self._prediction_worker.status_changed.connect(self.lbl_status.setText)
        self._prediction_worker.failed.connect(lambda message: QMessageBox.critical(self, "辨識失敗", message))
        self._prediction_worker.finished.connect(self._prediction_finished)
        self._prediction_worker.finished.connect(self._prediction_thread.quit)
        self._prediction_thread.finished.connect(self._prediction_worker.deleteLater)
        self._prediction_thread.finished.connect(self._clear_prediction_thread_refs)
        self._prediction_thread.finished.connect(self._prediction_thread.deleteLater)
        self._prediction_thread.start()

    def _stop_prediction(self) -> None:
        if self._prediction_worker is not None:
            self._prediction_worker.stop()
            self.lbl_status.setText("停止中...")

    def _prediction_finished(self, predictions_obj: object, summary_obj: object) -> None:
        self._predictions = list(predictions_obj) if isinstance(predictions_obj, list) else self._predictions
        self._summary = summary_obj if isinstance(summary_obj, DetectionSummary) else build_detection_summary(self._predictions)
        self._refresh_summary()
        self._set_running(False)
        self.lbl_status.setText(f"完成：{len(self._predictions)} 張")
        if self._has_export_selection() and self.ent_output.text().strip():
            self._export_results()

    def _export_results(self) -> None:
        if not self._predictions:
            QMessageBox.warning(self, "尚無結果", "請先完成辨識後再匯出。")
            return
        if not self._has_export_selection():
            QMessageBox.warning(self, "缺少匯出格式", "請至少選擇一種匯出格式。")
            return
        output_text = self.ent_output.text().strip()
        if not output_text:
            QMessageBox.warning(self, "缺少輸出位置", "請先選擇輸出資料夾。")
            return
        try:
            result = write_detection_exports(
                predictions=self._predictions,
                summary=self._summary,
                model_labels=self._model_labels,
                inference_settings={
                    "conf": float(self.sp_conf.value()),
                    "iou": float(self.sp_iou.value()),
                    "device": self.ent_device.text().strip(),
                },
                input_root=self._input_root or Path(self.ent_input.text().strip()),
                options=ExportOptions(
                    output_dir=Path(output_text),
                    write_csv=self.chk_csv.isChecked(),
                    write_json=self.chk_json.isChecked(),
                    write_annotated_images=self.chk_annotated.isChecked(),
                ),
            )
            self.lbl_status.setText(f"已匯出：{', '.join(result.files)}")
        except Exception as exc:
            QMessageBox.critical(self, "匯出失敗", str(exc))

    def _validated_model_path(self) -> Path | None:
        raw = self.ent_model.text().strip()
        if not raw:
            QMessageBox.warning(self, "缺少模型", "請先選擇 YOLO 模型。")
            return None
        path = Path(raw)
        if not path.exists():
            QMessageBox.critical(self, "模型不存在", f"找不到模型檔：{path}")
            return None
        return path

    def _set_model_labels(self, labels_obj: object) -> None:
        labels = dict(labels_obj) if isinstance(labels_obj, dict) else {}
        self._model_labels = {int(key): str(value) for key, value in labels.items()}
        lines = [f"{index}: {label}" for index, label in sorted(self._model_labels.items())]
        self.txt_labels.setPlainText("\n".join(lines))
        self.lbl_status.setText(f"已載入 labels：{len(lines)} 個")

    def _clear_model_thread_refs(self) -> None:
        """Clear model loading thread references after Qt finishes cleanup."""
        self._model_worker = None
        self._model_thread = None

    def _clear_prediction_thread_refs(self) -> None:
        """Clear prediction thread references after Qt finishes cleanup."""
        self._prediction_worker = None
        self._prediction_thread = None

    def closeEvent(self, event) -> None:  # noqa: N802
        """Handle window close by stopping background work first."""
        self.request_shutdown()
        QApplication.processEvents()
        super().closeEvent(event)
