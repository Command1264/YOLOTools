from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QTableWidgetItem

if __package__ in {None, ""}:
    from error_dialog import CopyableErrorMessageBox
    from image_preview import ImagePreviewDialog
    from models import ImagePrediction
    from statistics_service import build_detection_summary, classify_prediction, unique_prediction_labels
else:
    from .error_dialog import CopyableErrorMessageBox
    from .image_preview import ImagePreviewDialog
    from .models import ImagePrediction
    from .statistics_service import build_detection_summary, classify_prediction, unique_prediction_labels


class MainWindowResultsMixin:
    """Update result tables, image preview, and summary text."""

    def _add_prediction(self, prediction_obj: object) -> None:
        if not isinstance(prediction_obj, ImagePrediction):
            return
        self._predictions.append(prediction_obj)
        row = self.tbl_results.rowCount()
        self.tbl_results.insertRow(row)
        labels = unique_prediction_labels(prediction_obj)
        values = [
            str(prediction_obj.image_path),
            classify_prediction(prediction_obj),
            ", ".join(labels),
            str(len(prediction_obj.detections)),
            prediction_obj.error_message or "",
        ]
        for column, value in enumerate(values):
            self.tbl_results.setItem(row, column, QTableWidgetItem(value))
        self._summary = build_detection_summary(self._predictions)
        self._refresh_summary()
        if row == 0:
            self.tbl_results.selectRow(0)

    def _on_result_selected(self) -> None:
        selected = self.tbl_results.selectionModel().selectedRows()
        if not selected:
            return
        row = selected[0].row()
        if row < 0 or row >= len(self._predictions):
            return
        self._show_prediction(self._predictions[row])

    def _show_error_for_result_item(self, item: QTableWidgetItem) -> None:
        """Show a copyable error dialog when a result row has an error."""
        row = item.row()
        if row < 0 or row >= len(self._predictions):
            return
        error_message = self._predictions[row].error_message
        if not error_message:
            return
        CopyableErrorMessageBox(self, "辨識錯誤", error_message).exec()

    def _show_prediction(self, prediction: ImagePrediction) -> None:
        image_path = prediction.annotated_image_path or prediction.image_path
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self.lbl_image.clear_preview("無法載入圖片")
        else:
            self.lbl_image.set_source_pixmap(pixmap, image_path)
        self.tbl_boxes.setRowCount(0)
        for detection in prediction.detections:
            row = self.tbl_boxes.rowCount()
            self.tbl_boxes.insertRow(row)
            values = [
                detection.label,
                f"{detection.confidence:.3f}",
                str(detection.x1),
                str(detection.y1),
                str(detection.x2),
                str(detection.y2),
            ]
            for column, value in enumerate(values):
                self.tbl_boxes.setItem(row, column, QTableWidgetItem(value))

    def _refresh_summary(self) -> None:
        lines = [
            f"總圖片數：{self._summary.total_images}",
            f"沒有偵測：{self._summary.no_detection_count}",
            f"多個 label：{self._summary.multi_label_count}",
            "",
            "單一 label 統計：",
        ]
        lines.extend(f"- {label}: {count}" for label, count in self._summary.single_label_counts.items())
        lines.extend(["", "各自 label 數量："])
        lines.extend(f"- {label}: {count}" for label, count in self._summary.per_label_image_counts.items())
        self.txt_summary.setPlainText("\n".join(lines))

    def _set_progress(self, current: int, total: int) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(current)

    def _clear_results(self) -> None:
        self._predictions.clear()
        self._summary = build_detection_summary([])
        self.tbl_results.setRowCount(0)
        self.tbl_boxes.setRowCount(0)
        self.lbl_image.setText("尚未選擇圖片")
        self.lbl_image.clear_preview()
        self._refresh_summary()

    def _set_running(self, running: bool) -> None:
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.btn_export.setEnabled(not running and bool(self._predictions))
        self.btn_model.setEnabled(not running)
        self.btn_input_file.setEnabled(not running)
        self.btn_input_folder.setEnabled(not running)
        self.btn_load_labels.setEnabled(not running)

    def _has_export_selection(self) -> bool:
        return self.chk_csv.isChecked() or self.chk_json.isChecked() or self.chk_annotated.isChecked()

    def _open_full_image_preview(self, image_path) -> None:
        dialog = ImagePreviewDialog(image_path, self)
        dialog.destroyed.connect(lambda _=None, target=dialog: self._image_dialogs.remove(target) if target in self._image_dialogs else None)
        self._image_dialogs.append(dialog)
        dialog.show()
