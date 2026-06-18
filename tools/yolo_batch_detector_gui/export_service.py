from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Mapping

if __package__ in {None, ""}:
    from models import DetectionBox, DetectionSummary, ExportError, ExportOptions, ExportResult, ImagePrediction
    from statistics_service import MULTI_LABEL, NO_DETECTION_LABEL, classify_prediction, unique_prediction_labels
else:
    from .models import DetectionBox, DetectionSummary, ExportError, ExportOptions, ExportResult, ImagePrediction
    from .statistics_service import MULTI_LABEL, NO_DETECTION_LABEL, classify_prediction, unique_prediction_labels


def write_detection_exports(
    predictions: list[ImagePrediction],
    summary: DetectionSummary,
    model_labels: Mapping[int, str],
    inference_settings: Mapping[str, object],
    input_root: Path,
    options: ExportOptions,
) -> ExportResult:
    """Write selected detection exports.

    Args:
        predictions (list[ImagePrediction]): Image predictions to export.
        summary (DetectionSummary): Aggregate statistics.
        model_labels (Mapping[int, str]): Model label mapping.
        inference_settings (Mapping[str, object]): Inference settings used.
        input_root (Path): Original file or directory input root.
        options (ExportOptions): Export destination and enabled formats.

    Returns:
        ExportResult: Paths for generated files.

    Raises:
        ExportError: If writing one of the selected outputs fails.
    """
    try:
        options.output_dir.mkdir(parents=True, exist_ok=True)
        files: dict[str, Path] = {}
        if options.write_csv:
            csv_path = options.output_dir / "detections.csv"
            _write_csv(csv_path, predictions)
            files["csv"] = csv_path
        if options.write_json:
            json_path = options.output_dir / "detections.json"
            _write_json(json_path, predictions, summary, model_labels, inference_settings)
            files["json"] = json_path
        if options.write_annotated_images:
            annotated_dir = options.output_dir / "annotated"
            _copy_annotated_images(annotated_dir, predictions, input_root)
            files["annotated_images"] = annotated_dir
        return ExportResult(files=files)
    except Exception as exc:
        raise ExportError(f"匯出結果失敗：{exc}") from exc


def _write_csv(path: Path, predictions: list[ImagePrediction]) -> None:
    fieldnames = [
        "image_path",
        "status",
        "labels",
        "class_id",
        "label",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "error_message",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for prediction in predictions:
            detections = prediction.detections or (None,)
            for detection in detections:
                writer.writerow(_csv_row(prediction, detection))


def _csv_row(prediction: ImagePrediction, detection: DetectionBox | None) -> dict[str, object]:
    labels = unique_prediction_labels(prediction)
    row: dict[str, object] = {
        "image_path": str(prediction.image_path),
        "status": _export_status(prediction),
        "labels": ";".join(labels),
        "class_id": "",
        "label": "",
        "confidence": "",
        "x1": "",
        "y1": "",
        "x2": "",
        "y2": "",
        "error_message": prediction.error_message or "",
    }
    if detection is not None:
        row.update(
            {
                "class_id": detection.class_id,
                "label": detection.label,
                "confidence": f"{detection.confidence:.6f}",
                "x1": detection.x1,
                "y1": detection.y1,
                "x2": detection.x2,
                "y2": detection.y2,
            }
        )
    return row


def _write_json(
    path: Path,
    predictions: list[ImagePrediction],
    summary: DetectionSummary,
    model_labels: Mapping[int, str],
    inference_settings: Mapping[str, object],
) -> None:
    payload = {
        "model_labels": [model_labels[key] for key in sorted(model_labels)],
        "inference_settings": dict(inference_settings),
        "summary": {
            "total_images": summary.total_images,
            "no_detection_count": summary.no_detection_count,
            "single_label_counts": summary.single_label_counts,
            "multi_label_count": summary.multi_label_count,
            "per_label_image_counts": summary.per_label_image_counts,
        },
        "predictions": [_prediction_payload(prediction) for prediction in predictions],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _prediction_payload(prediction: ImagePrediction) -> dict[str, object]:
    return {
        "image_path": str(prediction.image_path),
        "status": _export_status(prediction),
        "labels": list(unique_prediction_labels(prediction)),
        "error_message": prediction.error_message,
        "detections": [
            {
                "class_id": detection.class_id,
                "label": detection.label,
                "confidence": detection.confidence,
                "xyxy": list(detection.xyxy),
            }
            for detection in prediction.detections
        ],
    }


def _copy_annotated_images(output_dir: Path, predictions: list[ImagePrediction], input_root: Path) -> None:
    for prediction in predictions:
        if prediction.annotated_image_path is None or not prediction.annotated_image_path.exists():
            continue
        target = output_dir / _relative_export_path(prediction.image_path, input_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(prediction.annotated_image_path, target)


def _relative_export_path(image_path: Path, input_root: Path) -> Path:
    root = input_root if input_root.is_dir() else input_root.parent
    try:
        return image_path.relative_to(root)
    except ValueError:
        return Path(image_path.name)


def _export_status(prediction: ImagePrediction) -> str:
    category = classify_prediction(prediction)
    if prediction.error_message:
        return "error"
    if category == NO_DETECTION_LABEL:
        return "no_detection"
    if category == MULTI_LABEL:
        return "multi_label"
    return category
