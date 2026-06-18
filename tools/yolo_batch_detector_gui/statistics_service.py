from __future__ import annotations

if __package__ in {None, ""}:
    from models import DetectionSummary, ImagePrediction
else:
    from .models import DetectionSummary, ImagePrediction

NO_DETECTION_LABEL = "沒有"
MULTI_LABEL = "多個 label"
ERROR_LABEL = "錯誤"


def unique_prediction_labels(prediction: ImagePrediction) -> tuple[str, ...]:
    """Return stable unique labels for one image prediction.

    Args:
        prediction (ImagePrediction): Prediction to inspect.

    Returns:
        tuple[str, ...]: Unique labels in detection order.
    """
    raw_labels = prediction.labels or tuple(detection.label for detection in prediction.detections)
    return tuple(dict.fromkeys(label for label in raw_labels if label))


def classify_prediction(prediction: ImagePrediction) -> str:
    """Classify one image into none, one label, multiple labels, or error.

    Args:
        prediction (ImagePrediction): Prediction to classify.

    Returns:
        str: Human-readable category.
    """
    if prediction.error_message:
        return ERROR_LABEL
    labels = unique_prediction_labels(prediction)
    if not labels:
        return NO_DETECTION_LABEL
    if len(labels) == 1:
        return labels[0]
    return MULTI_LABEL


def build_detection_summary(predictions: list[ImagePrediction]) -> DetectionSummary:
    """Build aggregate statistics from image predictions.

    Args:
        predictions (list[ImagePrediction]): Image predictions to summarize.

    Returns:
        DetectionSummary: Image-level detection counts.
    """
    no_detection_count = 0
    multi_label_count = 0
    single_label_counts: dict[str, int] = {}
    per_label_image_counts: dict[str, int] = {}

    for prediction in predictions:
        if prediction.error_message:
            continue
        labels = unique_prediction_labels(prediction)
        if not labels:
            no_detection_count += 1
            continue
        for label in labels:
            per_label_image_counts[label] = per_label_image_counts.get(label, 0) + 1
        if len(labels) == 1:
            label = labels[0]
            single_label_counts[label] = single_label_counts.get(label, 0) + 1
        else:
            multi_label_count += 1

    return DetectionSummary(
        total_images=len(predictions),
        no_detection_count=no_detection_count,
        single_label_counts=dict(sorted(single_label_counts.items())),
        multi_label_count=multi_label_count,
        per_label_image_counts=dict(sorted(per_label_image_counts.items())),
    )
