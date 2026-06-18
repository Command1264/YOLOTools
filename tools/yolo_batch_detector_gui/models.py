from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class YoloBatchDetectorError(Exception):
    """Base class for YOLO batch detector errors."""


class ImageSourceError(YoloBatchDetectorError):
    """Raised when image input sources are invalid."""


class YoloModelError(YoloBatchDetectorError):
    """Raised when YOLO model loading or inference fails."""


class ExportError(YoloBatchDetectorError):
    """Raised when result export fails."""


@dataclass(frozen=True)
class DetectionBox:
    """One YOLO detection bounding box.

    Args:
        class_id (int): YOLO class index.
        label (str): YOLO class label.
        confidence (float): Detection confidence.
        x1 (int): Left coordinate.
        y1 (int): Top coordinate.
        x2 (int): Right coordinate.
        y2 (int): Bottom coordinate.
    """

    class_id: int
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        """Return the box coordinates in xyxy order.

        Returns:
            tuple[int, int, int, int]: Left, top, right, bottom coordinates.
        """
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class ImagePrediction:
    """Prediction result for one source image.

    Args:
        image_path (Path): Source image path.
        detections (tuple[DetectionBox, ...]): Detected boxes.
        labels (tuple[str, ...]): Unique labels detected in this image.
        annotated_image_path (Path | None): Optional rendered bbox image path.
        error_message (str | None): Optional per-image error message.
    """

    image_path: Path
    detections: tuple[DetectionBox, ...]
    labels: tuple[str, ...]
    annotated_image_path: Path | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class DetectionSummary:
    """Aggregated image-level detection statistics.

    Args:
        total_images (int): Number of processed images.
        no_detection_count (int): Number of images without boxes.
        single_label_counts (dict[str, int]): Image counts grouped by one label.
        multi_label_count (int): Number of images with multiple unique labels.
        per_label_image_counts (dict[str, int]): Per-label image counts.
    """

    total_images: int
    no_detection_count: int
    single_label_counts: dict[str, int]
    multi_label_count: int
    per_label_image_counts: dict[str, int]


@dataclass(frozen=True)
class ExportOptions:
    """Export destination and format options.

    Args:
        output_dir (Path): Directory where selected exports are written.
        write_csv (bool): Whether to write `detections.csv`.
        write_json (bool): Whether to write `detections.json`.
        write_annotated_images (bool): Whether to copy annotated images.
    """

    output_dir: Path
    write_csv: bool
    write_json: bool
    write_annotated_images: bool


@dataclass(frozen=True)
class ExportResult:
    """Files produced by one export operation.

    Args:
        files (dict[str, Path]): Export kind to generated file path.
    """

    files: dict[str, Path]
