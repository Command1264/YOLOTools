from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class DetectorConfig:
    """Persistent GUI settings for the YOLO batch detector.

    Args:
        model_path (str): Last selected YOLO model path.
        input_path (str): Last selected image or image folder.
        output_dir (str): Last selected export directory.
        export_csv (bool): Whether CSV export is enabled.
        export_json (bool): Whether JSON export is enabled.
        export_annotated_images (bool): Whether annotated image export is enabled.
        conf (float): YOLO confidence threshold.
        iou (float): YOLO IoU threshold.
        device (str): Optional YOLO device string.
    """

    model_path: str = ""
    input_path: str = ""
    output_dir: str = ""
    export_csv: bool = True
    export_json: bool = True
    export_annotated_images: bool = False
    conf: float = 0.25
    iou: float = 0.45
    device: str = ""


def load_config(config_path: Path) -> DetectorConfig:
    """Load GUI configuration from JSON.

    Args:
        config_path (Path): JSON config path.

    Returns:
        DetectorConfig: Loaded config, or defaults when the file is absent/partial.
    """
    if not config_path.exists():
        return DetectorConfig()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return DetectorConfig()
    defaults = DetectorConfig()
    return DetectorConfig(
        model_path=str(payload.get("model_path", defaults.model_path)),
        input_path=str(payload.get("input_path", defaults.input_path)),
        output_dir=str(payload.get("output_dir", defaults.output_dir)),
        export_csv=bool(payload.get("export_csv", defaults.export_csv)),
        export_json=bool(payload.get("export_json", defaults.export_json)),
        export_annotated_images=bool(
            payload.get("export_annotated_images", defaults.export_annotated_images)
        ),
        conf=_clamp_threshold(payload.get("conf", defaults.conf), defaults.conf),
        iou=_clamp_threshold(payload.get("iou", defaults.iou), defaults.iou),
        device=str(payload.get("device", defaults.device)),
    )


def save_config(config_path: Path, config: DetectorConfig) -> None:
    """Save GUI configuration as JSON.

    Args:
        config_path (Path): JSON config path.
        config (DetectorConfig): Config to save.
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")


def _clamp_threshold(raw_value: object, default_value: float) -> float:
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return default_value
    return max(0.0, min(1.0, value))
