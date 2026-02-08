from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainerGuiConfig:
    """Serialized config model for trainer GUI."""

    task: str = "detect"
    dataset_zip: str = ""
    work_dir: str = ""
    out_zip_dir: str = ""
    model_pick: str = ""
    model_family: str = "all"
    model_size: str = "all"
    model_dir: str = ""
    use_custom_model: bool = False
    custom_model: str = ""
    epochs: int = 50
    imgsz: int = 640
    batch: int = 16
    eta_decimal_places: int = 1
    device: str = ""
    resume: bool = False
    skip_unlabeled: bool = True
    delete_temp: bool = True

    @classmethod
    def from_json_text(cls, text: str) -> "TrainerGuiConfig":
        """Deserialize config from JSON text."""
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError("Trainer config JSON must be an object")
        return cls(
            task=str(obj.get("task", "detect")),
            dataset_zip=str(obj.get("dataset_zip", "")),
            work_dir=str(obj.get("work_dir", "")),
            out_zip_dir=str(obj.get("out_zip_dir", "")),
            model_pick=str(obj.get("model_pick", "")),
            model_family=str(obj.get("model_family", "all")),
            model_size=str(obj.get("model_size", "all")),
            model_dir=str(obj.get("model_dir", "")),
            use_custom_model=bool(obj.get("use_custom_model", False)),
            custom_model=str(obj.get("custom_model", "")),
            epochs=int(obj.get("epochs", 50)),
            imgsz=int(obj.get("imgsz", 640)),
            batch=int(obj.get("batch", 16)),
            eta_decimal_places=int(obj.get("eta_decimal_places", 1)),
            device=str(obj.get("device", "")),
            resume=bool(obj.get("resume", False)),
            skip_unlabeled=bool(obj.get("skip_unlabeled", True)),
            delete_temp=bool(obj.get("delete_temp", True)),
        )

    @classmethod
    def from_file(cls, path: Path) -> "TrainerGuiConfig":
        """Load config from JSON file."""
        return cls.from_json_text(path.read_text(encoding="utf-8"))

    def to_json_text(self) -> str:
        """Serialize config as JSON text."""
        return json.dumps(self.__dict__, ensure_ascii=True, indent=2)

    def save(self, path: Path) -> None:
        """Write config file."""
        path.write_text(self.to_json_text(), encoding="utf-8")
