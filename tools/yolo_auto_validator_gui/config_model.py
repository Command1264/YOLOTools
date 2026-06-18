from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SavedSourceConfig:
    """已保存的資料集來源設定。"""

    source_path: str
    source_kind: str


@dataclass(frozen=True)
class ValidatorAppConfig:
    """自動驗證器設定模型。"""

    model_path: str = ""
    device: str = ""
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    temp_root: str = ""
    delete_temp_after_run: bool = False
    window_close_behavior: str = "tray_only"
    last_sources: tuple[SavedSourceConfig, ...] = ()
    saved_mappings: dict[str, list[dict[str, object]]] = field(default_factory=dict)

    @classmethod
    def from_json_text(cls, text: str) -> "ValidatorAppConfig":
        """由 JSON 文字反序列化。"""
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError("設定檔格式錯誤。")
        last_sources_raw = obj.get("last_sources", [])
        last_sources: list[SavedSourceConfig] = []
        if isinstance(last_sources_raw, list):
            for item in last_sources_raw:
                if not isinstance(item, dict):
                    continue
                last_sources.append(
                    SavedSourceConfig(
                        source_path=str(item.get("source_path", "")),
                        source_kind=str(item.get("source_kind", "")),
                    )
                )
        saved_mappings_raw = obj.get("saved_mappings", {})
        saved_mappings = saved_mappings_raw if isinstance(saved_mappings_raw, dict) else {}
        return cls(
            model_path=str(obj.get("model_path", "")),
            device=str(obj.get("device", "")),
            conf_threshold=float(obj.get("conf_threshold", 0.25)),
            iou_threshold=float(obj.get("iou_threshold", 0.45)),
            temp_root=str(obj.get("temp_root", "")),
            delete_temp_after_run=bool(obj.get("delete_temp_after_run", False)),
            window_close_behavior=str(obj.get("window_close_behavior", "tray_only")),
            last_sources=tuple(last_sources),
            saved_mappings=saved_mappings,
        )

    @classmethod
    def from_file(cls, path: Path) -> "ValidatorAppConfig":
        """由檔案載入設定。"""
        return cls.from_json_text(path.read_text(encoding="utf-8"))

    def to_json_text(self) -> str:
        """輸出 JSON 文字。"""
        payload = {
            "model_path": self.model_path,
            "device": self.device,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "temp_root": self.temp_root,
            "delete_temp_after_run": self.delete_temp_after_run,
            "window_close_behavior": self.window_close_behavior,
            "last_sources": [item.__dict__ for item in self.last_sources],
            "saved_mappings": self.saved_mappings,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def save(self, path: Path) -> None:
        """寫入設定檔。"""
        path.write_text(self.to_json_text(), encoding="utf-8")

