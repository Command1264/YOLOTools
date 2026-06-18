from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class YamlMappingModel:
    """YAML mapping 序列化模型。"""

    data: dict[str, object]

    @classmethod
    def from_text(cls, text: str) -> "YamlMappingModel":
        """由 YAML 文字建立 mapping。"""
        obj = yaml.safe_load(text)
        if not isinstance(obj, dict):
            raise ValueError("YAML 內容必須是 mapping。")
        return cls(data=dict(obj))

    @classmethod
    def from_file(cls, path: Path) -> "YamlMappingModel":
        """由檔案載入 YAML mapping。"""
        return cls.from_text(path.read_text(encoding="utf-8"))

    def to_text(self) -> str:
        """輸出 YAML 文字。"""
        return yaml.safe_dump(self.data, allow_unicode=True, sort_keys=False)

    def save(self, path: Path) -> None:
        """寫入 YAML 檔案。"""
        path.write_text(self.to_text(), encoding="utf-8")
