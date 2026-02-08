from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import yaml


@dataclass
class YamlMappingModel:
    """Generic YAML mapping model."""

    data: Dict[str, object]

    @classmethod
    def from_yaml_text(cls, text: str) -> "YamlMappingModel":
        """Deserialize YAML text into mapping model."""
        obj = yaml.safe_load(text)
        if not isinstance(obj, dict):
            raise ValueError("YAML document must be a mapping")
        return cls(data=dict(obj))

    @classmethod
    def from_file(cls, path: Path) -> "YamlMappingModel":
        """Load mapping model from YAML file."""
        return cls.from_yaml_text(path.read_text(encoding="utf-8"))

    def to_yaml_text(self) -> str:
        """Serialize mapping to YAML text."""
        return yaml.safe_dump(self.data, sort_keys=False, allow_unicode=True)

    def save(self, path: Path) -> None:
        """Write YAML mapping file."""
        path.write_text(self.to_yaml_text(), encoding="utf-8")
