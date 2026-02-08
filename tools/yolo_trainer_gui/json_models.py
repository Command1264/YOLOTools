from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class WeightsCacheModel:
    """JSON cache model for weights list."""

    weights: List[str]

    @classmethod
    def from_json_text(cls, text: str) -> "WeightsCacheModel":
        """Deserialize list payload from JSON text."""
        obj = json.loads(text)
        if not isinstance(obj, list):
            raise ValueError("Weights cache JSON must be a list")
        out = [str(item) for item in obj]
        return cls(weights=out)

    @classmethod
    def from_file(cls, path: Path) -> "WeightsCacheModel":
        """Load model from a JSON file."""
        return cls.from_json_text(path.read_text(encoding="utf-8"))

    def to_json_text(self) -> str:
        """Serialize weights list to JSON text."""
        return json.dumps(self.weights, ensure_ascii=False, indent=2)

    def save(self, path: Path) -> None:
        """Write weights cache file."""
        path.write_text(self.to_json_text(), encoding="utf-8")
