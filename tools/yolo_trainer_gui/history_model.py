from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict


@dataclass
class HistoryRecord:
    """Single JSONL history record wrapper."""

    data: Dict[str, object]

    @classmethod
    def from_json_line(cls, line: str) -> "HistoryRecord":
        """Deserialize one JSONL line into a record."""
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise ValueError("History JSON line must be an object")
        return cls(data=dict(obj))

    def to_json_line(self) -> str:
        """Serialize record as one JSONL line."""
        return json.dumps(self.data, ensure_ascii=False)
