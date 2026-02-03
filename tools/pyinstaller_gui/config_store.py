from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ConfigRecord:
    name: str
    data: Dict[str, Any]
    timestamp: str


class ConfigStore:
    def __init__(self, exec_dir: Path) -> None:
        self.exec_dir = exec_dir
        self.path = exec_dir / "pyinstaller_gui_config.jsonl"

    def list_names(self) -> List[str]:
        records = self._read_all()
        names = []
        for r in records:
            if r.name not in names:
                names.append(r.name)
        return names

    def load(self, name: str) -> Optional[Dict[str, Any]]:
        records = self._read_all()
        for r in reversed(records):
            if r.name == name:
                return r.data
        return None

    def save(self, name: str, data: Dict[str, Any]) -> None:
        record = {
            "name": name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data": data,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _read_all(self) -> List[ConfigRecord]:
        if not self.path.exists():
            return []
        records: List[ConfigRecord] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                records.append(
                    ConfigRecord(
                        name=str(raw.get("name", "")),
                        data=dict(raw.get("data", {})),
                        timestamp=str(raw.get("timestamp", "")),
                    )
                )
            except Exception:
                continue
        return records
