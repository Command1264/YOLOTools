from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import jsonlines

@dataclass
class ConfigRecord:
    name: str
    data: Dict[str, Any]
    timestamp: str


class ConfigStore:
    def __init__(self, exec_dir: Path) -> None:
        self.exec_dir = exec_dir
        self.path = exec_dir / "python_packager_gui_config.jsonl"

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
        record = self._build_record(name, data)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write_all([record])
            return
        records = self._read_all_raw()
        new_records: List[Dict[str, Any]] = []
        replaced = False
        for item in records:
            if str(item.get("name", "")) == name:
                if not replaced:
                    new_records.append(record)
                    replaced = True
                continue
            new_records.append(item)
        if not replaced:
            new_records.append(record)
        self._write_all(new_records)

    def delete(self, name: str) -> bool:
        if not self.path.exists():
            return False
        records = self._read_all_raw()
        new_records: List[Dict[str, Any]] = []
        removed = False
        for item in records:
            if str(item.get("name", "")) == name:
                removed = True
                continue
            new_records.append(item)
        if removed:
            self._write_all(new_records)
        return removed

    def _read_all(self) -> List[ConfigRecord]:
        records: List[ConfigRecord] = []
        for raw in self._read_all_raw():
            records.append(
                ConfigRecord(
                    name=str(raw.get("name", "")),
                    data=dict(raw.get("data", {})),
                    timestamp=str(raw.get("timestamp", "")),
                )
            )
        return records

    def _read_all_raw(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        records: List[Dict[str, Any]] = []
        with jsonlines.open(self.path, mode="r") as reader:
            for item in reader:
                if isinstance(item, dict):
                    records.append(item)
        normalized, changed = self._normalize_records(records)
        if changed:
            self._write_all(normalized)
        return normalized

    def _write_all(self, records: List[Dict[str, Any]]) -> None:
        with jsonlines.open(self.path, mode="w") as writer:
            writer.write_all(records)

    def _build_record(self, name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "name": name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data": data,
        }

    def _normalize_records(self, records: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], bool]:
        normalized: List[Dict[str, Any]] = []
        changed = False
        for item in records:
            new_item, item_changed = self._normalize_item(item)
            normalized.append(new_item)
            changed = changed or item_changed
        return normalized, changed

    def _normalize_item(self, item: Dict[str, Any]) -> tuple[Dict[str, Any], bool]:
        changed = False
        new_item: Dict[str, Any] = {}
        for key, value in item.items():
            new_value, value_changed = self._normalize_value(value)
            new_item[key] = new_value
            changed = changed or value_changed
        return new_item, changed

    def _normalize_value(self, value: Any) -> tuple[Any, bool]:
        if isinstance(value, str):
            if "\\" in value:
                return value.replace("\\", "/"), True
            return value, False
        if isinstance(value, dict):
            return self._normalize_item(value)
        if isinstance(value, list):
            changed = False
            new_list: List[Any] = []
            for item in value:
                new_item, item_changed = self._normalize_value(item)
                new_list.append(new_item)
                changed = changed or item_changed
            return new_list, changed
        return value, False
