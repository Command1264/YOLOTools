from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class ClassMappingRule:
    """單一來源類別的映射規則。"""

    source_name: str
    target_model_index: int | None
    ignore: bool
    rename_note: str = ""

    def to_dict(self) -> dict[str, object]:
        """輸出可序列化物件。"""
        return {
            "source_name": self.source_name,
            "target_model_index": self.target_model_index,
            "ignore": self.ignore,
            "rename_note": self.rename_note,
        }


def build_default_mapping(
    source_names: list[str],
    model_names: list[str],
) -> list[ClassMappingRule]:
    """建立預設映射。"""
    lowered_model_names = {name.casefold(): index for index, name in enumerate(model_names)}
    rules: list[ClassMappingRule] = []
    for name in source_names:
        target_index = lowered_model_names.get(name.casefold())
        rules.append(
            ClassMappingRule(
                source_name=name,
                target_model_index=target_index,
                ignore=target_index is None,
                rename_note="" if target_index is None else model_names[target_index],
            )
        )
    return rules


def mapping_signature(rules: list[ClassMappingRule]) -> str:
    """計算映射規則簽章。"""
    payload = [rule.to_dict() for rule in rules]
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
