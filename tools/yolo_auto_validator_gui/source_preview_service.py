from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

if __package__ in {None, ""}:
    from archive_service import detect_rar_backend, read_archive_yaml_preview
    from exceptions import PreviewError
    from logging_utils import get_logger
    from yaml_mapping_model import YamlMappingModel
else:
    from .archive_service import detect_rar_backend, read_archive_yaml_preview
    from .exceptions import PreviewError
    from .logging_utils import get_logger
    from .yaml_mapping_model import YamlMappingModel

VAL_KEYS = ("val", "valid", "validation", "vaild")
LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class DatasetSourcePreview:
    """資料集來源預檢結果。"""

    source_path: Path
    source_kind: str
    yaml_locator: str
    yaml_signature: str
    names: list[str]
    split_keys_present: tuple[str, ...]
    preview_error: str | None = None


def preview_source(source_path: Path) -> DatasetSourcePreview:
    """只讀取來源中的 data.yaml 進行預檢。"""
    source = source_path.resolve()
    LOGGER.info("Previewing dataset source. source=%s", source)
    if source.is_dir():
        yaml_path = _find_data_yaml_in_folder(source)
        if yaml_path is None:
            raise PreviewError(f"資料夾找不到 data.yaml 或 data.yml：{source}")
        yaml_text = yaml_path.read_text(encoding="utf-8", errors="ignore")
        locator = str(yaml_path.relative_to(source)).replace("\\", "/")
        LOGGER.debug("Folder preview resolved YAML. source=%s yaml=%s", source, yaml_path)
        return _build_preview(source, "folder", locator, yaml_text)
    if source.is_file() and source.suffix.lower() == ".zip":
        archive_preview = read_archive_yaml_preview(source)
        return _build_preview(source, "zip", archive_preview.member_path, archive_preview.text)
    if source.is_file() and source.suffix.lower() == ".rar":
        ok, message = detect_rar_backend()
        if not ok:
            return DatasetSourcePreview(
                source_path=source,
                source_kind="rar",
                yaml_locator="",
                yaml_signature="",
                names=[],
                split_keys_present=(),
                preview_error=message,
            )
        archive_preview = read_archive_yaml_preview(source)
        return _build_preview(source, "rar", archive_preview.member_path, archive_preview.text)
    raise PreviewError(f"不支援的來源：{source}")


def safe_preview_source(source_path: Path) -> DatasetSourcePreview:
    """回傳包含錯誤資訊的預檢結果。"""
    try:
        return preview_source(source_path)
    except Exception as exc:
        LOGGER.exception("Dataset preview failed. source=%s", source_path)
        source = source_path.resolve()
        kind = "folder" if source.is_dir() else source.suffix.lower().lstrip(".")
        return DatasetSourcePreview(
            source_path=source,
            source_kind=kind or "unknown",
            yaml_locator="",
            yaml_signature="",
            names=[],
            split_keys_present=(),
            preview_error=str(exc),
        )


def _build_preview(
    source_path: Path,
    source_kind: str,
    yaml_locator: str,
    yaml_text: str,
) -> DatasetSourcePreview:
    mapping = YamlMappingModel.from_text(yaml_text).data
    names = _parse_names(mapping.get("names"))
    if not names:
        raise PreviewError("data.yaml 缺少 names。")
    split_keys = tuple(key for key in ("train", "val", "test") if _pick_first(mapping, (key,)) is not None)
    if not split_keys and _pick_first(mapping, VAL_KEYS) is not None:
        split_keys = ("val",)
    signature = _build_yaml_signature(mapping, names)
    LOGGER.info(
        "Dataset preview ready. source=%s kind=%s names=%s splits=%s signature=%s",
        source_path,
        source_kind,
        names,
        split_keys,
        signature,
    )
    return DatasetSourcePreview(
        source_path=source_path,
        source_kind=source_kind,
        yaml_locator=yaml_locator,
        yaml_signature=signature,
        names=names,
        split_keys_present=split_keys,
        preview_error=None,
    )


def _find_data_yaml_in_folder(folder: Path) -> Path | None:
    for name in ("data.yaml", "data.yml"):
        direct = folder / name
        if direct.exists():
            return direct.resolve()
    for path in folder.rglob("data.yaml"):
        return path.resolve()
    for path in folder.rglob("data.yml"):
        return path.resolve()
    return None


def _pick_first(mapping: Mapping[str, object], keys: tuple[str, ...]) -> object | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _parse_names(raw_value: object) -> list[str]:
    if isinstance(raw_value, list):
        return [str(item) for item in raw_value]
    if isinstance(raw_value, dict):
        items: list[tuple[int, str]] = []
        for key, value in raw_value.items():
            items.append((int(key), str(value)))
        return [value for _, value in sorted(items, key=lambda item: item[0])]
    return []


def _normalize_split_value(raw_value: object) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return [str(item).replace("\\", "/") for item in raw_value]
    return [str(raw_value).replace("\\", "/")]


def _build_yaml_signature(mapping: Mapping[str, object], names: list[str]) -> str:
    normalized = {
        "path": str(mapping.get("path", ".")).replace("\\", "/"),
        "train": _normalize_split_value(mapping.get("train")),
        "val": _normalize_split_value(_pick_first(mapping, VAL_KEYS)),
        "test": _normalize_split_value(mapping.get("test")),
        "names": names,
    }
    raw = json.dumps(normalized, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
