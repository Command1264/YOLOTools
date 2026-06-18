from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    from archive_service import extract_archive
    from class_mapping_model import ClassMappingRule, mapping_signature
    from dataset_cache_service import (
        PreparedCacheMetadata,
        build_prepared_cache_key,
        compute_source_fingerprint,
        read_prepared_cache_metadata,
        write_prepared_cache_metadata,
    )
    from exceptions import DatasetPrepareError, OperationCancelledError
    from logging_utils import get_logger
    from source_preview_service import DatasetSourcePreview
    from yaml_mapping_model import YamlMappingModel
else:
    from .archive_service import extract_archive
    from .class_mapping_model import ClassMappingRule, mapping_signature
    from .dataset_cache_service import (
        PreparedCacheMetadata,
        build_prepared_cache_key,
        compute_source_fingerprint,
        read_prepared_cache_metadata,
        write_prepared_cache_metadata,
    )
    from .exceptions import DatasetPrepareError, OperationCancelledError
    from .logging_utils import get_logger
    from .source_preview_service import DatasetSourcePreview
    from .yaml_mapping_model import YamlMappingModel

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VAL_KEYS = ("val", "valid", "validation", "vaild")
LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class PreparedDatasetSpec:
    """Prepared dataset 規格。"""

    prepared_root: Path
    patched_yaml_path: Path
    active_model_names: tuple[str, ...]
    cache_key: str
    contains_unlabeled_images: bool
    empty_label_count: int
    dataset_key: str


def prepare_dataset(
    *,
    preview: DatasetSourcePreview,
    model_names: list[str],
    mapping_rules: list[ClassMappingRule],
    temp_root: Path,
    progress_cb=None,
    cancel_cb=None,
) -> PreparedDatasetSpec:
    """建立或重用 prepared dataset。"""
    LOGGER.info("Preparing dataset. source=%s kind=%s", preview.source_path, preview.source_kind)
    cache_root = temp_root / "cache"
    raw_cache_root = cache_root / "raw_cache"
    prepared_cache_root = cache_root / "prepared_cache"
    LOGGER.debug("Computing source fingerprint for prepare. source=%s", preview.source_path)
    source_fingerprint = compute_source_fingerprint(
        preview.source_path,
        progress_cb=progress_cb,
        cancel_cb=cancel_cb,
    )
    mapping_signature_value = mapping_signature(mapping_rules)
    prepared_key = build_prepared_cache_key(
        source_fingerprint,
        preview.yaml_signature,
        mapping_signature_value,
    )
    prepared_dir = prepared_cache_root / prepared_key
    metadata = read_prepared_cache_metadata(prepared_dir)
    if metadata is not None and metadata.source_fingerprint == source_fingerprint:
        yaml_path = prepared_dir / "data.yaml"
        if yaml_path.exists():
            LOGGER.info("Prepared dataset cache hit. source=%s cache_key=%s", preview.source_path, prepared_key)
            stats = _load_prepare_stats(prepared_dir)
            return PreparedDatasetSpec(
                prepared_root=prepared_dir,
                patched_yaml_path=yaml_path,
                active_model_names=tuple(model_names),
                cache_key=prepared_key,
                contains_unlabeled_images=bool(stats.get("contains_unlabeled_images", False)),
                empty_label_count=int(stats.get("empty_label_count", 0)),
                dataset_key=_dataset_key(preview.source_path),
            )
    LOGGER.info("Prepared dataset cache miss. source=%s cache_key=%s", preview.source_path, prepared_key)

    raw_dir = realize_raw_dataset(
        preview=preview,
        cache_root=raw_cache_root,
        source_fingerprint=source_fingerprint,
        progress_cb=progress_cb,
        cancel_cb=cancel_cb,
    )
    _prepare_dataset_from_raw(
        preview=preview,
        raw_dir=raw_dir,
        prepared_dir=prepared_dir,
        model_names=model_names,
        mapping_rules=mapping_rules,
        progress_cb=progress_cb,
        cancel_cb=cancel_cb,
    )
    write_prepared_cache_metadata(
        prepared_dir,
        PreparedCacheMetadata(
            source_fingerprint=source_fingerprint,
            yaml_signature=preview.yaml_signature,
            mapping_signature=mapping_signature_value,
            source_kind=preview.source_kind,
        ),
    )
    stats = _load_prepare_stats(prepared_dir)
    LOGGER.info(
        "Prepared dataset ready. source=%s prepared_dir=%s unlabeled=%s empty_label_count=%s",
        preview.source_path,
        prepared_dir,
        bool(stats.get("contains_unlabeled_images", False)),
        int(stats.get("empty_label_count", 0)),
    )
    return PreparedDatasetSpec(
        prepared_root=prepared_dir,
        patched_yaml_path=prepared_dir / "data.yaml",
        active_model_names=tuple(model_names),
        cache_key=prepared_key,
        contains_unlabeled_images=bool(stats.get("contains_unlabeled_images", False)),
        empty_label_count=int(stats.get("empty_label_count", 0)),
        dataset_key=_dataset_key(preview.source_path),
    )


def realize_raw_dataset(
    *,
    preview: DatasetSourcePreview,
    cache_root: Path,
    source_fingerprint: str,
    progress_cb=None,
    cancel_cb=None,
) -> Path:
    """建立或重用 raw cache。"""
    raw_dir = cache_root / source_fingerprint
    yaml_path = raw_dir / preview.yaml_locator
    if yaml_path.exists():
        LOGGER.info("Raw dataset cache hit. source=%s raw_dir=%s", preview.source_path, raw_dir)
        return raw_dir
    if raw_dir.exists():
        LOGGER.warning("Removing incomplete raw dataset cache before rebuild. raw_dir=%s", raw_dir)
        shutil.rmtree(raw_dir, ignore_errors=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    if preview.source_kind == "folder":
        LOGGER.info("Copying folder dataset into raw cache. source=%s raw_dir=%s", preview.source_path, raw_dir)
        _copy_tree(preview.source_path, raw_dir, progress_cb=progress_cb, cancel_cb=cancel_cb)
        LOGGER.info("Folder dataset copied into raw cache. raw_dir=%s", raw_dir)
        return raw_dir
    if preview.source_kind in {"zip", "rar"}:
        LOGGER.info("Extracting archive dataset into raw cache. source=%s raw_dir=%s", preview.source_path, raw_dir)
        extract_archive(preview.source_path, raw_dir, progress_cb=progress_cb, cancel_cb=cancel_cb)
        LOGGER.info("Archive dataset extracted into raw cache. raw_dir=%s", raw_dir)
        return raw_dir
    raise DatasetPrepareError(f"不支援的來源類型：{preview.source_kind}")


def build_aggregate_dataset(
    prepared_specs: list[PreparedDatasetSpec],
    output_root: Path,
    model_names: list[str],
    progress_cb=None,
    cancel_cb=None,
) -> Path:
    """建立 aggregate validation dataset。"""
    aggregate_root = output_root / "aggregate_dataset"
    if aggregate_root.exists():
        LOGGER.warning("Removing previous aggregate dataset directory. aggregate_root=%s", aggregate_root)
        shutil.rmtree(aggregate_root, ignore_errors=True)
    image_dir = aggregate_root / "images" / "val"
    label_dir = aggregate_root / "labels" / "val"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    input_images: list[Path] = []
    for spec in prepared_specs:
        input_images.extend(sorted((spec.prepared_root / "images" / "val").glob("*")))
    LOGGER.info("Building aggregate dataset. output_root=%s source_image_count=%s", output_root, len(input_images))
    total = max(1, len(input_images))
    counter = 0
    for spec in prepared_specs:
        for image_path in sorted((spec.prepared_root / "images" / "val").glob("*")):
            if cancel_cb is not None and cancel_cb():
                raise OperationCancelledError("作業已取消。")
            counter += 1
            name = f"{spec.dataset_key}_{counter:08d}{image_path.suffix.lower()}"
            target_image = image_dir / name
            target_label = label_dir / f"{Path(name).stem}.txt"
            _link_or_copy(image_path, target_image)
            _link_or_copy(spec.prepared_root / "labels" / "val" / f"{image_path.stem}.txt", target_label)
            if progress_cb is not None:
                progress_cb(counter, total)
    YamlMappingModel(
        data={
            "path": str(aggregate_root.resolve()),
            "train": "images/val",
            "val": "images/val",
            "names": list(model_names),
            "nc": len(model_names),
        }
    ).save(aggregate_root / "data.yaml")
    LOGGER.info("Aggregate dataset ready. aggregate_root=%s image_count=%s", aggregate_root, counter)
    return aggregate_root / "data.yaml"


def _prepare_dataset_from_raw(
    *,
    preview: DatasetSourcePreview,
    raw_dir: Path,
    prepared_dir: Path,
    model_names: list[str],
    mapping_rules: list[ClassMappingRule],
    progress_cb=None,
    cancel_cb=None,
) -> None:
    yaml_path = raw_dir / preview.yaml_locator
    if not yaml_path.exists():
        raise DatasetPrepareError(f"raw cache 內缺少 YAML：{yaml_path}")
    if prepared_dir.exists():
        LOGGER.warning("Removing previous prepared dataset directory before rebuild. prepared_dir=%s", prepared_dir)
        shutil.rmtree(prepared_dir, ignore_errors=True)
    prepared_image_dir = prepared_dir / "images" / "val"
    prepared_label_dir = prepared_dir / "labels" / "val"
    prepared_image_dir.mkdir(parents=True, exist_ok=True)
    prepared_label_dir.mkdir(parents=True, exist_ok=True)
    mapping = YamlMappingModel.from_file(yaml_path).data
    dataset_root = _resolve_dataset_root(yaml_path, str(mapping.get("path", ".")), raw_dir)
    active_split_key, split_entries = _resolve_active_split(mapping)
    images = _collect_split_images(split_entries, dataset_root, yaml_path.parent)
    LOGGER.info(
        "Preparing raw dataset into validation dataset. yaml=%s dataset_root=%s split=%s image_count=%s",
        yaml_path,
        dataset_root,
        active_split_key,
        len(images),
    )
    total = max(1, len(images))
    target_lookup = {rule.source_name: rule.target_model_index for rule in mapping_rules if not rule.ignore}
    ignored_names = {rule.source_name for rule in mapping_rules if rule.ignore}
    empty_label_count = 0
    contains_unlabeled = False
    for index, image_path in enumerate(images, start=1):
        if cancel_cb is not None and cancel_cb():
            raise OperationCancelledError("作業已取消。")
        out_name = f"{index:08d}{image_path.suffix.lower()}"
        shutil.copy2(image_path, prepared_image_dir / out_name)
        label_path = _find_label_path(image_path)
        out_lines = _rewrite_label_lines(label_path, mapping_rules, target_lookup, ignored_names)
        if not out_lines:
            empty_label_count += 1
            contains_unlabeled = True
        (prepared_label_dir / f"{Path(out_name).stem}.txt").write_text(
            "\n".join(out_lines) + ("\n" if out_lines else ""),
            encoding="utf-8",
        )
        if progress_cb is not None:
            progress_cb(index, total)
    YamlMappingModel(
        data={
            "path": str(prepared_dir.resolve()),
            "train": "images/val",
            "val": "images/val",
            "names": list(model_names),
            "nc": len(model_names),
            "source_split": active_split_key,
        }
    ).save(prepared_dir / "data.yaml")
    _write_prepare_stats(
        prepared_dir,
        {
            "contains_unlabeled_images": contains_unlabeled,
            "empty_label_count": empty_label_count,
        },
    )
    LOGGER.info(
        "Prepared validation dataset materialized. prepared_dir=%s image_count=%s empty_label_count=%s",
        prepared_dir,
        len(images),
        empty_label_count,
    )


def _resolve_active_split(mapping: dict[str, object]) -> tuple[str, list[str]]:
    raw_val = mapping.get("val")
    if raw_val is None:
        for key in VAL_KEYS:
            if key in mapping:
                raw_val = mapping[key]
                break
    if raw_val is not None:
        return "val", _normalize_split_entries(raw_val)
    raw_test = mapping.get("test")
    if raw_test is not None:
        return "test", _normalize_split_entries(raw_test)
    raise DatasetPrepareError("找不到可用的 val 或 test split。")


def _normalize_split_entries(raw_value: object) -> list[str]:
    if isinstance(raw_value, list):
        return [str(item) for item in raw_value]
    return [str(raw_value)]


def _resolve_dataset_root(yaml_path: Path, path_value: str, raw_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        if path.exists():
            LOGGER.debug("Using absolute dataset root from YAML. yaml=%s dataset_root=%s", yaml_path, path)
            return path.resolve()
        LOGGER.warning(
            "Absolute dataset root from YAML does not exist after relocation; falling back to raw dataset root. yaml=%s dataset_root=%s raw_root=%s",
            yaml_path,
            path,
            raw_root,
        )
        return raw_root.resolve()
    return (yaml_path.parent / path).resolve()


def _collect_split_images(entries: list[str], dataset_root: Path, yaml_dir: Path) -> list[Path]:
    images: list[Path] = []
    for raw_entry in entries:
        entry_path = _resolve_entry_path(raw_entry, dataset_root, yaml_dir)
        if entry_path.is_dir():
            images.extend([path for path in sorted(entry_path.rglob("*")) if path.suffix.lower() in IMAGE_EXTENSIONS])
            continue
        if entry_path.is_file() and entry_path.suffix.lower() == ".txt":
            text = entry_path.read_text(encoding="utf-8", errors="ignore")
            for line in text.splitlines():
                item = line.strip()
                if not item or item.startswith("#"):
                    continue
                images.append(_resolve_entry_path(item, dataset_root, entry_path.parent))
            continue
        if entry_path.is_file() and entry_path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(entry_path)
    if not images:
        raise DatasetPrepareError("找不到任何驗證圖片。")
    return images


def _resolve_entry_path(raw_entry: str, dataset_root: Path, yaml_dir: Path) -> Path:
    path = Path(raw_entry)
    if path.is_absolute():
        fallback = _fallback_absolute_path(path, dataset_root, yaml_dir)
        if fallback is not None:
            return fallback
        if path.exists():
            return path.resolve()
    candidate_root = (dataset_root / raw_entry).resolve()
    if candidate_root.exists():
        return candidate_root
    candidate_yaml = (yaml_dir / raw_entry).resolve()
    if candidate_yaml.exists():
        return candidate_yaml
    parts = [part for part in raw_entry.replace("\\", "/").split("/") if part not in {"", "."}]
    while parts and parts[0] == "..":
        parts.pop(0)
    fallback = dataset_root.joinpath(*parts).resolve() if parts else dataset_root
    return fallback


def _fallback_absolute_path(raw_path: Path, dataset_root: Path, yaml_dir: Path) -> Path | None:
    parts = [part for part in raw_path.parts if part not in {"", raw_path.anchor}]
    for base in (dataset_root, yaml_dir):
        for index in range(len(parts)):
            candidate = base.joinpath(*parts[index:]).resolve()
            if candidate.exists():
                return candidate
    return None


def _find_label_path(image_path: Path) -> Path | None:
    candidates = [image_path.with_suffix(".txt")]
    parts = list(image_path.parts)
    for index, part in enumerate(parts):
        if part.lower() != "images":
            continue
        replaced = parts[:]
        replaced[index] = "labels"
        candidates.insert(0, Path(*replaced).with_suffix(".txt"))
        break
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _rewrite_label_lines(
    label_path: Path | None,
    mapping_rules: list[ClassMappingRule],
    target_lookup: dict[str, int | None],
    ignored_names: set[str],
) -> list[str]:
    if label_path is None or not label_path.exists():
        return []
    rule_names = [rule.source_name for rule in mapping_rules]
    text = label_path.read_text(encoding="utf-8", errors="ignore")
    out_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            source_index = int(parts[0])
        except Exception:
            continue
        if source_index < 0 or source_index >= len(rule_names):
            continue
        source_name = rule_names[source_index]
        if source_name in ignored_names:
            continue
        target_index = target_lookup.get(source_name)
        if target_index is None:
            continue
        out_lines.append(" ".join([str(target_index), *parts[1:]]))
    return out_lines


def _copy_tree(source_dir: Path, target_dir: Path, progress_cb=None, cancel_cb=None) -> None:
    files = [path for path in source_dir.rglob("*") if path.is_file()]
    total = max(1, len(files))
    for index, path in enumerate(files, start=1):
        if cancel_cb is not None and cancel_cb():
            raise OperationCancelledError("作業已取消。")
        rel = path.relative_to(source_dir)
        destination = target_dir / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        if progress_cb is not None:
            progress_cb(index, total)


def _dataset_key(source_path: Path) -> str:
    raw = source_path.resolve().as_posix().encode("utf-8")
    digest = hashlib.sha1(raw).hexdigest()[:8]
    stem = "".join(ch.lower() if ch.isalnum() else "_" for ch in source_path.stem).strip("_") or "dataset"
    return f"{stem}_{digest}"


def _link_or_copy(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if os.path.exists(target_path):
            os.unlink(target_path)
        os.link(source_path, target_path)
        LOGGER.debug("Linked file into aggregate dataset. source=%s target=%s", source_path, target_path)
    except Exception:
        shutil.copy2(source_path, target_path)
        LOGGER.debug("Copied file into aggregate dataset after link fallback. source=%s target=%s", source_path, target_path)


def _write_prepare_stats(prepared_dir: Path, payload: dict[str, object]) -> None:
    (prepared_dir / "prepare_stats.yaml").write_text(
        YamlMappingModel(data=payload).to_text(),
        encoding="utf-8",
    )


def _load_prepare_stats(prepared_dir: Path) -> dict[str, object]:
    stats_path = prepared_dir / "prepare_stats.yaml"
    if not stats_path.exists():
        return {}
    return YamlMappingModel.from_text(stats_path.read_text(encoding="utf-8")).data
