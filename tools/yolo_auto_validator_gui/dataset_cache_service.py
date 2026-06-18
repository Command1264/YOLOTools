from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    from archive_service import compute_archive_sha256
    from exceptions import CacheError, OperationCancelledError
    from logging_utils import get_logger
else:
    from .archive_service import compute_archive_sha256
    from .exceptions import CacheError, OperationCancelledError
    from .logging_utils import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class PreparedCacheMetadata:
    """Prepared dataset 快取 metadata。"""

    source_fingerprint: str
    yaml_signature: str
    mapping_signature: str
    source_kind: str

    @classmethod
    def from_json_text(cls, text: str) -> "PreparedCacheMetadata":
        """由 JSON 建立 metadata。"""
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError("prepared cache metadata 格式錯誤。")
        return cls(
            source_fingerprint=str(obj.get("source_fingerprint", "")),
            yaml_signature=str(obj.get("yaml_signature", "")),
            mapping_signature=str(obj.get("mapping_signature", "")),
            source_kind=str(obj.get("source_kind", "")),
        )

    def to_json_text(self) -> str:
        """輸出 JSON。"""
        return json.dumps(self.__dict__, ensure_ascii=False, indent=2)


def compute_source_fingerprint(source_path: Path, progress_cb=None, cancel_cb=None) -> str:
    """依來源型別計算 fingerprint。"""
    source = source_path.resolve()
    LOGGER.debug("Computing source fingerprint. source=%s", source)
    if source.is_dir():
        fingerprint = _compute_folder_fingerprint(source, progress_cb=progress_cb, cancel_cb=cancel_cb)
        LOGGER.debug("Folder fingerprint completed. source=%s fingerprint=%s", source, fingerprint)
        return fingerprint
    if source.is_file() and source.suffix.lower() in {".zip", ".rar"}:
        fingerprint = compute_archive_sha256(source, progress_cb=progress_cb, cancel_cb=cancel_cb)
        LOGGER.debug("Archive fingerprint completed. source=%s fingerprint=%s", source, fingerprint)
        return fingerprint
    raise CacheError(f"不支援的來源：{source}")


def build_prepared_cache_key(
    source_fingerprint: str,
    yaml_signature: str,
    mapping_signature_value: str,
) -> str:
    """建立 prepared cache key。"""
    raw = f"{source_fingerprint}:{yaml_signature}:{mapping_signature_value}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def read_prepared_cache_metadata(cache_dir: Path) -> PreparedCacheMetadata | None:
    """讀取 prepared cache metadata。"""
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.exists():
        LOGGER.debug("Prepared cache metadata missing. cache_dir=%s", cache_dir)
        return None
    try:
        metadata = PreparedCacheMetadata.from_json_text(metadata_path.read_text(encoding="utf-8"))
        LOGGER.debug("Prepared cache metadata loaded. cache_dir=%s", cache_dir)
        return metadata
    except Exception:
        LOGGER.warning("Prepared cache metadata could not be parsed. cache_dir=%s", cache_dir, exc_info=True)
        return None


def write_prepared_cache_metadata(cache_dir: Path, metadata: PreparedCacheMetadata) -> None:
    """寫入 prepared cache metadata。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "metadata.json").write_text(metadata.to_json_text(), encoding="utf-8")
    LOGGER.debug("Prepared cache metadata written. cache_dir=%s", cache_dir)


def _compute_folder_fingerprint(folder: Path, progress_cb=None, cancel_cb=None) -> str:
    files = [path for path in folder.rglob("*") if path.is_file()]
    digest = hashlib.sha256()
    total = max(1, len(files))
    LOGGER.debug("Computing folder fingerprint. folder=%s file_count=%s", folder, len(files))
    for index, path in enumerate(sorted(files, key=lambda item: str(item).lower()), start=1):
        if cancel_cb is not None and cancel_cb():
            raise OperationCancelledError("作業已取消。")
        rel = str(path.relative_to(folder)).replace("\\", "/")
        stat = path.stat()
        digest.update(rel.encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
        if progress_cb is not None:
            progress_cb(index, total)
    return digest.hexdigest()
