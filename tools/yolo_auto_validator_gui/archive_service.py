from __future__ import annotations

import hashlib
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    from exceptions import ArchiveBackendError, OperationCancelledError
    from logging_utils import get_logger
else:
    from .exceptions import ArchiveBackendError, OperationCancelledError
    from .logging_utils import get_logger

_HASH_CHUNK_SIZE = 1024 * 1024
LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class ArchiveMemberPreview:
    """壓縮檔內 YAML 預覽。"""

    member_path: str
    text: str


def read_archive_yaml_preview(source_path: Path) -> ArchiveMemberPreview:
    """只讀取壓縮檔內的 data.yaml。"""
    ext = source_path.suffix.lower()
    LOGGER.debug("Reading archive YAML preview. source=%s ext=%s", source_path, ext)
    if ext == ".zip":
        with zipfile.ZipFile(source_path, "r") as archive:
            names = [_normalize_member_path(name) for name in archive.namelist()]
            member = _find_yaml_member([name for name in names if name.lower().endswith(("data.yaml", "data.yml"))])
            LOGGER.debug("Archive YAML preview resolved. source=%s member=%s", source_path, member)
            return ArchiveMemberPreview(member_path=member, text=archive.read(member).decode("utf-8", errors="ignore"))
    if ext == ".rar":
        rarfile = _import_rarfile()
        with rarfile.RarFile(source_path) as archive:
            names = [_normalize_member_path(info.filename) for info in archive.infolist()]
            member = _find_yaml_member([name for name in names if name.lower().endswith(("data.yaml", "data.yml"))])
            LOGGER.debug("Archive YAML preview resolved. source=%s member=%s", source_path, member)
            return ArchiveMemberPreview(member_path=member, text=archive.read(member).decode("utf-8", errors="ignore"))
    raise ArchiveBackendError(f"不支援的壓縮檔格式：{source_path.suffix}")


def extract_archive(
    source_path: Path,
    output_dir: Path,
    progress_cb=None,
    cancel_cb=None,
) -> None:
    """解壓縮來源到指定資料夾。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    ext = source_path.suffix.lower()
    LOGGER.info("Extracting archive. source=%s output_dir=%s ext=%s", source_path, output_dir, ext)
    if ext == ".zip":
        with zipfile.ZipFile(source_path, "r") as archive:
            members = archive.infolist()
            total = max(1, len(members))
            for index, member in enumerate(members, start=1):
                _check_cancel(cancel_cb)
                archive.extract(member, output_dir)
                if progress_cb is not None:
                    progress_cb(index, total)
        LOGGER.info("Archive extraction completed. source=%s members=%s", source_path, len(members))
        return
    if ext == ".rar":
        rarfile = _import_rarfile()
        with rarfile.RarFile(source_path) as archive:
            members = archive.infolist()
            total = max(1, len(members))
            for index, member in enumerate(members, start=1):
                _check_cancel(cancel_cb)
                archive.extract(member, output_dir)
                if progress_cb is not None:
                    progress_cb(index, total)
        LOGGER.info("Archive extraction completed. source=%s members=%s", source_path, len(members))
        return
    raise ArchiveBackendError(f"不支援的壓縮檔格式：{source_path.suffix}")


def compute_archive_sha256(
    source_path: Path,
    progress_cb=None,
    cancel_cb=None,
) -> str:
    """計算 archive SHA-256。"""
    total_size = max(1, source_path.stat().st_size)
    hashed_size = 0
    digest = hashlib.sha256()
    LOGGER.debug("Computing archive SHA-256. source=%s size=%s", source_path, total_size)
    with source_path.open("rb") as file:
        while True:
            _check_cancel(cancel_cb)
            chunk = file.read(_HASH_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            hashed_size += len(chunk)
            if progress_cb is not None:
                progress_cb(min(hashed_size, total_size), total_size)
    fingerprint = digest.hexdigest()
    LOGGER.debug("Archive SHA-256 completed. source=%s fingerprint=%s", source_path, fingerprint)
    return fingerprint


def detect_rar_backend() -> tuple[bool, str]:
    """檢查 RAR 後端是否可用。"""
    LOGGER.debug("Detecting RAR backend availability.")
    try:
        _import_rarfile()
    except ArchiveBackendError as exc:
        LOGGER.warning("RAR backend unavailable because rarfile import failed. message=%s", exc)
        return False, str(exc)
    tool_candidates = [
        "unrar",
        "rar",
        "unar",
        "bsdtar",
        str(Path("C:/Program Files/WinRAR/UnRAR.exe")),
        str(Path("C:/Program Files/WinRAR/WinRAR.exe")),
    ]
    for candidate in tool_candidates:
        resolved = shutil.which(candidate) or (candidate if Path(candidate).exists() else "")
        if not resolved:
            continue
        try:
            result = subprocess.run(
                [resolved],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except Exception:
            continue
        if result.returncode in {0, 1, 7}:
            LOGGER.info("RAR backend detected. executable=%s", resolved)
            return True, resolved
    LOGGER.warning("RAR backend detection failed. No usable extractor found.")
    return False, "找不到可用的 RAR 解壓工具（WinRAR/UnRAR/unar/bsdtar）。"


def _import_rarfile():
    try:
        import rarfile  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ArchiveBackendError("缺少 rarfile 套件，無法讀取 RAR。") from exc
    return rarfile


def _check_cancel(cancel_cb) -> None:
    if cancel_cb is not None and cancel_cb():
        raise OperationCancelledError("作業已取消。")


def _normalize_member_path(path_text: str) -> str:
    return path_text.replace("\\", "/")


def _find_yaml_member(candidates: list[str]) -> str:
    if not candidates:
        raise FileNotFoundError("壓縮檔內找不到 data.yaml 或 data.yml。")
    return sorted(candidates, key=lambda item: (item.count("/"), len(item), item))[0]
