# dataset_prep.py
from __future__ import annotations
import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import hashlib

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LBL_EXT = ".txt"

def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def extract_zip(zip_path: Path, out_dir: Path, name_prefix: str) -> Path:
    safe_mkdir(out_dir)
    root = out_dir / name_prefix
    safe_mkdir(root)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)

    entries = [p for p in root.iterdir()]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return root

def _iter_files_for_hash(root: Path) -> List[Path]:
    out = []
    for p in root.rglob("*"):
        if p.is_file():
            out.append(p)
    return out

def _hash_file_list(paths: List[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda x: str(x).lower()):
        try:
            rel = str(p).replace("\\", "/")
            h.update(rel.encode("utf-8", errors="ignore"))
            h.update(str(p.stat().st_size).encode("utf-8"))
        except Exception:
            continue
    return h.hexdigest()

def compute_dataset_manifest(root: Path) -> Dict[str, Any]:
    files = _iter_files_for_hash(root)
    return {
        "file_count": len(files),
        "hash": _hash_file_list(files),
    }

def manifest_matches(root: Path, expected: Dict[str, Any]) -> bool:
    try:
        cur = compute_dataset_manifest(root)
        return cur.get("file_count") == expected.get("file_count") and cur.get("hash") == expected.get("hash")
    except Exception:
        return False

def write_manifest(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

def read_manifest(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def find_data_yaml(root: Path) -> Optional[Path]:
    for p in (root / "data.yaml", root / "data.yml"):
        if p.exists():
            return p
    for p in list(root.rglob("data.yaml")) + list(root.rglob("data.yml")):
        return p
    return None

def load_data_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))

def _resolve_path(base_dir: Path, p: str) -> Path:
    # Ultralytics supports relative paths in yaml; treat relative to yaml directory
    pp = Path(p)
    return pp if pp.is_absolute() else (base_dir / pp).resolve()

def _iter_images(img_dir: Path) -> List[Path]:
    if not img_dir.exists():
        return []
    out = []
    for p in img_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
    return out

def _label_path_for_image(img_path: Path) -> Path:
    # Typical: images/.../xxx.jpg -> labels/.../xxx.txt
    # If dataset uses same-dir labels, still ok if you set yaml accordingly, but most are images/labels split.
    # We'll just replace suffix and later decide base labels dir by relative path transform.
    return img_path.with_suffix(LBL_EXT)

def filter_unlabeled_yolo_dataset(data_yaml_path: Path, log_cb, apply_to: Tuple[str, ...] = ("train", "val")) -> Dict[str, int]:
    """
    Remove images without corresponding label file (or empty label file) for each split in apply_to.
    Works for typical YOLO format (images/ labels/). For nonstandard formats, best-effort.
    """
    base = data_yaml_path.parent
    data = load_data_yaml(data_yaml_path)

    removed_total = 0
    checked_total = 0

    for split in apply_to:
        if split not in data:
            continue
        split_path = _resolve_path(base, str(data[split]))
        # If yaml points to images folder directly:
        img_dir = split_path
        imgs = _iter_images(img_dir)
        if not imgs:
            # If points to dataset root, try common 'images/split'
            cand = split_path / "images" / split
            imgs = _iter_images(cand)
            if imgs:
                img_dir = cand

        # infer labels dir: replace 'images' with 'labels' at same relative position if possible
        labels_dir = None
        parts = list(img_dir.parts)
        if "images" in parts:
            idx = parts.index("images")
            labels_dir = Path(*parts[:idx], "labels", *parts[idx+1:])
        else:
            labels_dir = img_dir.parent / "labels"

        removed = 0
        checked = 0

        for img in imgs:
            checked += 1
            checked_total += 1
            rel = None
            try:
                rel = img.relative_to(img_dir)
            except Exception:
                rel = img.name

            lbl = (labels_dir / rel).with_suffix(LBL_EXT)

            # unlabeled: missing or empty
            if (not lbl.exists()) or (lbl.exists() and lbl.stat().st_size == 0):
                try:
                    img.unlink(missing_ok=True)
                    removed += 1
                    removed_total += 1
                except Exception:
                    pass

        log_cb(f"  - split={split}: checked={checked}, removed_unlabeled={removed}\n")

    return {"checked": checked_total, "removed": removed_total}

def zip_folder(src_dir: Path, zip_path: Path, log_cb) -> None:
    safe_mkdir(zip_path.parent)
    if zip_path.exists():
        zip_path.unlink()

    log_cb(f"打包輸出 zip: {zip_path}\n")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in src_dir.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(src_dir)))

def remove_dir_safe(p: Path, log_cb) -> None:
    try:
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
            log_cb(f"已刪除暫存資料夾：{p}\n")
    except Exception as e:
        log_cb(f"刪除暫存資料夾失敗：{e}\n")


# --- add to dataset_prep.py (bottom) ---

def rewrite_data_yaml_to_extracted_root(
        orig_yaml: Path,
        extracted_root: Path,
        log_cb
) -> Path:
    """
    Create a patched data.yaml under extracted_root/_patched/data.yaml
    so that Ultralytics will read images from the extracted dataset, not the original absolute paths.

    Strategy:
    - Load yaml
    - If it has 'path': set to extracted_root
    - For 'train'/'val'/'test':
        - If absolute and contains 'images\\train' style, convert to relative 'images/train'
        - Else if absolute but doesn't contain images folder, try best-effort mapping by taking last parts
        - If already relative, keep
    """
    data = load_data_yaml(orig_yaml)

    patched_dir = extracted_root / "_patched"
    safe_mkdir(patched_dir)
    patched_yaml = patched_dir / "data.yaml"

    def to_posix_rel(p: str) -> str:
        return p.replace("\\", "/").lstrip("./")

    def best_rel_from_abs(abs_path: str) -> str:
        s = abs_path.replace("\\", "/")
        # Common patterns:
        # .../images/train  or .../images/val  or .../labels/train
        for key in ["/images/train", "/images/val", "/images/test",
                    "/images/Train", "/images/Val", "/images/Test",
                    "/labels/train", "/labels/val", "/labels/test"]:
            idx = s.lower().find(key.lower())
            if idx != -1:
                return to_posix_rel(s[idx + 1:])  # remove leading '/'
        # If no known marker, just take last 2 segments as a weak fallback
        parts = [x for x in s.split("/") if x]
        if len(parts) >= 2:
            return to_posix_rel("/".join(parts[-2:]))
        return to_posix_rel(parts[-1]) if parts else "images/train"

    # 1) force path to extracted_root (Ultralytics uses path as base)
    data["path"] = str(extracted_root)

    # 2) rewrite splits
    for split in ("train", "val", "test"):
        if split not in data:
            continue
        v = str(data[split])
        pv = Path(v)
        if pv.is_absolute():
            rel = best_rel_from_abs(v)
            data[split] = rel
            log_cb(f"  - rewrite {split}: ABS -> REL  {v}  =>  {rel}\n")
        else:
            # keep relative, but normalize slashes
            data[split] = to_posix_rel(v)
            log_cb(f"  - normalize {split}: {v} => {data[split]}\n")

    # 3) save
    patched_yaml.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    log_cb(f"[patched] data.yaml => {patched_yaml}\n")
    return patched_yaml

