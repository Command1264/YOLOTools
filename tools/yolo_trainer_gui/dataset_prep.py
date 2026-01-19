# dataset_prep.py
from __future__ import annotations
import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

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
