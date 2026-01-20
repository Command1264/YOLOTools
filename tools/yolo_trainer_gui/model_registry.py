# model_registry.py
from __future__ import annotations
import re
import json
from pathlib import Path
from typing import List, Dict, Optional

import requests

DEFAULT_FALLBACK = [
    # Detect (common)
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
    # Seg
    "yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt", "yolov8l-seg.pt", "yolov8x-seg.pt",
    "yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt", "yolo11l-seg.pt", "yolo11x-seg.pt",
    # Pose
    "yolov8n-pose.pt", "yolov8s-pose.pt", "yolov8m-pose.pt", "yolov8l-pose.pt", "yolov8x-pose.pt",
    "yolo11n-pose.pt", "yolo11s-pose.pt", "yolo11m-pose.pt", "yolo11l-pose.pt", "yolo11x-pose.pt",
    # OBB
    "yolov8n-obb.pt", "yolov8s-obb.pt", "yolov8m-obb.pt", "yolov8l-obb.pt", "yolov8x-obb.pt",
    "yolo11n-obb.pt", "yolo11s-obb.pt", "yolo11m-obb.pt", "yolo11l-obb.pt", "yolo11x-obb.pt",
    # Cls (common naming)
    "yolov8n-cls.pt", "yolov8s-cls.pt", "yolov8m-cls.pt", "yolov8l-cls.pt", "yolov8x-cls.pt",
]

DOCS_MODELS_URL = "https://docs.ultralytics.com/models/"  # lists supported families
GITHUB_README_URL = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/README.md"


def _extract_pt_names(text: str) -> List[str]:
    # Pull anything that looks like a weights filename e.g. yolo11n.pt, yolov8n-seg.pt etc.
    names = set(re.findall(r"\b[a-zA-Z0-9][a-zA-Z0-9_.-]*\.pt\b", text))
    # Filter very short / irrelevant
    keep = []
    for n in names:
        ln = n.lower()
        if "yolo" in ln and len(ln) <= 64:
            keep.append(n)
    return sorted(set(keep), key=lambda x: x.lower())

def fetch_ultralytics_weights_online(timeout_sec: int = 6) -> List[str]:
    """
    Best-effort online fetch. If it fails, raise.
    """
    texts = []
    for url in (DOCS_MODELS_URL, GITHUB_README_URL):
        r = requests.get(url, timeout=timeout_sec)
        r.raise_for_status()
        texts.append(r.text)

    out = set()
    for t in texts:
        out.update(_extract_pt_names(t))

    # If extraction yields too few, fallback to DEFAULT_FALLBACK upstream.
    return sorted(out, key=lambda x: x.lower())

def load_weights(cache_path: Path) -> List[str]:
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                return data
        except Exception:
            pass
    return DEFAULT_FALLBACK[:]

def save_weights(cache_path: Path, weights: List[str]) -> None:
    cache_path.write_text(json.dumps(weights, ensure_ascii=False, indent=2), encoding="utf-8")

def get_weights(cache_path: Path, try_online: bool = True) -> Dict[str, List[str]]:
    """
    Return dict with:
      - weights: list
      - source: 'online' | 'cache' | 'fallback'
    """
    if try_online:
        try:
            w = fetch_ultralytics_weights_online()
            if len(w) >= 10:
                save_weights(cache_path, w)
                return {"weights": w, "source": "online"}
        except Exception:
            pass

    w2 = load_weights(cache_path)
    src = "cache" if cache_path.exists() else "fallback"
    return {"weights": w2, "source": src}

# --- add to model_registry.py (bottom) ---

def infer_task_from_weight_name(weight: str) -> str:
    w = weight.lower()
    if "-seg.pt" in w:
        return "segment"
    if "-cls.pt" in w:
        return "classify"
    if "-pose.pt" in w:
        return "pose"
    if "-obb.pt" in w:
        return "obb"
    # heuristic: assume detect
    return "detect"

def filter_weights_by_task(weights: list[str], task: str) -> list[str]:
    task = task.lower().strip()
    if task not in {"detect", "segment", "classify", "pose", "obb"}:
        return _sort_weights_small_to_large(weights)

    out = []
    for w in weights:
        t = infer_task_from_weight_name(w)
        if task == "detect":
            # detect = anything that isn't explicitly seg/cls/pose/obb
            if t == "detect":
                out.append(w)
        else:
            if t == task:
                out.append(w)

    # fallback: if nothing matched, return original to avoid empty dropdown
    return _sort_weights_small_to_large(out if out else weights)


_SIZE_ORDER = {"n": 0, "s": 1, "m": 2, "l": 3, "x": 4}

def _size_rank(name: str) -> int:
    base = Path(name).name.lower()
    if base.endswith(".pt"):
        base = base[:-3]
    base = base.split("-")[0]
    for ch in reversed(base):
        if ch in _SIZE_ORDER:
            return _SIZE_ORDER[ch]
    return 99

def _sort_weights_small_to_large(weights: List[str]) -> List[str]:
    return sorted(weights, key=lambda w: (_size_rank(w), w.lower()))
