from __future__ import annotations

import json
import time
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

APP_DIR = Path(__file__).resolve().parent
GUI_DIR = APP_DIR.parent / "yolo_trainer_gui"
sys.path.insert(0, str(GUI_DIR))

from dataset_prep import (  # noqa: E402
    extract_zip, find_data_yaml, filter_unlabeled_yolo_dataset,
    zip_folder, remove_dir_safe, rewrite_data_yaml_to_extracted_root,
    compute_dataset_manifest, manifest_matches, read_manifest, write_manifest,
)
from trainer_worker import parse_results_csv, read_last_epoch_row  # noqa: E402


CONFIG_PATH = APP_DIR / "trainer_config_colab.json"


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class TrainConfig:
    task: str
    dataset_zip: str
    work_dir: str
    out_zip_dir: str
    model: str
    model_dir: str
    epochs: int
    imgsz: int
    batch: int
    device: str
    resume: bool
    skip_unlabeled: bool
    delete_temp: bool


def load_config(path: Path) -> TrainConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    return TrainConfig(
        task=str(data.get("task", "detect")),
        dataset_zip=str(data.get("dataset_zip", "")),
        work_dir=str(data.get("work_dir", str(APP_DIR / "workdir"))),
        out_zip_dir=str(data.get("out_zip_dir", str(APP_DIR / "output_zips"))),
        model=str(data.get("model", "")),
        model_dir=str(data.get("model_dir", "")),
        epochs=int(data.get("epochs", 50)),
        imgsz=int(data.get("imgsz", 640)),
        batch=int(data.get("batch", 16)),
        device=str(data.get("device", "")),
        resume=bool(data.get("resume", False)),
        skip_unlabeled=bool(data.get("skip_unlabeled", True)),
        delete_temp=bool(data.get("delete_temp", True)),
    )


def log(s: str):
    print(s, end="", flush=True)


def ensure_dirs(*paths: Path):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def run():
    if not CONFIG_PATH.exists():
        raise RuntimeError(f"找不到設定檔：{CONFIG_PATH}")
    cfg = load_config(CONFIG_PATH)

    dataset_zip = Path(cfg.dataset_zip).expanduser().resolve()
    if not dataset_zip.exists():
        raise RuntimeError("dataset.zip 不存在")

    work_dir = Path(cfg.work_dir).expanduser().resolve()
    out_zip_dir = Path(cfg.out_zip_dir).expanduser().resolve()
    model_dir = Path(cfg.model_dir).expanduser().resolve() if cfg.model_dir else None
    ensure_dirs(work_dir, out_zip_dir)
    if model_dir:
        ensure_dirs(model_dir)

    log(f"[{now_str()}] ===== 開始訓練 =====\n")
    log(f"[{now_str()}] task={cfg.task}\n")
    log(f"[{now_str()}] dataset.zip={dataset_zip}\n")
    log(f"[{now_str()}] work_dir={work_dir}\n")
    log(f"[{now_str()}] out_zip_dir={out_zip_dir}\n")
    log(f"[{now_str()}] model={cfg.model}\n")
    log(f"[{now_str()}] model_dir={cfg.model_dir}\n")
    log(f"[{now_str()}] epochs={cfg.epochs}, imgsz={cfg.imgsz}, batch={cfg.batch}, device={cfg.device}, resume={cfg.resume}\n\n")

    # 1) extract or reuse
    name_prefix = f"ds_{dataset_zip.stem}"
    extracted_root = work_dir / name_prefix
    manifest_path = extracted_root / "_manifest.yaml"
    reuse = False
    if extracted_root.exists():
        log(f"[{now_str()}] 檢查資料集（可否重用）...\n")
        expect = read_manifest(manifest_path)
        if expect and manifest_matches(extracted_root, expect):
            reuse = True
            log(f"[{now_str()}] 使用既有解壓資料：{extracted_root}\n")
    if not reuse:
        log(f"[{now_str()}] 解壓縮資料集...\n")
        extracted_root = extract_zip(dataset_zip, work_dir, name_prefix)
        log(f"[{now_str()}] 解壓縮完成：{extracted_root}\n")
        try:
            manifest = compute_dataset_manifest(extracted_root)
            write_manifest(manifest_path, manifest)
        except Exception:
            pass

    # 2) find yaml
    data_yaml = find_data_yaml(extracted_root)
    if not data_yaml:
        raise RuntimeError(f"找不到 data.yaml：{extracted_root}")
    log(f"[{now_str()}] 找到原始 data.yaml：{data_yaml}\n")

    log(f"[{now_str()}] 修正 data.yaml 路徑（指向解壓資料夾）...\n")
    patched_yaml = rewrite_data_yaml_to_extracted_root(data_yaml, extracted_root, log)
    log(f"[{now_str()}] 使用 patched data.yaml：{patched_yaml}\n")
    data_yaml = patched_yaml

    # 3) optional filter unlabeled
    if cfg.skip_unlabeled and cfg.task in ("detect", "segment", "pose", "obb"):
        log(f"[{now_str()}] 清理未標記圖片（skip unlabeled）...\n")
        stat = filter_unlabeled_yolo_dataset(data_yaml, log, apply_to=("train", "val"))
        log(f"[{now_str()}] 清理完成：checked={stat['checked']}, removed={stat['removed']}\n")

    # 4) load model
    from ultralytics import YOLO
    model_path = cfg.model
    try:
        if model_dir and not Path(cfg.model).expanduser().exists():
            model_path = str(model_dir / Path(cfg.model).name)
    except Exception:
        pass
    model = YOLO(model_path)
    log(f"[{now_str()}] 模型：{model_path}\n")

    # 5) train
    run_name = f"{cfg.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    train_kwargs = dict(
        data=str(data_yaml) if cfg.task != "classify" else str(extracted_root),
        epochs=int(cfg.epochs),
        imgsz=int(cfg.imgsz),
        batch=int(cfg.batch),
        project=str(work_dir / "runs"),
        name=run_name,
        exist_ok=False,
        verbose=True,
    )
    if cfg.device.strip():
        train_kwargs["device"] = cfg.device.strip()
    if cfg.resume:
        train_kwargs["resume"] = True

    results = model.train(**train_kwargs)

    # 6) resolve run_dir
    run_dir = None
    try:
        run_dir = Path(results.save_dir)
    except Exception:
        try:
            run_dir = Path(model.trainer.save_dir)
        except Exception:
            run_dir = None
    if not run_dir or not run_dir.exists():
        candidates = sorted((work_dir / "runs").rglob("results.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            run_dir = candidates[0].parent
    if not run_dir or not run_dir.exists():
        raise RuntimeError("訓練完成但找不到 run_dir（無法整理輸出）")

    log(f"[{now_str()}] run_dir：{run_dir}\n")

    metrics_payload = parse_results_csv(run_dir)
    last_row = metrics_payload.get("last_row") if metrics_payload else None
    log(f"[{now_str()}] 上一輪指標：{read_last_epoch_row(run_dir) or last_row}\n")

    # 7) zip output
    out_zip_path = out_zip_dir / f"{run_dir.name}.zip"
    zip_folder(run_dir, out_zip_path, log)

    # 8) optional delete temp
    if cfg.delete_temp:
        remove_dir_safe(extracted_root, log)

    log(f"\n[{now_str()}] ===== 完成 =====\n")


if __name__ == "__main__":
    run()
