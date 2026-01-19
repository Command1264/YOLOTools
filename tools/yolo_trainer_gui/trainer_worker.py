# trainer_worker.py
from __future__ import annotations
import time
import queue
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from dataset_prep import (
    extract_zip, find_data_yaml, filter_unlabeled_yolo_dataset,
    zip_folder, remove_dir_safe, rewrite_data_yaml_to_extracted_root
)

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def parse_results_csv(run_dir: Path) -> Dict[str, Optional[float]]:
    """
    Try to parse last-row metrics from results.csv (Ultralytics standard output).
    """
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return {}

    import csv
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}

    last = rows[-1]

    def get_float(keys):
        for k in keys:
            if k in last and last[k] not in (None, ""):
                try:
                    return float(last[k])
                except Exception:
                    pass
        return None

    p = get_float(["metrics/precision(B)", "metrics/precision", "precision"])
    r = get_float(["metrics/recall(B)", "metrics/recall", "recall"])
    map50 = get_float(["metrics/mAP50(B)", "metrics/mAP50", "mAP50"])
    map5095 = get_float(["metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50-95", "mAP5095"])

    f1 = None
    if p is not None and r is not None and (p + r) > 0:
        f1 = 2 * p * r / (p + r)

    return {"precision": p, "recall": r, "f1": f1, "mAP50": map50, "mAP50-95": map5095}


@dataclass
class TrainConfig:
    task: str              # detect / segment / classify / pose / obb
    dataset_zip: str
    work_dir: str
    model: str             # yolov8n.pt or custom path
    epochs: int
    imgsz: int
    batch: int
    device: str            # "" / "cpu" / "0" / "0,1"
    resume: bool
    skip_unlabeled: bool
    delete_temp: bool
    out_zip_dir: str       # where to put zipped run_dir


class TrainerWorker(threading.Thread):
    def __init__(self, cfg: TrainConfig, msg_q: queue.Queue):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.q = msg_q
        self.stop_requested = False

        # internal progress
        self._epoch_total = max(1, int(cfg.epochs))
        self._epoch_cur = 0
        self._batch_total = 0
        self._batch_cur = 0

    # --------- UI messages ----------
    def log(self, s: str):
        self.q.put(("log", s))

    def status(self, s: str):
        self.q.put(("status", s))

    def progress_epoch(self, cur: int, total: int):
        self.q.put(("progress_epoch", cur, total))

    def progress_batch(self, cur: int, total: int):
        self.q.put(("progress_batch", cur, total))

    def done(self, ok: bool, payload: Dict[str, Any]):
        self.q.put(("done", ok, payload))

    def request_stop(self):
        self.stop_requested = True

    # --------- worker ----------
    def run(self):
        t0 = time.time()
        cfg = self.cfg

        try:
            from ultralytics import YOLO

            dataset_zip = Path(cfg.dataset_zip).expanduser().resolve()
            work_dir = Path(cfg.work_dir).expanduser().resolve()
            out_zip_dir = Path(cfg.out_zip_dir).expanduser().resolve()
            work_dir.mkdir(parents=True, exist_ok=True)
            out_zip_dir.mkdir(parents=True, exist_ok=True)

            # 1) extract
            self.status("解壓縮資料集 ...")
            name_prefix = f"ds_{dataset_zip.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            extracted_root = extract_zip(dataset_zip, work_dir, name_prefix)
            self.log(f"[{now_str()}] 解壓縮完成：{extracted_root}\n")

            # 2) find yaml
            data_yaml = find_data_yaml(extracted_root)
            if not data_yaml:
                raise RuntimeError(f"找不到 data.yaml：{extracted_root}")
            self.log(f"[{now_str()}] 找到原始 data.yaml：{data_yaml}\n")

            # ✅ NEW: rewrite yaml so it points to extracted_root
            self.status("修正 data.yaml 路徑（指向解壓資料夾） ...")
            patched_yaml = rewrite_data_yaml_to_extracted_root(data_yaml, extracted_root, self.log)
            self.log(f"[{now_str()}] 使用 patched data.yaml：{patched_yaml}\n")

            # 後面訓練要用 patched_yaml
            data_yaml = patched_yaml

            # 3) optional filter unlabeled
            if cfg.skip_unlabeled and cfg.task in ("detect", "segment", "pose", "obb"):
                self.status("清理未標記圖片（skip unlabeled） ...")
                self.log(f"[{now_str()}] 開始移除沒有 label 的圖片...\n")
                stat = filter_unlabeled_yolo_dataset(data_yaml, self.log, apply_to=("train", "val"))
                self.log(f"[{now_str()}] 清理完成：checked={stat['checked']}, removed={stat['removed']}\n")

            # 4) load model
            self.status("載入模型 ...")
            model = YOLO(cfg.model)
            self.log(f"[{now_str()}] 模型：{cfg.model}\n")

            # 5) callbacks
            # NOTE: callback events list can be seen via get_default_callbacks() in docs. :contentReference[oaicite:6]{index=6}
            def stop_if_needed(trainer):
                # BaseTrainer has `stop` flag concept. We'll set it to stop. :contentReference[oaicite:7]{index=7}
                if self.stop_requested:
                    setattr(trainer, "stop", True)

            def on_train_start(trainer):
                self._epoch_cur = 0
                self.progress_epoch(0, self._epoch_total)

            def on_train_epoch_start(trainer):
                stop_if_needed(trainer)
                # try update batch_total for this epoch
                try:
                    self._batch_total = len(trainer.train_loader)
                except Exception:
                    self._batch_total = 0
                self._batch_cur = 0
                if self._batch_total > 0:
                    self.progress_batch(0, self._batch_total)

            def on_train_batch_end(trainer):
                stop_if_needed(trainer)
                if self._batch_total > 0:
                    self._batch_cur += 1
                    self.progress_batch(self._batch_cur, self._batch_total)

            def on_train_epoch_end(trainer):
                stop_if_needed(trainer)
                # trainer.epoch is 0-based
                ep = int(getattr(trainer, "epoch", 0)) + 1
                self._epoch_cur = ep
                self.progress_epoch(ep, self._epoch_total)

            def on_train_end(trainer):
                self.progress_epoch(self._epoch_total, self._epoch_total)

            # attach callbacks (best-effort)
            try:
                model.add_callback("on_train_start", on_train_start)
                model.add_callback("on_train_epoch_start", on_train_epoch_start)
                model.add_callback("on_train_batch_end", on_train_batch_end)
                model.add_callback("on_train_epoch_end", on_train_epoch_end)
                model.add_callback("on_train_end", on_train_end)
            except Exception:
                self.log(f"[{now_str()}] 警告：此 ultralytics 版本不支援 add_callback，進度/停止可能較不完整。\n")

            # 6) train kwargs
            self.status("訓練中 ...")
            run_name = f"{cfg.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            train_kwargs = dict(
                data=str(data_yaml) if cfg.task != "classify" else str(extracted_root),  # classify often uses folder structure
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
                train_kwargs["resume"] = True  # supports resume in cfg. :contentReference[oaicite:8]{index=8}

            # NOTE: For classify/segment/pose/obb, Ultralytics uses task inferred by model or args.
            # We can pass `task=` in some versions; safest is CLI style not required here.
            # We'll rely on model + data format; still keep cfg.task for naming/UI.

            results = model.train(**train_kwargs)

            # 7) resolve run_dir
            run_dir = None
            try:
                run_dir = Path(results.save_dir)
            except Exception:
                try:
                    run_dir = Path(model.trainer.save_dir)
                except Exception:
                    run_dir = None

            if not run_dir or not run_dir.exists():
                # fallback search latest results.csv
                candidates = sorted((work_dir / "runs").rglob("results.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
                if candidates:
                    run_dir = candidates[0].parent

            if not run_dir or not run_dir.exists():
                raise RuntimeError("訓練完成但找不到 run_dir（無法整理輸出）")

            self.log(f"[{now_str()}] run_dir：{run_dir}\n")

            # 8) metrics
            metrics = parse_results_csv(run_dir)

            # 9) zip output
            self.status("打包輸出 ...")
            out_zip_path = out_zip_dir / f"{run_dir.name}.zip"
            zip_folder(run_dir, out_zip_path, self.log)

            # 10) optional delete temp dataset extraction
            if cfg.delete_temp:
                self.status("清理暫存 ...")
                remove_dir_safe(extracted_root, self.log)

            elapsed = time.time() - t0
            payload = {
                "time": now_str(),
                "elapsed_sec": elapsed,
                "task": cfg.task,
                "dataset_zip": str(dataset_zip),
                "work_dir": str(work_dir),
                "extracted_root": str(extracted_root),
                "data_yaml": str(data_yaml),
                "model": cfg.model,
                "epochs": cfg.epochs,
                "imgsz": cfg.imgsz,
                "batch": cfg.batch,
                "device": cfg.device,
                "resume": cfg.resume,
                "skip_unlabeled": cfg.skip_unlabeled,
                "delete_temp": cfg.delete_temp,
                "run_dir": str(run_dir),
                "out_zip": str(out_zip_path),
                "metrics": metrics,
                "stopped": bool(self.stop_requested),
            }
            self.done(True, payload)

        except Exception as e:
            self.done(False, {"error": str(e)})
