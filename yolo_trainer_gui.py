import os
import re
import json
import time
import queue
import zipfile
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# -----------------------------
# Utilities
# -----------------------------
APP_DIR = Path(__file__).resolve().parent
HISTORY_PATH = APP_DIR / "train_history.jsonl"


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def default_dialog_dir(last_path: Optional[str]) -> str:
    """
    Rule:
    - if no selection => open APP_DIR
    - if has selection => open selection path (folder if folder, parent if file)
    """
    if not last_path:
        return str(APP_DIR)
    lp = Path(last_path)
    if lp.exists():
        return str(lp if lp.is_dir() else lp.parent)
    return str(APP_DIR)


def read_last_history(n: int = 200):
    if not HISTORY_PATH.exists():
        return []
    items = []
    with HISTORY_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items[-n:]


def append_history(record: Dict[str, Any]) -> None:
    with HISTORY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_zip(zip_path: Path, out_dir: Path, log_cb) -> Path:
    """
    Extract dataset.zip to out_dir/<zip_stem>_<timestamp>/
    Return extracted root folder path.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = out_dir / f"{zip_path.stem}_{ts}"
    safe_mkdir(root)
    log_cb(f"[{now_str()}] 解壓縮: {zip_path} -> {root}\n")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)

    # If zip contains a single top-level folder, normalize to that folder
    entries = [p for p in root.iterdir()]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return root


def find_data_yaml(extracted_root: Path) -> Optional[Path]:
    # common: data.yaml at root or in dataset folder
    for p in [extracted_root / "data.yaml", extracted_root / "data.yml"]:
        if p.exists():
            return p
    # search
    for p in extracted_root.rglob("data.yaml"):
        return p
    for p in extracted_root.rglob("data.yml"):
        return p
    return None


def parse_results_csv(run_dir: Path) -> Optional[Dict[str, float]]:
    """
    Ultralytics usually produces: run_dir/results.csv
    Columns often include:
      metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B)
    We'll take the last row.
    """
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return None

    try:
        import csv
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return None
        last = rows[-1]

        def get_float(key_candidates):
            for k in key_candidates:
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

        if p is None and r is None and map50 is None and map5095 is None:
            return None

        f1 = None
        if p is not None and r is not None and (p + r) > 0:
            f1 = 2 * p * r / (p + r)

        return {
            "precision": p,
            "recall": r,
            "f1": f1,
            "mAP50": map50,
            "mAP50-95": map5095,
        }
    except Exception:
        return None


# -----------------------------
# Training Worker
# -----------------------------
@dataclass
class TrainConfig:
    dataset_zip: str
    work_dir: str
    model_source: str  # model preset name or custom .pt
    epochs: int
    imgsz: int
    batch: int
    device: str  # "", "cpu", "0", "0,1"


class TrainerWorker(threading.Thread):
    def __init__(self, cfg: TrainConfig, msg_q: queue.Queue):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.q = msg_q
        self._stop_flag = False

    def log(self, s: str):
        self.q.put(("log", s))

    def progress(self, cur: int, total: int):
        self.q.put(("progress", cur, total))

    def status(self, s: str):
        self.q.put(("status", s))

    def done(self, ok: bool, payload: Dict[str, Any]):
        self.q.put(("done", ok, payload))

    def run(self):
        start_ts = time.time()
        cfg = self.cfg

        try:
            self.status("準備中：匯入 ultralytics ...")
            try:
                from ultralytics import YOLO
            except Exception as e:
                raise RuntimeError(
                    "找不到 ultralytics。請先 pip install ultralytics\n"
                    f"詳細錯誤：{e}"
                )

            dataset_zip = Path(cfg.dataset_zip).expanduser().resolve()
            work_dir = Path(cfg.work_dir).expanduser().resolve()
            safe_mkdir(work_dir)

            # 1) Extract dataset
            self.status("解壓縮資料集 ...")
            extracted_root = extract_zip(dataset_zip, work_dir, self.log)

            # 2) Find data.yaml
            data_yaml = find_data_yaml(extracted_root)
            if not data_yaml:
                raise RuntimeError(f"在解壓後的資料夾找不到 data.yaml：{extracted_root}")

            self.log(f"[{now_str()}] 找到 data.yaml: {data_yaml}\n")

            # 3) Build model
            self.status("載入模型 ...")
            model_src = cfg.model_source.strip()
            if not model_src:
                raise RuntimeError("模型來源是空的，請選擇模型（預設或自訂 .pt）。")

            self.log(f"[{now_str()}] 使用模型：{model_src}\n")
            model = YOLO(model_src)

            # 4) Training callbacks
            total_epochs = int(cfg.epochs)

            def on_train_epoch_end(trainer):
                # trainer.epoch is 0-based
                ep = int(getattr(trainer, "epoch", 0)) + 1
                self.progress(ep, total_epochs)

            def on_train_start(trainer):
                self.progress(0, total_epochs)

            def on_train_end(trainer):
                # ensure progress full
                self.progress(total_epochs, total_epochs)

            # Attach callbacks (ultralytics supports add_callback)
            # Works for modern ultralytics; if not available, it will just not show fine-grained updates.
            try:
                model.add_callback("on_train_start", on_train_start)
                model.add_callback("on_train_epoch_end", on_train_epoch_end)
                model.add_callback("on_train_end", on_train_end)
            except Exception:
                # fallback: we will update only start/end
                self.log(f"[{now_str()}] 警告：目前 ultralytics 版本不支援 add_callback，進度條可能只在開始/結束更新。\n")

            # 5) Train
            self.status("訓練中 ...")
            self.log(f"[{now_str()}] 開始訓練：epochs={cfg.epochs}, imgsz={cfg.imgsz}, batch={cfg.batch}, device={cfg.device}\n")

            train_kwargs = dict(
                data=str(data_yaml),
                epochs=int(cfg.epochs),
                imgsz=int(cfg.imgsz),
                batch=int(cfg.batch),
                project=str(work_dir / "runs"),
                name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                exist_ok=False,
                verbose=True,
            )
            if cfg.device.strip() != "":
                train_kwargs["device"] = cfg.device.strip()

            results = model.train(**train_kwargs)

            # 6) Locate run_dir
            run_dir = None
            # Ultralytics results usually has "save_dir"
            try:
                run_dir = Path(results.save_dir)
            except Exception:
                # try model trainer
                try:
                    run_dir = Path(model.trainer.save_dir)
                except Exception:
                    run_dir = None

            if run_dir is None or not run_dir.exists():
                # fallback: find newest in work_dir/runs/detect or runs/segment etc
                runs_dir = work_dir / "runs"
                if runs_dir.exists():
                    candidates = sorted(runs_dir.rglob("results.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
                    if candidates:
                        run_dir = candidates[0].parent

            if run_dir is None or not run_dir.exists():
                raise RuntimeError("訓練完成但找不到 run_dir（無法讀取 results.csv）。")

            self.log(f"[{now_str()}] 訓練輸出資料夾：{run_dir}\n")

            # 7) Parse metrics (last epoch)
            metrics = parse_results_csv(run_dir)
            if not metrics:
                metrics = {}

            elapsed = time.time() - start_ts

            payload = {
                "time": now_str(),
                "elapsed_sec": elapsed,
                "dataset_zip": str(dataset_zip),
                "extracted_root": str(extracted_root),
                "data_yaml": str(data_yaml),
                "work_dir": str(work_dir),
                "run_dir": str(run_dir),
                "model_source": model_src,
                "epochs": cfg.epochs,
                "imgsz": cfg.imgsz,
                "batch": cfg.batch,
                "device": cfg.device,
                "metrics": metrics,
            }

            append_history(payload)
            self.done(True, payload)

        except Exception as e:
            self.done(False, {"error": str(e)})


# -----------------------------
# GUI
# -----------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("YOLO Dataset.zip 訓練工具")
        self.geometry("980x700")

        self.msg_q = queue.Queue()

        # keep last paths for dialog rules
        self.last_dataset_zip = ""
        self.last_work_dir = ""
        self.last_custom_model = ""

        self._build_ui()
        self._load_history()
        self._poll_queue()

    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        self.tab_train = ttk.Frame(nb)
        self.tab_hist = ttk.Frame(nb)
        nb.add(self.tab_train, text="訓練")
        nb.add(self.tab_hist, text="歷史紀錄")

        # ---------------- Train Tab ----------------
        frm = ttk.Frame(self.tab_train, padding=12)
        frm.pack(fill="both", expand=True)

        # Dataset zip
        row = 0
        ttk.Label(frm, text="資料集 dataset.zip：").grid(row=row, column=0, sticky="w", pady=6)
        self.var_dataset_zip = tk.StringVar()
        ent = ttk.Entry(frm, textvariable=self.var_dataset_zip)
        ent.grid(row=row, column=1, sticky="we", padx=8)
        ttk.Button(frm, text="選擇...", command=self.pick_dataset_zip).grid(row=row, column=2, sticky="e")

        # Work dir
        row += 1
        ttk.Label(frm, text="訓練工作資料夾（解壓/輸出）：").grid(row=row, column=0, sticky="w", pady=6)
        self.var_work_dir = tk.StringVar()
        ent2 = ttk.Entry(frm, textvariable=self.var_work_dir)
        ent2.grid(row=row, column=1, sticky="we", padx=8)
        ttk.Button(frm, text="選擇...", command=self.pick_work_dir).grid(row=row, column=2, sticky="e")

        # Model family/version (UI only; weights decide actual version)
        row += 1
        ttk.Label(frm, text="YOLO 版本/家族：").grid(row=row, column=0, sticky="w", pady=6)
        self.var_family = tk.StringVar(value="Ultralytics (YOLOv8/v9/v10/v11...)")
        fam = ttk.Combobox(frm, textvariable=self.var_family, state="readonly",
                           values=["Ultralytics (YOLOv8/v9/v10/v11...)"])
        fam.grid(row=row, column=1, sticky="we", padx=8)

        # Model preset
        row += 1
        ttk.Label(frm, text="模型（預設權重名稱）：").grid(row=row, column=0, sticky="w", pady=6)
        self.var_model_preset = tk.StringVar(value="yolov8n.pt")
        presets = [
            # v8
            "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
            # v9 (if available in your ultralytics build)
            "yolov9c.pt", "yolov9e.pt",
            # v10 (names may vary by release)
            "yolov10n.pt", "yolov10s.pt", "yolov10m.pt", "yolov10l.pt", "yolov10x.pt",
            # v11 (names may vary by release)
            "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
        ]
        cmb = ttk.Combobox(frm, textvariable=self.var_model_preset, values=presets)
        cmb.grid(row=row, column=1, sticky="we", padx=8)

        # Custom model
        row += 1
        ttk.Label(frm, text="或自訂模型權重（.pt）：").grid(row=row, column=0, sticky="w", pady=6)
        self.var_custom_model = tk.StringVar()
        ent3 = ttk.Entry(frm, textvariable=self.var_custom_model)
        ent3.grid(row=row, column=1, sticky="we", padx=8)
        ttk.Button(frm, text="選擇...", command=self.pick_custom_model).grid(row=row, column=2, sticky="e")

        # Train params
        row += 1
        param_box = ttk.LabelFrame(frm, text="訓練參數", padding=10)
        param_box.grid(row=row, column=0, columnspan=3, sticky="we", pady=10)

        ttk.Label(param_box, text="epochs").grid(row=0, column=0, sticky="w")
        self.var_epochs = tk.IntVar(value=50)
        ttk.Entry(param_box, textvariable=self.var_epochs, width=10).grid(row=0, column=1, sticky="w", padx=8)

        ttk.Label(param_box, text="imgsz").grid(row=0, column=2, sticky="w")
        self.var_imgsz = tk.IntVar(value=640)
        ttk.Entry(param_box, textvariable=self.var_imgsz, width=10).grid(row=0, column=3, sticky="w", padx=8)

        ttk.Label(param_box, text="batch").grid(row=0, column=4, sticky="w")
        self.var_batch = tk.IntVar(value=16)
        ttk.Entry(param_box, textvariable=self.var_batch, width=10).grid(row=0, column=5, sticky="w", padx=8)

        ttk.Label(param_box, text="device (空白=自動, cpu, 0, 0,1)").grid(row=1, column=0, sticky="w", pady=8)
        self.var_device = tk.StringVar(value="")
        ttk.Entry(param_box, textvariable=self.var_device, width=18).grid(row=1, column=1, sticky="w", padx=8)

        for c in range(6):
            param_box.grid_columnconfigure(c, weight=1)

        # Progress and controls
        row += 1
        ctrl = ttk.Frame(frm)
        ctrl.grid(row=row, column=0, columnspan=3, sticky="we", pady=8)
        self.btn_start = ttk.Button(ctrl, text="開始訓練", command=self.start_train)
        self.btn_start.pack(side="left")
        self.var_status = tk.StringVar(value="就緒")
        ttk.Label(ctrl, textvariable=self.var_status).pack(side="left", padx=12)

        row += 1
        prog_box = ttk.LabelFrame(frm, text="訓練進度", padding=10)
        prog_box.grid(row=row, column=0, columnspan=3, sticky="we", pady=8)
        self.var_prog_text = tk.StringVar(value="0 / 0 epochs")
        ttk.Label(prog_box, textvariable=self.var_prog_text).pack(anchor="w")
        self.progress = ttk.Progressbar(prog_box, orient="horizontal", mode="determinate")
        self.progress.pack(fill="x", expand=True, pady=6)

        # Metrics
        row += 1
        met_box = ttk.LabelFrame(frm, text="最後一輪指標", padding=10)
        met_box.grid(row=row, column=0, columnspan=3, sticky="we", pady=8)

        self.var_metrics = tk.StringVar(value="(尚未訓練)")
        ttk.Label(met_box, textvariable=self.var_metrics, justify="left").pack(anchor="w")

        # Log area
        row += 1
        log_box = ttk.LabelFrame(frm, text="Log", padding=10)
        log_box.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=8)
        self.txt_log = tk.Text(log_box, height=14, wrap="word")
        self.txt_log.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(log_box, command=self.txt_log.yview)
        sb.pack(side="right", fill="y")
        self.txt_log.configure(yscrollcommand=sb.set)

        frm.grid_columnconfigure(1, weight=1)
        frm.grid_rowconfigure(row, weight=1)

        # ---------------- History Tab ----------------
        hf = ttk.Frame(self.tab_hist, padding=12)
        hf.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(hf, columns=("time", "model", "dataset", "map50", "recall", "run"), show="headings")
        self.tree.heading("time", text="時間")
        self.tree.heading("model", text="模型")
        self.tree.heading("dataset", text="dataset.zip")
        self.tree.heading("map50", text="mAP50")
        self.tree.heading("recall", text="Recall")
        self.tree.heading("run", text="run_dir")

        self.tree.column("time", width=150, anchor="w")
        self.tree.column("model", width=120, anchor="w")
        self.tree.column("dataset", width=240, anchor="w")
        self.tree.column("map50", width=80, anchor="e")
        self.tree.column("recall", width=80, anchor="e")
        self.tree.column("run", width=260, anchor="w")

        self.tree.pack(fill="both", expand=True)

        btnrow = ttk.Frame(hf)
        btnrow.pack(fill="x", pady=10)
        ttk.Button(btnrow, text="重新載入", command=self._load_history).pack(side="left")
        ttk.Button(btnrow, text="開啟 run_dir", command=self.open_selected_run_dir).pack(side="left", padx=8)
        ttk.Button(btnrow, text="查看詳細", command=self.show_selected_detail).pack(side="left", padx=8)

    # ---------------- Dialog Pickers ----------------
    def pick_dataset_zip(self):
        initial = default_dialog_dir(self.last_dataset_zip)
        p = filedialog.askopenfilename(
            title="選擇 dataset.zip",
            initialdir=initial,
            filetypes=[("Zip files", "*.zip"), ("All files", "*.*")]
        )
        if p:
            self.var_dataset_zip.set(p)
            self.last_dataset_zip = p

    def pick_work_dir(self):
        initial = default_dialog_dir(self.last_work_dir)
        p = filedialog.askdirectory(
            title="選擇訓練工作資料夾",
            initialdir=initial
        )
        if p:
            self.var_work_dir.set(p)
            self.last_work_dir = p

    def pick_custom_model(self):
        initial = default_dialog_dir(self.last_custom_model)
        p = filedialog.askopenfilename(
            title="選擇模型權重 .pt",
            initialdir=initial,
            filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")]
        )
        if p:
            self.var_custom_model.set(p)
            self.last_custom_model = p

    # ---------------- Training ----------------
    def start_train(self):
        dataset_zip = self.var_dataset_zip.get().strip()
        work_dir = self.var_work_dir.get().strip()

        if not dataset_zip:
            messagebox.showerror("缺少資料", "請先選擇 dataset.zip")
            return
        if not work_dir:
            # if user didn't select, default to APP_DIR
            work_dir = str(APP_DIR / "workdir")
            self.var_work_dir.set(work_dir)

        custom_model = self.var_custom_model.get().strip()
        preset = self.var_model_preset.get().strip()
        model_src = custom_model if custom_model else preset

        try:
            epochs = int(self.var_epochs.get())
            imgsz = int(self.var_imgsz.get())
            batch = int(self.var_batch.get())
        except Exception:
            messagebox.showerror("參數錯誤", "epochs/imgsz/batch 必須是整數")
            return

        if epochs <= 0:
            messagebox.showerror("參數錯誤", "epochs 必須 > 0")
            return

        cfg = TrainConfig(
            dataset_zip=dataset_zip,
            work_dir=work_dir,
            model_source=model_src,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=self.var_device.get().strip(),
        )

        self._reset_ui_for_train()
        self._log(f"[{now_str()}] ===== 開始訓練 =====\n")
        self._log(f"[{now_str()}] dataset.zip: {cfg.dataset_zip}\n")
        self._log(f"[{now_str()}] work_dir: {cfg.work_dir}\n")
        self._log(f"[{now_str()}] model: {cfg.model_source}\n\n")

        self.btn_start.config(state="disabled")
        self.var_status.set("啟動訓練執行緒 ...")

        self.worker = TrainerWorker(cfg, self.msg_q)
        self.worker.start()

    def _reset_ui_for_train(self):
        self.progress["value"] = 0
        self.progress["maximum"] = 1
        self.var_prog_text.set("0 / 0 epochs")
        self.var_metrics.set("(訓練中...)")
        self.txt_log.delete("1.0", "end")

    # ---------------- Queue / UI updates ----------------
    def _poll_queue(self):
        try:
            while True:
                msg = self.msg_q.get_nowait()
                kind = msg[0]

                if kind == "log":
                    self._log(msg[1])
                elif kind == "progress":
                    cur, total = msg[1], msg[2]
                    self._set_progress(cur, total)
                elif kind == "status":
                    self.var_status.set(msg[1])
                elif kind == "done":
                    ok, payload = msg[1], msg[2]
                    self._on_done(ok, payload)
                else:
                    pass
        except queue.Empty:
            pass

        self.after(120, self._poll_queue)

    def _log(self, s: str):
        self.txt_log.insert("end", s)
        self.txt_log.see("end")

    def _set_progress(self, cur: int, total: int):
        total = max(1, int(total))
        cur = max(0, min(int(cur), total))
        self.progress["maximum"] = total
        self.progress["value"] = cur
        self.var_prog_text.set(f"{cur} / {total} epochs")

    def _on_done(self, ok: bool, payload: Dict[str, Any]):
        self.btn_start.config(state="normal")
        if ok:
            self.var_status.set("完成")
            self._log(f"\n[{now_str()}] ===== 訓練完成 =====\n")
            self._log(f"[{now_str()}] run_dir: {payload.get('run_dir')}\n")
            self._log(f"[{now_str()}] elapsed: {payload.get('elapsed_sec'):.1f} sec\n")

            m = payload.get("metrics", {}) or {}
            p = m.get("precision")
            r = m.get("recall")
            f1 = m.get("f1")
            map50 = m.get("mAP50")
            map5095 = m.get("mAP50-95")

            def fmt(x):
                return "-" if x is None else f"{x:.4f}"

            self.var_metrics.set(
                "Precision: " + fmt(p) + "\n"
                                         "Recall:    " + fmt(r) + "\n"
                                                                  "F1:        " + fmt(f1) + "\n"
                                                                                            "mAP50:     " + fmt(map50) + "\n"
                                                                                                                         "mAP50-95:  " + fmt(map5095)
            )

            # refresh history tab
            self._load_history()
        else:
            self.var_status.set("失敗")
            err = payload.get("error", "Unknown error")
            self._log(f"\n[{now_str()}] ===== 訓練失敗 =====\n{err}\n")
            messagebox.showerror("訓練失敗", err)

    # ---------------- History Tab ----------------
    def _load_history(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        items = read_last_history(500)
        for rec in reversed(items):
            t = rec.get("time", "")
            model = rec.get("model_source", "")
            dataset = rec.get("dataset_zip", "")
            run_dir = rec.get("run_dir", "")

            met = rec.get("metrics", {}) or {}
            map50 = met.get("mAP50", None)
            recall = met.get("recall", None)

            def fmt(x):
                return "" if x is None else f"{x:.4f}"

            self.tree.insert("", "end", values=(t, model, dataset, fmt(map50), fmt(recall), run_dir))

    def _selected_record(self) -> Optional[Dict[str, Any]]:
        sel = self.tree.selection()
        if not sel:
            return None
        values = self.tree.item(sel[0], "values")
        # match by time+run_dir (good enough)
        time_val = values[0]
        run_dir_val = values[5]
        items = read_last_history(2000)
        for rec in reversed(items):
            if rec.get("time") == time_val and rec.get("run_dir") == run_dir_val:
                return rec
        return None

    def open_selected_run_dir(self):
        rec = self._selected_record()
        if not rec:
            messagebox.showinfo("提示", "請先在歷史紀錄中選擇一筆")
            return
        run_dir = rec.get("run_dir")
        if not run_dir or not Path(run_dir).exists():
            messagebox.showerror("錯誤", "run_dir 不存在")
            return

        # open folder (Windows/macOS/Linux)
        p = Path(run_dir)
        try:
            if os.name == "nt":
                os.startfile(str(p))
            elif sys.platform == "darwin":
                os.system(f'open "{p}"')
            else:
                os.system(f'xdg-open "{p}"')
        except Exception as e:
            messagebox.showerror("錯誤", f"無法開啟資料夾：{e}")

    def show_selected_detail(self):
        rec = self._selected_record()
        if not rec:
            messagebox.showinfo("提示", "請先在歷史紀錄中選擇一筆")
            return

        top = tk.Toplevel(self)
        top.title("訓練詳細資訊")
        top.geometry("800x520")

        txt = tk.Text(top, wrap="word")
        txt.pack(fill="both", expand=True)

        txt.insert("end", json.dumps(rec, ensure_ascii=False, indent=2))
        txt.configure(state="disabled")


if __name__ == "__main__":
    import sys
    app = App()
    app.mainloop()
