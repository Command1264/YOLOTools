# app.py
from __future__ import annotations

import os
import sys
import json
import time
import queue
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from model_registry import get_weights, filter_weights_by_task
from trainer_worker import TrainConfig, TrainerWorker
from hardcore_view import HardcorePanel

APP_DIR = Path(__file__).resolve().parent
CACHE_WEIGHTS = APP_DIR / "weights_cache.json"
HISTORY_PATH = APP_DIR / "train_history.jsonl"
CONFIG_PATH = APP_DIR / "trainer_config.json"


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def default_dialog_dir(last_path: str) -> str:
    """
    Rule:
    - if no selection => open APP_DIR
    - if has selection => open selected path (folder if folder, parent if file)
    """
    if not last_path:
        return str(APP_DIR)
    p = Path(last_path)
    if p.exists():
        return str(p if p.is_dir() else p.parent)
    return str(APP_DIR)


def append_history(record: Dict[str, Any]) -> None:
    with HISTORY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_history(limit: int = 500):
    if not HISTORY_PATH.exists():
        return []
    out = []
    with HISTORY_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out[-limit:]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Ultralytics YOLO GUI Trainer")
        self.geometry("1200x800")
        self.minsize(1200, 800)
        # self.state('zoomed')   # Windows only

        self.msg_q = queue.Queue()
        self.worker: Optional[TrainerWorker] = None
        self._pending_msgs = []
        self._flush_after_id = None
        self._is_resizing = False
        self._resize_after_id = None
        self._eta_epoch_sec = None
        self._eta_batch_sec = None
        self._eta_epoch_last_update = 0.0
        self._eta_batch_last_update = 0.0
        self._eta_tick_after_id = None
        self._eta_decimal_places = 1
        self._last_epoch_progress = (0, 1)
        self._last_batch_progress = (0, 1)
        self._loading_config = False
        self._force_stop_locked = False
        self.var_epoch_summary = tk.StringVar(value="(尚未訓練)")
        self._metrics_log = ""

        # remember last paths for dialog rule
        self.last_dataset_zip = ""
        self.last_work_dir = ""
        self.last_out_zip_dir = ""
        self.last_custom_model = ""
        self.last_model_dir = ""

        self._build_ui()
        self.bind("<Configure>", self._on_resize)
        self._bind_config_traces()
        self._bind_eta_decimal_trace()

        self._load_config()

        # load model list (try online first)
        self._load_weights(try_online=True)

        # apply task filter once after weights loaded
        self._apply_task_filter()

        # trace task changes -> re-filter
        self.var_task.trace_add("write", lambda *args: self._apply_task_filter())

        self._load_history()
        self._poll_queue()
        self._start_eta_tick()

    def _bind_eta_decimal_trace(self):
        try:
            self.var_eta_decimal_places.trace_add("write", lambda *args: self._sync_eta_decimal_places())
        except Exception:
            pass

    def _sync_eta_decimal_places(self):
        try:
            dp = int(self.var_eta_decimal_places.get())
        except Exception:
            return
        self._eta_decimal_places = dp
        ep_cur, ep_total = self._last_epoch_progress
        ba_cur, ba_total = self._last_batch_progress
        if self._eta_epoch_sec is None:
            eta_ep = " 剩餘時間：--:--:--"
        else:
            eta_ep = f" 剩餘時間：{self._format_eta_seconds(self._eta_epoch_sec, self._eta_decimal_places)}"
        if self._eta_batch_sec is None:
            eta_ba = " 剩餘時間：--:--:--"
        else:
            eta_ba = f" 剩餘時間：{self._format_eta_seconds(self._eta_batch_sec, self._eta_decimal_places)}"
        self.var_ep_text.set(f"Epoch: {ep_cur}/{ep_total}{eta_ep}")
        self.var_ba_text.set(f"Batch: {ba_cur}/{ba_total}{eta_ba}")

    # ---------------- UI ----------------
    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        self.tab_train = ttk.Frame(nb)
        self.tab_hist = ttk.Frame(nb)
        self.tab_hardcore = ttk.Frame(nb)

        nb.add(self.tab_train, text="訓練")
        nb.add(self.tab_hist, text="歷史")
        nb.add(self.tab_hardcore, text="硬核視覺化")

        # -------- Train Tab --------
        frm = ttk.Frame(self.tab_train, padding=12)
        frm.pack(fill="both", expand=True)

        r = 0
        ttk.Label(frm, text="任務（Task）：").grid(row=r, column=0, sticky="w", pady=6)
        self.var_task = tk.StringVar(value="detect")
        cmb_task = ttk.Combobox(
            frm,
            textvariable=self.var_task,
            state="readonly",
            values=["detect", "segment", "classify", "pose", "obb"]
        )
        cmb_task.grid(row=r, column=1, sticky="we", padx=8)
        ttk.Button(frm, text="更新官方模型清單（線上）", command=lambda: self._load_weights(True)).grid(
            row=r, column=2, sticky="e"
        )

        r += 1
        ttk.Label(frm, text="資料集（zip）：").grid(row=r, column=0, sticky="w", pady=6)
        self.var_dataset_zip = tk.StringVar()
        ttk.Entry(frm, textvariable=self.var_dataset_zip).grid(row=r, column=1, sticky="we", padx=8)
        ttk.Button(frm, text="選擇...", command=self.pick_dataset_zip).grid(row=r, column=2, sticky="e")

        r += 1
        ttk.Label(frm, text="訓練工作資料夾（解壓縮/訓練模型）：").grid(row=r, column=0, sticky="w", pady=6)
        self.var_work_dir = tk.StringVar(value=str(APP_DIR / "workdir"))
        ttk.Entry(frm, textvariable=self.var_work_dir).grid(row=r, column=1, sticky="we", padx=8)
        ttk.Button(frm, text="選擇...", command=self.pick_work_dir).grid(row=r, column=2, sticky="e")

        r += 1
        ttk.Label(frm, text="輸出 zip 存放資料夾：").grid(row=r, column=0, sticky="w", pady=6)
        self.var_out_zip_dir = tk.StringVar(value=str(APP_DIR / "output_zips"))
        ttk.Entry(frm, textvariable=self.var_out_zip_dir).grid(row=r, column=1, sticky="we", padx=8)
        ttk.Button(frm, text="選擇...", command=self.pick_out_zip_dir).grid(row=r, column=2, sticky="e")

        r += 1
        ttk.Label(frm, text="模型（依 Task 過濾後清單）：").grid(row=r, column=0, sticky="w", pady=6)
        self.var_model_pick = tk.StringVar()
        self.cmb_model = ttk.Combobox(frm, textvariable=self.var_model_pick)
        self.cmb_model.grid(row=r, column=1, sticky="we", padx=8)
        self.var_model_source = tk.StringVar(value="(尚未載入)")
        ttk.Label(frm, textvariable=self.var_model_source).grid(row=r, column=2, sticky="e")

        r += 1
        ttk.Label(frm, text="模型存放資料夾(下載/快取)").grid(row=r, column=0, sticky="w", pady=6)
        self.var_model_dir = tk.StringVar(value=str(APP_DIR / "models"))
        ttk.Entry(frm, textvariable=self.var_model_dir).grid(row=r, column=1, sticky="we", padx=8)
        ttk.Button(frm, text="選擇...", command=self.pick_model_dir).grid(row=r, column=2, sticky="e")

        r += 1
        ttk.Label(frm, text="或自訂權重（.pt）：").grid(row=r, column=0, sticky="w", pady=6)
        self.var_use_custom_model = tk.BooleanVar(value=False)
        self.var_custom_model = tk.StringVar()
        ttk.Entry(frm, textvariable=self.var_custom_model).grid(row=r, column=1, sticky="we", padx=8)
        ttk.Checkbutton(frm, text="使用自訂權重", variable=self.var_use_custom_model).grid(row=r, column=2, sticky="e")

        r += 1
        ttk.Button(frm, text="選擇...", command=self.pick_custom_model).grid(row=r, column=1, sticky="w", padx=8)

        # Params
        r += 1
        box = ttk.LabelFrame(frm, text="訓練參數", padding=10)
        box.grid(row=r, column=0, columnspan=3, sticky="we", pady=10)

        self.var_epochs = tk.IntVar(value=50)
        self.var_imgsz = tk.IntVar(value=640)
        self.var_batch = tk.IntVar(value=16)
        self.var_eta_decimal_places = tk.IntVar(value=self._eta_decimal_places)
        self.var_device = tk.StringVar(value="")
        self.var_resume = tk.BooleanVar(value=False)
        self.var_skip_unlabeled = tk.BooleanVar(value=True)
        self.var_delete_temp = tk.BooleanVar(value=True)

        ttk.Label(box, text="epochs").grid(row=0, column=0, sticky="w")
        ttk.Entry(box, textvariable=self.var_epochs, width=10).grid(row=0, column=1, sticky="w", padx=8)

        ttk.Label(box, text="imgsz").grid(row=0, column=2, sticky="w")
        ttk.Entry(box, textvariable=self.var_imgsz, width=10).grid(row=0, column=3, sticky="w", padx=8)

        ttk.Label(box, text="batch").grid(row=0, column=4, sticky="w")
        ttk.Entry(box, textvariable=self.var_batch, width=10).grid(row=0, column=5, sticky="w", padx=8)

        ttk.Label(box, text="device（空白自動 / cpu / 0 / 0,1）").grid(row=1, column=0, sticky="w", pady=8)
        ttk.Entry(box, textvariable=self.var_device, width=18).grid(row=1, column=1, sticky="w", padx=8)

        ttk.Label(box, text="ETA 小數位數（<=0 不顯示）").grid(row=1, column=2, sticky="w", pady=8)
        ttk.Entry(box, textvariable=self.var_eta_decimal_places, width=10).grid(row=1, column=3, sticky="w", padx=8)

        ttk.Checkbutton(box, text="resume（接著上次中斷續跑）", variable=self.var_resume).grid(row=2, column=0, sticky="w")
        ttk.Checkbutton(box, text="跳過無標記圖片（detect/seg/pose/obb）", variable=self.var_skip_unlabeled).grid(row=2, column=1, sticky="w")
        ttk.Checkbutton(box, text="訓練後刪除解壓暫存", variable=self.var_delete_temp).grid(row=2, column=2, sticky="w")

        # Controls
        r += 1
        ctrl = ttk.Frame(frm)
        ctrl.grid(row=r, column=0, columnspan=3, sticky="we", pady=8)
        self.btn_start = ttk.Button(ctrl, text="開始訓練", command=self.start_train)
        self.btn_start.pack(side="left")
        self.btn_stop = ttk.Button(ctrl, text="停止", command=self.stop_train, state="disabled")
        self.btn_stop.pack(side="left", padx=8)
        self.var_status = tk.StringVar(value="就緒")
        ttk.Label(ctrl, textvariable=self.var_status).pack(side="left", padx=12)
        self.var_run_device = tk.StringVar(value="裝置：未知")
        ttk.Label(ctrl, textvariable=self.var_run_device).pack(side="left", padx=12)
        self.btn_stop.bind("<Double-Button-1>", lambda _e: self.force_stop_train())

        # Progress (epoch + batch)
        r += 1
        pb = ttk.LabelFrame(frm, text="進度", padding=10)
        pb.grid(row=r, column=0, columnspan=3, sticky="we", pady=8)

        self.var_ep_text = tk.StringVar(value="Epoch: 0/0")
        self.var_ba_text = tk.StringVar(value="Batch: 0/0")
        ttk.Label(pb, textvariable=self.var_ep_text).pack(anchor="w")
        self.pbar_epoch = ttk.Progressbar(pb, mode="determinate")
        self.pbar_epoch.pack(fill="x", expand=True, pady=6)

        ttk.Label(pb, textvariable=self.var_ba_text).pack(anchor="w")
        self.pbar_batch = ttk.Progressbar(pb, mode="determinate")
        self.pbar_batch.pack(fill="x", expand=True, pady=6)

        # Metrics
        r += 1
        paned_metrics_log = tk.PanedWindow(frm, orient="vertical")
        paned_metrics_log.grid(row=r, column=0, columnspan=3, sticky="nsew", pady=8)

        mb = ttk.LabelFrame(paned_metrics_log, text="上一輪指標 / 每輪摘要", padding=10)
        self.txt_metrics = tk.Text(mb, height=8, wrap="word")
        self.txt_metrics.pack(side="left", fill="both", expand=True)
        sbm = ttk.Scrollbar(mb, command=self.txt_metrics.yview)
        sbm.pack(side="right", fill="y")
        self.txt_metrics.configure(yscrollcommand=sbm.set)

        lb = ttk.LabelFrame(paned_metrics_log, text="Log", padding=10)
        self.txt_log = tk.Text(lb, height=3, wrap="word")
        self.txt_log.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(lb, command=self.txt_log.yview)
        sb.pack(side="right", fill="y")
        self.txt_log.configure(yscrollcommand=sb.set)

        paned_metrics_log.add(mb)
        paned_metrics_log.add(lb)
        paned_metrics_log.paneconfigure(lb, minsize=60)
        self._metrics_log_paned = paned_metrics_log

        frm.grid_columnconfigure(1, weight=1)
        frm.grid_rowconfigure(r, weight=1)

        # -------- History Tab --------
        hf = ttk.Frame(self.tab_hist, padding=12)
        hf.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(
            hf,
            columns=("time", "task", "model", "dataset", "map50", "recall", "zip", "run"),
            show="headings"
        )
        for c, t in [
            ("time", "時間"),
            ("task", "Task"),
            ("model", "模型"),
            ("dataset", "dataset.zip"),
            ("map50", "mAP50"),
            ("recall", "Recall"),
            ("zip", "輸出zip"),
            ("run", "run_dir"),
        ]:
            self.tree.heading(c, text=t)

        self.tree.column("time", width=150, anchor="w")
        self.tree.column("task", width=80, anchor="w")
        self.tree.column("model", width=160, anchor="w")
        self.tree.column("dataset", width=260, anchor="w")
        self.tree.column("map50", width=80, anchor="e")
        self.tree.column("recall", width=80, anchor="e")
        self.tree.column("zip", width=240, anchor="w")
        self.tree.column("run", width=260, anchor="w")

        self.tree.pack(fill="both", expand=True)

        br = ttk.Frame(hf)
        br.pack(fill="x", pady=10)
        ttk.Button(br, text="重新載入", command=self._load_history).pack(side="left")
        ttk.Button(br, text="查看詳細", command=self.show_selected_detail).pack(side="left", padx=8)
        ttk.Button(br, text="開啟輸出 zip", command=self.open_selected_zip).pack(side="left", padx=8)
        ttk.Button(br, text="開啟輸出 zip 資料夾", command=self.open_selected_zip_dir).pack(side="left", padx=8)
        ttk.Button(br, text="載入到硬核視覺化", command=self.load_selected_to_hardcore).pack(side="left", padx=8)
        ttk.Button(br, text="刪除選取", command=self.delete_selected_history).pack(side="left", padx=8)

        # -------- Hardcore Tab --------
        hp = ttk.Frame(self.tab_hardcore, padding=12)
        hp.pack(fill="both", expand=True)

        ttk.Label(
            hp,
            text="此頁會顯示 run_dir 的 results.csv（最後 50 epochs）與常見輸出圖（results/PR/confusion matrix）。"
        ).pack(anchor="w", pady=6)

        self.hardcore_panel = HardcorePanel(hp)
        self.hardcore_panel.pack(fill="both", expand=True)

    # ---------------- model list ----------------
    def _load_weights(self, try_online: bool):
        info = get_weights(CACHE_WEIGHTS, try_online=try_online)
        w = info["weights"]
        src = info["source"]

        # 暫存「原始清單」到 self，以免 combobox values 已被過濾後越濾越少
        self._base_weights = list(w)

        self.var_model_source.set(f"模型清單來源：{src}（已自動依 Task 過濾）")
        self._apply_task_filter()

    def _apply_task_filter(self):
        # 永遠從 base weights 過濾
        base = getattr(self, "_base_weights", None)
        if not base:
            info = get_weights(CACHE_WEIGHTS, try_online=False)
            base = info["weights"]
            self._base_weights = list(base)

        task = self.var_task.get().strip()
        filtered = filter_weights_by_task(list(base), task)

        self.cmb_model["values"] = filtered
        if filtered:
            cur = self.var_model_pick.get().strip()
            if cur not in filtered:
                self.var_model_pick.set(filtered[0])

    # ---------------- file dialogs ----------------
    def pick_dataset_zip(self):
        p = filedialog.askopenfilename(
            title="選擇 dataset.zip",
            initialdir=default_dialog_dir(self.last_dataset_zip),
            filetypes=[("Zip files", "*.zip"), ("All files", "*.*")]
        )
        if p:
            self.var_dataset_zip.set(p)
            self.last_dataset_zip = p

    def pick_work_dir(self):
        p = filedialog.askdirectory(
            title="選擇工作資料夾",
            initialdir=default_dialog_dir(self.last_work_dir)
        )
        if p:
            self.var_work_dir.set(p)
            self.last_work_dir = p

    def pick_out_zip_dir(self):
        p = filedialog.askdirectory(
            title="選擇輸出 zip 資料夾",
            initialdir=default_dialog_dir(self.last_out_zip_dir)
        )
        if p:
            self.var_out_zip_dir.set(p)
            self.last_out_zip_dir = p

    def pick_custom_model(self):
        p = filedialog.askopenfilename(
            title="選擇自訂模型 .pt",
            initialdir=default_dialog_dir(self.last_custom_model),
            filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")]
        )
        if p:
            self.var_custom_model.set(p)
            self.last_custom_model = p

    # ---------------- training ----------------

    def pick_model_dir(self):
        p = filedialog.askdirectory(
            title="選擇模型資料夾",
            initialdir=default_dialog_dir(self.last_model_dir)
        )
        if p:
            self.var_model_dir.set(p)
            self.last_model_dir = p

    def _log(self, s: str):
        self.txt_log.insert("end", s)
        self.txt_log.see("end")

    def start_train(self):
        self._save_config()
        dataset_zip = self.var_dataset_zip.get().strip()
        if not dataset_zip:
            messagebox.showerror("缺少資料", "請選擇訓練集")
            return

        work_dir = self.var_work_dir.get().strip() or str(APP_DIR / "workdir")
        out_zip_dir = self.var_out_zip_dir.get().strip() or str(APP_DIR / "output_zips")

        use_custom = bool(self.var_use_custom_model.get())
        custom = self.var_custom_model.get().strip()
        if use_custom and not custom:
            messagebox.showerror("缺少自訂權重", "請選擇自訂權重")
            return
        model = custom if use_custom else self.var_model_pick.get().strip()
        if not model:
            messagebox.showerror("缺少模型", "請選擇模型或指定自訂權重")
            return

        try:
            epochs = int(self.var_epochs.get())
            if epochs <= 0:
                messagebox.showerror("無效參數", "epochs 必須是大於 0")
                return
        except Exception:
            messagebox.showerror("無效參數", "epochs 必須是整數")
            return

        try:
            imgsz = int(self.var_imgsz.get())
            if imgsz <= 0:
                messagebox.showerror("無效參數", "imgsz 必須是大於 0")
                return
        except Exception:
            messagebox.showerror("無效參數", "imgsz 必須是整數")
            return

        try:
            batch = int(self.var_batch.get())
            if batch <= 0:
                messagebox.showerror("無效參數", "batch 必須是大於 0")
                return
        except Exception:
            messagebox.showerror("無效參數", "batch 必須是整數")
            return

        try:
            dp = int(self.var_eta_decimal_places.get())
        except Exception:
            messagebox.showerror("無效參數", "ETA 小數位數必須是整數")
            return
        self._eta_decimal_places = dp



        cfg = TrainConfig(
            task=self.var_task.get().strip(),
            dataset_zip=dataset_zip,
            work_dir=work_dir,
            out_zip_dir=out_zip_dir,
            model=model,
            model_dir=self.var_model_dir.get().strip(),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=self.var_device.get().strip(),
            resume=bool(self.var_resume.get()),
            skip_unlabeled=bool(self.var_skip_unlabeled.get()),
            delete_temp=bool(self.var_delete_temp.get()),
        )

        self.txt_log.delete("1.0", "end")
        self._log(f"[{now_str()}] ===== 開始訓練 =====\n")
        self._log(f"[{now_str()}] task={cfg.task}\n")
        self._log(f"[{now_str()}] dataset.zip={cfg.dataset_zip}\n")
        self._log(f"[{now_str()}] work_dir={cfg.work_dir}\n")
        self._log(f"[{now_str()}] out_zip_dir={cfg.out_zip_dir}\n")
        self._log(f"[{now_str()}] model={cfg.model}\n")
        self._log(f"[{now_str()}] model_dir={cfg.model_dir}\n")
        self._log(f"[{now_str()}] epochs={cfg.epochs}, imgsz={cfg.imgsz}, batch={cfg.batch}, device={cfg.device}, resume={cfg.resume}\n\n")

        self._set_epoch_progress(0, max(1, cfg.epochs))
        self._set_batch_progress(0, 1)
        self._metrics_log = ""
        self._update_metrics_text()
        self.var_epoch_summary.set("(訓練中...)")
        self.var_status.set("準備中 ...")
        self.var_run_device.set("裝置：偵測中")

        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")

        self.worker = TrainerWorker(cfg, self.msg_q)
        self.worker.start()

    def stop_train(self):
        if self.worker:
            self.worker.request_stop()
            self._log(f"\n[{now_str()}] 已送出停止請求（將在下一個 callback 觸發時停下）\n")
            self.var_status.set("停止中 ...")


    def force_stop_train(self):
        if self.worker:
            if self._force_stop_locked:
                messagebox.showinfo("強制停止", "目前正在下載/載入模型，強制停止暫不可用。")
                return
            if messagebox.askyesno("強制停止", "確定要強制停止嗎？"):
                self.worker.request_force_stop()
                self._log(f"\n[{now_str()}] 已送出強制停止，訓練將立即中止。\n")
                self.var_status.set("強制停止中 ...")
                messagebox.showinfo("強制停止", "強制停止成功！")

    def _set_epoch_progress(self, cur: int, total: int, eta_seconds: Optional[float] = None):
        total = max(1, int(total))
        cur = max(0, min(int(cur), total))
        self._last_epoch_progress = (cur, total)
        self.pbar_epoch["maximum"] = total
        self.pbar_epoch["value"] = cur
        if eta_seconds is not None:
            if not (eta_seconds <= 0 and cur >= total):
                self._eta_epoch_sec = max(0.0, float(eta_seconds))
                self._eta_epoch_last_update = time.time()
        if self._eta_epoch_sec is None:
            eta_txt = " 剩餘時間：--:--:--"
        else:
            eta_txt = f" 剩餘時間：{self._format_eta_seconds(self._eta_epoch_sec, self._eta_decimal_places)}"
            # print(f"更新 epoch eta：{self._eta_epoch_sec}")
        self.var_ep_text.set(f"Epoch: {cur}/{total}{eta_txt}")

    def _set_batch_progress(self, cur: int, total: int, eta_seconds: Optional[float] = None):
        total = max(1, int(total))
        cur = max(0, min(int(cur), total))
        self._last_batch_progress = (cur, total)
        self.pbar_batch["maximum"] = total
        self.pbar_batch["value"] = cur
        if eta_seconds is not None:
            if not (eta_seconds <= 0 and cur >= total):
                self._eta_batch_sec = max(0.0, float(eta_seconds))
                self._eta_batch_last_update = time.time()
        if self._eta_batch_sec is None:
            eta_txt = " 剩餘時間：--:--:--"
        else:
            eta_txt = f" 剩餘時間：{self._format_eta_seconds(self._eta_batch_sec, self._eta_decimal_places)}"
            # print(f"更新 batch eta：{self._eta_batch_sec}")
        self.var_ba_text.set(f"Batch: {cur}/{total}{eta_txt}")

    def _format_eta_seconds(self, seconds: Optional[float], decimal_places: int = 1) -> str:
        if seconds is None: return ""

        sec_f = max(0.0, float(seconds))
        sec_i = int(sec_f)

        if decimal_places <= 0:
            ms = 0
        else:
            unit = 10 ** decimal_places
            frac = sec_f - sec_i
            ms = int(round(frac * unit))

        if decimal_places > 0 and ms >= 10 ** decimal_places:
            sec_i += 1
            ms = 0
        if sec_i == 0 and ms == 0:
            if decimal_places <= 0: return "00:00:00"
            return f"00:00:00.{0:0{decimal_places}d}"

        mins, s = divmod(sec_i, 60)
        hrs, m = divmod(mins, 60)
        days, h = divmod(hrs, 24)
        months, d = divmod(days, 30)
        years, mo = divmod(months, 12)

        if decimal_places <= 0:
            if years or mo or d: return f"{years:04d}:{mo:02d}:{d:02d} {h:02d}:{m:02d}:{s:02d}"
            return f"{h:02d}:{m:02d}:{s:02d}"

        frac = f"{ms:0{decimal_places}d}"
        if years or mo or d: return f"{years:04d}:{mo:02d}:{d:02d} {h:02d}:{m:02d}:{s:02d}.{frac}"
        return f"{h:02d}:{m:02d}:{s:02d}.{frac}"



    def _start_eta_tick(self):
        if self._eta_tick_after_id is not None:
            return
        self._eta_tick_after_id = self.after(1000, self._tick_eta)

    def _tick_eta(self):
        now = time.time()
        if self._eta_epoch_sec is not None and self._eta_epoch_sec > 0:
            delta = now - self._eta_epoch_last_update
            if delta > 0:
                self._eta_epoch_sec = max(0.0, self._eta_epoch_sec - delta)
                self._eta_epoch_last_update = now
                cur, total = self._last_epoch_progress
                eta_txt = f" 剩餘時間：{self._format_eta_seconds(self._eta_epoch_sec, self._eta_decimal_places)}"
                self.var_ep_text.set(f"Epoch: {cur}/{total}{eta_txt}")

        if self._eta_batch_sec is not None and self._eta_batch_sec > 0:
            delta = now - self._eta_batch_last_update
            if delta > 0:
                self._eta_batch_sec = max(0.0, self._eta_batch_sec - delta)
                self._eta_batch_last_update = now
                cur, total = self._last_batch_progress
                eta_txt = f" 剩餘時間：{self._format_eta_seconds(self._eta_batch_sec, self._eta_decimal_places)}"
                self.var_ba_text.set(f"Batch: {cur}/{total}{eta_txt}")

        self._eta_tick_after_id = self.after(1000, self._tick_eta)

    def _format_prev_row(self, row: Optional[Dict[str, Any]]) -> str:
        if not row:
            return "(無)"
        def val(keys):
            for k in keys:
                if k in row and row[k] not in ("", None):
                    return row[k]
            return None

        fields = [
            ("GPU_mem", ["GPU_mem", "gpu_mem"]),
            ("box_loss", ["train/box_loss", "box_loss"]),
            ("cls_loss", ["train/cls_loss", "cls_loss"]),
            ("dfl_loss", ["train/dfl_loss", "dfl_loss"]),
            ("Instances", ["Instances"]),
            ("Size", ["Size"]),
            ("Class", ["Class"]),
            ("Images", ["Images"]),
            ("Box(P)", ["metrics/precision(B)", "metrics/precision", "precision"]),
            ("R", ["metrics/recall(B)", "metrics/recall", "recall"]),
            ("mAP50", ["metrics/mAP50(B)", "metrics/mAP50", "mAP50"]),
            ("mAP50-95", ["metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50-95", "mAP5095"]),
        ]
        lines = []
        for label, keys in fields:
            v = val(keys)
            if v is None:
                continue
            lines.append(f"{label}: {v}")
        return "\n".join(lines) if lines else "(無)"

    def _format_epoch_summary(self, row: Optional[Dict[str, Any]]) -> str:
        if not row:
            return "(無)"

        def val(keys):
            for k in keys:
                if k in row and row[k] not in ("", None):
                    return row[k]
            return None

        def fmt(x):
            try:
                return f"{float(x):.4f}"
            except Exception:
                return "-" if x is None else str(x)

        p = val(["metrics/precision(B)", "metrics/precision", "precision"])
        r = val(["metrics/recall(B)", "metrics/recall", "recall"])
        map50 = val(["metrics/mAP50(B)", "metrics/mAP50", "mAP50"])
        map95 = val(["metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50-95", "mAP5095"])

        f1 = None
        try:
            pf = float(p)
            rf = float(r)
            if pf + rf > 0:
                f1 = 2 * pf * rf / (pf + rf)
        except Exception:
            f1 = None

        lines = []
        if p is not None: lines.append("精確率 (Precision):   " + fmt(p))
        if r is not None: lines.append("召回率 (Recall):      " + fmt(r))
        if f1 is not None: lines.append("F1 分數 (F1-score):   " + fmt(f1))
        if map50 is not None: lines.append("mAP50:               " + fmt(map50))
        if map95 is not None: lines.append("mAP50-95:            " + fmt(map95))
        return "\n".join(lines) if lines else "(無)"

    def _update_metrics_text(self):
        self.txt_metrics.delete("1.0", "end")
        self.txt_metrics.insert("end", self._metrics_log or "(無)")
        self.txt_metrics.see("end")

    def _bind_config_traces(self):
        def _trace(v):
            try:
                v.trace_add("write", lambda *args: self._save_config())
            except Exception:
                pass

        for v in [
            self.var_task,
            self.var_dataset_zip,
            self.var_work_dir,
            self.var_out_zip_dir,
            self.var_model_pick,
            self.var_model_dir,
            self.var_use_custom_model,
            self.var_custom_model,
            self.var_epochs,
            self.var_imgsz,
            self.var_batch,
            self.var_eta_decimal_places,
            self.var_device,
            self.var_resume,
            self.var_skip_unlabeled,
            self.var_delete_temp,
        ]:
            _trace(v)

    def _load_config(self):
        if not CONFIG_PATH.exists():
            return
        try:
            self._loading_config = True
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            self.var_task.set(data.get("task", self.var_task.get()))
            self.var_dataset_zip.set(data.get("dataset_zip", self.var_dataset_zip.get()))
            self.var_work_dir.set(data.get("work_dir", self.var_work_dir.get()))
            self.var_out_zip_dir.set(data.get("out_zip_dir", self.var_out_zip_dir.get()))
            self.var_model_pick.set(data.get("model_pick", self.var_model_pick.get()))
            self.var_model_dir.set(data.get("model_dir", self.var_model_dir.get()))
            self.var_use_custom_model.set(bool(data.get("use_custom_model", self.var_use_custom_model.get())))
            self.var_custom_model.set(data.get("custom_model", self.var_custom_model.get()))
            self.var_device.set(data.get("device", self.var_device.get()))
            self.var_resume.set(bool(data.get("resume", self.var_resume.get())))
            self.var_skip_unlabeled.set(bool(data.get("skip_unlabeled", self.var_skip_unlabeled.get())))
            self.var_delete_temp.set(bool(data.get("delete_temp", self.var_delete_temp.get())))
            try:
                dp = int(data.get("eta_decimal_places", self.var_eta_decimal_places.get()))
                self.var_eta_decimal_places.set(dp)
                self._eta_decimal_places = dp
            except Exception:
                pass
            try:
                self.var_epochs.set(int(data.get("epochs", self.var_epochs.get())))
            except Exception:
                pass
            try:
                self.var_imgsz.set(int(data.get("imgsz", self.var_imgsz.get())))
            except Exception:
                pass
            try:
                self.var_batch.set(int(data.get("batch", self.var_batch.get())))
            except Exception:
                pass
        except Exception:
            return
        finally:
            self._loading_config = False

    def _save_config(self):
        if self._loading_config:
            return
        try:
            data = {
                "task": self.var_task.get(),
                "dataset_zip": self.var_dataset_zip.get(),
                "work_dir": self.var_work_dir.get(),
                "out_zip_dir": self.var_out_zip_dir.get(),
                "model_pick": self.var_model_pick.get(),
                "model_dir": self.var_model_dir.get(),
                "use_custom_model": bool(self.var_use_custom_model.get()),
                "custom_model": self.var_custom_model.get(),
                "epochs": int(self.var_epochs.get()),
                "imgsz": int(self.var_imgsz.get()),
                "batch": int(self.var_batch.get()),
                "eta_decimal_places": int(self.var_eta_decimal_places.get()),
                "device": self.var_device.get(),
                "resume": bool(self.var_resume.get()),
                "skip_unlabeled": bool(self.var_skip_unlabeled.get()),
                "delete_temp": bool(self.var_delete_temp.get()),
            }
            CONFIG_PATH.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")
        except Exception:
            return

    def _poll_queue(self):
        try:
            while True:
                msg = self.msg_q.get_nowait()
                self._pending_msgs.append(msg)
        except queue.Empty:
            pass
        if not self._is_resizing:
            self._flush_pending_msgs()
        self.after(120, self._poll_queue)

    def _flush_pending_msgs(self):
        if self._flush_after_id is not None:
            try:
                self.after_cancel(self._flush_after_id)
            except Exception:
                pass
            self._flush_after_id = None

        while self._pending_msgs:
            msg = self._pending_msgs.pop(0)
            kind = msg[0]
            if kind == "log":
                self._log(msg[1])
            elif kind == "status":
                self.var_status.set(msg[1])
            elif kind == "progress_epoch":
                self._set_epoch_progress(msg[1], msg[2], msg[3] if len(msg) > 3 else None)
            elif kind == "progress_batch":
                self._set_batch_progress(msg[1], msg[2], msg[3] if len(msg) > 3 else None)
            elif kind == "device":
                self.var_run_device.set(f"裝置：{msg[1]}")
            elif kind == "prev_epoch_metrics":
                row = msg[1]
                self._metrics_log = self._format_prev_row(row) + "\n\n" + self._format_epoch_summary(row)
                self._update_metrics_text()
            elif kind == "force_stop_lock":
                self._force_stop_locked = bool(msg[1])
            elif kind == "done":
                self._on_done(msg[1], msg[2])

    def _on_resize(self, event):
        if event.widget is not self:
            return
        self._is_resizing = True
        if self._resize_after_id is not None:
            try:
                self.after_cancel(self._resize_after_id)
            except Exception:
                pass
        self._resize_after_id = self.after(50, self._end_resize)

    def _end_resize(self):
        self._is_resizing = False
        self._resize_after_id = None
        if self._pending_msgs:
            self._flush_after_id = self.after(50, self._flush_pending_msgs)

    def _on_done(self, ok: bool, payload: Dict[str, Any]):
        self._force_stop_locked = False
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")

        # Force ETA to zero before any completion messagebox.
        self._eta_epoch_sec = 0.0
        self._eta_batch_sec = 0.0
        self._eta_epoch_last_update = time.time()
        self._eta_batch_last_update = time.time()
        ep_cur, ep_total = self._last_epoch_progress
        ba_cur, ba_total = self._last_batch_progress
        self.var_ep_text.set(
            f"Epoch: {ep_cur}/{ep_total} 剩餘時間：{self._format_eta_seconds(0.0, self._eta_decimal_places)}"
        )
        self.var_ba_text.set(
            f"Batch: {ba_cur}/{ba_total} 剩餘時間：{self._format_eta_seconds(0.0, self._eta_decimal_places)}"
        )

        if ok:
            append_history(payload)
            self._load_history()

            rows = payload.get("metrics_rows", {}) or {}
            last_row = rows.get("last")
            self._metrics_log = self._format_prev_row(last_row) + "\n\n" + self._format_epoch_summary(last_row)
            summary = (
                f"\n輸出 ZIP 檔案:       {payload.get('out_zip','')}\n"
                f"是否中途停止:         {payload.get('stopped', False)}"
            )
            self._metrics_log += summary
            self._update_metrics_text()
            self.var_status.set("完成")
            self._log(f"\n[{now_str()}] ===== 完成 =====\n")

            # Auto load hardcore view
            run_dir = payload.get("run_dir", "")
            if run_dir:
                try:
                    self.hardcore_panel.load_run(run_dir)
                except Exception as e:
                    self._log(f"[{now_str()}] 硬核視覺化載入失敗：{e}\n")
            if payload.get("stopped", False):
                messagebox.showinfo("訓練結束", "訓練已停止。")
            else:
                messagebox.showinfo("訓練完成", "訓練完成。")
        else:
            err = payload.get("error", "Unknown error")
            self.var_status.set("失敗")
            self._log(f"\n[{now_str()}] ===== 失敗 =====\n{err}\n")
            messagebox.showerror("訓練失敗", err)

        self._eta_epoch_sec = None
        self._eta_batch_sec = None
        self.pbar_epoch["value"] = 0
        self.pbar_batch["value"] = 0
        self.var_ep_text.set("Epoch: 0/0")
        self.var_ba_text.set("Batch: 0/0")

    # ---------------- history ----------------
    def _load_history(self):
        for it in self.tree.get_children():
            self.tree.delete(it)

        items = read_history(800)
        for rec in reversed(items):
            met = rec.get("metrics", {}) or {}
            map50 = met.get("mAP50", None)
            recall = met.get("recall", None)

            def fmt(x):
                return "" if x is None else f"{x:.4f}"

            self.tree.insert(
                "",
                "end",
                values=(
                    rec.get("time", ""),
                    rec.get("task", ""),
                    rec.get("model", ""),
                    rec.get("dataset_zip", ""),
                    fmt(map50),
                    fmt(recall),
                    rec.get("out_zip", ""),
                    rec.get("run_dir", ""),
                ),
            )

    def _selected_record(self) -> Optional[Dict[str, Any]]:
        sel = self.tree.selection()
        if not sel:
            return None
        vals = self.tree.item(sel[0], "values")
        time_val = vals[0]
        out_zip = vals[6]
        run_dir = vals[7]

        items = read_history(3000)
        for rec in reversed(items):
            if rec.get("time") == time_val and rec.get("out_zip") == out_zip and rec.get("run_dir") == run_dir:
                return rec
        return None

    def show_selected_detail(self):
        if not self._ensure_single_selection("查看詳細"):
            return
        rec = self._selected_record()
        if not rec:
            messagebox.showinfo("提示", "請先選擇一筆歷史紀錄")
            return
        top = tk.Toplevel(self)
        top.title("訓練詳細")
        top.geometry("920x580")
        txt = tk.Text(top, wrap="word")
        txt.pack(fill="both", expand=True)
        txt.insert("end", json.dumps(rec, ensure_ascii=False, indent=2))
        txt.configure(state="disabled")

    def open_selected_zip_dir(self):
        sels = self.tree.selection()
        if not sels:
            messagebox.showinfo("提示", "請先選擇一筆歷史紀錄")
            return
        # If single selection and zip exists, open folder and select zip.
        if len(sels) == 1:
            rec = self._selected_record()
            if rec:
                zp = rec.get("out_zip", "")
                if zp:
                    p = Path(zp)
                    if p.exists() and os.name == "nt":
                        try:
                            os.system(f'explorer /select,"{p}"')
                            return
                        except Exception:
                            pass
        # fallback: open folder from first selected
        rec = self._selected_record()
        if not rec:
            messagebox.showinfo("提示", "請先選擇一筆歷史紀錄")
            return
        zp = rec.get("out_zip", "")
        if not zp:
            messagebox.showerror("找不到歷史紀錄", "找不到歷史紀錄資料夾")
            return
        p = Path(zp)
        folder = p.parent
        self._open_folder(folder)

    def load_selected_to_hardcore(self):
        if not self._ensure_single_selection("載入硬核視覺化"):
            return
        rec = self._selected_record()
        if not rec:
            messagebox.showinfo("提示", "請先選擇一筆歷史紀錄")
            return
        run_dir = rec.get("run_dir", "")
        if not run_dir:
            messagebox.showerror("載入失敗", "找不到此訓練結果")
            return
        try:
            self.hardcore_panel.load_run(run_dir)
            messagebox.showinfo("載入成功", "成功載入至硬核視覺化")
        except Exception as e:
            messagebox.showerror("載入失敗", f"硬核視覺化載入失敗：\n{e}")

    def _ensure_single_selection(self, action_name: str) -> bool:
        sels = self.tree.selection()
        if len(sels) > 1:
            messagebox.showwarning("多選限制", f"{action_name} 只能單選，請只選擇一筆紀錄。")
            return False
        return True

    def open_selected_zip(self):
        if not self._ensure_single_selection("開啟輸出 zip"):
            return
        rec = self._selected_record()
        if not rec:
            messagebox.showinfo("提示", "請先選擇一筆歷史紀錄")
            return
        zp = rec.get("out_zip", "")
        if not zp:
            messagebox.showwarning("找不到輸出", "找不到輸出 zip。")
            return
        p = Path(zp)
        if not p.exists():
            messagebox.showwarning("找不到輸出", "輸出 zip 不存在。")
            return
        try:
            if os.name == "nt":
                os.startfile(str(p))
                return
        except Exception:
            pass
        # fallback: open folder and select the zip
        try:
            if os.name == "nt":
                os.system(f'explorer /select,"{p}"')
            else:
                self._open_folder(p.parent)
        except Exception:
            self._open_folder(p.parent)

    def _confirm_delete_dialog(self, count: int) -> Optional[bool]:
        top = tk.Toplevel(self)
        top.title("刪除確認")
        top.resizable(False, False)
        top.grab_set()

        top.columnconfigure(0, weight=1)

        msg = ttk.Label(
            top,
            text=f"即將刪除 {count} 筆歷史紀錄。\n此操作無法復原，是否繼續？",
            justify="left"
        )
        msg.grid(row=0, column=0, sticky="w", padx=16, pady=(10, 6))

        var_del_zip = tk.BooleanVar(value=False)
        cb = ttk.Checkbutton(top, text="同時刪除對應的輸出 zip", variable=var_del_zip)
        cb.grid(row=1, column=0, sticky="w", padx=16, pady=(0, 6))

        ret = {"ok": None}
        btns = ttk.Frame(top)
        btns.grid(row=2, column=0, sticky="e", padx=16, pady=(0, 10))
        ttk.Button(btns, text="取消", command=lambda: _close(False)).pack(side="right")
        ttk.Button(btns, text="刪除", command=lambda: _close(True)).pack(side="right", padx=8)

        def _close(ok: bool):
            ret["ok"] = ok
            top.destroy()

        # Center and size to content.
        top.update_idletasks()
        w = top.winfo_reqwidth()
        h = top.winfo_reqheight()
        x = self.winfo_rootx() + (self.winfo_width() - w) // 2
        y = self.winfo_rooty() + (self.winfo_height() - h) // 2
        top.geometry(f"{w}x{h}+{x}+{y}")
        top.minsize(w, h)
        top.maxsize(w, h)

        top.wait_window()
        if ret["ok"] is None:
            return None
        return ret["ok"], bool(var_del_zip.get())

    def delete_selected_history(self):
        sels = self.tree.selection()
        if not sels:
            messagebox.showinfo("提示", "請先選擇要刪除的歷史紀錄")
            return
        confirm = self._confirm_delete_dialog(len(sels))
        if not confirm:
            return
        ok, del_zip = confirm
        if not ok:
            return

        items = read_history(100000)
        selected_keys = set()
        for sel in sels:
            vals = self.tree.item(sel, "values")
            if not vals:
                continue
            selected_keys.add((vals[0], vals[6], vals[7]))

        kept = []
        removed = []
        for rec in items:
            key = (rec.get("time", ""), rec.get("out_zip", ""), rec.get("run_dir", ""))
            if key in selected_keys:
                removed.append(rec)
            else:
                kept.append(rec)

        # rewrite history
        try:
            HISTORY_PATH.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in kept) + ("\n" if kept else ""),
                encoding="utf-8",
            )
        except Exception as e:
            messagebox.showerror("刪除失敗", f"無法更新歷史檔：{e}")
            return

        if del_zip:
            for rec in removed:
                zp = rec.get("out_zip", "")
                if not zp:
                    continue
                try:
                    Path(zp).unlink(missing_ok=True)
                except Exception:
                    pass

        self._load_history()
        messagebox.showinfo("完成", f"已刪除 {len(removed)} 筆歷史紀錄。")

    # ---------------- open folder helper ----------------
    def _open_folder(self, folder: Path):
        try:
            folder = folder.resolve()
        except Exception:
            pass

        try:
            if os.name == "nt":
                os.startfile(str(folder))
                return

            if sys.platform == "darwin":
                os.system(f'open "{folder}"')
            else:
                os.system(f'xdg-open "{folder}"')
        except Exception as e:
            messagebox.showerror("開啟失敗", f"無法開啟資料夾：\n{e}")


if __name__ == "__main__":
    app = App()
    app.mainloop()
