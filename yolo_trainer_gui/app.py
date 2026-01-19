# app.py
from __future__ import annotations

import os
import sys
import json
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
        self.title("Ultralytics YOLO GUI Trainer（dataset.zip）")
        self.geometry("1120x800")

        self.msg_q = queue.Queue()
        self.worker: Optional[TrainerWorker] = None

        # remember last paths for dialog rule
        self.last_dataset_zip = ""
        self.last_work_dir = ""
        self.last_out_zip_dir = ""
        self.last_custom_model = ""

        self._build_ui()

        # load model list (try online first)
        self._load_weights(try_online=True)

        # apply task filter once after weights loaded
        self._apply_task_filter()

        # trace task changes -> re-filter
        self.var_task.trace_add("write", lambda *args: self._apply_task_filter())

        self._load_history()
        self._poll_queue()

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
        ttk.Label(frm, text="dataset.zip：").grid(row=r, column=0, sticky="w", pady=6)
        self.var_dataset_zip = tk.StringVar()
        ttk.Entry(frm, textvariable=self.var_dataset_zip).grid(row=r, column=1, sticky="we", padx=8)
        ttk.Button(frm, text="選擇...", command=self.pick_dataset_zip).grid(row=r, column=2, sticky="e")

        r += 1
        ttk.Label(frm, text="訓練工作資料夾（解壓/跑 runs）：").grid(row=r, column=0, sticky="w", pady=6)
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
        ttk.Label(frm, text="或自訂權重（.pt）：").grid(row=r, column=0, sticky="w", pady=6)
        self.var_custom_model = tk.StringVar()
        ttk.Entry(frm, textvariable=self.var_custom_model).grid(row=r, column=1, sticky="we", padx=8)
        ttk.Button(frm, text="選擇...", command=self.pick_custom_model).grid(row=r, column=2, sticky="e")

        # Params
        r += 1
        box = ttk.LabelFrame(frm, text="訓練參數", padding=10)
        box.grid(row=r, column=0, columnspan=3, sticky="we", pady=10)

        self.var_epochs = tk.IntVar(value=50)
        self.var_imgsz = tk.IntVar(value=640)
        self.var_batch = tk.IntVar(value=16)
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

        ttk.Checkbutton(box, text="resume（接著上次中斷續跑）", variable=self.var_resume).grid(row=1, column=2, sticky="w")
        ttk.Checkbutton(box, text="跳過無標記圖片（detect/seg/pose/obb）", variable=self.var_skip_unlabeled).grid(row=1, column=3, sticky="w")
        ttk.Checkbutton(box, text="訓練後刪除解壓暫存", variable=self.var_delete_temp).grid(row=1, column=4, sticky="w")

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
        mb = ttk.LabelFrame(frm, text="最後一輪指標", padding=10)
        mb.grid(row=r, column=0, columnspan=3, sticky="we", pady=8)
        self.var_metrics = tk.StringVar(value="(尚未訓練)")
        ttk.Label(mb, textvariable=self.var_metrics, justify="left").pack(anchor="w")

        # Log
        r += 1
        lb = ttk.LabelFrame(frm, text="Log", padding=10)
        lb.grid(row=r, column=0, columnspan=3, sticky="nsew", pady=8)
        self.txt_log = tk.Text(lb, height=14, wrap="word")
        self.txt_log.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(lb, command=self.txt_log.yview)
        sb.pack(side="right", fill="y")
        self.txt_log.configure(yscrollcommand=sb.set)

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
        ttk.Button(br, text="開啟輸出 zip 資料夾", command=self.open_selected_zip_dir).pack(side="left", padx=8)
        ttk.Button(br, text="載入到硬核視覺化", command=self.load_selected_to_hardcore).pack(side="left", padx=8)

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
    def _log(self, s: str):
        self.txt_log.insert("end", s)
        self.txt_log.see("end")

    def start_train(self):
        dataset_zip = self.var_dataset_zip.get().strip()
        if not dataset_zip:
            messagebox.showerror("缺少資料", "請選擇 dataset.zip")
            return

        work_dir = self.var_work_dir.get().strip() or str(APP_DIR / "workdir")
        out_zip_dir = self.var_out_zip_dir.get().strip() or str(APP_DIR / "output_zips")

        custom = self.var_custom_model.get().strip()
        model = custom if custom else self.var_model_pick.get().strip()
        if not model:
            messagebox.showerror("缺少模型", "請選擇模型或指定自訂 .pt")
            return

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
            task=self.var_task.get().strip(),
            dataset_zip=dataset_zip,
            work_dir=work_dir,
            out_zip_dir=out_zip_dir,
            model=model,
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
        self._log(f"[{now_str()}] epochs={cfg.epochs}, imgsz={cfg.imgsz}, batch={cfg.batch}, device={cfg.device}, resume={cfg.resume}\n\n")

        self._set_epoch_progress(0, max(1, cfg.epochs))
        self._set_batch_progress(0, 1)
        self.var_metrics.set("(訓練中...)")
        self.var_status.set("訓練中 ...")

        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")

        self.worker = TrainerWorker(cfg, self.msg_q)
        self.worker.start()

    def stop_train(self):
        if self.worker:
            self.worker.request_stop()
            self._log(f"\n[{now_str()}] 已送出停止請求（將在下一個 callback 觸發時停下）\n")
            self.var_status.set("停止中 ...")

    def _set_epoch_progress(self, cur: int, total: int):
        total = max(1, int(total))
        cur = max(0, min(int(cur), total))
        self.pbar_epoch["maximum"] = total
        self.pbar_epoch["value"] = cur
        self.var_ep_text.set(f"Epoch: {cur}/{total}")

    def _set_batch_progress(self, cur: int, total: int):
        total = max(1, int(total))
        cur = max(0, min(int(cur), total))
        self.pbar_batch["maximum"] = total
        self.pbar_batch["value"] = cur
        self.var_ba_text.set(f"Batch: {cur}/{total}")

    # ---------------- queue polling ----------------
    def _poll_queue(self):
        try:
            while True:
                msg = self.msg_q.get_nowait()
                kind = msg[0]
                if kind == "log":
                    self._log(msg[1])
                elif kind == "status":
                    self.var_status.set(msg[1])
                elif kind == "progress_epoch":
                    self._set_epoch_progress(msg[1], msg[2])
                elif kind == "progress_batch":
                    self._set_batch_progress(msg[1], msg[2])
                elif kind == "done":
                    self._on_done(msg[1], msg[2])
        except queue.Empty:
            pass
        self.after(120, self._poll_queue)

    def _on_done(self, ok: bool, payload: Dict[str, Any]):
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")

        if ok:
            append_history(payload)
            self._load_history()

            m = payload.get("metrics", {}) or {}

            def fmt(x):
                return "-" if x is None else f"{x:.4f}"

            self.var_metrics.set(
                "Precision: " + fmt(m.get("precision")) + "\n"
                                                          "Recall:    " + fmt(m.get("recall")) + "\n"
                                                                                                 "F1:        " + fmt(m.get("f1")) + "\n"
                                                                                                                                    "mAP50:     " + fmt(m.get("mAP50")) + "\n"
                                                                                                                                                                          "mAP50-95:  " + fmt(m.get("mAP50-95")) + "\n"
                                                                                                                                                                                                                   f"輸出zip:   {payload.get('out_zip','')}\n"
                                                                                                                                                                                                                   f"stopped:   {payload.get('stopped', False)}"
            )
            self.var_status.set("完成")
            self._log(f"\n[{now_str()}] ===== 完成 =====\n")

            # Auto load hardcore view
            run_dir = payload.get("run_dir", "")
            if run_dir:
                try:
                    self.hardcore_panel.load_run(run_dir)
                except Exception as e:
                    self._log(f"[{now_str()}] 硬核視覺化載入失敗：{e}\n")
        else:
            err = payload.get("error", "Unknown error")
            self.var_status.set("失敗")
            self._log(f"\n[{now_str()}] ===== 失敗 =====\n{err}\n")
            messagebox.showerror("訓練失敗", err)

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
        rec = self._selected_record()
        if not rec:
            messagebox.showinfo("提示", "請先選擇一筆歷史紀錄")
            return
        zp = rec.get("out_zip", "")
        if not zp:
            messagebox.showerror("錯誤", "找不到 out_zip")
            return
        p = Path(zp)
        folder = p.parent
        self._open_folder(folder)

    def load_selected_to_hardcore(self):
        rec = self._selected_record()
        if not rec:
            messagebox.showinfo("提示", "請先選擇一筆歷史紀錄")
            return
        run_dir = rec.get("run_dir", "")
        if not run_dir:
            messagebox.showerror("錯誤", "此紀錄沒有 run_dir")
            return
        try:
            self.hardcore_panel.load_run(run_dir)
        except Exception as e:
            messagebox.showerror("錯誤", f"載入硬核視覺化失敗：{e}")

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
            messagebox.showerror("錯誤", f"無法開啟資料夾：{e}")


if __name__ == "__main__":
    app = App()
    app.mainloop()
