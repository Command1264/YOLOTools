# hardcore_view.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, List

import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

def load_results_rows(run_dir: Path) -> List[dict]:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return []
    import csv
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)

def pick_plot_files(run_dir: Path) -> Dict[str, Path]:
    # best-effort: show if exists
    candidates = {
        "results.png": run_dir / "results.png",
        "PR_curve.png": run_dir / "PR_curve.png",
        "P_curve.png": run_dir / "P_curve.png",
        "R_curve.png": run_dir / "R_curve.png",
        "F1_curve.png": run_dir / "F1_curve.png",
        "confusion_matrix.png": run_dir / "confusion_matrix.png",
        "confusion_matrix_normalized.png": run_dir / "confusion_matrix_normalized.png",
    }
    return {k: v for k, v in candidates.items() if v.exists()}

class HardcorePanel(ttk.Frame):
    """
    A panel that shows:
      - epoch table from results.csv
      - plot thumbnails (click to open large window)
    """
    def __init__(self, master):
        super().__init__(master, padding=8)

        self._thumb_refs = []  # keep PhotoImage refs alive

        # top: table
        self.table = ttk.Treeview(self, show="headings")
        self.table.pack(fill="both", expand=True)

        # bottom: images
        self.img_box = ttk.LabelFrame(self, text="訓練輸出圖表（縮圖）", padding=8)
        self.img_box.pack(fill="x", pady=10)

        self.img_frame = ttk.Frame(self.img_box)
        self.img_frame.pack(fill="x")

    def clear(self):
        for col in self.table["columns"]:
            self.table.heading(col, text="")
        self.table["columns"] = ()
        for item in self.table.get_children():
            self.table.delete(item)

        for w in self.img_frame.winfo_children():
            w.destroy()

        self._thumb_refs.clear()

    def load_run(self, run_dir: str):
        self.clear()
        rd = Path(run_dir)
        rows = load_results_rows(rd)
        if rows:
            # Choose a reasonable subset of columns if too many
            cols = list(rows[0].keys())
            preferred = [
                "epoch",
                "train/box_loss", "train/cls_loss", "train/dfl_loss",
                "val/box_loss", "val/cls_loss", "val/dfl_loss",
                "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"
            ]
            chosen = [c for c in preferred if c in cols]
            if not chosen:
                chosen = cols[:12]

            self.table["columns"] = chosen
            for c in chosen:
                self.table.heading(c, text=c)
                self.table.column(c, width=120, anchor="w")

            # insert last N rows
            for r in rows[-50:]:
                self.table.insert("", "end", values=[r.get(c, "") for c in chosen])

        plots = pick_plot_files(rd)
        if not plots:
            ttk.Label(self.img_frame, text="(找不到 results.png / PR_curve.png / confusion_matrix.png 等輸出圖)").pack(anchor="w")
            return

        # create thumbnails row
        for name, path in plots.items():
            self._add_thumb(name, path)

    def _add_thumb(self, name: str, path: Path):
        # thumb size
        max_w, max_h = 320, 200
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_w, max_h))
        tkimg = ImageTk.PhotoImage(img)
        self._thumb_refs.append(tkimg)

        card = ttk.Frame(self.img_frame)
        card.pack(side="left", padx=8, pady=6)

        lbl = ttk.Label(card, text=name)
        lbl.pack()

        btn = ttk.Button(card, image=tkimg, command=lambda: self._open_full(name, path))
        btn.pack()

    def _open_full(self, name: str, path: Path):
        top = tk.Toplevel(self)
        top.title(name)
        top.geometry("1000x700")

        canvas = tk.Canvas(top, bg="black")
        canvas.pack(fill="both", expand=True)

        img = Image.open(path).convert("RGB")
        # fit to window initially
        w, h = img.size
        scale = min(1000 / max(w, 1), 700 / max(h, 1), 1.0)
        img = img.resize((int(w * scale), int(h * scale)))
        tkimg = ImageTk.PhotoImage(img)

        canvas.create_image(10, 10, anchor="nw", image=tkimg)
        canvas.image = tkimg  # keep ref
