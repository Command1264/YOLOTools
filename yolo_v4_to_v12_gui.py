import os
import re
import uuid
import time
import shutil
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Sample:
    img_path: Path
    label_path: Path
    # top-level category under input root, e.g., Fire / Smoke
    category: str


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def parse_yolo_label_lines(text: str) -> List[List[str]]:
    """
    Parses YOLO label file lines.
    Keeps tokens as strings to preserve formatting precision.
    Expected format per line: class x y w h (all normalized floats), but we don't over-validate.
    """
    lines = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = re.split(r"\s+", raw)
        if len(parts) < 5:
            # Keep as-is, but caller may choose to skip/raise
            lines.append(parts)
        else:
            lines.append(parts)
    return lines


def rewrite_label_class_ids(label_in: Path, label_out: Path, new_class_id: int) -> Tuple[bool, str]:
    """
    Replace class_id (first token) with new_class_id on each non-empty line.
    Returns (ok, message).
    """
    try:
        txt = label_in.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return False, f"讀取標註失敗: {label_in} ({e})"

    lines = parse_yolo_label_lines(txt)
    out_lines = []
    for parts in lines:
        if not parts:
            continue
        # If line is malformed, still attempt rewrite if possible
        parts[0] = str(new_class_id)
        out_lines.append(" ".join(parts))

    try:
        label_out.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    except Exception as e:
        return False, f"寫入標註失敗: {label_out} ({e})"

    return True, "OK"


def copy_label_keep_original(label_in: Path, label_out: Path) -> Tuple[bool, str]:
    try:
        shutil.copyfile(label_in, label_out)
        return True, "OK"
    except Exception as e:
        return False, f"複製標註失敗: {label_in} -> {label_out} ({e})"


def generate_unique_name(mode: str, stem: str, ext: str, used: set) -> str:
    """
    mode:
      - 'uuid': always uuid
      - 'avoid': keep stem unless collision, then append _1, _2...
    used contains full filename (with ext)
    """
    if mode == "uuid":
        while True:
            name = f"{uuid.uuid4().hex}{ext}"
            if name not in used:
                used.add(name)
                return name

    # avoid collisions
    base = stem
    candidate = f"{base}{ext}"
    if candidate not in used:
        used.add(candidate)
        return candidate

    i = 1
    while True:
        candidate = f"{base}_{i}{ext}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        i += 1


def find_samples(input_root: Path, logger) -> List[Sample]:
    """
    Recursively find image+label pairs.
    Category is the first folder under input_root for that file.
    Example: input_root/Fire/Train/Class1/Obj0_0.png -> category='Fire'
    """
    samples: List[Sample] = []
    if not input_root.exists():
        return samples

    # index all txt by stem for quick match? safer: match per image directly
    for img_path in input_root.rglob("*"):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in IMG_EXTS:
            continue

        label_path = img_path.with_suffix(".txt")
        if not label_path.exists():
            logger(f"[略過] 找不到標註: {img_path}")
            continue

        # category = first path component relative to root
        try:
            rel = img_path.relative_to(input_root)
            category = rel.parts[0] if len(rel.parts) >= 2 else "unknown"
        except Exception:
            category = "unknown"

        samples.append(Sample(img_path=img_path, label_path=label_path, category=category))

    return samples


def split_counts(total: int, p_train: float, p_val: float, p_test: float) -> Tuple[int, int, int]:
    """
    Percentages sum <= 100.
    Compute base counts via floor(total * p/100), then distribute remainder to train>val>test.
    """
    # base
    train = int(total * p_train / 100.0)
    val = int(total * p_val / 100.0)
    test = int(total * p_test / 100.0)

    used = train + val + test
    rem = total - used
    # distribute remainder by priority train>val>test
    while rem > 0:
        if rem <= 0:
            break
        train += 1
        rem -= 1
        if rem <= 0:
            break
        val += 1
        rem -= 1
        if rem <= 0:
            break
        test += 1
        rem -= 1

    # In rare case sum > total due to weird floating, clip:
    if train + val + test > total:
        overflow = train + val + test - total
        # remove from test then val then train
        t = min(test, overflow); test -= t; overflow -= t
        v = min(val, overflow); val -= v; overflow -= v
        train = max(0, train - overflow)

    return train, val, test


def write_dataset_yaml(out_dir: Path, class_names: List[str]):
    # Ultralytics dataset YAML
    # Use relative paths from yaml file location
    yaml_text = (
        f"path: {out_dir.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n\n"
        f"names:\n"
    )
    for i, n in enumerate(class_names):
        yaml_text += f"  {i}: {n}\n"

    (out_dir / "dataset.yaml").write_text(yaml_text, encoding="utf-8")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("YOLOv4 Darknet 資料集 → YOLOv12 (Ultralytics) 格式轉換器")
        self.geometry("980x700")
        self.minsize(900, 640)

        # Vars
        self.var_input = tk.StringVar()
        self.var_output = tk.StringVar()

        self.var_train = tk.StringVar(value="80")
        self.var_val = tk.StringVar(value="10")
        self.var_test = tk.StringVar(value="10")

        self.var_seed = tk.StringVar(value="42")
        self.var_shuffle = tk.BooleanVar(value=True)

        self.var_rename_mode = tk.StringVar(value="uuid")  # uuid / avoid
        self.var_class_mode = tk.StringVar(value="folder")  # folder / keep

        self._stop_flag = False
        self._worker: Optional[threading.Thread] = None

        self._build_ui()

    def _build_ui(self):
        pad = 10
        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True, padx=pad, pady=pad)

        # Paths
        path_box = ttk.LabelFrame(frm, text="路徑設定")
        path_box.pack(fill="x", padx=2, pady=6)

        row = 0
        ttk.Label(path_box, text="來源資料夾 (TaskFire):").grid(row=row, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(path_box, textvariable=self.var_input).grid(row=row, column=1, sticky="ew", padx=8, pady=6)
        ttk.Button(path_box, text="瀏覽...", command=self.browse_input).grid(row=row, column=2, padx=8, pady=6)

        row += 1
        ttk.Label(path_box, text="輸出資料夾:").grid(row=row, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(path_box, textvariable=self.var_output).grid(row=row, column=1, sticky="ew", padx=8, pady=6)
        ttk.Button(path_box, text="瀏覽...", command=self.browse_output).grid(row=row, column=2, padx=8, pady=6)

        path_box.columnconfigure(1, weight=1)

        # Options
        opt = ttk.LabelFrame(frm, text="轉換設定")
        opt.pack(fill="x", padx=2, pady=6)

        # Split ratios
        ttk.Label(opt, text="train (%):").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(opt, width=8, textvariable=self.var_train).grid(row=0, column=1, sticky="w", padx=8, pady=6)
        ttk.Label(opt, text="val (%):").grid(row=0, column=2, sticky="w", padx=8, pady=6)
        ttk.Entry(opt, width=8, textvariable=self.var_val).grid(row=0, column=3, sticky="w", padx=8, pady=6)
        ttk.Label(opt, text="test (%):").grid(row=0, column=4, sticky="w", padx=8, pady=6)
        ttk.Entry(opt, width=8, textvariable=self.var_test).grid(row=0, column=5, sticky="w", padx=8, pady=6)

        # Shuffle + seed
        ttk.Checkbutton(opt, text="先打亂資料集 (shuffle)", variable=self.var_shuffle).grid(
            row=1, column=0, columnspan=2, sticky="w", padx=8, pady=6
        )
        ttk.Label(opt, text="隨機種子 seed:").grid(row=1, column=2, sticky="w", padx=8, pady=6)
        ttk.Entry(opt, width=10, textvariable=self.var_seed).grid(row=1, column=3, sticky="w", padx=8, pady=6)

        # Rename mode
        rename_box = ttk.LabelFrame(opt, text="檔名處理")
        rename_box.grid(row=2, column=0, columnspan=6, sticky="ew", padx=8, pady=8)
        ttk.Radiobutton(rename_box, text="全部改成 UUID", value="uuid", variable=self.var_rename_mode).grid(
            row=0, column=0, sticky="w", padx=8, pady=6
        )
        ttk.Radiobutton(rename_box, text="保留原名，遇到重複自動避重", value="avoid", variable=self.var_rename_mode).grid(
            row=0, column=1, sticky="w", padx=8, pady=6
        )

        # Class mode
        class_box = ttk.LabelFrame(opt, text="類別來源 / 標註處理")
        class_box.grid(row=3, column=0, columnspan=6, sticky="ew", padx=8, pady=8)
        ttk.Radiobutton(
            class_box,
            text="用最上層資料夾當類別 (Fire/Smoke/...)，並重寫每行 class_id",
            value="folder",
            variable=self.var_class_mode,
        ).grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Radiobutton(
            class_box,
            text="保留原 txt 內容 (不改 class_id)，只搬移/改名",
            value="keep",
            variable=self.var_class_mode,
        ).grid(row=1, column=0, sticky="w", padx=8, pady=6)

        # Buttons + progress
        act = ttk.Frame(frm)
        act.pack(fill="x", padx=2, pady=8)

        self.btn_start = ttk.Button(act, text="開始轉換", command=self.start)
        self.btn_start.pack(side="left", padx=6)

        self.btn_stop = ttk.Button(act, text="停止", command=self.stop, state="disabled")
        self.btn_stop.pack(side="left", padx=6)

        self.prog = ttk.Progressbar(act, orient="horizontal", mode="determinate")
        self.prog.pack(side="left", fill="x", expand=True, padx=10)

        # Log
        log_box = ttk.LabelFrame(frm, text="Log")
        log_box.pack(fill="both", expand=True, padx=2, pady=6)

        self.txt = tk.Text(log_box, height=18, wrap="word")
        self.txt.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(log_box, orient="vertical", command=self.txt.yview)
        sb.pack(side="right", fill="y")
        self.txt.configure(yscrollcommand=sb.set)

        self.log("就緒。選擇來源資料夾(TaskFire)與輸出資料夾後開始。")

    def log(self, s: str):
        ts = time.strftime("%H:%M:%S")
        self.txt.insert("end", f"[{ts}] {s}\n")
        self.txt.see("end")
        self.update_idletasks()

    def browse_input(self):
        p = filedialog.askdirectory(title="選擇來源資料夾 (TaskFire)")
        if p:
            self.var_input.set(p)

    def browse_output(self):
        p = filedialog.askdirectory(title="選擇輸出資料夾")
        if p:
            self.var_output.set(p)

    def stop(self):
        self._stop_flag = True
        self.log("已要求停止：會在下一個檔案處理點中止。")

    def start(self):
        if self._worker and self._worker.is_alive():
            messagebox.showwarning("執行中", "目前正在執行轉換。")
            return

        input_root = Path(self.var_input.get().strip())
        output_root = Path(self.var_output.get().strip())

        if not input_root.exists():
            messagebox.showerror("錯誤", "來源資料夾不存在。")
            return
        if not output_root.exists():
            messagebox.showerror("錯誤", "輸出資料夾不存在（請先建立或選擇存在的資料夾）。")
            return

        # Parse ratios
        try:
            p_train = float(self.var_train.get().strip() or "0")
            p_val = float(self.var_val.get().strip() or "0")
            p_test = float(self.var_test.get().strip() or "0")
        except Exception:
            messagebox.showerror("錯誤", "train/val/test 請輸入數字百分比。")
            return

        if p_train < 0 or p_val < 0 or p_test < 0:
            messagebox.showerror("錯誤", "百分比不可為負數。")
            return
        if p_train + p_val + p_test > 100.0 + 1e-9:
            messagebox.showerror("錯誤", "train + val + test 的總和不可超過 100%。")
            return

        try:
            seed = int(self.var_seed.get().strip() or "42")
        except Exception:
            messagebox.showerror("錯誤", "seed 請輸入整數。")
            return

        rename_mode = self.var_rename_mode.get()
        class_mode = self.var_class_mode.get()
        do_shuffle = bool(self.var_shuffle.get())

        # Start worker
        self._stop_flag = False
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.prog["value"] = 0

        self._worker = threading.Thread(
            target=self._run_convert,
            args=(input_root, output_root, p_train, p_val, p_test, do_shuffle, seed, rename_mode, class_mode),
            daemon=True,
        )
        self._worker.start()

    def _run_convert(
            self,
            input_root: Path,
            output_root: Path,
            p_train: float,
            p_val: float,
            p_test: float,
            do_shuffle: bool,
            seed: int,
            rename_mode: str,
            class_mode: str,
    ):
        try:
            out_dir = output_root / f"yolo12_dataset_{time.strftime('%Y%m%d_%H%M%S')}"
            safe_mkdir(out_dir)

            # Create dirs
            for split in ("train", "val", "test"):
                safe_mkdir(out_dir / "images" / split)
                safe_mkdir(out_dir / "labels" / split)

            self.log(f"輸出資料夾：{out_dir}")

            self.log("掃描資料集中（遞迴尋找圖片+同名txt）...")
            samples = find_samples(input_root, self.log)
            if not samples:
                self.log("找不到任何有效的 圖片+標註 配對。")
                raise RuntimeError("No samples found")

            self.log(f"找到 {len(samples)} 組樣本。")

            # Determine classes
            class_names: List[str] = []
            class_map: Dict[str, int] = {}
            if class_mode == "folder":
                # Build class list by category (stable sorted for reproducibility)
                cats = sorted({s.category for s in samples})
                class_names = cats
                class_map = {c: i for i, c in enumerate(class_names)}
                self.log(f"類別（由最上層資料夾推導）: {class_names}")
            else:
                # keep existing class ids: unknown names
                class_names = ["class0"]
                self.log("類別模式：保留原 txt class_id（dataset.yaml 會先給 placeholder class0；你可自行修改）。")

            # Shuffle
            idxs = list(range(len(samples)))
            if do_shuffle:
                rnd = random.Random(seed)
                rnd.shuffle(idxs)
                self.log(f"已打亂資料集（seed={seed}）。")
            else:
                self.log("未打亂資料集（依掃描順序）。")

            total = len(samples)
            n_train, n_val, n_test = split_counts(total, p_train, p_val, p_test)
            self.log(f"切分數量: train={n_train}, val={n_val}, test={n_test}（總數={total}）")

            # Assign splits
            split_tags = (["train"] * n_train) + (["val"] * n_val) + (["test"] * n_test)
            # If sum < total due to percentages < 100, we still must place all samples somewhere:
            # distribute remaining by train>val>test as requested
            if len(split_tags) < total:
                rem = total - len(split_tags)
                self.log(f"占比總和 < 100%，剩餘 {rem} 會依 train>val>test 分配。")
                while rem > 0:
                    split_tags.append("train"); rem -= 1
                    if rem <= 0: break
                    split_tags.append("val"); rem -= 1
                    if rem <= 0: break
                    split_tags.append("test"); rem -= 1

            split_tags = split_tags[:total]

            used_names = set()
            self.prog["maximum"] = total

            ok_cnt = 0
            skip_cnt = 0

            for i, (sample_idx, split) in enumerate(zip(idxs, split_tags), start=1):
                if self._stop_flag:
                    self.log("已停止。")
                    break

                s = samples[sample_idx]
                ext = s.img_path.suffix.lower()

                new_img_name = generate_unique_name(rename_mode, s.img_path.stem, ext, used_names)
                new_lbl_name = Path(new_img_name).with_suffix(".txt").name

                dst_img = out_dir / "images" / split / new_img_name
                dst_lbl = out_dir / "labels" / split / new_lbl_name

                # copy image
                try:
                    shutil.copyfile(s.img_path, dst_img)
                except Exception as e:
                    self.log(f"[失敗] 複製圖片: {s.img_path} -> {dst_img} ({e})")
                    skip_cnt += 1
                    self.prog["value"] = i
                    continue

                # labels
                if class_mode == "folder":
                    new_id = class_map.get(s.category, 0)
                    ok, msg = rewrite_label_class_ids(s.label_path, dst_lbl, new_id)
                else:
                    ok, msg = copy_label_keep_original(s.label_path, dst_lbl)

                if not ok:
                    self.log(f"[失敗] 標註處理: {msg}")
                    # If label failed, remove copied image to keep dataset consistent
                    try:
                        dst_img.unlink(missing_ok=True)
                    except Exception:
                        pass
                    skip_cnt += 1
                else:
                    ok_cnt += 1

                if i % 50 == 0 or i == total:
                    self.log(f"進度：{i}/{total}（成功 {ok_cnt}，略過/失敗 {skip_cnt}）")

                self.prog["value"] = i

            # dataset.yaml
            if class_mode == "folder":
                write_dataset_yaml(out_dir, class_names)
                self.log("已產生 dataset.yaml（類別=最上層資料夾）。")
            else:
                # Placeholder; user can edit later
                write_dataset_yaml(out_dir, class_names)
                self.log("已產生 dataset.yaml（placeholder）。若你保留原 class_id，請自行改 names 對應。")

            self.log(f"完成：成功 {ok_cnt}，略過/失敗 {skip_cnt}")
            self.log(f"請使用這個 YAML：{out_dir / 'dataset.yaml'}")

        except Exception as e:
            self.log(f"[錯誤] {e}")
        finally:
            self.btn_start.config(state="normal")
            self.btn_stop.config(state="disabled")


if __name__ == "__main__":
    app = App()
    app.mainloop()
