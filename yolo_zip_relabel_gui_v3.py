import os
import re
import shutil
import zipfile
import tempfile
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# ---- Optional dependency: PyYAML ----
try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None


# =========================
# YAML parsing helpers
# =========================
def _parse_yaml_fallback(text: str) -> dict:
    """
    Minimal YAML parser fallback.
    Supports:
      - key: value
      - names: ['a','b']  or names: [a, b]
      - names:
          0: a
          1: b
    """
    data = {}
    lines = text.splitlines()
    cleaned = []
    for ln in lines:
        ln = ln.split("#", 1)[0].rstrip("\n")
        if ln.strip():
            cleaned.append(ln)

    i = 0
    while i < len(cleaned):
        ln = cleaned[i]
        if ":" not in ln:
            i += 1
            continue
        key, rest = ln.split(":", 1)
        key = key.strip()
        rest = rest.strip()

        if key == "names" and rest == "":
            names_map = {}
            i += 1
            while i < len(cleaned):
                sub = cleaned[i]
                if re.match(r"^\s+\d+\s*:\s*", sub):
                    m = re.match(r"^\s+(\d+)\s*:\s*(.+)$", sub)
                    if m:
                        idx = int(m.group(1))
                        val = m.group(2).strip().strip("'\"")
                        names_map[idx] = val
                    i += 1
                else:
                    break
            if names_map:
                data["names"] = [names_map[k] for k in sorted(names_map.keys())]
            continue

        if key == "names" and rest.startswith("["):
            inside = rest.strip()[1:-1]
            parts = [p.strip().strip("'\"") for p in inside.split(",") if p.strip()]
            data["names"] = parts
            i += 1
            continue

        data[key] = rest.strip().strip("'\"")
        i += 1

    return data


def parse_data_yaml(text: str) -> dict:
    if yaml is not None:
        try:
            obj = yaml.safe_load(text)
            if isinstance(obj, dict):
                names = obj.get("names")
                if isinstance(names, dict):
                    obj["names"] = [names[k] for k in sorted(names.keys())]
                return obj
        except Exception:
            pass
    return _parse_yaml_fallback(text)


def _dump_yaml_simple(obj: dict) -> str:
    lines = []
    for k, v in obj.items():
        if k == "names" and isinstance(v, list):
            quoted = ", ".join([f"'{x}'" for x in v])
            lines.append(f"names: [{quoted}]")
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines) + "\n"


# =========================
# ZIP helpers
# =========================
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def norm_zip_path(p: str) -> str:
    return p.replace("\\", "/").lstrip("./")


def join_zip(base: str, rel: str) -> str:
    base = norm_zip_path(base)
    rel = norm_zip_path(rel)
    if base == "" or base == ".":
        return rel
    return f"{base.rstrip('/')}/{rel}"


def find_data_yaml_in_zip(zf: zipfile.ZipFile) -> str:
    candidates = [n for n in zf.namelist() if n.lower().endswith("data.yaml")]
    if not candidates:
        raise FileNotFoundError("Zip 裡找不到 data.yaml")
    candidates.sort(key=lambda s: (s.count("/"), len(s)))
    return candidates[0]


def zip_read_text(zf: zipfile.ZipFile, member: str) -> str:
    with zf.open(member, "r") as f:
        raw = f.read()
    for enc in ("utf-8", "utf-8-sig", "cp950", "big5"):
        try:
            return raw.decode(enc)
        except Exception:
            pass
    return raw.decode("utf-8", errors="replace")


def infer_labels_dir(images_dir: str) -> str:
    """
    Common convention:
      .../images/... -> .../labels/...
    """
    p = norm_zip_path(images_dir)
    if "/images/" in p:
        return p.replace("/images/", "/labels/")
    if p.endswith("/images"):
        return p[:-len("/images")] + "/labels"
    if p.startswith("images/"):
        return p.replace("images/", "labels/", 1)
    return p.replace("images", "labels")


def ensure_parent_dir(fp: Path):
    fp.parent.mkdir(parents=True, exist_ok=True)


def list_images_in_zip(all_names: set[str], images_dir: str) -> list[str]:
    """
    Return file members under images_dir (recursive) with image extensions.
    """
    prefix = norm_zip_path(images_dir).rstrip("/") + "/"
    imgs = []
    for n in all_names:
        nn = norm_zip_path(n)
        if nn.startswith(prefix) and nn.lower().endswith(IMG_EXTS):
            imgs.append(nn)
    imgs.sort()
    return imgs


def find_label_for_image(all_names: set[str], labels_dir: str, image_member: str) -> str | None:
    """
    For image: <labels_dir>/<stem>.txt (same filename stem)
    """
    stem = Path(image_member).stem
    cand = norm_zip_path(labels_dir).rstrip("/") + "/" + stem + ".txt"
    return cand if cand in all_names else None


# =========================
# Drag & drop listbox
# =========================
class DragListbox(tk.Listbox):
    def __init__(self, master, **kw):
        super().__init__(master, kw)
        self.curIndex = None
        self.bind("<Button-1>", self.set_current)
        self.bind("<B1-Motion>", self.shift_selection)

    def set_current(self, event):
        self.curIndex = self.nearest(event.y)

    def shift_selection(self, event):
        i = self.nearest(event.y)
        if i < 0 or self.curIndex is None:
            return
        if i != self.curIndex:
            x = self.get(self.curIndex)
            self.delete(self.curIndex)
            self.insert(i, x)
            self.curIndex = i


# =========================
# Main GUI App
# =========================
class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("YOLO Zip Label Remapper (通用) v3")
        self.root.geometry("980x760")

        self.zip_path = tk.StringVar(value="")
        self.out_dir = tk.StringVar(value="")  # empty => script dir
        self.out_name = tk.StringVar(value="converted_dataset.zip")
        self.keep_unlabeled = tk.BooleanVar(value=True)  # <-- new option

        self.status = tk.StringVar(value="尚未載入 dataset")
        self.progress_text = tk.StringVar(value="0/0")

        self.data_yaml_member = None
        self.data_yaml_base = ""     # folder containing data.yaml inside zip
        self.orig_yaml = {}          # parsed yaml dict
        self.orig_names = []         # list[str]
        self.items = []              # list[dict]: {"old_id": int, "name": str}

        self._build_ui()

    # -----------------
    # Path rules
    # -----------------
    def _script_dir(self) -> Path:
        return Path(os.getcwd())

    def _initial_dir(self, chosen_path: str) -> str:
        if chosen_path and Path(chosen_path).exists():
            return str(Path(chosen_path))
        return str(self._script_dir())

    # -----------------
    # UI
    # -----------------
    def _build_ui(self):
        pad = 10
        frm = tk.Frame(self.root)
        frm.pack(fill="both", expand=True, padx=pad, pady=pad)

        # Zip selection
        zip_row = tk.Frame(frm)
        zip_row.pack(fill="x")
        tk.Label(zip_row, text="輸入 Zip（載入時只讀 data.yaml；不整包解壓）").pack(anchor="w")
        tk.Entry(zip_row, textvariable=self.zip_path).pack(side="left", fill="x", expand=True)
        tk.Button(zip_row, text="選擇 Zip", command=self.pick_zip).pack(side="left", padx=6)

        # Output
        out_box = tk.LabelFrame(frm, text="輸出設定（沒填路徑=程式執行位置；輸出為 zip）")
        out_box.pack(fill="x", pady=(12, 0))

        out_row1 = tk.Frame(out_box)
        out_row1.pack(fill="x", padx=8, pady=6)
        tk.Label(out_row1, text="輸出資料夾：").pack(side="left")
        tk.Entry(out_row1, textvariable=self.out_dir).pack(side="left", fill="x", expand=True)
        tk.Button(out_row1, text="選擇資料夾", command=self.pick_out_dir).pack(side="left", padx=6)

        out_row2 = tk.Frame(out_box)
        out_row2.pack(fill="x", padx=8, pady=(0, 8))
        tk.Label(out_row2, text="輸出 zip 檔名：").pack(side="left")
        tk.Entry(out_row2, textvariable=self.out_name, width=52).pack(side="left")

        # Options
        opt_box = tk.LabelFrame(frm, text="處理選項")
        opt_box.pack(fill="x", pady=(12, 0))
        tk.Checkbutton(
            opt_box,
            text="保留沒有標註/過濾後為空的圖片（會產生空的 .txt label，最相容）",
            variable=self.keep_unlabeled
        ).pack(anchor="w", padx=8, pady=6)

        # Middle
        mid = tk.Frame(frm)
        mid.pack(fill="both", expand=True, pady=(12, 0))

        left = tk.LabelFrame(mid, text="Labels（拖曳改順序；顯示：原 id → 新 id；可刪除；可改名）")
        left.pack(side="left", fill="both", expand=True)

        self.listbox = DragListbox(left, selectmode=tk.SINGLE, height=18)
        self.listbox.pack(fill="both", expand=True, padx=8, pady=8)
        self.listbox.bind("<<ListboxSelect>>", self.on_select)

        btn_row = tk.Frame(left)
        btn_row.pack(fill="x", padx=8, pady=(0, 8))
        tk.Button(btn_row, text="刪除選取 label", command=self.delete_selected).pack(side="left")
        tk.Button(btn_row, text="清空列表", command=self.clear_all).pack(side="left", padx=6)
        tk.Button(btn_row, text="重新計算 → 顯示新 id", command=self.refresh_display).pack(side="left", padx=6)

        right = tk.LabelFrame(mid, text="編輯 label 名稱")
        right.pack(side="left", fill="y", padx=(12, 0))

        tk.Label(right, text="目前選取：").pack(anchor="w", padx=8, pady=(8, 0))
        self.sel_info = tk.StringVar(value="(無)")
        tk.Label(right, textvariable=self.sel_info, wraplength=280, justify="left").pack(anchor="w", padx=8)

        tk.Label(right, text="新名稱：").pack(anchor="w", padx=8, pady=(12, 0))
        self.rename_var = tk.StringVar(value="")
        tk.Entry(right, textvariable=self.rename_var, width=34).pack(anchor="w", padx=8)
        tk.Button(right, text="套用改名", command=self.apply_rename).pack(anchor="w", padx=8, pady=8)

        tk.Label(
            right,
            text="顯示格式：\n  old_id label  →  new_id\n\n說明：\n- old_id 只用來對照原始資料\n- new_id 由目前順序決定\n- 刪除=不輸出該 class",
            justify="left"
        ).pack(anchor="w", padx=8, pady=(12, 8))

        # Progress
        bottom = tk.Frame(frm)
        bottom.pack(fill="x", pady=(10, 0))

        self.pbar = ttk.Progressbar(bottom, orient="horizontal", mode="determinate")
        self.pbar.pack(side="left", fill="x", expand=True, padx=(0, 8))
        tk.Label(bottom, textvariable=self.progress_text, width=14).pack(side="left")
        tk.Button(bottom, text="🚀 一鍵處理並輸出 Zip", command=self.run).pack(side="left", padx=8)

        tk.Label(frm, textvariable=self.status, fg="#333").pack(anchor="w", pady=(10, 0))

        # Sync drag to data
        self._wire_drag_sync()

    # -----------------
    # Drag sync
    # -----------------
    def _wire_drag_sync(self):
        lb = self.listbox

        def shift_selection(event):
            i = lb.nearest(event.y)
            if i < 0 or lb.curIndex is None:
                return
            if i != lb.curIndex:
                # move in data
                item = self.items.pop(lb.curIndex)
                self.items.insert(i, item)
                lb.curIndex = i
                self.refresh_display()
                lb.selection_clear(0, tk.END)
                lb.selection_set(i)

        lb.shift_selection = shift_selection
        lb.bind("<B1-Motion>", lb.shift_selection)

    # -----------------
    # Load zip / yaml (no full extract)
    # -----------------
    def pick_zip(self):
        initdir = self._initial_dir(self.zip_path.get())
        path = filedialog.askopenfilename(
            initialdir=initdir,
            filetypes=[("Zip files", "*.zip"), ("All files", "*.*")]
        )
        if not path:
            return
        self.zip_path.set(path)
        self.load_yaml_from_zip(Path(path))

    def pick_out_dir(self):
        initdir = self._initial_dir(self.out_dir.get())
        path = filedialog.askdirectory(initialdir=initdir)
        if not path:
            return
        self.out_dir.set(path)

    def load_yaml_from_zip(self, zpath: Path):
        try:
            with zipfile.ZipFile(zpath, "r") as zf:
                member = find_data_yaml_in_zip(zf)
                text = zip_read_text(zf, member)
                data = parse_data_yaml(text)

            names = data.get("names")
            if not isinstance(names, list) or not names:
                raise ValueError("data.yaml 解析不到 names（需為 list 或 mapping）")

            self.data_yaml_member = member
            base = norm_zip_path(str(Path(member).parent))
            self.data_yaml_base = "" if base in ("", ".", "./") else base

            self.orig_yaml = data
            self.orig_names = names[:]
            self.items = [{"old_id": i, "name": n} for i, n in enumerate(names)]
            self.refresh_display()

            self.status.set(f"已載入：{zpath.name} | data.yaml: {member} | classes: {len(names)}")
        except Exception as e:
            messagebox.showerror("錯誤", str(e))
            self.status.set("載入失敗")

    # -----------------
    # List display (old -> new arrow)
    # -----------------
    def refresh_display(self):
        sel = self._selected_index()
        self.listbox.delete(0, tk.END)
        for new_id, item in enumerate(self.items):
            old_id = item["old_id"]
            name = item["name"]
            self.listbox.insert(tk.END, f"{old_id:>3}  {name}   →   {new_id}")
        if sel is not None and sel < self.listbox.size():
            self.listbox.selection_set(sel)
        self.on_select()

        self.progress_text.set("0/0")
        self.pbar["value"] = 0
        self.pbar["maximum"] = 1

    def _selected_index(self):
        sel = self.listbox.curselection()
        if not sel:
            return None
        return int(sel[0])

    def on_select(self, _evt=None):
        idx = self._selected_index()
        if idx is None:
            self.sel_info.set("(無)")
            self.rename_var.set("")
            return
        item = self.items[idx]
        self.sel_info.set(f"old {item['old_id']}  name '{item['name']}'  (目前 new id = {idx})")
        self.rename_var.set(item["name"])

    def delete_selected(self):
        idx = self._selected_index()
        if idx is None:
            return
        del self.items[idx]
        self.refresh_display()

    def clear_all(self):
        if messagebox.askyesno("確認", "確定要清空列表嗎？（你可以再載入 zip 重新來）"):
            self.items = []
            self.refresh_display()

    def apply_rename(self):
        idx = self._selected_index()
        if idx is None:
            return
        new_name = self.rename_var.get().strip()
        if not new_name:
            messagebox.showwarning("提示", "新名稱不能是空白")
            return
        self.items[idx]["name"] = new_name
        self.refresh_display()

    # -----------------
    # Run processing
    # -----------------
    def run(self):
        if not self.zip_path.get():
            messagebox.showerror("錯誤", "請先選擇輸入 zip")
            return
        if not self.data_yaml_member or not self.orig_names:
            messagebox.showerror("錯誤", "尚未成功載入 data.yaml")
            return
        if len(self.items) == 0:
            messagebox.showerror("錯誤", "labels 列表是空的（你可能刪光了）")
            return

        out_dir = Path(self.out_dir.get().strip()) if self.out_dir.get().strip() else self._script_dir()
        out_dir.mkdir(parents=True, exist_ok=True)

        out_name = self.out_name.get().strip() or "converted_dataset.zip"
        if not out_name.lower().endswith(".zip"):
            out_name += ".zip"
        out_zip_path = out_dir / out_name

        new_names = [it["name"] for it in self.items]
        kept_old_ids = [it["old_id"] for it in self.items]
        removed = set(range(len(self.orig_names))) - set(kept_old_ids)
        id_map = {old_id: new_id for new_id, old_id in enumerate(kept_old_ids)}

        keep_unlabeled = bool(self.keep_unlabeled.get())

        threading.Thread(
            target=self._process_zip_worker,
            args=(Path(self.zip_path.get()), out_zip_path, id_map, removed, new_names, keep_unlabeled),
            daemon=True
        ).start()

    def _process_zip_worker(self, zip_path: Path, out_zip_path: Path,
                            id_map: dict, removed: set, new_names: list, keep_unlabeled: bool):
        try:
            self._set_progress(0, 1, "準備中...")

            with zipfile.ZipFile(zip_path, "r") as zf:
                yaml_text = zip_read_text(zf, self.data_yaml_member)
                data = parse_data_yaml(yaml_text)

                base = self.data_yaml_base

                # IMPORTANT:
                # Output folder structure should follow these relative paths.
                # We'll use exactly the same train/val/test strings in output yaml.
                train_rel = str(data.get("train", "images/train"))
                val_rel = str(data.get("val", "images/val"))
                test_rel = str(data.get("test", "images/test"))

                train_images = norm_zip_path(join_zip(base, train_rel))
                val_images = norm_zip_path(join_zip(base, val_rel))
                test_images = norm_zip_path(join_zip(base, test_rel))

                split_images = {"train": train_images, "val": val_images, "test": test_images}
                split_labels = {k: infer_labels_dir(v) for k, v in split_images.items()}

                all_names = set(map(norm_zip_path, zf.namelist()))

                # Collect images for progress (scan images dirs)
                split_image_members = {}
                total_images = 0
                for split, img_dir in split_images.items():
                    imgs = list_images_in_zip(all_names, img_dir)
                    split_image_members[split] = imgs
                    total_images += len(imgs)

                if total_images == 0:
                    raise RuntimeError("在 zip 裡找不到任何圖片（請確認 data.yaml 的 train/val/test 指向正確資料夾）")

                # Temp workspace auto cleaned
                with tempfile.TemporaryDirectory() as td:
                    td = Path(td)
                    out_root = td / "dataset"

                    # Phase 1: iterate images (so we can keep/skip unlabeled)
                    kept = 0
                    processed = 0
                    self._set_progress(0, total_images, f"處理 images/labels... (0/{total_images})")

                    for split, imgs in split_image_members.items():
                        img_out_dir = out_root / split_images[split]  # keep same structure as input
                        lbl_out_dir = out_root / split_labels[split]
                        img_out_dir.mkdir(parents=True, exist_ok=True)
                        lbl_out_dir.mkdir(parents=True, exist_ok=True)

                        for img_member in imgs:
                            processed += 1

                            # Find label member (may not exist)
                            lbl_member = find_label_for_image(all_names, split_labels[split], img_member)
                            new_lines = []

                            if lbl_member is not None:
                                text = zip_read_text(zf, lbl_member)
                                for line in text.splitlines():
                                    parts = line.strip().split()
                                    if not parts:
                                        continue
                                    try:
                                        old_id = int(parts[0])
                                    except Exception:
                                        continue
                                    if old_id in removed:
                                        continue
                                    if old_id not in id_map:
                                        continue
                                    new_id = id_map[old_id]
                                    new_lines.append(str(new_id) + " " + " ".join(parts[1:]))

                            # Decide keep or skip
                            if (lbl_member is None and not keep_unlabeled) or (lbl_member is not None and not new_lines and not keep_unlabeled):
                                # skip unlabeled or emptied
                                if processed % 10 == 0 or processed == total_images:
                                    self._set_progress(processed, total_images,
                                                       f"處理 images/labels... ({processed}/{total_images}) | 已保留 {kept}")
                                continue

                            # Copy image
                            out_img_path = out_root / img_member
                            ensure_parent_dir(out_img_path)
                            with zf.open(img_member, "r") as src, open(out_img_path, "wb") as dst:
                                shutil.copyfileobj(src, dst)

                            # Write label:
                            # - If no label exists or becomes empty:
                            #   - keep_unlabeled True => write empty .txt (most compatible)
                            #   - else already skipped
                            out_lbl_path = out_root / split_labels[split] / (Path(img_member).stem + ".txt")
                            ensure_parent_dir(out_lbl_path)
                            out_lbl_path.write_text("\n".join(new_lines), encoding="utf-8")

                            kept += 1

                            if processed % 10 == 0 or processed == total_images:
                                self._set_progress(processed, total_images,
                                                   f"處理 images/labels... ({processed}/{total_images}) | 已保留 {kept}")

                    # Write new data.yaml (keep original train/val/test strings!)
                    new_yaml = dict(data) if isinstance(data, dict) else {}
                    new_yaml["path"] = "."
                    new_yaml["names"] = new_names
                    new_yaml["nc"] = len(new_names)
                    new_yaml["train"] = train_rel
                    new_yaml["val"] = val_rel
                    new_yaml["test"] = test_rel

                    yaml_str = yaml.safe_dump(new_yaml, allow_unicode=True, sort_keys=False) if yaml else _dump_yaml_simple(new_yaml)
                    (out_root / join_zip(base, "data.yaml")).parent.mkdir(parents=True, exist_ok=True)
                    (out_root / join_zip(base, "data.yaml")).write_text(yaml_str, encoding="utf-8")

                    # Phase 2: zip packaging with progress
                    files = [fp for fp in out_root.rglob("*") if fp.is_file()]
                    total_files = len(files)
                    if total_files == 0:
                        raise RuntimeError("輸出資料夾沒有任何檔案可打包（可能全部被略過）")

                    if out_zip_path.exists():
                        out_zip_path.unlink()

                    self._set_progress(0, total_files, f"📦 打包 zip... (0/{total_files})")
                    with zipfile.ZipFile(out_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as outzf:
                        for j, fp in enumerate(files, start=1):
                            rel = fp.relative_to(out_root).as_posix()
                            outzf.write(fp, arcname=rel)
                            if j % 20 == 0 or j == total_files:
                                self._set_progress(j, total_files, f"📦 打包 zip... ({j}/{total_files})")

            self._set_progress(1, 1, f"完成：輸出 {out_zip_path}")
            messagebox.showinfo("完成", f"已輸出：\n{out_zip_path}")

        except Exception as e:
            self._set_progress(0, 1, "失敗")
            messagebox.showerror("錯誤", str(e))

    def _set_progress(self, cur: int, total: int, msg: str):
        def _ui():
            self.pbar["maximum"] = max(total, 1)
            self.pbar["value"] = min(cur, total)
            self.progress_text.set(f"{cur}/{total}")
            self.status.set(msg)
        self.root.after(0, _ui)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
