from __future__ import annotations

import re
import shutil
import tempfile
import threading
import zipfile
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

try:
    import yaml
except Exception:
    yaml = None


def _parse_yaml_fallback(text: str) -> dict:
    data: dict = {}
    lines = []
    for ln in text.splitlines():
        ln = ln.split("#", 1)[0].rstrip("\n")
        if ln.strip():
            lines.append(ln)

    i = 0
    while i < len(lines):
        ln = lines[i]
        if ":" not in ln:
            i += 1
            continue
        key, rest = ln.split(":", 1)
        key = key.strip()
        rest = rest.strip()
        if key == "names" and rest == "":
            names_map = {}
            i += 1
            while i < len(lines):
                sub = lines[i]
                if re.match(r"^\s+\d+\s*:\s*", sub):
                    m = re.match(r"^\s+(\d+)\s*:\s*(.+)$", sub)
                    if m:
                        names_map[int(m.group(1))] = m.group(2).strip().strip("'\"")
                    i += 1
                else:
                    break
            if names_map:
                data["names"] = [names_map[k] for k in sorted(names_map.keys())]
            continue
        if key == "names" and rest.startswith("["):
            inside = rest[1:-1]
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
            lines.append(f"names: [{', '.join([repr(x) for x in v])}]")
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines) + "\n"


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def norm_zip_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def join_zip(base: str, rel: str) -> str:
    base = norm_zip_path(base)
    rel = norm_zip_path(rel)
    if not base:
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
    path = norm_zip_path(images_dir)
    if "/images/" in path:
        return path.replace("/images/", "/labels/")
    if path.endswith("/images"):
        return path[:-len("/images")] + "/labels"
    if path.startswith("images/"):
        return path.replace("images/", "labels/", 1)
    return path.replace("images", "labels")


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def list_images_in_zip(all_names: set[str], images_dir: str) -> list[str]:
    prefix = norm_zip_path(images_dir).rstrip("/") + "/"
    imgs = []
    for name in all_names:
        nn = norm_zip_path(name)
        if nn.startswith(prefix) and nn.lower().endswith(IMG_EXTS):
            imgs.append(nn)
    imgs.sort()
    return imgs


def find_label_for_image(all_names: set[str], labels_dir: str, image_member: str) -> Optional[str]:
    stem = Path(image_member).stem
    cand = norm_zip_path(labels_dir).rstrip("/") + "/" + stem + ".txt"
    return cand if cand in all_names else None


class ReorderListWidget(QListWidget):
    order_changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QListWidget.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)

    def dropEvent(self, event) -> None:
        super().dropEvent(event)
        self.order_changed.emit()


class MainWindow(QMainWindow):
    """YOLO zip label remapper GUI."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("YOLO Zip Label Remapper v3")
        self.resize(1020, 760)
        self.setMinimumSize(900, 640)

        self.data_yaml_member: Optional[str] = None
        self.data_yaml_base = ""
        self.orig_yaml: dict = {}
        self.orig_names: list[str] = []
        self.items: list[dict] = []

        self._worker: Optional[threading.Thread] = None
        self._running = False
        self._status_buffer: list[str] = []
        self._progress_cur = 0
        self._progress_total = 1
        self._lock = threading.Lock()
        self._finish_dialog: Optional[tuple[str, str, str]] = None

        self._build_ui()

        self._timer = QTimer(self)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._flush_ui)
        self._timer.start()

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        path_box = QGroupBox("輸入與輸出", root)
        path_layout = QFormLayout(path_box)

        row_zip = QWidget(path_box)
        row_zip_l = QHBoxLayout(row_zip)
        row_zip_l.setContentsMargins(0, 0, 0, 0)
        self.ent_zip = QLineEdit(row_zip)
        btn_zip = QPushButton("選擇 Zip", row_zip)
        btn_zip.clicked.connect(self.pick_zip)
        row_zip_l.addWidget(self.ent_zip, 1)
        row_zip_l.addWidget(btn_zip)
        path_layout.addRow("輸入 Zip:", row_zip)

        row_out_dir = QWidget(path_box)
        row_out_dir_l = QHBoxLayout(row_out_dir)
        row_out_dir_l.setContentsMargins(0, 0, 0, 0)
        self.ent_out_dir = QLineEdit(row_out_dir)
        btn_out = QPushButton("選擇資料夾", row_out_dir)
        btn_out.clicked.connect(self.pick_out_dir)
        row_out_dir_l.addWidget(self.ent_out_dir, 1)
        row_out_dir_l.addWidget(btn_out)
        path_layout.addRow("輸出資料夾（空白=目前目錄）:", row_out_dir)

        self.ent_out_name = QLineEdit("converted_dataset.zip", path_box)
        path_layout.addRow("輸出 zip 檔名:", self.ent_out_name)
        layout.addWidget(path_box)

        self.chk_keep_unlabeled = QCheckBox(
            "保留沒有標註/過濾後為空的圖片（會產生空的 .txt label）",
            root,
        )
        self.chk_keep_unlabeled.setChecked(True)
        layout.addWidget(self.chk_keep_unlabeled)

        mid = QWidget(root)
        mid_l = QHBoxLayout(mid)
        mid_l.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(mid, 1)

        left = QGroupBox("Labels（拖曳改順序；可刪除；可改名）", mid)
        left_l = QVBoxLayout(left)
        self.listbox = ReorderListWidget(left)
        self.listbox.currentRowChanged.connect(self.on_select)
        self.listbox.order_changed.connect(self._sync_items_from_view)
        left_l.addWidget(self.listbox, 1)

        row_left_btn = QWidget(left)
        row_left_btn_l = QHBoxLayout(row_left_btn)
        row_left_btn_l.setContentsMargins(0, 0, 0, 0)
        btn_del = QPushButton("刪除選取 label", row_left_btn)
        btn_del.clicked.connect(self.delete_selected)
        btn_clear = QPushButton("清空列表", row_left_btn)
        btn_clear.clicked.connect(self.clear_all)
        btn_refresh = QPushButton("重新整理顯示", row_left_btn)
        btn_refresh.clicked.connect(self.refresh_display)
        row_left_btn_l.addWidget(btn_del)
        row_left_btn_l.addWidget(btn_clear)
        row_left_btn_l.addWidget(btn_refresh)
        left_l.addWidget(row_left_btn)
        mid_l.addWidget(left, 2)

        right = QGroupBox("編輯 label 名稱", mid)
        right_l = QVBoxLayout(right)
        self.lbl_sel = QLabel("(無)", right)
        right_l.addWidget(self.lbl_sel)
        self.ent_rename = QLineEdit(right)
        right_l.addWidget(self.ent_rename)
        btn_apply = QPushButton("套用改名", right)
        btn_apply.clicked.connect(self.apply_rename)
        right_l.addWidget(btn_apply)
        right_l.addStretch(1)
        mid_l.addWidget(right, 1)

        row_bottom = QWidget(root)
        row_bottom_l = QHBoxLayout(row_bottom)
        row_bottom_l.setContentsMargins(0, 0, 0, 0)
        self.progress = QProgressBar(row_bottom)
        self.progress_text = QLabel("0/0", row_bottom)
        self.btn_run = QPushButton("一鍵處理並輸出 Zip", row_bottom)
        self.btn_run.clicked.connect(self.run)
        row_bottom_l.addWidget(self.progress, 1)
        row_bottom_l.addWidget(self.progress_text)
        row_bottom_l.addWidget(self.btn_run)
        layout.addWidget(row_bottom)

        self.lbl_status = QLabel("尚未載入 dataset", root)
        layout.addWidget(self.lbl_status)

    def _post_status(self, text: str) -> None:
        with self._lock:
            self._status_buffer.append(text)

    def _set_progress(self, cur: int, total: int) -> None:
        with self._lock:
            self._progress_cur = cur
            self._progress_total = max(1, total)

    def _flush_ui(self) -> None:
        with self._lock:
            msgs = self._status_buffer[:]
            self._status_buffer.clear()
            cur = self._progress_cur
            total = self._progress_total
        if msgs:
            self.lbl_status.setText(msgs[-1])
        self.progress.setRange(0, total)
        self.progress.setValue(min(cur, total))
        self.progress_text.setText(f"{cur}/{total}")
        if self._worker and (not self._worker.is_alive()):
            if self._running or self._finish_dialog is not None:
                self._running = False
                self.btn_run.setEnabled(True)
                if self._finish_dialog is not None:
                    level, title, message = self._finish_dialog
                    self._finish_dialog = None
                    if level == "error":
                        QMessageBox.critical(self, title, message)
                    elif level == "warn":
                        QMessageBox.warning(self, title, message)
                    else:
                        QMessageBox.information(self, title, message)

    def pick_zip(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "選擇 Zip", str(Path.cwd()), "Zip files (*.zip)")
        if not path:
            return
        self.ent_zip.setText(path)
        self.load_yaml_from_zip(Path(path))

    def pick_out_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "選擇輸出資料夾", str(Path.cwd()))
        if path:
            self.ent_out_dir.setText(path)

    def load_yaml_from_zip(self, zpath: Path) -> None:
        try:
            with zipfile.ZipFile(zpath, "r") as zf:
                member = find_data_yaml_in_zip(zf)
                text = zip_read_text(zf, member)
            data = parse_data_yaml(text)
            names = data.get("names")
            if not isinstance(names, list) or not names:
                raise ValueError("data.yaml 解析不到 names")

            self.data_yaml_member = member
            base = norm_zip_path(str(Path(member).parent))
            self.data_yaml_base = "" if base in ("", ".", "./") else base
            self.orig_yaml = data
            self.orig_names = names[:]
            self.items = [{"old_id": i, "name": n} for i, n in enumerate(names)]
            self.refresh_display()
            self._post_status(f"已載入：{zpath.name} | data.yaml: {member} | classes: {len(names)}")
        except Exception as exc:
            QMessageBox.critical(self, "錯誤", str(exc))

    def refresh_display(self) -> None:
        self.listbox.clear()
        for new_id, item in enumerate(self.items):
            old_id = item["old_id"]
            name = item["name"]
            text = f"{old_id:>3}  {name}   ->   {new_id}"
            list_item = QListWidgetItem(text)
            list_item.setData(Qt.UserRole, {"old_id": old_id, "name": name})
            self.listbox.addItem(list_item)
        if self.listbox.count() > 0:
            self.listbox.setCurrentRow(0)
        self._set_progress(0, 1)

    def _sync_items_from_view(self) -> None:
        new_items = []
        for i in range(self.listbox.count()):
            item = self.listbox.item(i)
            data = item.data(Qt.UserRole) or {}
            old_id = int(data.get("old_id"))
            name = str(data.get("name"))
            new_items.append({"old_id": old_id, "name": name})
        self.items = new_items
        self.refresh_display()

    def on_select(self, row: int) -> None:
        if row < 0 or row >= len(self.items):
            self.lbl_sel.setText("(無)")
            self.ent_rename.clear()
            return
        item = self.items[row]
        self.lbl_sel.setText(f"old {item['old_id']} | name '{item['name']}' | new id {row}")
        self.ent_rename.setText(item["name"])

    def delete_selected(self) -> None:
        row = self.listbox.currentRow()
        if row < 0:
            return
        del self.items[row]
        self.refresh_display()

    def clear_all(self) -> None:
        if QMessageBox.question(self, "確認", "確定要清空列表嗎？") != QMessageBox.Yes:
            return
        self.items = []
        self.refresh_display()

    def apply_rename(self) -> None:
        row = self.listbox.currentRow()
        if row < 0:
            return
        new_name = self.ent_rename.text().strip()
        if not new_name:
            QMessageBox.warning(self, "提示", "新名稱不能是空白")
            return
        self.items[row]["name"] = new_name
        self.refresh_display()
        self.listbox.setCurrentRow(row)

    def run(self) -> None:
        if self._worker and self._worker.is_alive():
            QMessageBox.warning(self, "執行中", "目前正在處理中。")
            return
        if not self.ent_zip.text().strip():
            QMessageBox.critical(self, "錯誤", "請先選擇輸入 zip")
            return
        if not self.data_yaml_member or not self.orig_names:
            QMessageBox.critical(self, "錯誤", "尚未成功載入 data.yaml")
            return
        if len(self.items) == 0:
            QMessageBox.critical(self, "錯誤", "labels 列表是空的")
            return

        out_dir = Path(self.ent_out_dir.text().strip()) if self.ent_out_dir.text().strip() else Path.cwd()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = self.ent_out_name.text().strip() or "converted_dataset.zip"
        if not out_name.lower().endswith(".zip"):
            out_name += ".zip"
        out_zip_path = out_dir / out_name

        new_names = [it["name"] for it in self.items]
        kept_old_ids = [it["old_id"] for it in self.items]
        removed = set(range(len(self.orig_names))) - set(kept_old_ids)
        id_map = {old_id: new_id for new_id, old_id in enumerate(kept_old_ids)}

        self.btn_run.setEnabled(False)
        self._running = True
        self._worker = threading.Thread(
            target=self._process_zip_worker,
            args=(
                Path(self.ent_zip.text().strip()),
                out_zip_path,
                id_map,
                removed,
                new_names,
                self.chk_keep_unlabeled.isChecked(),
            ),
            daemon=True,
        )
        self._worker.start()

    def _process_zip_worker(
        self,
        zip_path: Path,
        out_zip_path: Path,
        id_map: dict,
        removed: set,
        new_names: list,
        keep_unlabeled: bool,
    ) -> None:
        try:
            self._set_progress(0, 1)
            self._post_status("準備中...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                data = parse_data_yaml(zip_read_text(zf, self.data_yaml_member))
                base = self.data_yaml_base

                train_rel = str(data.get("train", "images/train"))
                val_rel = str(data.get("val", "images/val"))
                test_rel = str(data.get("test", "images/test"))
                split_images = {
                    "train": norm_zip_path(join_zip(base, train_rel)),
                    "val": norm_zip_path(join_zip(base, val_rel)),
                    "test": norm_zip_path(join_zip(base, test_rel)),
                }
                split_labels = {k: infer_labels_dir(v) for k, v in split_images.items()}

                all_names = set(map(norm_zip_path, zf.namelist()))
                split_image_members: dict[str, list[str]] = {}
                total_images = 0
                for split, img_dir in split_images.items():
                    imgs = list_images_in_zip(all_names, img_dir)
                    split_image_members[split] = imgs
                    total_images += len(imgs)
                if total_images == 0:
                    raise RuntimeError("找不到任何圖片，請確認 data.yaml 的路徑設定。")

                with tempfile.TemporaryDirectory() as td:
                    out_root = Path(td) / "dataset"
                    kept = 0
                    processed = 0
                    self._set_progress(0, total_images)

                    for split, imgs in split_image_members.items():
                        for img_member in imgs:
                            processed += 1
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
                                    if old_id in removed or old_id not in id_map:
                                        continue
                                    new_id = id_map[old_id]
                                    new_lines.append(str(new_id) + " " + " ".join(parts[1:]))

                            should_skip = (
                                (lbl_member is None and not keep_unlabeled)
                                or (lbl_member is not None and not new_lines and not keep_unlabeled)
                            )
                            if should_skip:
                                if processed % 10 == 0 or processed == total_images:
                                    self._post_status(f"處理中... ({processed}/{total_images}) | 已保留 {kept}")
                                    self._set_progress(processed, total_images)
                                continue

                            out_img_path = out_root / img_member
                            ensure_parent_dir(out_img_path)
                            with zf.open(img_member, "r") as src, open(out_img_path, "wb") as dst:
                                shutil.copyfileobj(src, dst)

                            out_lbl_path = out_root / split_labels[split] / (Path(img_member).stem + ".txt")
                            ensure_parent_dir(out_lbl_path)
                            out_lbl_path.write_text("\n".join(new_lines), encoding="utf-8")

                            kept += 1
                            if processed % 10 == 0 or processed == total_images:
                                self._post_status(f"處理中... ({processed}/{total_images}) | 已保留 {kept}")
                                self._set_progress(processed, total_images)

                    new_yaml = dict(data) if isinstance(data, dict) else {}
                    new_yaml["path"] = "."
                    new_yaml["names"] = new_names
                    new_yaml["nc"] = len(new_names)
                    new_yaml["train"] = train_rel
                    new_yaml["val"] = val_rel
                    new_yaml["test"] = test_rel

                    if yaml is not None:
                        yaml_str = yaml.safe_dump(new_yaml, allow_unicode=True, sort_keys=False)
                    else:
                        yaml_str = _dump_yaml_simple(new_yaml)
                    yaml_path = out_root / join_zip(base, "data.yaml")
                    yaml_path.parent.mkdir(parents=True, exist_ok=True)
                    yaml_path.write_text(yaml_str, encoding="utf-8")

                    files = [fp for fp in out_root.rglob("*") if fp.is_file()]
                    total_files = len(files)
                    if total_files == 0:
                        raise RuntimeError("輸出資料夾沒有任何檔案可打包。")
                    if out_zip_path.exists():
                        out_zip_path.unlink()

                    self._set_progress(0, total_files)
                    self._post_status("打包 zip 中...")
                    with zipfile.ZipFile(out_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as out_zf:
                        for i, fp in enumerate(files, start=1):
                            out_zf.write(fp, arcname=fp.relative_to(out_root).as_posix())
                            if i % 20 == 0 or i == total_files:
                                self._set_progress(i, total_files)
                                self._post_status(f"打包中... ({i}/{total_files})")

            self._set_progress(1, 1)
            self._post_status(f"完成：{out_zip_path}")
            self._finish_dialog = ("info", "完成", f"已輸出：\n{out_zip_path}")
        except Exception as exc:
            self._set_progress(0, 1)
            self._post_status("失敗")
            self._finish_dialog = ("error", "錯誤", str(exc))
        finally:
            self._running = False


def main() -> None:
    app = QApplication([])
    win = MainWindow()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
