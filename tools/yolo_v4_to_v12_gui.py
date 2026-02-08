from __future__ import annotations

import random
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Sample:
    img_path: Path
    label_path: Path
    category: str


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_yolo_label_lines(text: str) -> List[List[str]]:
    lines: List[List[str]] = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        lines.append(re.split(r"\s+", raw))
    return lines


def rewrite_label_class_ids(label_in: Path, label_out: Path, new_class_id: int) -> Tuple[bool, str]:
    try:
        txt = label_in.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        return False, f"讀取標註失敗: {label_in} ({exc})"

    lines = parse_yolo_label_lines(txt)
    out_lines: List[str] = []
    for parts in lines:
        if not parts:
            continue
        parts[0] = str(new_class_id)
        out_lines.append(" ".join(parts))

    try:
        label_out.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    except Exception as exc:
        return False, f"寫入標註失敗: {label_out} ({exc})"
    return True, "OK"


def copy_label_keep_original(label_in: Path, label_out: Path) -> Tuple[bool, str]:
    try:
        shutil.copyfile(label_in, label_out)
        return True, "OK"
    except Exception as exc:
        return False, f"複製標註失敗: {label_in} -> {label_out} ({exc})"


def generate_unique_name(mode: str, stem: str, ext: str, used: set[str]) -> str:
    if mode == "uuid":
        while True:
            name = f"{uuid.uuid4().hex}{ext}"
            if name not in used:
                used.add(name)
                return name

    candidate = f"{stem}{ext}"
    if candidate not in used:
        used.add(candidate)
        return candidate

    i = 1
    while True:
        candidate = f"{stem}_{i}{ext}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        i += 1


def find_samples(input_root: Path, logger) -> List[Sample]:
    samples: List[Sample] = []
    if not input_root.exists():
        return samples
    for img_path in input_root.rglob("*"):
        if not img_path.is_file() or img_path.suffix.lower() not in IMG_EXTS:
            continue
        label_path = img_path.with_suffix(".txt")
        if not label_path.exists():
            logger(f"[略過] 找不到標註: {img_path}")
            continue
        try:
            rel = img_path.relative_to(input_root)
            category = rel.parts[0] if len(rel.parts) >= 2 else "unknown"
        except Exception:
            category = "unknown"
        samples.append(Sample(img_path=img_path, label_path=label_path, category=category))
    return samples


def split_counts(total: int, p_train: float, p_val: float, p_test: float) -> Tuple[int, int, int]:
    train = int(total * p_train / 100.0)
    val = int(total * p_val / 100.0)
    test = int(total * p_test / 100.0)
    rem = total - (train + val + test)
    while rem > 0:
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
    if train + val + test > total:
        overflow = train + val + test - total
        t = min(test, overflow)
        test -= t
        overflow -= t
        v = min(val, overflow)
        val -= v
        overflow -= v
        train = max(0, train - overflow)
    return train, val, test


def write_data_yaml(out_dir: Path, class_names: List[str]) -> None:
    yaml_text = (
        f"path: {out_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
    )
    for i, name in enumerate(class_names):
        yaml_text += f"  {i}: {name}\n"
    (out_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")


class MainWindow(QMainWindow):
    """YOLOv4 Darknet to YOLOv12 dataset converter GUI."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("YOLOv4 Darknet 資料集 -> YOLOv12 格式轉換器")
        self.resize(980, 720)
        self.setMinimumSize(860, 620)

        self._worker: Optional[threading.Thread] = None
        self._stop_flag = False
        self._log_queue: List[str] = []
        self._lock = threading.Lock()
        self._progress_now = 0
        self._progress_max = 100
        self._running = False
        self._finish_dialog: Optional[tuple[str, str, str]] = None

        self._build_ui()

        self._ui_timer = QTimer(self)
        self._ui_timer.setInterval(120)
        self._ui_timer.timeout.connect(self._flush_ui_events)
        self._ui_timer.start()

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        path_box = QGroupBox("路徑設定", root)
        path_layout = QFormLayout(path_box)

        row_input = QWidget(path_box)
        row_input_l = QHBoxLayout(row_input)
        row_input_l.setContentsMargins(0, 0, 0, 0)
        self.ent_input = QLineEdit(row_input)
        btn_input = QPushButton("瀏覽...", row_input)
        btn_input.clicked.connect(self.browse_input)
        row_input_l.addWidget(self.ent_input, 1)
        row_input_l.addWidget(btn_input)
        path_layout.addRow("來源資料夾 (TaskFire):", row_input)

        row_output = QWidget(path_box)
        row_output_l = QHBoxLayout(row_output)
        row_output_l.setContentsMargins(0, 0, 0, 0)
        self.ent_output = QLineEdit(row_output)
        btn_output = QPushButton("瀏覽...", row_output)
        btn_output.clicked.connect(self.browse_output)
        row_output_l.addWidget(self.ent_output, 1)
        row_output_l.addWidget(btn_output)
        path_layout.addRow("輸出資料夾:", row_output)
        layout.addWidget(path_box)

        opt_box = QGroupBox("轉換設定", root)
        opt_layout = QFormLayout(opt_box)

        split_row = QWidget(opt_box)
        split_row_l = QHBoxLayout(split_row)
        split_row_l.setContentsMargins(0, 0, 0, 0)
        self.ent_train = QLineEdit("80", split_row)
        self.ent_val = QLineEdit("10", split_row)
        self.ent_test = QLineEdit("10", split_row)
        split_row_l.addWidget(QLabel("train (%):", split_row))
        split_row_l.addWidget(self.ent_train)
        split_row_l.addWidget(QLabel("val (%):", split_row))
        split_row_l.addWidget(self.ent_val)
        split_row_l.addWidget(QLabel("test (%):", split_row))
        split_row_l.addWidget(self.ent_test)
        split_row_l.addStretch(1)
        opt_layout.addRow(split_row)

        row_seed = QWidget(opt_box)
        row_seed_l = QHBoxLayout(row_seed)
        row_seed_l.setContentsMargins(0, 0, 0, 0)
        self.chk_shuffle = QCheckBox("先打亂資料集 (shuffle)", row_seed)
        self.chk_shuffle.setChecked(True)
        self.ent_seed = QLineEdit("42", row_seed)
        self.ent_seed.setMaximumWidth(120)
        row_seed_l.addWidget(self.chk_shuffle)
        row_seed_l.addWidget(QLabel("seed:", row_seed))
        row_seed_l.addWidget(self.ent_seed)
        row_seed_l.addStretch(1)
        opt_layout.addRow(row_seed)

        rename_box = QGroupBox("檔名處理", opt_box)
        rename_layout = QVBoxLayout(rename_box)
        self.rd_uuid = QRadioButton("全部改成 UUID", rename_box)
        self.rd_avoid = QRadioButton("保留原名，遇到重複自動避重", rename_box)
        self.rd_uuid.setChecked(True)
        rename_layout.addWidget(self.rd_uuid)
        rename_layout.addWidget(self.rd_avoid)
        self.rename_group = QButtonGroup(rename_box)
        self.rename_group.addButton(self.rd_uuid)
        self.rename_group.addButton(self.rd_avoid)
        opt_layout.addRow(rename_box)

        class_box = QGroupBox("類別來源 / 標註處理", opt_box)
        class_layout = QVBoxLayout(class_box)
        self.rd_folder = QRadioButton("用最上層資料夾當類別，並重寫每行 class_id", class_box)
        self.rd_keep = QRadioButton("保留原 txt 內容（不改 class_id），只搬移/改名", class_box)
        self.rd_folder.setChecked(True)
        class_layout.addWidget(self.rd_folder)
        class_layout.addWidget(self.rd_keep)
        self.class_group = QButtonGroup(class_box)
        self.class_group.addButton(self.rd_folder)
        self.class_group.addButton(self.rd_keep)
        opt_layout.addRow(class_box)
        layout.addWidget(opt_box)

        act_row = QWidget(root)
        act_row_l = QHBoxLayout(act_row)
        act_row_l.setContentsMargins(0, 0, 0, 0)
        self.btn_start = QPushButton("開始轉換", act_row)
        self.btn_start.clicked.connect(self.start)
        self.btn_stop = QPushButton("停止", act_row)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_stop.setEnabled(False)
        self.progress = QProgressBar(act_row)
        self.progress.setRange(0, 100)
        act_row_l.addWidget(self.btn_start)
        act_row_l.addWidget(self.btn_stop)
        act_row_l.addWidget(self.progress, 1)
        layout.addWidget(act_row)

        self.txt_log = QPlainTextEdit(root)
        self.txt_log.setReadOnly(True)
        layout.addWidget(self.txt_log, 1)
        self.log("就緒。選擇來源資料夾(TaskFire)與輸出資料夾後開始。")

    def log(self, text: str) -> None:
        with self._lock:
            self._log_queue.append(f"[{time.strftime('%H:%M:%S')}] {text}")

    def _flush_ui_events(self) -> None:
        with self._lock:
            lines = self._log_queue[:]
            self._log_queue.clear()
            p_now = self._progress_now
            p_max = self._progress_max
        for line in lines:
            self.txt_log.appendPlainText(line)
        self.progress.setRange(0, max(1, p_max))
        self.progress.setValue(min(p_now, p_max))
        if self._worker and (not self._worker.is_alive()):
            if self._running or self._finish_dialog is not None:
                self._running = False
                self.btn_start.setEnabled(True)
                self.btn_stop.setEnabled(False)
                if self._finish_dialog is not None:
                    level, title, message = self._finish_dialog
                    self._finish_dialog = None
                    if level == "error":
                        QMessageBox.critical(self, title, message)
                    elif level == "warn":
                        QMessageBox.warning(self, title, message)
                    else:
                        QMessageBox.information(self, title, message)

    def browse_input(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "選擇來源資料夾 (TaskFire)", str(Path.cwd()))
        if path:
            self.ent_input.setText(path)

    def browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "選擇輸出資料夾", str(Path.cwd()))
        if path:
            self.ent_output.setText(path)

    def stop(self) -> None:
        self._stop_flag = True
        self.log("已要求停止：會在下一個檔案處理點中止。")

    def _set_progress(self, value: int, maximum: int) -> None:
        with self._lock:
            self._progress_now = value
            self._progress_max = maximum

    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            QMessageBox.warning(self, "執行中", "目前正在執行轉換。")
            return

        input_root = Path(self.ent_input.text().strip())
        output_root = Path(self.ent_output.text().strip())
        if not input_root.exists():
            QMessageBox.critical(self, "錯誤", "來源資料夾不存在。")
            return
        if not output_root.exists():
            QMessageBox.critical(self, "錯誤", "輸出資料夾不存在。")
            return

        try:
            p_train = float(self.ent_train.text().strip() or "0")
            p_val = float(self.ent_val.text().strip() or "0")
            p_test = float(self.ent_test.text().strip() or "0")
        except Exception:
            QMessageBox.critical(self, "錯誤", "train/val/test 請輸入數字百分比。")
            return
        if p_train < 0 or p_val < 0 or p_test < 0:
            QMessageBox.critical(self, "錯誤", "百分比不可為負數。")
            return
        if p_train + p_val + p_test > 100.0 + 1e-9:
            QMessageBox.critical(self, "錯誤", "train + val + test 總和不可超過 100%。")
            return

        try:
            seed = int(self.ent_seed.text().strip() or "42")
        except Exception:
            QMessageBox.critical(self, "錯誤", "seed 請輸入整數。")
            return

        rename_mode = "uuid" if self.rd_uuid.isChecked() else "avoid"
        class_mode = "folder" if self.rd_folder.isChecked() else "keep"
        do_shuffle = self.chk_shuffle.isChecked()

        self._stop_flag = False
        self._set_progress(0, 100)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._running = True

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
    ) -> None:
        try:
            out_dir = output_root / f"yolo12_dataset_{time.strftime('%Y%m%d_%H%M%S')}"
            safe_mkdir(out_dir)
            for split in ("train", "val", "test"):
                safe_mkdir(out_dir / "images" / split)
                safe_mkdir(out_dir / "labels" / split)

            self.log(f"輸出資料夾：{out_dir}")
            self.log("掃描資料集中...")
            samples = find_samples(input_root, self.log)
            if not samples:
                raise RuntimeError("找不到任何有效樣本（圖片+標註）。")
            self.log(f"找到 {len(samples)} 組樣本。")

            class_names: List[str]
            class_map: Dict[str, int]
            if class_mode == "folder":
                cats = sorted({s.category for s in samples})
                class_names = cats
                class_map = {c: i for i, c in enumerate(class_names)}
                self.log(f"類別: {class_names}")
            else:
                class_names = ["class0"]
                class_map = {}
                self.log("類別模式：保留原 class_id")

            idxs = list(range(len(samples)))
            if do_shuffle:
                rnd = random.Random(seed)
                rnd.shuffle(idxs)
                self.log(f"已打亂資料集（seed={seed}）。")
            else:
                self.log("未打亂資料集。")

            total = len(samples)
            n_train, n_val, n_test = split_counts(total, p_train, p_val, p_test)
            self.log(f"切分數量: train={n_train}, val={n_val}, test={n_test}, total={total}")

            split_tags = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
            if len(split_tags) < total:
                rem = total - len(split_tags)
                while rem > 0:
                    split_tags.append("train")
                    rem -= 1
                    if rem <= 0:
                        break
                    split_tags.append("val")
                    rem -= 1
                    if rem <= 0:
                        break
                    split_tags.append("test")
                    rem -= 1
            split_tags = split_tags[:total]

            used_names: set[str] = set()
            ok_cnt = 0
            fail_cnt = 0
            self._set_progress(0, total)

            for i, (sample_idx, split) in enumerate(zip(idxs, split_tags), start=1):
                if self._stop_flag:
                    self.log("已停止。")
                    self._finish_dialog = ("info", "轉換訓練資料", "已停止轉換")
                    break

                sample = samples[sample_idx]
                ext = sample.img_path.suffix.lower()
                new_img_name = generate_unique_name(rename_mode, sample.img_path.stem, ext, used_names)
                new_lbl_name = Path(new_img_name).with_suffix(".txt").name

                dst_img = out_dir / "images" / split / new_img_name
                dst_lbl = out_dir / "labels" / split / new_lbl_name

                try:
                    shutil.copyfile(sample.img_path, dst_img)
                except Exception as exc:
                    self.log(f"[失敗] 複製圖片: {sample.img_path} -> {dst_img} ({exc})")
                    fail_cnt += 1
                    self._set_progress(i, total)
                    continue

                if class_mode == "folder":
                    new_id = class_map.get(sample.category, 0)
                    ok, msg = rewrite_label_class_ids(sample.label_path, dst_lbl, new_id)
                else:
                    ok, msg = copy_label_keep_original(sample.label_path, dst_lbl)
                if not ok:
                    self.log(f"[失敗] 標註處理: {msg}")
                    try:
                        dst_img.unlink(missing_ok=True)
                    except Exception:
                        pass
                    fail_cnt += 1
                else:
                    ok_cnt += 1

                if i % 50 == 0 or i == total:
                    self.log(f"進度：{i}/{total}（成功 {ok_cnt}，失敗 {fail_cnt}）")
                self._set_progress(i, total)

            write_data_yaml(out_dir, class_names)
            self.log("已產生 data.yaml。")
            self.log(f"完成：成功 {ok_cnt}，失敗 {fail_cnt}")
            self.log(f"YAML：{out_dir / 'data.yaml'}")
            if self._finish_dialog is None:
                self._finish_dialog = ("info", "轉換訓練資料", "已完成轉換")
        except Exception as exc:
            self.log(f"[錯誤] {exc}")
            self._finish_dialog = ("error", "轉換訓練資料", f"轉換失敗：{exc}")
        finally:
            self._running = False


def main() -> None:
    app = QApplication([])
    win = MainWindow()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
