from __future__ import annotations

import shutil
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Optional

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")
VAL_KEYS = ("val", "valid", "validation", "vaild")


class MergeStoppedError(Exception):
    """使用者主動停止合併。"""


def _path_text(path: Path) -> str:
    return path.resolve().as_posix()


def _norm_text_path(text: str) -> str:
    return text.strip().replace("\\", "/")


@dataclass
class DataYamlModel:
    """YOLO data.yaml 標準模型。"""

    path: str
    train: list[str]
    val: list[str]
    test: list[str]
    names: list[str]

    @classmethod
    def from_mapping(cls, src: Mapping[str, object]) -> "DataYamlModel":
        """由 mapping 建立模型。"""
        path_value = str(src.get("path", ".") or ".")
        train_values = _to_str_list(src.get("train"))
        val_values = _to_str_list(_pick_first(src, VAL_KEYS))
        test_values = _to_str_list(src.get("test"))
        names = _parse_names(src.get("names"))
        if not names:
            raise ValueError("data.yaml 缺少 names 或 names 為空")
        return cls(path=path_value, train=train_values, val=val_values, test=test_values, names=names)


@dataclass
class DataSetPreview:
    """來源預覽資訊（只讀 data.yaml）。"""

    source_path: Path
    source_kind: str
    yaml_path: Optional[Path]
    yaml_member: Optional[str]
    names: list[str]
    dataset_path_value: str


@dataclass
class DataSetRuntimeSpec:
    """可用於實際合併的來源設定。"""

    source_path: Path
    yaml_path: Path
    yaml_dir: Path
    dataset_root: Path
    model: DataYamlModel


@dataclass
class PlannedSample:
    """盤點後待輸出的樣本。"""

    split: str
    image_path: Path
    label_path: Optional[Path]
    source_names_len: int
    output_name: str = ""


def _pick_first(src: Mapping[str, object], keys: Iterable[str]) -> object:
    for key in keys:
        if key in src:
            return src[key]
    return None


def _to_str_list(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        out: list[str] = []
        for item in raw:
            if item is None:
                continue
            out.append(str(item))
        return out
    return [str(raw)]


def _parse_names(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, dict):
        pairs: list[tuple[int, str]] = []
        for key, value in raw.items():
            try:
                idx = int(key)
            except Exception as exc:
                raise ValueError(f"names key 必須可轉成整數: {key}") from exc
            pairs.append((idx, str(value)))
        pairs.sort(key=lambda p: p[0])
        return [value for _, value in pairs]
    raise ValueError("names 必須是 list 或 dict")


def _find_data_yaml_in_folder(folder: Path) -> Optional[Path]:
    for name in ("data.yaml", "data.yml"):
        direct = folder / name
        if direct.exists():
            return direct.resolve()
    for p in folder.rglob("data.yaml"):
        return p.resolve()
    for p in folder.rglob("data.yml"):
        return p.resolve()
    return None


def _find_data_yaml_member_in_zip(zf: zipfile.ZipFile) -> str:
    candidates: list[str] = []
    for name in zf.namelist():
        low = name.lower()
        if low.endswith("data.yaml") or low.endswith("data.yml"):
            candidates.append(name)
    if not candidates:
        raise FileNotFoundError("zip 裡找不到 data.yaml/data.yml")
    candidates.sort(key=lambda s: (s.count("/"), len(s), s))
    return candidates[0]


def _read_yaml_model_from_text(text: str) -> DataYamlModel:
    if yaml is None:
        raise RuntimeError("需要安裝 PyYAML 才能讀取 data.yaml")
    obj = yaml.safe_load(text)
    if not isinstance(obj, dict):
        raise ValueError("data.yaml 內容不是 mapping")
    return DataYamlModel.from_mapping(obj)


def _read_preview_from_source(source_path: Path) -> DataSetPreview:
    source = source_path.resolve()
    if source.is_dir():
        yaml_path = _find_data_yaml_in_folder(source)
        if yaml_path is None:
            raise FileNotFoundError(f"資料夾找不到 data.yaml/data.yml: {_path_text(source)}")
        text = yaml_path.read_text(encoding="utf-8", errors="ignore")
        model = _read_yaml_model_from_text(text)
        return DataSetPreview(source, "folder", yaml_path, None, model.names, model.path)
    if source.is_file():
        ext = source.suffix.lower()
        if ext in {".yaml", ".yml"}:
            text = source.read_text(encoding="utf-8", errors="ignore")
            model = _read_yaml_model_from_text(text)
            return DataSetPreview(source, "yaml", source, None, model.names, model.path)
        if ext == ".zip":
            with zipfile.ZipFile(source, "r") as zf:
                member = _find_data_yaml_member_in_zip(zf)
                text = zf.read(member).decode("utf-8", errors="ignore")
            model = _read_yaml_model_from_text(text)
            return DataSetPreview(source, "zip", None, member.replace("\\", "/"), model.names, model.path)
    raise ValueError(f"不支援的來源: {_path_text(source)}（需為資料夾、yaml 或 zip）")


def _is_prefix_names(shorter: list[str], longer: list[str]) -> bool:
    if len(shorter) > len(longer):
        return False
    for idx, name in enumerate(shorter):
        if name != longer[idx]:
            return False
    return True


def _pick_canonical_names(all_names: list[list[str]]) -> list[str]:
    if not all_names:
        raise ValueError("未提供任何 names")
    max_len = max(len(x) for x in all_names)
    candidates = [x for x in all_names if len(x) == max_len]
    for candidate in candidates:
        if all(_is_prefix_names(other, candidate) for other in all_names):
            return candidate
    raise ValueError("names 比對失敗：必須同大小寫、同順序，僅允許前綴缺少")


def _resolve_dataset_root(yaml_path: Path, path_value: str) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p.resolve()
    return (yaml_path.parent / p).resolve()


def _resolve_entry_path(raw_entry: str, dataset_root: Path, yaml_dir: Path) -> Path:
    def _fallback_by_tail(base: Path, abs_path: Path) -> Optional[Path]:
        parts = list(abs_path.parts)
        start = 1 if parts and (parts[0].endswith("\\") or parts[0] == "/") else 0
        for idx in range(start, len(parts)):
            cand = (base.joinpath(*parts[idx:])).resolve()
            if cand.exists():
                return cand
        return None

    p = Path(raw_entry)
    if p.is_absolute():
        resolved = p.resolve()
        if resolved.exists():
            return resolved
        by_root_tail = _fallback_by_tail(dataset_root, p)
        if by_root_tail is not None:
            return by_root_tail
        by_yaml_tail = _fallback_by_tail(yaml_dir, p)
        if by_yaml_tail is not None:
            return by_yaml_tail
        return resolved
    by_root = (dataset_root / raw_entry).resolve()
    by_yaml = (yaml_dir / raw_entry).resolve()
    if by_root.exists():
        return by_root
    if by_yaml.exists():
        return by_yaml

    # Some exported datasets (e.g. Roboflow zips) may keep paths like
    # "../train/images" even when data.yaml is already at dataset root.
    # If direct resolution fails, trim leading ../ or ./ and retry.
    rel_txt = raw_entry.replace("\\", "/")
    trimmed = rel_txt
    changed = False
    while True:
        if trimmed.startswith("../"):
            trimmed = trimmed[3:]
            changed = True
            continue
        if trimmed.startswith("./"):
            trimmed = trimmed[2:]
            changed = True
            continue
        break
    if changed and trimmed:
        by_root_trim = (dataset_root / trimmed).resolve()
        if by_root_trim.exists():
            return by_root_trim
        by_yaml_trim = (yaml_dir / trimmed).resolve()
        if by_yaml_trim.exists():
            return by_yaml_trim
    return by_root


def _replace_images_with_labels(raw_entry: str) -> str:
    txt = raw_entry.replace("\\", "/")
    parts = txt.split("/")
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx].lower() == "images":
            parts[idx] = "labels"
            return "/".join(parts)
    return txt


def _iter_images_from_list_file(list_file: Path, dataset_root: Path, yaml_dir: Path) -> Iterator[Path]:
    text = list_file.read_text(encoding="utf-8", errors="ignore")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        p = Path(line)
        if p.is_absolute():
            img = p.resolve()
        else:
            a = (list_file.parent / line).resolve()
            b = (dataset_root / line).resolve()
            c = (yaml_dir / line).resolve()
            if a.exists():
                img = a
            elif b.exists():
                img = b
            else:
                img = c
        if img.is_file() and img.suffix.lower() in IMAGE_EXTS:
            yield img


def _iter_images(entry_path: Path, dataset_root: Path, yaml_dir: Path) -> Iterator[Path]:
    if entry_path.is_file() and entry_path.suffix.lower() == ".txt":
        yield from _iter_images_from_list_file(entry_path, dataset_root, yaml_dir)
        return
    if entry_path.is_file() and entry_path.suffix.lower() in IMAGE_EXTS:
        yield entry_path
        return
    if entry_path.is_dir():
        for p in entry_path.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                yield p


def _find_label_for_image(image_path: Path, image_root: Path, label_root: Optional[Path]) -> Optional[Path]:
    candidates: list[Path] = []
    if label_root is not None and image_root.is_dir():
        try:
            rel = image_path.relative_to(image_root)
            candidates.append((label_root / rel).with_suffix(".txt"))
        except Exception:
            pass
    parts = list(image_path.parts)
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx].lower() == "images":
            repl = parts[:]
            repl[idx] = "labels"
            candidates.append(Path(*repl).with_suffix(".txt"))
            break
    candidates.append(image_path.with_suffix(".txt"))
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    return None


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


class MainWindow(QMainWindow):
    """YOLO 資料集合併 GUI。"""

    def __init__(self) -> None:
        """初始化主視窗。"""
        super().__init__()
        self.setWindowTitle("YOLO 資料集合併工具")
        self.resize(1120, 760)
        self.setMinimumSize(940, 660)

        self._worker: Optional[threading.Thread] = None
        self._running = False
        self._stop_flag = False
        self._lock = threading.Lock()
        self._logs: list[str] = []
        self._progress_now = 0
        self._progress_max = 100
        self._progress_busy = False
        self._finish_dialog: Optional[tuple[str, str, str]] = None

        self._uid_counter = 0
        self._used_names: dict[str, set[str]] = {k: set() for k in SPLITS}

        self._build_ui()

        self._timer = QTimer(self)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._flush_ui)
        self._timer.start()

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        src_box = QGroupBox("資料集來源（支援資料夾 / data.yaml / zip）", root)
        src_layout = QVBoxLayout(src_box)
        btn_row = QWidget(src_box)
        btn_row_l = QHBoxLayout(btn_row)
        btn_row_l.setContentsMargins(0, 0, 0, 0)
        self.btn_add_files = QPushButton("加入檔案（yaml/zip）", btn_row)
        self.btn_add_folder = QPushButton("加入資料夾", btn_row)
        self.btn_remove = QPushButton("移除選取", btn_row)
        self.btn_clear = QPushButton("清空", btn_row)
        self.btn_add_files.clicked.connect(self.add_source_files)
        self.btn_add_folder.clicked.connect(self.add_source_folder)
        self.btn_remove.clicked.connect(self.remove_selected)
        self.btn_clear.clicked.connect(self.clear_all)
        btn_row_l.addWidget(self.btn_add_files)
        btn_row_l.addWidget(self.btn_add_folder)
        btn_row_l.addWidget(self.btn_remove)
        btn_row_l.addWidget(self.btn_clear)
        btn_row_l.addStretch(1)
        src_layout.addWidget(btn_row)

        self.list_source = QListWidget(src_box)
        src_layout.addWidget(self.list_source, 1)
        layout.addWidget(src_box, 1)

        out_box = QGroupBox("輸出與暫存設定", root)
        out_form = QFormLayout(out_box)

        row_out_dir = QWidget(out_box)
        row_out_dir_l = QHBoxLayout(row_out_dir)
        row_out_dir_l.setContentsMargins(0, 0, 0, 0)
        self.ent_out_dir = QLineEdit(_path_text(Path.cwd()), row_out_dir)
        btn_out_dir = QPushButton("瀏覽...", row_out_dir)
        btn_out_dir.clicked.connect(self.pick_out_dir)
        row_out_dir_l.addWidget(self.ent_out_dir, 1)
        row_out_dir_l.addWidget(btn_out_dir)
        out_form.addRow("輸出根目錄:", row_out_dir)

        self.ent_out_name = QLineEdit(f"merged_dataset_{time.strftime('%Y%m%d_%H%M%S')}", out_box)
        out_form.addRow("輸出資料夾名稱:", self.ent_out_name)

        row_tmp_dir = QWidget(out_box)
        row_tmp_dir_l = QHBoxLayout(row_tmp_dir)
        row_tmp_dir_l.setContentsMargins(0, 0, 0, 0)
        default_tmp = Path.cwd() / ".tmp" / "yolo_dataset_merge"
        self.ent_temp_dir = QLineEdit(_path_text(default_tmp), row_tmp_dir)
        btn_tmp_dir = QPushButton("瀏覽...", row_tmp_dir)
        btn_tmp_dir.clicked.connect(self.pick_temp_dir)
        row_tmp_dir_l.addWidget(self.ent_temp_dir, 1)
        row_tmp_dir_l.addWidget(btn_tmp_dir)
        out_form.addRow("暫存資料夾:", row_tmp_dir)

        self.cmb_output_mode = QComboBox(out_box)
        self.cmb_output_mode.addItems(["只輸出資料夾", "只輸出 zip", "同時輸出資料夾與 zip"])
        self.cmb_output_mode.setCurrentIndex(2)
        out_form.addRow("輸出模式:", self.cmb_output_mode)

        self.ent_zip_name = QLineEdit("merged_dataset.zip", out_box)
        out_form.addRow("zip 檔名:", self.ent_zip_name)
        layout.addWidget(out_box)

        naming_box = QGroupBox("檔名策略", root)
        naming_layout = QFormLayout(naming_box)

        self.rd_uid = QRadioButton("全部改為 UID", naming_box)
        self.rd_keep = QRadioButton("保留原名，衝突時補後綴", naming_box)
        self.rd_uid.setChecked(True)
        self.name_group = QButtonGroup(naming_box)
        self.name_group.addButton(self.rd_uid)
        self.name_group.addButton(self.rd_keep)
        row_mode = QWidget(naming_box)
        row_mode_l = QHBoxLayout(row_mode)
        row_mode_l.setContentsMargins(0, 0, 0, 0)
        row_mode_l.addWidget(self.rd_uid)
        row_mode_l.addWidget(self.rd_keep)
        row_mode_l.addStretch(1)
        naming_layout.addRow(row_mode)

        row_uid = QWidget(naming_box)
        row_uid_l = QHBoxLayout(row_uid)
        row_uid_l.setContentsMargins(0, 0, 0, 0)
        self.ent_uid_prefix = QLineEdit("img_", row_uid)
        self.sp_uid_width = QSpinBox(row_uid)
        self.sp_uid_width.setRange(1, 12)
        self.sp_uid_width.setValue(8)
        row_uid_l.addWidget(QLabel("UID 前綴:", row_uid))
        row_uid_l.addWidget(self.ent_uid_prefix)
        row_uid_l.addWidget(QLabel("位數:", row_uid))
        row_uid_l.addWidget(self.sp_uid_width)
        row_uid_l.addStretch(1)
        naming_layout.addRow(row_uid)

        row_dup = QWidget(naming_box)
        row_dup_l = QHBoxLayout(row_dup)
        row_dup_l.setContentsMargins(0, 0, 0, 0)
        self.ent_dup_token = QLineEdit("_dup", row_dup)
        self.sp_dup_start = QSpinBox(row_dup)
        self.sp_dup_start.setRange(0, 999999)
        self.sp_dup_start.setValue(1)
        self.sp_dup_pad = QSpinBox(row_dup)
        self.sp_dup_pad.setRange(0, 8)
        self.sp_dup_pad.setValue(0)
        row_dup_l.addWidget(QLabel("衝突後綴符號:", row_dup))
        row_dup_l.addWidget(self.ent_dup_token)
        row_dup_l.addWidget(QLabel("起始序號:", row_dup))
        row_dup_l.addWidget(self.sp_dup_start)
        row_dup_l.addWidget(QLabel("補零位數:", row_dup))
        row_dup_l.addWidget(self.sp_dup_pad)
        row_dup_l.addStretch(1)
        naming_layout.addRow(row_dup)
        layout.addWidget(naming_box)

        act_row = QWidget(root)
        act_row_l = QHBoxLayout(act_row)
        act_row_l.setContentsMargins(0, 0, 0, 0)
        self.btn_precheck = QPushButton("快速預檢", act_row)
        self.btn_precheck.clicked.connect(self.start_precheck)
        self.btn_start = QPushButton("開始讀取 + 合併", act_row)
        self.btn_start.clicked.connect(self.start_merge)
        self.btn_stop = QPushButton("停止", act_row)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_merge)
        self.progress = QProgressBar(act_row)
        self.progress.setRange(0, 100)
        act_row_l.addWidget(self.btn_precheck)
        act_row_l.addWidget(self.btn_start)
        act_row_l.addWidget(self.btn_stop)
        act_row_l.addWidget(self.progress, 1)
        layout.addWidget(act_row)

        self.txt_log = QPlainTextEdit(root)
        self.txt_log.setReadOnly(True)
        layout.addWidget(self.txt_log, 1)
        self.log("就緒。先加入來源，再按開始。讀取階段只解析 data.yaml。")

    def log(self, text: str) -> None:
        with self._lock:
            self._logs.append(f"[{time.strftime('%H:%M:%S')}] {text}")

    def _set_progress(self, value: int, maximum: int) -> None:
        with self._lock:
            self._progress_now = value
            self._progress_max = maximum
            self._progress_busy = False

    def _set_progress_busy(self) -> None:
        with self._lock:
            self._progress_busy = True

    def _flush_ui(self) -> None:
        with self._lock:
            logs = self._logs[:]
            self._logs.clear()
            now = self._progress_now
            maxv = self._progress_max
            busy = self._progress_busy
        for line in logs:
            self.txt_log.appendPlainText(line)
        if busy:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, max(1, maxv))
            self.progress.setValue(min(now, maxv))
        if self._worker and not self._worker.is_alive() and (self._running or self._finish_dialog):
            self._running = False
            self.btn_precheck.setEnabled(True)
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            if self._finish_dialog is not None:
                level, title, msg = self._finish_dialog
                self._finish_dialog = None
                if level == "error":
                    QMessageBox.critical(self, title, msg)
                elif level == "warn":
                    QMessageBox.warning(self, title, msg)
                else:
                    QMessageBox.information(self, title, msg)

    def add_source_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "選擇來源檔案",
            _norm_text_path(self.ent_out_dir.text()) or _path_text(Path.cwd()),
            "Dataset files (*.yaml *.yml *.zip);;All files (*.*)",
        )
        if not paths:
            return
        for path in paths:
            self._insert_source(Path(path))

    def add_source_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "選擇來源資料夾",
            _norm_text_path(self.ent_out_dir.text()) or _path_text(Path.cwd()),
        )
        if not folder:
            return
        self._insert_source(Path(folder))

    def _insert_source(self, source_path: Path) -> None:
        rpath = _path_text(source_path)
        for i in range(self.list_source.count()):
            if self.list_source.item(i).text() == rpath:
                return
        self.list_source.addItem(QListWidgetItem(rpath))

    def remove_selected(self) -> None:
        for item in self.list_source.selectedItems():
            row = self.list_source.row(item)
            self.list_source.takeItem(row)

    def clear_all(self) -> None:
        self.list_source.clear()

    def pick_out_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "選擇輸出根目錄",
            _norm_text_path(self.ent_out_dir.text()) or _path_text(Path.cwd()),
        )
        if path:
            self.ent_out_dir.setText(_norm_text_path(path))

    def pick_temp_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "選擇暫存資料夾",
            _norm_text_path(self.ent_temp_dir.text()) or _path_text(Path.cwd()),
        )
        if path:
            self.ent_temp_dir.setText(_norm_text_path(path))

    def stop_merge(self) -> None:
        self._stop_flag = True
        self.log("已要求停止，將於下一個處理點中止。")

    def _source_paths(self) -> list[Path]:
        out: list[Path] = []
        for i in range(self.list_source.count()):
            out.append(Path(self.list_source.item(i).text()))
        return out

    def start_precheck(self) -> None:
        if self._worker and self._worker.is_alive():
            QMessageBox.warning(self, "執行中", "目前已有作業在執行。")
            return
        if yaml is None:
            QMessageBox.critical(self, "錯誤", "找不到 PyYAML，請先安裝：pip install pyyaml")
            return
        sources = self._source_paths()
        if not sources:
            QMessageBox.critical(self, "錯誤", "請至少加入一個來源。")
            return
        self._stop_flag = False
        self._running = True
        self._set_progress_busy()
        self.btn_precheck.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._worker = threading.Thread(
            target=self._run_precheck,
            args=(sources,),
            daemon=True,
        )
        self._worker.start()

    def start_merge(self) -> None:
        if self._worker and self._worker.is_alive():
            QMessageBox.warning(self, "執行中", "目前已有作業在執行。")
            return
        if yaml is None:
            QMessageBox.critical(self, "錯誤", "找不到 PyYAML，請先安裝：pip install pyyaml")
            return
        sources = self._source_paths()
        if not sources:
            QMessageBox.critical(self, "錯誤", "請至少加入一個來源。")
            return

        out_root = Path(_norm_text_path(self.ent_out_dir.text()))
        if not out_root.exists():
            QMessageBox.critical(self, "錯誤", "輸出根目錄不存在。")
            return
        temp_root = Path(_norm_text_path(self.ent_temp_dir.text()))
        out_name = self.ent_out_name.text().strip() or f"merged_dataset_{time.strftime('%Y%m%d_%H%M%S')}"

        output_mode = int(self.cmb_output_mode.currentIndex())  # 0=folder,1=zip,2=both
        zip_name = self.ent_zip_name.text().strip()
        if output_mode in (1, 2) and not zip_name:
            zip_name = f"{out_name}.zip"
        if output_mode in (1, 2) and not zip_name.lower().endswith(".zip"):
            zip_name += ".zip"

        self._stop_flag = False
        self._running = True
        self._uid_counter = 0
        self._used_names = {k: set() for k in SPLITS}
        self._set_progress(0, 100)
        self.btn_precheck.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self._worker = threading.Thread(
            target=self._run_merge,
            args=(sources, out_root, out_name, temp_root, output_mode, zip_name),
            daemon=True,
        )
        self._worker.start()

    def _run_precheck(self, sources: list[Path]) -> None:
        try:
            self._set_progress_busy()
            self.log("快速預檢開始（只讀 data.yaml）...")
            previews: list[DataSetPreview] = []
            for idx, source in enumerate(sources, start=1):
                if self._stop_flag:
                    raise MergeStoppedError("使用者停止")
                preview = _read_preview_from_source(source)
                previews.append(preview)
                if preview.source_kind == "zip":
                    yaml_desc = f"{_path_text(preview.source_path)}!/{preview.yaml_member}"
                else:
                    yaml_desc = _path_text(preview.yaml_path) if preview.yaml_path else _path_text(preview.source_path)
                self.log(f"[{idx}/{len(sources)}] {yaml_desc}")
                self.log(f"  names={preview.names}")
                self.log(f"  path={preview.dataset_path_value}")
            canonical_names = _pick_canonical_names([p.names for p in previews])
            self.log(f"names 比對通過，主類別順序: {canonical_names}")
            self._finish_dialog = (
                "info",
                "快速預檢",
                f"預檢完成。\n來源數: {len(previews)}\n主 names: {canonical_names}",
            )
        except MergeStoppedError:
            self._finish_dialog = ("warn", "快速預檢", "已停止預檢作業")
        except Exception as exc:
            self.log(f"[錯誤] {exc}")
            self._finish_dialog = ("error", "快速預檢", f"預檢失敗：{exc}")
        finally:
            self._running = False
            self._set_progress(0, 100)

    def _next_output_name(self, split: str, original_name: str) -> str:
        used = self._used_names[split]
        ext = Path(original_name).suffix.lower()
        stem = Path(original_name).stem
        if self.rd_uid.isChecked():
            prefix = self.ent_uid_prefix.text().strip()
            width = int(self.sp_uid_width.value())
            while True:
                self._uid_counter += 1
                name = f"{prefix}{self._uid_counter:0{width}d}{ext}"
                if name not in used:
                    used.add(name)
                    return name
        candidate = f"{stem}{ext}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        token = self.ent_dup_token.text()
        start_no = int(self.sp_dup_start.value())
        pad = int(self.sp_dup_pad.value())
        n = start_no
        while True:
            no = f"{n:0{pad}d}" if pad > 0 else str(n)
            name = f"{stem}{token}{no}{ext}"
            if name not in used:
                used.add(name)
                return name
            n += 1

    def _ensure_output_tree(self, out_dir: Path) -> None:
        for split in SPLITS:
            _safe_mkdir(out_dir / "images" / split)
            _safe_mkdir(out_dir / "labels" / split)

    def _write_output_yaml(self, out_dir: Path, names: list[str]) -> Path:
        if yaml is None:
            raise RuntimeError("需要 PyYAML")
        return self._write_output_yaml_with_counts(
            out_dir,
            names,
            {"train": 0, "val": 0, "test": 0},
        )

    def _write_output_yaml_with_counts(
        self,
        out_dir: Path,
        names: list[str],
        counts: dict[str, int],
    ) -> Path:
        if yaml is None:
            raise RuntimeError("需要 PyYAML")
        doc = {
            "path": ".",
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": names,
            "nc": len(names),
            "train_count": int(counts.get("train", 0)),
            "val_count": int(counts.get("val", 0)),
            "test_count": int(counts.get("test", 0)),
        }
        ypath = out_dir / "data.yaml"
        ypath.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return ypath

    def _zip_output_dir(self, src_dir: Path, zip_path: Path) -> None:
        _safe_mkdir(zip_path.parent)
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file in src_dir.rglob("*"):
                if file.is_file():
                    zf.write(file, file.relative_to(src_dir))

    def _resolve_runtime_spec(
        self,
        preview: DataSetPreview,
        temp_run_dir: Path,
        source_index: int,
    ) -> DataSetRuntimeSpec:
        if preview.source_kind in {"folder", "yaml"}:
            if preview.yaml_path is None:
                raise RuntimeError("yaml_path 遺失")
            text = preview.yaml_path.read_text(encoding="utf-8", errors="ignore")
            model = _read_yaml_model_from_text(text)
            return DataSetRuntimeSpec(
                source_path=preview.source_path,
                yaml_path=preview.yaml_path.resolve(),
                yaml_dir=preview.yaml_path.parent.resolve(),
                dataset_root=_resolve_dataset_root(preview.yaml_path, model.path),
                model=model,
            )
        if preview.source_kind == "zip":
            if not preview.yaml_member:
                raise RuntimeError("zip 來源缺少 data.yaml member")
            extract_dir = temp_run_dir / f"src_{source_index:03d}_{preview.source_path.stem}"
            _safe_mkdir(extract_dir)
            self.log(f"解壓 zip: {_path_text(preview.source_path)} -> {_path_text(extract_dir)}")
            with zipfile.ZipFile(preview.source_path, "r") as zf:
                zf.extractall(extract_dir)
            yaml_path = (extract_dir / preview.yaml_member).resolve()
            text = yaml_path.read_text(encoding="utf-8", errors="ignore")
            model = _read_yaml_model_from_text(text)
            return DataSetRuntimeSpec(
                source_path=preview.source_path,
                yaml_path=yaml_path,
                yaml_dir=yaml_path.parent.resolve(),
                dataset_root=_resolve_dataset_root(yaml_path, model.path),
                model=model,
            )
        raise RuntimeError(f"未知來源類型: {preview.source_kind}")

    def _collect_samples_from_spec(
        self,
        spec: DataSetRuntimeSpec,
        seen_inputs: set[Path],
    ) -> list[PlannedSample]:
        planned: list[PlannedSample] = []
        split_entries: dict[str, list[str]] = {
            "train": spec.model.train,
            "val": spec.model.val,
            "test": spec.model.test,
        }
        for split in SPLITS:
            for raw_entry in split_entries[split]:
                if self._stop_flag:
                    raise MergeStoppedError("使用者停止")
                image_root = _resolve_entry_path(raw_entry, spec.dataset_root, spec.yaml_dir)
                self.log(
                    f"  split={split} raw={raw_entry.replace(chr(92), '/')} "
                    f"-> resolved={_path_text(image_root)}"
                )
                label_root_raw = _replace_images_with_labels(raw_entry)
                label_root = _resolve_entry_path(label_root_raw, spec.dataset_root, spec.yaml_dir)
                if not label_root.exists():
                    label_root = None
                    self.log(f"    labels={label_root_raw} -> resolved=（不存在）")
                else:
                    self.log(f"    labels={label_root_raw} -> resolved={_path_text(label_root)}")
                for image_path in _iter_images(image_root, spec.dataset_root, spec.yaml_dir):
                    if self._stop_flag:
                        raise MergeStoppedError("使用者停止")
                    rimg = image_path.resolve()
                    if rimg in seen_inputs:
                        continue
                    seen_inputs.add(rimg)
                    label_path = _find_label_for_image(rimg, image_root, label_root)
                    planned.append(
                        PlannedSample(
                            split=split,
                            image_path=rimg,
                            label_path=label_path,
                            source_names_len=len(spec.model.names),
                        )
                    )
        return planned

    def _assign_output_names(self, samples: list[PlannedSample]) -> None:
        total = len(samples)
        self._set_progress(0, max(1, total))
        self.log("階段 3/4：處理全域檔名衝突...")
        for idx, sample in enumerate(samples, start=1):
            if self._stop_flag:
                raise MergeStoppedError("使用者停止")
            sample.output_name = self._next_output_name(sample.split, sample.image_path.name)
            if idx % 200 == 0 or idx == total:
                self._set_progress(idx, total)
                self.log(f"  檔名決策 {idx}/{total}")

    def _write_samples(
        self,
        samples: list[PlannedSample],
        out_dir: Path,
    ) -> dict[str, int]:
        counters: dict[str, int] = {"train": 0, "val": 0, "test": 0, "warn": 0}
        total = len(samples)
        self._set_progress(0, max(1, total))
        self.log("階段 4/4：輸出資料...")
        for idx, sample in enumerate(samples, start=1):
            if self._stop_flag:
                raise MergeStoppedError("使用者停止")
            out_name = sample.output_name or self._next_output_name(sample.split, sample.image_path.name)
            out_img = out_dir / "images" / sample.split / out_name
            out_lbl = out_dir / "labels" / sample.split / Path(out_name).with_suffix(".txt").name
            shutil.copyfile(sample.image_path, out_img)

            if sample.label_path is None or not sample.label_path.exists():
                out_lbl.write_text("", encoding="utf-8")
                counters["warn"] += 1
            else:
                text = sample.label_path.read_text(encoding="utf-8", errors="ignore")
                out_lines: list[str] = []
                for raw in text.splitlines():
                    line = raw.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if not parts:
                        continue
                    try:
                        cls_id = int(parts[0])
                    except Exception:
                        counters["warn"] += 1
                        continue
                    if cls_id < 0 or cls_id >= sample.source_names_len:
                        counters["warn"] += 1
                        continue
                    out_lines.append(" ".join(parts))
                out_lbl.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
            counters[sample.split] += 1
            if idx % 200 == 0 or idx == total:
                self._set_progress(idx, total)
                self.log(f"  輸出 {idx}/{total}")
        return counters

    def _run_merge(
        self,
        sources: list[Path],
        out_root: Path,
        out_name: str,
        temp_root: Path,
        output_mode: int,
        zip_name: str,
    ) -> None:
        temp_run_dir: Optional[Path] = None
        try:
            self.log("階段 1/4：讀取來源（只解析 data.yaml）...")
            previews: list[DataSetPreview] = []
            self._set_progress(0, len(sources))
            for idx, source in enumerate(sources, start=1):
                if self._stop_flag:
                    raise MergeStoppedError("使用者停止")
                preview = _read_preview_from_source(source)
                previews.append(preview)
                if preview.source_kind == "zip":
                    yaml_desc = f"{_path_text(preview.source_path)}!/{preview.yaml_member}"
                else:
                    yaml_desc = _path_text(preview.yaml_path) if preview.yaml_path else _path_text(preview.source_path)
                self.log(f"讀取完成: {yaml_desc}")
                self.log(f"  names={preview.names}")
                self.log(f"  path={preview.dataset_path_value}")
                self._set_progress(idx, len(sources))

            canonical_names = _pick_canonical_names([p.names for p in previews])
            self.log(f"names 比對通過，主類別順序: {canonical_names}")

            temp_run_dir = (temp_root / f"run_{time.strftime('%Y%m%d_%H%M%S')}").resolve()
            _safe_mkdir(temp_run_dir)
            self.log(f"暫存資料夾: {_path_text(temp_run_dir)}")

            self.log("階段 2/4：盤點全部來源樣本（先不輸出）...")
            seen_inputs: set[Path] = set()
            planned_samples: list[PlannedSample] = []
            self._set_progress(0, len(previews))
            for idx, preview in enumerate(previews, start=1):
                if self._stop_flag:
                    raise MergeStoppedError("使用者停止")
                spec = self._resolve_runtime_spec(preview, temp_run_dir, idx)
                self.log(f"處理來源 {idx}/{len(previews)}: {_path_text(spec.source_path)}")
                self.log(f"  dataset_root={_path_text(spec.dataset_root)}")
                source_planned = self._collect_samples_from_spec(spec, seen_inputs)
                planned_samples.extend(source_planned)
                t_count = sum(1 for s in source_planned if s.split == "train")
                v_count = sum(1 for s in source_planned if s.split == "val")
                te_count = sum(1 for s in source_planned if s.split == "test")
                self.log(
                    f"  來源樣本數: {len(source_planned)} "
                    f"(train={t_count}, val={v_count}, test={te_count}) "
                    f"（累計: {len(planned_samples)}）"
                )
                if len(source_planned) == 0:
                    self.log("  [警告] 此來源盤點結果為 0，請檢查 data.yaml 的 path/train/val/test 寫法")
                self._set_progress(idx, len(previews))
            if not planned_samples:
                raise RuntimeError("盤點後樣本為 0，請確認 data.yaml 路徑設定")

            self._assign_output_names(planned_samples)

            out_dir = (out_root / out_name).resolve()
            stage_dir = out_dir
            if output_mode == 1:
                stage_dir = (temp_run_dir / out_name).resolve()
            _safe_mkdir(stage_dir)
            self._ensure_output_tree(stage_dir)
            self.log(f"輸出工作資料夾: {_path_text(stage_dir)}")

            counters = self._write_samples(planned_samples, stage_dir)
            ypath = self._write_output_yaml_with_counts(stage_dir, canonical_names, counters)
            self.log(f"已寫入: {_path_text(ypath)}")

            zip_path: Optional[Path] = None
            if output_mode in (1, 2):
                zip_path = (out_root / zip_name).resolve()
                self.log("打包 zip 中...")
                self._set_progress_busy()
                self._zip_output_dir(stage_dir, zip_path)
                self.log(f"zip 打包完成: {_path_text(zip_path)}")

            if output_mode == 1:
                folder_line = "輸出資料夾: （僅輸出 zip，未保留資料夾）"
            else:
                folder_line = f"輸出資料夾:\n{_path_text(out_dir)}"

            summary = (
                f"完成。\n{folder_line}\n"
                f"{f'zip:\n{_path_text(zip_path)}\n' if zip_path else ''}\n"
                f"樣本數: {len(planned_samples)}\n"
                f"train/val/test: {counters['train']}/{counters['val']}/{counters['test']}\n"
                f"警告數: {counters['warn']}"
            )
            self._finish_dialog = ("info", "資料集合併", summary)
        except MergeStoppedError:
            self._finish_dialog = ("warn", "資料集合併", "已停止合併作業")
        except Exception as exc:
            self.log(f"[錯誤] {exc}")
            self._finish_dialog = ("error", "資料集合併", f"合併失敗：{exc}")
        finally:
            if temp_run_dir is not None and temp_run_dir.exists():
                try:
                    shutil.rmtree(temp_run_dir, ignore_errors=True)
                except Exception:
                    pass
            self._running = False
            self._set_progress(0, 100)


def main() -> None:
    """啟動 GUI。"""
    app = QApplication([])
    win = MainWindow()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
