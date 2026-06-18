from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QThread, Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
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
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

if __package__ in {None, ""}:
    from class_mapping_model import ClassMappingRule, build_default_mapping
    from config_model import SavedSourceConfig, ValidatorAppConfig
    from config_store import ConfigStore
    from dataset_prepare_service import PreparedDatasetSpec, build_aggregate_dataset, prepare_dataset
    from dataset_worker import DatasetTaskWorker
    from engine_client import ValidationEngineClient
    from engine_protocol import EngineEvent
    from exceptions import DatasetPrepareError, EngineClientError
    from logging_utils import configure_logging, get_logger
    from results_store import (
        DatasetResultRecord,
        ValidationRunResult,
        list_saved_run_indexes,
        load_saved_run,
        record_from_engine_payload,
        save_validation_run_result,
    )
    from results_view import ResultsView
    from source_preview_service import DatasetSourcePreview, safe_preview_source
    from tray_controller import TrayController
else:
    from .class_mapping_model import ClassMappingRule, build_default_mapping
    from .config_model import SavedSourceConfig, ValidatorAppConfig
    from .config_store import ConfigStore
    from .dataset_prepare_service import PreparedDatasetSpec, build_aggregate_dataset, prepare_dataset
    from .dataset_worker import DatasetTaskWorker
    from .engine_client import ValidationEngineClient
    from .engine_protocol import EngineEvent
    from .exceptions import DatasetPrepareError, EngineClientError
    from .logging_utils import configure_logging, get_logger
    from .results_store import (
        DatasetResultRecord,
        ValidationRunResult,
        list_saved_run_indexes,
        load_saved_run,
        record_from_engine_payload,
        save_validation_run_result,
    )
    from .results_view import ResultsView
    from .source_preview_service import DatasetSourcePreview, safe_preview_source
    from .tray_controller import TrayController

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "yolo_auto_validator_config.json"
RUNS_DIR = APP_DIR / "runs"
ICON_PATH = Path(__file__).resolve().parent.parent / "yolo_server_gui" / "yolo_server_icon.png"
LOGGER = get_logger(__name__)


@dataclass
class SourceEntry:
    """UI 內部來源狀態。"""

    source_path: Path
    preview: DatasetSourcePreview | None = None
    mapping_rules: list[ClassMappingRule] = field(default_factory=list)
    status_text: str = "待預檢"


class MainWindow(QMainWindow):
    """YOLO 自動驗證器主視窗。"""

    def __init__(self) -> None:
        super().__init__()
        configure_logging("yolo_auto_validator.main_window")
        self.setWindowTitle("YOLO 自動驗證器")
        self.resize(1360, 920)
        self.setMinimumSize(1120, 760)

        self._allow_exit = False
        self._config_store = ConfigStore(CONFIG_PATH)
        self._config = self._config_store.load()
        self._sources: list[SourceEntry] = []
        self._model_names: list[str] = []
        self._worker_thread: QThread | None = None
        self._worker: DatasetTaskWorker | None = None
        self._engine = ValidationEngineClient()
        self._engine.event_received.connect(self._on_engine_event)
        self._tray = TrayController(self)
        self._current_run_root: Path | None = None
        self._pending_validation_payloads: list[dict[str, object]] = []
        self._validation_request_meta: dict[str, dict[str, object]] = {}
        self._dataset_records: list[DatasetResultRecord] = []
        self._aggregate_record: DatasetResultRecord | None = None
        self._running_validation = False
        self._worker_finished_handler: Callable[[object], None] | None = None

        self._build_ui()
        self._tray.setup(ICON_PATH if ICON_PATH.exists() else None)
        self._apply_config_to_ui()
        self._restore_saved_sources()
        self._restore_latest_run()
        LOGGER.info("Main window initialized.")
        try:
            request_id = self._engine.warmup_runtime()
            LOGGER.info("Requested engine warmup. request_id=%s", request_id)
        except EngineClientError as exc:
            LOGGER.exception("Failed to request engine warmup during initialization.")
            self._append_log(f"背景引擎啟動失敗：{exc}")
            self.lbl_model_state.setText("背景引擎啟動失敗。")

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        upper_splitter = QSplitter(self)
        layout.addWidget(upper_splitter, 2)

        left_holder = QWidget(upper_splitter)
        left_layout = QVBoxLayout(left_holder)
        left_layout.addWidget(self._build_model_group())
        left_layout.addWidget(self._build_source_group(), 1)
        left_layout.addWidget(self._build_run_group())

        right_holder = QWidget(upper_splitter)
        right_layout = QVBoxLayout(right_holder)
        right_layout.addWidget(self._build_mapping_group(), 1)
        right_layout.addWidget(self._build_log_group(), 1)

        lower_tabs = QTabWidget(self)
        self.results_view = ResultsView(lower_tabs)
        lower_tabs.addTab(self.results_view, "結果")
        layout.addWidget(lower_tabs, 2)
        upper_splitter.setSizes([760, 520])

    def _build_model_group(self) -> QWidget:
        box = QGroupBox("模型設定", self)
        form = QFormLayout(box)

        row_model = QWidget(box)
        row_model_layout = QHBoxLayout(row_model)
        row_model_layout.setContentsMargins(0, 0, 0, 0)
        self.ent_model = QLineEdit(row_model)
        self.ent_model.editingFinished.connect(self._request_model_info)
        btn_model = QPushButton("瀏覽...", row_model)
        btn_model.clicked.connect(self._pick_model)
        row_model_layout.addWidget(self.ent_model, 1)
        row_model_layout.addWidget(btn_model)
        form.addRow("YOLO 模型：", row_model)

        self.lbl_model_state = QLabel("模型尚未載入。", box)
        form.addRow("模型狀態：", self.lbl_model_state)

        self.txt_model_names = QPlainTextEdit(box)
        self.txt_model_names.setReadOnly(True)
        self.txt_model_names.setFixedHeight(90)
        form.addRow("模型 names：", self.txt_model_names)

        row_threshold = QWidget(box)
        row_threshold_layout = QHBoxLayout(row_threshold)
        row_threshold_layout.setContentsMargins(0, 0, 0, 0)
        self.sp_conf = QDoubleSpinBox(row_threshold)
        self.sp_conf.setRange(0.0, 1.0)
        self.sp_conf.setDecimals(2)
        self.sp_conf.setSingleStep(0.01)
        self.sp_conf.valueChanged.connect(self._save_config)
        self.sp_iou = QDoubleSpinBox(row_threshold)
        self.sp_iou.setRange(0.0, 1.0)
        self.sp_iou.setDecimals(2)
        self.sp_iou.setSingleStep(0.01)
        self.sp_iou.valueChanged.connect(self._save_config)
        self.ent_device = QLineEdit(row_threshold)
        self.ent_device.editingFinished.connect(self._save_config)
        row_threshold_layout.addWidget(QLabel("conf", row_threshold))
        row_threshold_layout.addWidget(self.sp_conf)
        row_threshold_layout.addWidget(QLabel("iou", row_threshold))
        row_threshold_layout.addWidget(self.sp_iou)
        row_threshold_layout.addWidget(QLabel("device", row_threshold))
        row_threshold_layout.addWidget(self.ent_device, 1)
        form.addRow("驗證參數：", row_threshold)
        return box

    def _build_source_group(self) -> QWidget:
        box = QGroupBox("資料集來源", self)
        layout = QVBoxLayout(box)
        btn_row = QWidget(box)
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_add_folder = QPushButton("加入資料夾", btn_row)
        btn_add_folder.clicked.connect(self._pick_folders)
        btn_add_archive = QPushButton("加入 zip / rar", btn_row)
        btn_add_archive.clicked.connect(self._pick_archives)
        btn_precheck = QPushButton("重新預檢", btn_row)
        btn_precheck.clicked.connect(self._start_preview_for_all_sources)
        btn_remove = QPushButton("移除選取", btn_row)
        btn_remove.clicked.connect(self._remove_selected_source)
        btn_layout.addWidget(btn_add_folder)
        btn_layout.addWidget(btn_add_archive)
        btn_layout.addWidget(btn_precheck)
        btn_layout.addWidget(btn_remove)
        btn_layout.addStretch(1)
        layout.addWidget(btn_row)

        self.tbl_sources = QTableWidget(0, 4, box)
        self.tbl_sources.setHorizontalHeaderLabels(["來源", "類型", "狀態", "類別"])
        self.tbl_sources.itemSelectionChanged.connect(self._refresh_mapping_table)
        layout.addWidget(self.tbl_sources, 1)
        return box

    def _build_mapping_group(self) -> QWidget:
        box = QGroupBox("類別校正", self)
        layout = QVBoxLayout(box)
        self.lbl_mapping_hint = QLabel("請先載入模型與資料集。", box)
        layout.addWidget(self.lbl_mapping_hint)
        self.tbl_mapping = QTableWidget(0, 3, box)
        self.tbl_mapping.setHorizontalHeaderLabels(["來源類別", "目標模型類別", "忽略"])
        layout.addWidget(self.tbl_mapping, 1)
        return box

    def _build_run_group(self) -> QWidget:
        box = QGroupBox("暫存與執行", self)
        form = QFormLayout(box)

        row_temp = QWidget(box)
        row_temp_layout = QHBoxLayout(row_temp)
        row_temp_layout.setContentsMargins(0, 0, 0, 0)
        self.ent_temp_root = QLineEdit(row_temp)
        self.ent_temp_root.editingFinished.connect(self._save_config)
        btn_temp = QPushButton("瀏覽...", row_temp)
        btn_temp.clicked.connect(self._pick_temp_root)
        row_temp_layout.addWidget(self.ent_temp_root, 1)
        row_temp_layout.addWidget(btn_temp)
        form.addRow("暫存根目錄：", row_temp)

        self.chk_delete_temp = QCheckBox("驗證後刪除暫存資料", box)
        self.chk_delete_temp.toggled.connect(self._save_config)
        form.addRow("暫存清理：", self.chk_delete_temp)

        row_action = QWidget(box)
        row_action_layout = QHBoxLayout(row_action)
        row_action_layout.setContentsMargins(0, 0, 0, 0)
        self.btn_start = QPushButton("開始驗證", row_action)
        self.btn_start.clicked.connect(self._start_validation)
        self.btn_stop = QPushButton("停止", row_action)
        self.btn_stop.clicked.connect(self._stop_current_work)
        self.btn_stop.setEnabled(False)
        self.progress = QProgressBar(row_action)
        self.progress.setRange(0, 1)
        self.lbl_status = QLabel("就緒", row_action)
        row_action_layout.addWidget(self.btn_start)
        row_action_layout.addWidget(self.btn_stop)
        row_action_layout.addWidget(self.progress, 1)
        row_action_layout.addWidget(self.lbl_status)
        form.addRow("執行：", row_action)
        return box

    def _build_log_group(self) -> QWidget:
        box = QGroupBox("執行日誌", self)
        layout = QVBoxLayout(box)
        self.txt_log = QPlainTextEdit(box)
        self.txt_log.setReadOnly(True)
        layout.addWidget(self.txt_log, 1)
        return box

    def _apply_config_to_ui(self) -> None:
        self.ent_model.setText(self._config.model_path)
        self.ent_device.setText(self._config.device)
        self.sp_conf.setValue(self._config.conf_threshold)
        self.sp_iou.setValue(self._config.iou_threshold)
        default_temp = APP_DIR / ".tmp" / "yolo_auto_validator"
        self.ent_temp_root.setText(self._config.temp_root or str(default_temp))
        self.chk_delete_temp.setChecked(self._config.delete_temp_after_run)

    def _restore_saved_sources(self) -> None:
        for item in self._config.last_sources:
            source_path = Path(item.source_path)
            if source_path.exists():
                self._sources.append(SourceEntry(source_path=source_path, status_text="待預檢"))
        self._refresh_sources_table()
        if self._sources:
            self._start_preview_for_all_sources()
        if self.ent_model.text().strip():
            self._request_model_info()

    def _restore_latest_run(self) -> None:
        try:
            indexes = list_saved_run_indexes(RUNS_DIR)
            if indexes:
                self.results_view.set_run_result(load_saved_run(indexes[0]))
                LOGGER.info("Restored latest validation run. index=%s", indexes[0])
        except Exception:
            LOGGER.exception("Failed to restore latest validation run.")
            return

    def _pick_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "選擇模型", str(Path.cwd()), "YOLO weights (*.pt);;All files (*.*)")
        if not path:
            return
        self.ent_model.setText(path)
        self._save_config()
        self._request_model_info()

    def _pick_folders(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "選擇資料集資料夾", str(Path.cwd()))
        if not path:
            return
        self._append_sources([Path(path)])

    def _pick_archives(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "選擇壓縮資料集",
            str(Path.cwd()),
            "Archives (*.zip *.rar);;All files (*.*)",
        )
        if not paths:
            return
        self._append_sources([Path(path) for path in paths])

    def _pick_temp_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "選擇暫存根目錄", self.ent_temp_root.text().strip() or str(Path.cwd()))
        if not path:
            return
        self.ent_temp_root.setText(path)
        self._save_config()

    def _append_sources(self, paths: list[Path]) -> None:
        added = False
        existing = {entry.source_path.resolve() for entry in self._sources}
        for path in paths:
            resolved = path.resolve()
            if resolved in existing:
                continue
            self._sources.append(SourceEntry(source_path=resolved, status_text="待預檢"))
            existing.add(resolved)
            added = True
        if not added:
            return
        self._refresh_sources_table()
        self._save_config()
        self._start_preview_for_all_sources()

    def _remove_selected_source(self) -> None:
        row = self.tbl_sources.currentRow()
        if row < 0 or row >= len(self._sources):
            return
        del self._sources[row]
        self._refresh_sources_table()
        self._refresh_mapping_table()
        self._save_config()

    def _refresh_sources_table(self) -> None:
        self.tbl_sources.setRowCount(len(self._sources))
        for row, entry in enumerate(self._sources):
            preview = entry.preview
            kind = preview.source_kind if preview is not None else ("folder" if entry.source_path.is_dir() else entry.source_path.suffix.lower().lstrip("."))
            class_text = ", ".join(preview.names) if preview is not None and preview.names else "-"
            self.tbl_sources.setItem(row, 0, QTableWidgetItem(str(entry.source_path)))
            self.tbl_sources.setItem(row, 1, QTableWidgetItem(kind))
            self.tbl_sources.setItem(row, 2, QTableWidgetItem(entry.status_text))
            self.tbl_sources.setItem(row, 3, QTableWidgetItem(class_text))
        self.tbl_sources.resizeColumnsToContents()

    def _refresh_mapping_table(self) -> None:
        row = self.tbl_sources.currentRow()
        self.tbl_mapping.setRowCount(0)
        if row < 0 or row >= len(self._sources):
            self.lbl_mapping_hint.setText("請選擇資料集來源。")
            return
        entry = self._sources[row]
        if entry.preview is None:
            self.lbl_mapping_hint.setText("此來源尚未完成預檢。")
            return
        if entry.preview.preview_error:
            self.lbl_mapping_hint.setText(entry.preview.preview_error)
            return
        if not self._model_names:
            self.lbl_mapping_hint.setText("請先載入模型，才能校正類別。")
            return
        if not entry.mapping_rules:
            entry.mapping_rules = self._build_mapping_rules(entry.preview)
        self.lbl_mapping_hint.setText(f"來源：{entry.source_path.name}")
        self.tbl_mapping.setRowCount(len(entry.mapping_rules))
        for index, rule in enumerate(entry.mapping_rules):
            self.tbl_mapping.setItem(index, 0, QTableWidgetItem(rule.source_name))
            combo = QComboBox(self.tbl_mapping)
            combo.addItem("(未指定)", None)
            for model_index, name in enumerate(self._model_names):
                combo.addItem(name, model_index)
            combo.setCurrentIndex(0 if rule.target_model_index is None else rule.target_model_index + 1)
            combo.currentIndexChanged.connect(lambda _=0, r=index: self._on_mapping_target_changed(r))
            self.tbl_mapping.setCellWidget(index, 1, combo)
            checkbox = QCheckBox(self.tbl_mapping)
            checkbox.setChecked(rule.ignore)
            checkbox.toggled.connect(lambda _=False, r=index: self._on_mapping_ignore_changed(r))
            self.tbl_mapping.setCellWidget(index, 2, checkbox)
        self.tbl_mapping.resizeColumnsToContents()

    def _on_mapping_target_changed(self, row: int) -> None:
        source_row = self.tbl_sources.currentRow()
        if source_row < 0 or source_row >= len(self._sources):
            return
        entry = self._sources[source_row]
        if row < 0 or row >= len(entry.mapping_rules):
            return
        combo = self.tbl_mapping.cellWidget(row, 1)
        if not isinstance(combo, QComboBox):
            return
        target_index = combo.currentData()
        current = entry.mapping_rules[row]
        entry.mapping_rules[row] = ClassMappingRule(
            source_name=current.source_name,
            target_model_index=None if target_index is None else int(target_index),
            ignore=current.ignore,
            rename_note="" if target_index is None else self._model_names[int(target_index)],
        )
        self._save_config()

    def _on_mapping_ignore_changed(self, row: int) -> None:
        source_row = self.tbl_sources.currentRow()
        if source_row < 0 or source_row >= len(self._sources):
            return
        entry = self._sources[source_row]
        if row < 0 or row >= len(entry.mapping_rules):
            return
        checkbox = self.tbl_mapping.cellWidget(row, 2)
        if not isinstance(checkbox, QCheckBox):
            return
        current = entry.mapping_rules[row]
        entry.mapping_rules[row] = ClassMappingRule(
            source_name=current.source_name,
            target_model_index=current.target_model_index,
            ignore=checkbox.isChecked(),
            rename_note=current.rename_note,
        )
        self._save_config()

    def _request_model_info(self) -> None:
        model_path = self.ent_model.text().strip()
        if not model_path:
            return
        if not Path(model_path).exists():
            QMessageBox.warning(self, "模型不存在", "請先選擇有效的模型檔案。")
            return
        self._set_progress_unknown("背景讀取模型資訊中...")
        LOGGER.info("Requesting model info. model_path=%s", model_path)
        try:
            request_id = self._engine.load_model_info(model_path)
        except EngineClientError as exc:
            LOGGER.exception("Failed to dispatch model info request. model_path=%s", model_path)
            self._handle_engine_dispatch_failure("模型資訊讀取請求失敗", exc, reset_validation=False)
            return
        LOGGER.debug("Model info request dispatched. request_id=%s", request_id)
        self.lbl_model_state.setText("模型資訊載入中...")

    def _start_preview_for_all_sources(self) -> None:
        if not self._sources or self._worker_thread is not None:
            return
        paths = [entry.source_path for entry in self._sources]
        LOGGER.info("Starting dataset preview for all sources. source_count=%s", len(paths))

        def task(log_cb, progress_known_cb, progress_unknown_cb, cancel_cb):
            previews: list[tuple[Path, DatasetSourcePreview]] = []
            progress_unknown_cb("資料集預檢中...")
            total = max(1, len(paths))
            for index, path in enumerate(paths, start=1):
                if cancel_cb():
                    raise RuntimeError("使用者已取消。")
                preview = safe_preview_source(path)
                previews.append((path, preview))
                log_cb(f"預檢完成：{path.name}")
                progress_known_cb(index, total)
            return previews

        self._start_worker(task, self._on_preview_finished)

    def _start_validation(self) -> None:
        if self._worker_thread is not None or self._running_validation:
            return
        if not self._model_names:
            QMessageBox.warning(self, "模型未就緒", "請先完成模型載入。")
            return
        valid_entries = [entry for entry in self._sources if entry.preview is not None and not entry.preview.preview_error]
        if not valid_entries:
            QMessageBox.warning(self, "資料集未就緒", "請先完成資料集預檢。")
            return
        if any(not entry.mapping_rules for entry in valid_entries):
            QMessageBox.warning(self, "類別映射未就緒", "請先確認所有資料集的類別映射。")
            return

        temp_root = Path(self.ent_temp_root.text().strip()).resolve()
        run_dir = RUNS_DIR / time.strftime("%Y%m%d_%H%M%S")
        self._current_run_root = run_dir
        self._dataset_records = []
        self._aggregate_record = None
        self._pending_validation_payloads.clear()
        self._validation_request_meta.clear()
        self.results_view.clear_result()
        LOGGER.info(
            "Starting validation run. run_dir=%s source_count=%s temp_root=%s conf=%s iou=%s device=%s",
            run_dir,
            len(valid_entries),
            temp_root,
            self.sp_conf.value(),
            self.sp_iou.value(),
            self.ent_device.text().strip(),
        )

        def task(log_cb, progress_known_cb, progress_unknown_cb, cancel_cb):
            progress_unknown_cb("準備驗證資料中...")
            prepared_specs: list[PreparedDatasetSpec] = []
            total = max(1, len(valid_entries) + 1)
            for index, entry in enumerate(valid_entries, start=1):
                preview = entry.preview
                if preview is None:
                    raise DatasetPrepareError(f"資料集預檢資料遺失：{entry.source_path}")
                log_cb(f"準備資料集：{entry.source_path.name}")
                prepared_specs.append(
                    prepare_dataset(
                        preview=preview,
                        model_names=self._model_names,
                        mapping_rules=entry.mapping_rules,
                        temp_root=temp_root,
                        cancel_cb=cancel_cb,
                    )
                )
                progress_known_cb(index, total)
            if self._current_run_root is None:
                raise DatasetPrepareError("驗證輸出目錄尚未建立。")
            aggregate_yaml = build_aggregate_dataset(
                prepared_specs,
                output_root=self._current_run_root,
                model_names=self._model_names,
                cancel_cb=cancel_cb,
            )
            progress_known_cb(total, total)
            return {
                "prepared_specs": prepared_specs,
                "aggregate_yaml": aggregate_yaml,
            }

        self._start_worker(task, self._on_prepare_finished)

    def _start_worker(self, task, on_finished) -> None:
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._worker_thread = QThread(self)
        self._worker = DatasetTaskWorker(task)
        self._worker_finished_handler = on_finished
        LOGGER.debug("Starting dataset worker thread.")
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.log.connect(self._append_log, Qt.ConnectionType.QueuedConnection)
        self._worker.progress_known.connect(self._set_progress_known, Qt.ConnectionType.QueuedConnection)
        self._worker.progress_unknown.connect(self._set_progress_unknown, Qt.ConnectionType.QueuedConnection)
        self._worker.failed.connect(self._on_worker_failed, Qt.ConnectionType.QueuedConnection)
        self._worker.cancelled.connect(self._on_worker_cancelled, Qt.ConnectionType.QueuedConnection)
        self._worker.finished.connect(self._on_worker_finished_signal, Qt.ConnectionType.QueuedConnection)
        self._worker_thread.start()

    def _on_worker_finished_signal(self, payload: object) -> None:
        on_finished = self._worker_finished_handler
        if on_finished is None:
            LOGGER.warning("Worker finished signal received without a registered callback.")
            self._cleanup_worker()
            return
        try:
            on_finished(payload)
        except Exception as exc:
            LOGGER.exception("Worker finished callback failed. callback=%s", getattr(on_finished, "__name__", repr(on_finished)))
            self._append_log(f"背景作業完成後處理失敗：{exc}")
            QMessageBox.critical(self, "背景作業錯誤", f"{exc}\n\n詳細內容請查看 terminal log。")
        finally:
            self._cleanup_worker()

    def _cleanup_worker(self) -> None:
        if self._worker_thread is not None:
            LOGGER.debug("Cleaning up dataset worker thread.")
            self._worker_thread.quit()
            self._worker_thread.wait(3000)
            self._worker_thread.deleteLater()
            self._worker_thread = None
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        self._worker_finished_handler = None
        if not self._running_validation:
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self._set_progress_known(0, 1)

    def _on_preview_finished(self, payload: object) -> None:
        previews = list(payload)
        preview_map = {path.resolve(): preview for path, preview in previews}
        for entry in self._sources:
            preview = preview_map.get(entry.source_path.resolve())
            if preview is None:
                continue
            entry.preview = preview
            if preview.preview_error:
                entry.status_text = preview.preview_error
                entry.mapping_rules = []
                continue
            entry.status_text = "預檢完成"
            entry.mapping_rules = self._build_mapping_rules(preview)
        self._refresh_sources_table()
        self._refresh_mapping_table()
        self._save_config()
        self._append_log("資料集預檢完成。")

    def _on_prepare_finished(self, payload: object) -> None:
        payload_dict = dict(payload)
        prepared_specs = list(payload_dict.get("prepared_specs", []))
        aggregate_yaml_value = payload_dict.get("aggregate_yaml")
        if aggregate_yaml_value is None:
            message = "驗證準備結果缺少 aggregate YAML 路徑。"
            LOGGER.error(message)
            self._append_log(message)
            QMessageBox.critical(self, "驗證準備失敗", f"{message}\n\n詳細內容請查看 terminal log。")
            return
        aggregate_yaml = Path(aggregate_yaml_value)
        if self._current_run_root is None:
            message = "驗證輸出目錄遺失，無法開始驗證。"
            LOGGER.error("%s aggregate_yaml=%s prepared_count=%s", message, aggregate_yaml, len(prepared_specs))
            self._append_log(message)
            QMessageBox.critical(self, "驗證準備失敗", f"{message}\n\n詳細內容請查看 terminal log。")
            return
        LOGGER.info(
            "Validation preparation finished. prepared_count=%s aggregate_yaml=%s run_root=%s",
            len(prepared_specs),
            aggregate_yaml,
            self._current_run_root,
        )
        dataset_project_dir = self._current_run_root / "datasets"
        self._pending_validation_payloads = []
        for spec in prepared_specs:
            self._pending_validation_payloads.append(
                {
                    "role": "dataset",
                    "title": spec.dataset_key,
                    "dataset_key": spec.dataset_key,
                    "model_path": self.ent_model.text().strip(),
                    "data_yaml_path": str(spec.patched_yaml_path),
                    "project_dir": str(dataset_project_dir),
                    "run_name": spec.dataset_key,
                    "device": self.ent_device.text().strip(),
                    "conf_threshold": self.sp_conf.value(),
                    "iou_threshold": self.sp_iou.value(),
                    "contains_unlabeled_images": spec.contains_unlabeled_images,
                    "empty_label_count": spec.empty_label_count,
                }
            )
        self._pending_validation_payloads.append(
            {
                "role": "aggregate",
                "title": "全部資料集彙總",
                "dataset_key": "aggregate",
                "model_path": self.ent_model.text().strip(),
                "data_yaml_path": str(aggregate_yaml),
                "project_dir": str(self._current_run_root),
                "run_name": "aggregate",
                "device": self.ent_device.text().strip(),
                "conf_threshold": self.sp_conf.value(),
                "iou_threshold": self.sp_iou.value(),
                "contains_unlabeled_images": any(spec.contains_unlabeled_images for spec in prepared_specs),
                "empty_label_count": sum(spec.empty_label_count for spec in prepared_specs),
            }
        )
        self._running_validation = True
        self._start_next_validation()

    def _start_next_validation(self) -> None:
        if not self._pending_validation_payloads:
            self._running_validation = False
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self._set_progress_known(0, 1)
            self._finalize_run()
            return
        payload = self._pending_validation_payloads.pop(0)
        LOGGER.info(
            "Dispatching validation request. title=%s dataset_key=%s data_yaml=%s",
            payload.get("title", ""),
            payload.get("dataset_key", ""),
            payload.get("data_yaml_path", ""),
        )
        try:
            request_id = self._engine.validate_dataset(payload)
        except EngineClientError as exc:
            LOGGER.exception("Failed to dispatch validation request. payload=%s", payload)
            self._handle_engine_dispatch_failure("送出驗證請求失敗", exc, reset_validation=True)
            return
        self._validation_request_meta[request_id] = payload
        LOGGER.debug("Validation request dispatched. request_id=%s", request_id)
        self._set_progress_unknown(f"驗證中：{payload.get('title', '')}")

    def _finalize_run(self) -> None:
        if self._current_run_root is None or self._aggregate_record is None:
            LOGGER.warning(
                "Skipping validation run finalization because required state is missing. run_root=%s aggregate_record=%s",
                self._current_run_root,
                self._aggregate_record is not None,
            )
            return
        run_result = ValidationRunResult(
            run_dir=str(self._current_run_root),
            dataset_results=self._dataset_records,
            aggregate_result=self._aggregate_record,
            warnings=[],
        )
        try:
            save_validation_run_result(run_result)
            self.results_view.set_run_result(run_result)
        except Exception as exc:
            LOGGER.exception("Failed to finalize validation run. run_dir=%s", self._current_run_root)
            self._append_log(f"驗證結果保存失敗：{exc}")
            QMessageBox.critical(self, "驗證結果保存失敗", f"{exc}\n\n詳細內容請查看 terminal log。")
            return
        self._append_log(f"驗證完成：{self._current_run_root}")
        LOGGER.info("Validation run finalized. run_dir=%s dataset_count=%s", self._current_run_root, len(self._dataset_records))
        if self.chk_delete_temp.isChecked():
            temp_root = Path(self.ent_temp_root.text().strip())
            LOGGER.warning("Deleting temporary root after validation run. temp_root=%s", temp_root)
            __import__("shutil").rmtree(temp_root, ignore_errors=True)

    def _stop_current_work(self) -> None:
        if self._worker is not None:
            self._worker.request_cancel()
            LOGGER.info("Stop requested for dataset worker.")
            self._append_log("已要求停止背景資料處理。")
            return
        if self._running_validation:
            self._pending_validation_payloads.clear()
            self._validation_request_meta.clear()
            self._running_validation = False
            self._engine.terminate_and_restart()
            try:
                request_id = self._engine.warmup_runtime()
                LOGGER.info("Validation engine restarted after stop request. request_id=%s", request_id)
            except EngineClientError as exc:
                LOGGER.exception("Failed to warm up validation engine after restart.")
                self._append_log(f"背景驗證引擎重啟後預熱失敗：{exc}")
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self._set_progress_known(0, 1)
            LOGGER.info("Stop requested for running validation.")
            self._append_log("已中止背景驗證引擎。")

    def _build_mapping_rules(self, preview: DatasetSourcePreview) -> list[ClassMappingRule]:
        saved = self._config.saved_mappings.get(preview.yaml_signature)
        if isinstance(saved, list):
            restored: list[ClassMappingRule] = []
            for item in saved:
                if not isinstance(item, dict):
                    continue
                restored.append(
                    ClassMappingRule(
                        source_name=str(item.get("source_name", "")),
                        target_model_index=item.get("target_model_index"),
                        ignore=bool(item.get("ignore", False)),
                        rename_note=str(item.get("rename_note", "")),
                    )
                )
            if len(restored) == len(preview.names):
                return restored
        return build_default_mapping(preview.names, self._model_names)

    def _save_config(self) -> None:
        saved_sources = tuple(
            SavedSourceConfig(
                source_path=str(entry.source_path),
                source_kind=entry.preview.source_kind if entry.preview is not None else "",
            )
            for entry in self._sources
        )
        saved_mappings: dict[str, list[dict[str, object]]] = {}
        for entry in self._sources:
            if entry.preview is None or not entry.mapping_rules:
                continue
            saved_mappings[entry.preview.yaml_signature] = [rule.to_dict() for rule in entry.mapping_rules]
        self._config = ValidatorAppConfig(
            model_path=self.ent_model.text().strip(),
            device=self.ent_device.text().strip(),
            conf_threshold=float(self.sp_conf.value()),
            iou_threshold=float(self.sp_iou.value()),
            temp_root=self.ent_temp_root.text().strip(),
            delete_temp_after_run=self.chk_delete_temp.isChecked(),
            window_close_behavior="tray_only",
            last_sources=saved_sources,
            saved_mappings=saved_mappings,
        )
        self._config_store.save(self._config)

    def _append_log(self, message: str) -> None:
        self.txt_log.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {message}")

    def _set_progress_known(self, value: int, maximum: int) -> None:
        self.progress.setRange(0, max(1, maximum))
        self.progress.setValue(min(value, maximum))
        if maximum > 0 and value >= maximum:
            self.lbl_status.setText("完成")

    def _set_progress_unknown(self, message: str) -> None:
        self.progress.setRange(0, 0)
        self.lbl_status.setText(message)

    def _on_worker_failed(self, message: str) -> None:
        self._cleanup_worker()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._set_progress_known(0, 1)
        LOGGER.error("Background worker failed. message=%s", message)
        self._append_log(f"背景作業失敗：{message}")
        QMessageBox.critical(self, "錯誤", message)

    def _on_worker_cancelled(self) -> None:
        self._cleanup_worker()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._set_progress_known(0, 1)
        LOGGER.info("Background worker cancelled.")
        self._append_log("背景作業已停止。")

    def _on_engine_event(self, event: EngineEvent) -> None:
        LOGGER.debug("Handling engine event. request_id=%s kind=%s", event.request_id, event.kind)
        if event.kind == "log":
            self._append_log(str(event.payload.get("message", "")))
            return
        if event.kind == "progress_unknown":
            self._set_progress_unknown(str(event.payload.get("message", "執行中...")))
            return
        if event.kind == "completed":
            LOGGER.debug("Engine request completed. request_id=%s", event.request_id)
            return
        if event.kind == "stopped":
            LOGGER.info("Engine reported stopped state.")
            return
        if event.kind == "model_info":
            self._model_names = [str(item) for item in list(event.payload.get("names", []))]
            self.lbl_model_state.setText("模型資訊已載入。")
            self.txt_model_names.setPlainText("\n".join(self._model_names))
            LOGGER.info("Model info loaded. class_count=%s", len(self._model_names))
            for entry in self._sources:
                if entry.preview is not None and not entry.preview.preview_error:
                    entry.mapping_rules = self._build_mapping_rules(entry.preview)
            self._refresh_mapping_table()
            self._save_config()
            return
        if event.kind == "validation_done":
            payload = self._validation_request_meta.pop(event.request_id, {})
            try:
                record = record_from_engine_payload(event.payload)
            except Exception as exc:
                LOGGER.exception("Failed to deserialize validation result payload. request_id=%s", event.request_id)
                self._handle_engine_dispatch_failure("驗證結果解析失敗", exc, reset_validation=True)
                return
            if payload.get("role") == "aggregate":
                self._aggregate_record = record
            else:
                self._dataset_records.append(record)
            LOGGER.info("Validation result received. title=%s role=%s", record.title, payload.get("role", "dataset"))
            self._append_log(f"驗證完成：{record.title}")
            self._start_next_validation()
            return
        if event.kind == "error":
            self._handle_engine_error_event(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._save_config()
        if self._tray.handle_close_event(event):
            return
        super().closeEvent(event)

    def _handle_engine_dispatch_failure(self, title: str, exc: BaseException, *, reset_validation: bool) -> None:
        message = f"{title}：{exc}"
        LOGGER.error(message, exc_info=True)
        if reset_validation:
            self._validation_request_meta.clear()
            self._pending_validation_payloads.clear()
            self._running_validation = False
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self._set_progress_known(0, 1)
        self._append_log(message)
        QMessageBox.critical(self, "背景引擎錯誤", f"{message}\n\n詳細內容請查看 terminal log。")

    def _handle_engine_error_event(self, event: EngineEvent) -> None:
        message = str(event.payload.get("message", "未知錯誤"))
        action = str(event.payload.get("action", ""))
        stage = str(event.payload.get("stage", ""))
        traceback_text = str(event.payload.get("traceback", ""))
        if traceback_text:
            LOGGER.error(
                "Engine error event received. request_id=%s action=%s stage=%s\n%s",
                event.request_id,
                action,
                stage,
                traceback_text.rstrip(),
            )
        else:
            LOGGER.error(
                "Engine error event received. request_id=%s action=%s stage=%s message=%s",
                event.request_id,
                action,
                stage,
                message,
            )
        is_validation_error = action == "validate_dataset" or event.request_id in self._validation_request_meta or self._running_validation
        self._validation_request_meta.pop(event.request_id, None)
        if is_validation_error:
            self._pending_validation_payloads.clear()
            self._running_validation = False
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self._set_progress_known(0, 1)
        if action == "load_model_info":
            self.lbl_model_state.setText("模型資訊載入失敗。")
        self._append_log(f"引擎錯誤：{message}")
        detail = f"{message}\n\n階段：{stage or 'unknown'}\n\n詳細內容請查看 terminal log。"
        QMessageBox.critical(self, "驗證引擎錯誤", detail)

    def request_shutdown(self) -> None:
        """關閉前清理。"""
        LOGGER.info("Main window shutdown requested.")
        if self._worker is not None:
            self._worker.request_cancel()
        self._engine.shutdown()
