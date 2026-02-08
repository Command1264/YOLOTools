from __future__ import annotations

import queue
from typing import Optional

from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from app_services import ALL_OPTION, APP_DIR, CACHE_WEIGHTS, CONFIG_PATH, default_dialog_dir, normalize_path
from config_model import TrainerGuiConfig
from hardcore_view import HardcorePanel
from history_controller import HistoryController
from model_registry import filter_weights, get_weights, list_model_families, list_model_sizes
from train_controller import TrainController
from trainer_worker import TrainerWorker


class App(QMainWindow):
    """Main window that wires UI to controllers."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ultralytics YOLO GUI Trainer")
        self.resize(1200, 800)
        self.setMinimumSize(1200, 800)

        self.msg_q: queue.Queue = queue.Queue()
        self.worker: Optional[TrainerWorker] = None
        self._pending_msgs: list[tuple] = []
        self._is_resizing = False
        self._eta_epoch_sec: Optional[float] = None
        self._eta_batch_sec: Optional[float] = None
        self._eta_epoch_last_update = 0.0
        self._eta_batch_last_update = 0.0
        self._eta_decimal_places = 1
        self._last_epoch_progress = (0, 1)
        self._last_batch_progress = (0, 1)
        self._loading_config = False
        self._force_stop_locked = False
        self._updating_model_filters = False
        self._metrics_log = ""
        self._base_weights: list[str] = []

        self.last_dataset_zip = ""
        self.last_work_dir = ""
        self.last_out_zip_dir = ""
        self.last_custom_model = ""
        self.last_model_dir = ""

        self._build_ui()

        self.train_controller = TrainController(self)
        self.history_controller = HistoryController(self)

        self._bind_config_traces()
        self._load_config()
        self._sync_last_paths()
        self._sync_model_mode()
        self._load_weights(try_online=True)
        self._load_history()

        self._queue_timer = QTimer(self)
        self._queue_timer.setInterval(120)
        self._queue_timer.timeout.connect(self._poll_queue)
        self._queue_timer.start()

        self._eta_timer = QTimer(self)
        self._eta_timer.setInterval(1000)
        self._eta_timer.timeout.connect(self._tick_eta)
        self._eta_timer.start()

        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(50)
        self._resize_timer.timeout.connect(self._end_resize)

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        self.tabs = QTabWidget(central)
        root.addWidget(self.tabs)

        self.tab_train = QWidget(self.tabs)
        self.tab_hist = QWidget(self.tabs)
        self.tab_hardcore = QWidget(self.tabs)
        self.tabs.addTab(self.tab_train, "訓練")
        self.tabs.addTab(self.tab_hist, "歷史")
        self.tabs.addTab(self.tab_hardcore, "硬核視覺化")

        self._build_train_tab()
        self._build_history_tab()
        self._build_hardcore_tab()

    def _build_train_tab(self) -> None:
        outer = QVBoxLayout(self.tab_train)
        scroll = QScrollArea(self.tab_train)
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)

        holder = QWidget(scroll)
        scroll.setWidget(holder)
        grid = QGridLayout(holder)
        row = 0

        box_model = QGroupBox("模型選擇", holder)
        mgb = QGridLayout(box_model)
        self.cmb_task = QComboBox(box_model)
        self.cmb_task.addItems(["detect", "segment", "classify", "pose", "obb"])
        self.cmb_model_family = QComboBox(box_model)
        self.cmb_model_size = QComboBox(box_model)
        self.cmb_model = QComboBox(box_model)
        self.cmb_model.setEditable(True)
        self.chk_use_custom_model = QCheckBox("使用自訂權重", box_model)
        self.edt_custom_model = QLineEdit(box_model)
        self.btn_pick_custom_model = QPushButton("選擇...", box_model)
        self.btn_update_models = QPushButton("更新官方模型清單（線上）", box_model)
        self.lbl_model_source = QLabel("(尚未載入)", box_model)

        mgb.addWidget(QLabel("任務", box_model), 0, 0)
        mgb.addWidget(self.cmb_task, 0, 1)
        mgb.addWidget(QLabel("版本", box_model), 0, 2)
        mgb.addWidget(self.cmb_model_family, 0, 3)
        mgb.addWidget(QLabel("大小", box_model), 0, 4)
        mgb.addWidget(self.cmb_model_size, 0, 5)
        mgb.addWidget(self.btn_update_models, 0, 6)
        mgb.addWidget(QLabel("模型", box_model), 1, 0)
        mgb.addWidget(self.cmb_model, 1, 1, 1, 5)
        mgb.addWidget(self.lbl_model_source, 1, 6)
        mgb.addWidget(QLabel("自訂權重", box_model), 2, 0)
        custom = QHBoxLayout()
        custom.addWidget(self.edt_custom_model)
        custom.addWidget(self.btn_pick_custom_model)
        custom_w = QWidget(box_model)
        custom_w.setLayout(custom)
        mgb.addWidget(custom_w, 2, 1, 1, 5)
        mgb.addWidget(self.chk_use_custom_model, 2, 6)
        mgb.setColumnStretch(1, 1)
        mgb.setColumnStretch(3, 1)
        grid.addWidget(box_model, row, 0, 1, 3)

        self.cmb_task.currentTextChanged.connect(self._apply_model_filters)
        self.cmb_model_family.currentTextChanged.connect(self._apply_model_filters)
        self.cmb_model_size.currentTextChanged.connect(self._apply_model_filters)
        self.chk_use_custom_model.toggled.connect(self._sync_model_mode)
        self.btn_update_models.clicked.connect(lambda: self._load_weights(True))
        self.btn_pick_custom_model.clicked.connect(self.pick_custom_model)

        row += 1
        self.edt_dataset_zip = QLineEdit(holder)
        btn_dataset = QPushButton("選擇...", holder)
        btn_dataset.clicked.connect(self.pick_dataset_zip)
        grid.addWidget(QLabel("資料集（zip）：", holder), row, 0)
        grid.addWidget(self.edt_dataset_zip, row, 1)
        grid.addWidget(btn_dataset, row, 2)

        row += 1
        self.edt_work_dir = QLineEdit(normalize_path(str(APP_DIR / "workdir")), holder)
        btn_work = QPushButton("選擇...", holder)
        btn_work.clicked.connect(self.pick_work_dir)
        grid.addWidget(QLabel("訓練工作資料夾（解壓縮/訓練模型）：", holder), row, 0)
        grid.addWidget(self.edt_work_dir, row, 1)
        grid.addWidget(btn_work, row, 2)

        row += 1
        self.edt_out_zip_dir = QLineEdit(normalize_path(str(APP_DIR / "output_zips")), holder)
        btn_out = QPushButton("選擇...", holder)
        btn_out.clicked.connect(self.pick_out_zip_dir)
        grid.addWidget(QLabel("輸出 zip 存放資料夾：", holder), row, 0)
        grid.addWidget(self.edt_out_zip_dir, row, 1)
        grid.addWidget(btn_out, row, 2)

        row += 1
        self.edt_model_dir = QLineEdit(normalize_path(str(APP_DIR / "models")), holder)
        btn_model = QPushButton("選擇...", holder)
        btn_model.clicked.connect(self.pick_model_dir)
        grid.addWidget(QLabel("模型存放資料夾(下載/快取)", holder), row, 0)
        grid.addWidget(self.edt_model_dir, row, 1)
        grid.addWidget(btn_model, row, 2)

        row += 1
        box_params = QGroupBox("訓練參數", holder)
        pgb = QGridLayout(box_params)
        self.edt_epochs = QLineEdit("50", box_params)
        self.edt_imgsz = QLineEdit("640", box_params)
        self.edt_batch = QLineEdit("16", box_params)
        self.edt_eta_dp = QLineEdit("1", box_params)
        self.edt_device = QLineEdit("", box_params)
        self.chk_resume = QCheckBox("resume（接著上次中斷續跑）", box_params)
        self.chk_skip_unlabeled = QCheckBox("跳過無標記圖片（detect/seg/pose/obb）", box_params)
        self.chk_delete_temp = QCheckBox("訓練後刪除解壓暫存", box_params)
        self.chk_skip_unlabeled.setChecked(True)
        self.chk_delete_temp.setChecked(True)

        pgb.addWidget(QLabel("epochs", box_params), 0, 0)
        pgb.addWidget(self.edt_epochs, 0, 1)
        pgb.addWidget(QLabel("imgsz", box_params), 0, 2)
        pgb.addWidget(self.edt_imgsz, 0, 3)
        pgb.addWidget(QLabel("batch", box_params), 0, 4)
        pgb.addWidget(self.edt_batch, 0, 5)
        pgb.addWidget(QLabel("device（空白自動 / cpu / 0 / 0,1）", box_params), 1, 0)
        pgb.addWidget(self.edt_device, 1, 1)
        pgb.addWidget(QLabel("ETA 小數位數（<=0 不顯示）", box_params), 1, 2)
        pgb.addWidget(self.edt_eta_dp, 1, 3)
        pgb.addWidget(self.chk_resume, 2, 0)
        pgb.addWidget(self.chk_skip_unlabeled, 2, 1, 1, 3)
        pgb.addWidget(self.chk_delete_temp, 2, 4, 1, 2)
        pgb.setColumnStretch(1, 1)
        pgb.setColumnStretch(3, 1)
        pgb.setColumnStretch(5, 1)
        grid.addWidget(box_params, row, 0, 1, 3)

        row += 1
        ctrl = QHBoxLayout()
        self.btn_start = QPushButton("開始訓練", holder)
        self.btn_stop = QPushButton("停止", holder)
        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self.start_train)
        self.btn_stop.clicked.connect(self.stop_train)
        self.btn_stop.installEventFilter(self)
        self.lbl_status = QLabel("就緒", holder)
        self.lbl_run_device = QLabel("裝置：未知", holder)
        ctrl.addWidget(self.btn_start)
        ctrl.addWidget(self.btn_stop)
        ctrl.addWidget(self.lbl_status)
        ctrl.addWidget(self.lbl_run_device)
        ctrl.addStretch(1)
        ctrl_w = QWidget(holder)
        ctrl_w.setLayout(ctrl)
        grid.addWidget(ctrl_w, row, 0, 1, 3)

        row += 1
        box_progress = QGroupBox("進度", holder)
        p = QVBoxLayout(box_progress)
        self.lbl_ep_text = QLabel("Epoch: 0/0", box_progress)
        self.pbar_epoch = QProgressBar(box_progress)
        self.lbl_ba_text = QLabel("Batch: 0/0", box_progress)
        self.pbar_batch = QProgressBar(box_progress)
        p.addWidget(self.lbl_ep_text)
        p.addWidget(self.pbar_epoch)
        p.addWidget(self.lbl_ba_text)
        p.addWidget(self.pbar_batch)
        grid.addWidget(box_progress, row, 0, 1, 3)

        row += 1
        split = QSplitter(Qt.Vertical, holder)
        b1 = QGroupBox("上一輪指標 / 每輪摘要", split)
        l1 = QVBoxLayout(b1)
        self.txt_metrics = QPlainTextEdit(b1)
        self.txt_metrics.setReadOnly(True)
        l1.addWidget(self.txt_metrics)
        b2 = QGroupBox("Log", split)
        l2 = QVBoxLayout(b2)
        self.txt_log = QPlainTextEdit(b2)
        self.txt_log.setReadOnly(True)
        l2.addWidget(self.txt_log)
        split.addWidget(b1)
        split.addWidget(b2)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        grid.addWidget(split, row, 0, 1, 3)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(row, 1)

    def _build_history_tab(self) -> None:
        layout = QVBoxLayout(self.tab_hist)
        self.tree = QTableWidget(self.tab_hist)
        self.tree.setColumnCount(8)
        self.tree.setHorizontalHeaderLabels(["時間", "Task", "模型", "dataset.zip", "mAP50", "Recall", "輸出zip", "run_dir"])
        self.tree.setSelectionBehavior(QTableWidget.SelectRows)
        self.tree.setSelectionMode(QTableWidget.ExtendedSelection)
        self.tree.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.tree)

        row = QHBoxLayout()
        for text, fn in [
            ("重新載入", self._load_history),
            ("查看詳細", self.show_selected_detail),
            ("開啟輸出 zip", self.open_selected_zip),
            ("開啟輸出 zip 資料夾", self.open_selected_zip_dir),
            ("載入到硬核視覺化", self.load_selected_to_hardcore),
            ("刪除選取", self.delete_selected_history),
        ]:
            btn = QPushButton(text, self.tab_hist)
            btn.clicked.connect(fn)
            row.addWidget(btn)
        row.addStretch(1)
        layout.addLayout(row)

    def _build_hardcore_tab(self) -> None:
        layout = QVBoxLayout(self.tab_hardcore)
        layout.addWidget(QLabel("此頁會顯示 run_dir 的 results.csv（最後 50 epochs）與常見輸出圖（results/PR/confusion matrix）。", self.tab_hardcore))
        self.hardcore_panel = HardcorePanel(self.tab_hardcore)
        layout.addWidget(self.hardcore_panel)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self.btn_stop and event.type() == QEvent.MouseButtonDblClick:
            self.force_stop_train()
            return True
        return super().eventFilter(watched, event)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._is_resizing = True
        self._resize_timer.start()

    def _end_resize(self) -> None:
        self._is_resizing = False
        if self._pending_msgs:
            self._flush_pending_msgs()

    def _show_info(self, title: str, text: str) -> None:
        QMessageBox.information(self, title, text)

    def _show_warning(self, title: str, text: str) -> None:
        QMessageBox.warning(self, title, text)

    def _show_error(self, title: str, text: str) -> None:
        QMessageBox.critical(self, title, text)

    def _bind_config_traces(self) -> None:
        for widget in [
            self.edt_dataset_zip,
            self.edt_work_dir,
            self.edt_out_zip_dir,
            self.edt_model_dir,
            self.edt_custom_model,
            self.edt_epochs,
            self.edt_imgsz,
            self.edt_batch,
            self.edt_eta_dp,
            self.edt_device,
        ]:
            widget.textChanged.connect(self._save_config)
        for combo in [self.cmb_task, self.cmb_model_family, self.cmb_model_size, self.cmb_model]:
            combo.currentTextChanged.connect(self._save_config)
        for chk in [self.chk_use_custom_model, self.chk_resume, self.chk_skip_unlabeled, self.chk_delete_temp]:
            chk.toggled.connect(self._save_config)

    def _sync_model_mode(self) -> None:
        use_custom = self.chk_use_custom_model.isChecked()
        for widget in [self.cmb_task, self.cmb_model_family, self.cmb_model_size, self.cmb_model, self.btn_update_models]:
            widget.setEnabled(not use_custom)
        self.edt_custom_model.setEnabled(use_custom)
        self.btn_pick_custom_model.setEnabled(use_custom)
        self._save_config()

    def _load_weights(self, try_online: bool) -> None:
        info = get_weights(CACHE_WEIGHTS, try_online=try_online)
        self._base_weights = list(info["weights"])
        self.lbl_model_source.setText(f"模型清單來源：{info['source']}（已依 任務/版本/大小 過濾）")
        self._apply_model_filters()

    def _apply_model_filters(self) -> None:
        if self._updating_model_filters:
            return
        self._updating_model_filters = True
        try:
            base = self._base_weights or get_weights(CACHE_WEIGHTS, try_online=False)["weights"]
            task = self.cmb_task.currentText().strip() or "detect"

            cur_family = self.cmb_model_family.currentText()
            families = [ALL_OPTION] + list_model_families(list(base), task)
            self.cmb_model_family.blockSignals(True)
            self.cmb_model_family.clear()
            self.cmb_model_family.addItems(families)
            self.cmb_model_family.setCurrentText(cur_family if cur_family in families else ALL_OPTION)
            self.cmb_model_family.blockSignals(False)

            family = "" if self.cmb_model_family.currentText() in ["", ALL_OPTION, "all", "ALL"] else self.cmb_model_family.currentText().strip()
            cur_size = self.cmb_model_size.currentText()
            sizes = [ALL_OPTION] + list_model_sizes(list(base), task, family)
            self.cmb_model_size.blockSignals(True)
            self.cmb_model_size.clear()
            self.cmb_model_size.addItems(sizes)
            self.cmb_model_size.setCurrentText(cur_size if cur_size in sizes else ALL_OPTION)
            self.cmb_model_size.blockSignals(False)

            size = "" if self.cmb_model_size.currentText() in ["", ALL_OPTION, "all", "ALL"] else self.cmb_model_size.currentText().strip()
            filtered = filter_weights(list(base), task, family, size)
            cur_model = self.cmb_model.currentText().strip()
            if cur_model and cur_model.lower() not in {x.lower() for x in base}:
                filtered = [cur_model] + filtered

            self.cmb_model.blockSignals(True)
            self.cmb_model.clear()
            self.cmb_model.addItems(filtered)
            if filtered:
                self.cmb_model.setCurrentText(cur_model if cur_model in filtered else filtered[0])
            self.cmb_model.blockSignals(False)
        finally:
            self._updating_model_filters = False

    def pick_dataset_zip(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "選擇資料集", default_dialog_dir(self.last_dataset_zip, self.edt_dataset_zip.text()), "Zip files (*.zip);;All files (*.*)")
        if path:
            path = normalize_path(path)
            self.edt_dataset_zip.setText(path)
            self.last_dataset_zip = path

    def pick_work_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "選擇工作資料夾", default_dialog_dir(self.last_work_dir, self.edt_work_dir.text()))
        if path:
            path = normalize_path(path)
            self.edt_work_dir.setText(path)
            self.last_work_dir = path

    def pick_out_zip_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "選擇輸出 zip 資料夾", default_dialog_dir(self.last_out_zip_dir, self.edt_out_zip_dir.text()))
        if path:
            path = normalize_path(path)
            self.edt_out_zip_dir.setText(path)
            self.last_out_zip_dir = path

    def pick_custom_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "選擇自訂模型 .pt", default_dialog_dir(self.last_custom_model, self.edt_custom_model.text()), "PyTorch weights (*.pt);;All files (*.*)")
        if path:
            path = normalize_path(path)
            self.edt_custom_model.setText(path)
            self.last_custom_model = path

    def pick_model_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "選擇模型資料夾", default_dialog_dir(self.last_model_dir, self.edt_model_dir.text()))
        if path:
            path = normalize_path(path)
            self.edt_model_dir.setText(path)
            self.last_model_dir = path

    def _sync_last_paths(self) -> None:
        self.last_dataset_zip = normalize_path(self.edt_dataset_zip.text())
        self.last_work_dir = normalize_path(self.edt_work_dir.text())
        self.last_out_zip_dir = normalize_path(self.edt_out_zip_dir.text())
        self.last_custom_model = normalize_path(self.edt_custom_model.text())
        self.last_model_dir = normalize_path(self.edt_model_dir.text())

    def _load_config(self) -> None:
        if not CONFIG_PATH.exists():
            return
        try:
            self._loading_config = True
            cfg = TrainerGuiConfig.from_file(CONFIG_PATH)
            self.cmb_task.setCurrentText(cfg.task or self.cmb_task.currentText())
            self.edt_dataset_zip.setText(normalize_path(cfg.dataset_zip or self.edt_dataset_zip.text()))
            self.edt_work_dir.setText(normalize_path(cfg.work_dir or self.edt_work_dir.text()))
            self.edt_out_zip_dir.setText(normalize_path(cfg.out_zip_dir or self.edt_out_zip_dir.text()))
            self.cmb_model.setCurrentText(cfg.model_pick or self.cmb_model.currentText())
            self.cmb_model_family.setCurrentText(ALL_OPTION if str(cfg.model_family).lower() == "all" else str(cfg.model_family))
            self.cmb_model_size.setCurrentText(ALL_OPTION if str(cfg.model_size).lower() == "all" else str(cfg.model_size))
            self.edt_model_dir.setText(normalize_path(cfg.model_dir or self.edt_model_dir.text()))
            self.chk_use_custom_model.setChecked(bool(cfg.use_custom_model))
            self.edt_custom_model.setText(normalize_path(cfg.custom_model or self.edt_custom_model.text()))
            self.edt_device.setText(cfg.device or self.edt_device.text())
            self.chk_resume.setChecked(bool(cfg.resume))
            self.chk_skip_unlabeled.setChecked(bool(cfg.skip_unlabeled))
            self.chk_delete_temp.setChecked(bool(cfg.delete_temp))
            self.edt_epochs.setText(str(int(cfg.epochs)))
            self.edt_imgsz.setText(str(int(cfg.imgsz)))
            self.edt_batch.setText(str(int(cfg.batch)))
            dp = int(cfg.eta_decimal_places)
            self.edt_eta_dp.setText(str(dp))
            self._eta_decimal_places = dp
        except Exception:
            pass
        finally:
            self._loading_config = False

    def _save_config(self) -> None:
        if self._loading_config:
            return
        try:
            cfg = TrainerGuiConfig(
                task=self.cmb_task.currentText(),
                dataset_zip=normalize_path(self.edt_dataset_zip.text()),
                work_dir=normalize_path(self.edt_work_dir.text()),
                out_zip_dir=normalize_path(self.edt_out_zip_dir.text()),
                model_pick=self.cmb_model.currentText(),
                model_family=("all" if self.cmb_model_family.currentText() in ["", ALL_OPTION] else self.cmb_model_family.currentText()),
                model_size=("all" if self.cmb_model_size.currentText() in ["", ALL_OPTION] else self.cmb_model_size.currentText()),
                model_dir=normalize_path(self.edt_model_dir.text()),
                use_custom_model=self.chk_use_custom_model.isChecked(),
                custom_model=normalize_path(self.edt_custom_model.text()),
                epochs=int(self.edt_epochs.text()),
                imgsz=int(self.edt_imgsz.text()),
                batch=int(self.edt_batch.text()),
                eta_decimal_places=int(self.edt_eta_dp.text()),
                device=self.edt_device.text(),
                resume=self.chk_resume.isChecked(),
                skip_unlabeled=self.chk_skip_unlabeled.isChecked(),
                delete_temp=self.chk_delete_temp.isChecked(),
            )
            cfg.save(CONFIG_PATH)
        except Exception:
            pass

    def start_train(self) -> None:
        self.train_controller.start_train()

    def stop_train(self) -> None:
        self.train_controller.stop_train()

    def force_stop_train(self) -> None:
        self.train_controller.force_stop_train()

    def _poll_queue(self) -> None:
        self.train_controller.poll_queue()

    def _flush_pending_msgs(self) -> None:
        self.train_controller.flush_pending_msgs()

    def _tick_eta(self) -> None:
        self.train_controller.tick_eta()

    def _load_history(self) -> None:
        self.history_controller.load_history()

    def show_selected_detail(self) -> None:
        self.history_controller.show_selected_detail()

    def open_selected_zip_dir(self) -> None:
        self.history_controller.open_selected_zip_dir()

    def load_selected_to_hardcore(self) -> None:
        self.history_controller.load_selected_to_hardcore()

    def open_selected_zip(self) -> None:
        self.history_controller.open_selected_zip()

    def delete_selected_history(self) -> None:
        self.history_controller.delete_selected_history()

    def closeEvent(self, event: QCloseEvent) -> None:
        self._save_config()
        super().closeEvent(event)
