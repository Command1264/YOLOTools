from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

if __package__ in {None, ""}:
    from image_preview import ImagePreviewLabel
else:
    from .image_preview import ImagePreviewLabel


class MainWindowUiMixin:
    """Build the main window widgets and layouts."""

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.addWidget(self._build_input_group())
        layout.addWidget(self._build_action_group())

        splitter = QSplitter(Qt.Horizontal, root)
        splitter.addWidget(self._build_results_group())
        splitter.addWidget(self._build_detail_group())
        splitter.setSizes([620, 620])
        layout.addWidget(splitter, 1)

    def _build_input_group(self) -> QWidget:
        box = QGroupBox("模型、來源與輸出", self)
        form = QFormLayout(box)
        self.ent_model = QLineEdit(box)
        self.btn_model = QPushButton("瀏覽...", box)
        self.btn_load_labels = QPushButton("載入 labels", box)
        self.btn_model.clicked.connect(self._pick_model)
        self.btn_load_labels.clicked.connect(self._load_model_labels)
        form.addRow("YOLO 模型：", self._row(self.ent_model, self.btn_model, self.btn_load_labels))

        self.txt_labels = QPlainTextEdit(box)
        self.txt_labels.setReadOnly(True)
        self.txt_labels.setFixedHeight(92)
        form.addRow("模型 labels：", self.txt_labels)

        self.ent_input = QLineEdit(box)
        self.btn_input_file = QPushButton("圖片...", box)
        self.btn_input_folder = QPushButton("資料夾...", box)
        self.btn_input_file.clicked.connect(self._pick_image)
        self.btn_input_folder.clicked.connect(self._pick_folder)
        form.addRow("輸入來源：", self._row(self.ent_input, self.btn_input_file, self.btn_input_folder))

        self.ent_output = QLineEdit(box)
        self.btn_output = QPushButton("輸出資料夾...", box)
        self.btn_output.clicked.connect(self._pick_output_dir)
        form.addRow("輸出位置：", self._row(self.ent_output, self.btn_output))

        self.chk_csv = QCheckBox("CSV", box)
        self.chk_json = QCheckBox("JSON", box)
        self.chk_annotated = QCheckBox("標註圖片", box)
        self.chk_csv.setChecked(True)
        self.chk_json.setChecked(True)
        form.addRow("匯出格式：", self._compact_row(self.chk_csv, self.chk_json, self.chk_annotated))
        return box

    def _build_action_group(self) -> QWidget:
        box = QGroupBox("推論設定與執行", self)
        layout = QHBoxLayout(box)
        self.sp_conf = QDoubleSpinBox(box)
        self.sp_conf.setRange(0.0, 1.0)
        self.sp_conf.setDecimals(2)
        self.sp_conf.setSingleStep(0.01)
        self.sp_conf.setValue(0.25)
        self.sp_iou = QDoubleSpinBox(box)
        self.sp_iou.setRange(0.0, 1.0)
        self.sp_iou.setDecimals(2)
        self.sp_iou.setSingleStep(0.01)
        self.sp_iou.setValue(0.45)
        self.ent_device = QLineEdit(box)
        self.ent_device.setPlaceholderText("留空自動判斷")
        self.btn_start = QPushButton("開始辨識", box)
        self.btn_stop = QPushButton("停止", box)
        self.btn_export = QPushButton("匯出結果", box)
        self.progress = QProgressBar(box)
        self.lbl_status = QLabel("就緒", box)
        self.btn_start.clicked.connect(self._start_prediction)
        self.btn_stop.clicked.connect(self._stop_prediction)
        self.btn_export.clicked.connect(self._export_results)
        for label, widget in (("conf", self.sp_conf), ("iou", self.sp_iou), ("device", self.ent_device)):
            layout.addWidget(QLabel(label, box))
            layout.addWidget(widget)
        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_stop)
        layout.addWidget(self.btn_export)
        layout.addWidget(self.progress, 1)
        layout.addWidget(self.lbl_status)
        return box

    def _build_results_group(self) -> QWidget:
        box = QGroupBox("所有圖片辨識結果", self)
        layout = QVBoxLayout(box)
        self.tbl_results = QTableWidget(0, 5, box)
        self.tbl_results.setHorizontalHeaderLabels(["圖片", "分類", "labels", "bbox", "錯誤"])
        self.tbl_results.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl_results.setSelectionBehavior(QTableWidget.SelectRows)
        self.tbl_results.setSelectionMode(QTableWidget.SingleSelection)
        self.tbl_results.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tbl_results.itemSelectionChanged.connect(self._on_result_selected)
        self.tbl_results.itemDoubleClicked.connect(self._show_error_for_result_item)
        layout.addWidget(self.tbl_results, 1)
        return box

    def _build_detail_group(self) -> QWidget:
        box = QWidget(self)
        layout = QVBoxLayout(box)
        image_box = QGroupBox("單張圖片結果", box)
        image_layout = QVBoxLayout(image_box)
        self.image_scroll = QScrollArea(image_box)
        self.image_scroll.setWidgetResizable(True)
        self.lbl_image = ImagePreviewLabel(self.image_scroll)
        self.lbl_image.open_requested.connect(self._open_full_image_preview)
        self.image_scroll.setWidget(self.lbl_image)
        image_layout.addWidget(self.image_scroll, 1)
        layout.addWidget(image_box, 2)

        self.tbl_boxes = QTableWidget(0, 6, box)
        self.tbl_boxes.setHorizontalHeaderLabels(["label", "conf", "x1", "y1", "x2", "y2"])
        self.tbl_boxes.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl_boxes.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.tbl_boxes, 1)

        self.txt_summary = QPlainTextEdit(box)
        self.txt_summary.setReadOnly(True)
        self.txt_summary.setMinimumHeight(150)
        layout.addWidget(self.txt_summary, 1)
        return box

    def _row(self, *widgets: QWidget) -> QWidget:
        row = QWidget(self)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        for index, widget in enumerate(widgets):
            layout.addWidget(widget, 1 if index == 0 else 0)
        return row

    def _compact_row(self, *widgets: QWidget) -> QWidget:
        row = QWidget(self)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        for widget in widgets:
            layout.addWidget(widget, 0)
        layout.addStretch(1)
        return row
