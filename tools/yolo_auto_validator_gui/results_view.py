from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
    QComboBox,
)

if __package__ in {None, ""}:
    from results_store import DatasetResultRecord, ValidationRunResult
else:
    from .results_store import DatasetResultRecord, ValidationRunResult


class ResultsView(QWidget):
    """驗證結果檢視元件。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._records: dict[str, DatasetResultRecord] = {}
        self._pixmaps: list[QPixmap] = []
        self._dialogs: list[QWidget] = []

        layout = QVBoxLayout(self)
        self.cmb_result = QComboBox(self)
        self.cmb_result.currentIndexChanged.connect(self._on_result_changed)
        layout.addWidget(self.cmb_result)

        splitter = QSplitter(Qt.Vertical, self)
        layout.addWidget(splitter, 1)

        self.txt_summary = QPlainTextEdit(self)
        self.txt_summary.setReadOnly(True)
        splitter.addWidget(self.txt_summary)

        image_holder = QWidget(self)
        image_layout = QVBoxLayout(image_holder)
        image_layout.setContentsMargins(0, 0, 0, 0)
        self._image_scroll = QScrollArea(self)
        self._image_scroll.setWidgetResizable(True)
        self._image_content = QWidget(self._image_scroll)
        self._image_layout = QHBoxLayout(self._image_content)
        self._image_layout.setContentsMargins(0, 0, 0, 0)
        self._image_scroll.setWidget(self._image_content)
        image_layout.addWidget(self._image_scroll)
        splitter.addWidget(image_holder)
        splitter.setSizes([220, 340])

    def set_run_result(self, run_result: ValidationRunResult) -> None:
        """載入整輪驗證結果。"""
        self._records.clear()
        self.cmb_result.blockSignals(True)
        self.cmb_result.clear()
        for record in run_result.dataset_results:
            self._records[record.dataset_key] = record
            self.cmb_result.addItem(f"資料集：{record.title}", record.dataset_key)
        aggregate_key = f"{run_result.aggregate_result.dataset_key}__aggregate"
        self._records[aggregate_key] = run_result.aggregate_result
        self.cmb_result.addItem(f"總結果：{run_result.aggregate_result.title}", aggregate_key)
        self.cmb_result.blockSignals(False)
        if self.cmb_result.count() > 0:
            self.cmb_result.setCurrentIndex(self.cmb_result.count() - 1)
            self._render_record(self._records[self.cmb_result.currentData()])

    def clear_result(self) -> None:
        """清空顯示內容。"""
        self._records.clear()
        self.cmb_result.clear()
        self.txt_summary.clear()
        self._clear_images()

    def _on_result_changed(self) -> None:
        key = self.cmb_result.currentData()
        if not key or key not in self._records:
            return
        self._render_record(self._records[key])

    def _render_record(self, record: DatasetResultRecord) -> None:
        lines = [
            f"標題：{record.title}",
            f"結果資料夾：{record.run_dir}",
            "",
            "主要指標：",
        ]
        for key, value in sorted(record.metrics.items()):
            lines.append(f"- {key}: {value:.5f}")
        lines.extend(
            [
                "",
                f"空標註圖片：{'是' if record.contains_unlabeled_images else '否'}",
                f"空標註數量：{record.empty_label_count}",
                f"真正 Background -> Background：{record.background_true_negative_count}",
                f"Background -> 前景總數（背景誤報）：{int(round(sum(row[-1] for row in record.confusion.raw_matrix[:-1])))}",
                f"前景 -> Background 總數（漏檢）：{int(round(sum(record.confusion.raw_matrix[-1][:-1])))}",
                "",
                "混淆矩陣標籤：",
                ", ".join(record.confusion.labels),
            ]
        )
        self.txt_summary.setPlainText("\n".join(lines))
        self._render_images(record)

    def _render_images(self, record: DatasetResultRecord) -> None:
        self._clear_images()
        if not record.plot_files:
            label = QLabel("目前找不到輸出圖表。", self._image_content)
            self._image_layout.addWidget(label)
            self._image_layout.addStretch(1)
            return
        for name, path_text in record.plot_files.items():
            pixmap = QPixmap(path_text)
            if pixmap.isNull():
                continue
            thumb = pixmap.scaled(320, 220, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self._pixmaps.append(thumb)
            card = QFrame(self._image_content)
            card_layout = QVBoxLayout(card)
            card_layout.addWidget(QLabel(name, card))
            button = QPushButton(card)
            button.setIcon(QIcon(thumb))
            button.setIconSize(thumb.size())
            button.setFixedSize(thumb.width() + 8, thumb.height() + 8)
            button.clicked.connect(lambda _=False, p=Path(path_text), n=name: self._open_plot(n, p))
            card_layout.addWidget(button)
            self._image_layout.addWidget(card)
        self._image_layout.addStretch(1)

    def _clear_images(self) -> None:
        while self._image_layout.count():
            item = self._image_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._pixmaps.clear()

    def _open_plot(self, title: str, path: Path) -> None:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            return
        dialog = QWidget(self, Qt.Window)
        dialog.setWindowTitle(title)
        dialog.resize(1280, 900)
        layout = QVBoxLayout(dialog)
        label = QLabel(dialog)
        label.setAlignment(Qt.AlignCenter)
        scaled = pixmap.scaled(1240, 860, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled)
        layout.addWidget(label)
        dialog.destroyed.connect(lambda _=None, d=dialog: self._dialogs.remove(d) if d in self._dialogs else None)
        self._dialogs.append(dialog)
        dialog.show()
