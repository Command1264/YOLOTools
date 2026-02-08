from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


def load_results_rows(run_dir: Path) -> List[dict]:
    """Load rows from Ultralytics `results.csv`."""
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return []
    import csv

    with csv_path.open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def pick_plot_files(run_dir: Path) -> Dict[str, Path]:
    """Pick common output plot files if they exist."""
    candidates = {
        "results.png": run_dir / "results.png",
        "PR_curve.png": run_dir / "PR_curve.png",
        "P_curve.png": run_dir / "P_curve.png",
        "R_curve.png": run_dir / "R_curve.png",
        "F1_curve.png": run_dir / "F1_curve.png",
        "confusion_matrix.png": run_dir / "confusion_matrix.png",
        "confusion_matrix_normalized.png": run_dir / "confusion_matrix_normalized.png",
    }
    return {name: path for name, path in candidates.items() if path.exists()}


class _ImageDialog(QDialog):
    """Fullscreen-like image preview dialog with responsive scaling."""

    def __init__(self, name: str, pixmap: QPixmap, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(name)
        self.resize(1200, 800)
        self._orig = pixmap

        self._label = QLabel(self)
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.addWidget(self._label)
        self._redraw()

    def resizeEvent(self, event):  # type: ignore[override]
        super().resizeEvent(event)
        self._redraw()

    def _redraw(self) -> None:
        if self._orig.isNull():
            self._label.clear()
            return
        scaled = self._orig.scaled(
            max(1, self.width() - 24),
            max(1, self.height() - 24),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._label.setPixmap(scaled)


class HardcorePanel(QWidget):
    """
    Panel showing:
    - epoch table from `results.csv`
    - plot thumbnails that can be opened in a larger window
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._thumb_refs: List[QPixmap] = []
        self._active_dialogs: List[QDialog] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        self.table = QTableWidget(self)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        root.addWidget(self.table, stretch=3)

        self.img_box = QFrame(self)
        self.img_box.setFrameShape(QFrame.StyledPanel)
        img_box_layout = QVBoxLayout(self.img_box)
        img_box_layout.setContentsMargins(8, 8, 8, 8)
        img_box_layout.setSpacing(6)
        img_box_layout.addWidget(QLabel("訓練輸出圖表（縮圖）", self.img_box))

        self._img_scroll = QScrollArea(self.img_box)
        self._img_scroll.setWidgetResizable(True)
        self._img_content = QWidget(self._img_scroll)
        self.img_layout = QHBoxLayout(self._img_content)
        self.img_layout.setContentsMargins(0, 0, 0, 0)
        self.img_layout.setSpacing(8)
        self._img_scroll.setWidget(self._img_content)
        img_box_layout.addWidget(self._img_scroll)

        root.addWidget(self.img_box, stretch=2)

    def clear(self) -> None:
        """Clear table and thumbnails."""
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        while self.img_layout.count():
            item = self.img_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._thumb_refs.clear()

    def load_run(self, run_dir: str) -> None:
        """Load run outputs from `run_dir`."""
        self.clear()
        rd = Path(run_dir)
        rows = load_results_rows(rd)
        if rows:
            cols = list(rows[0].keys())
            preferred = [
                "epoch",
                "train/box_loss",
                "train/cls_loss",
                "train/dfl_loss",
                "val/box_loss",
                "val/cls_loss",
                "val/dfl_loss",
                "metrics/precision(B)",
                "metrics/recall(B)",
                "metrics/mAP50(B)",
                "metrics/mAP50-95(B)",
            ]
            chosen = [col for col in preferred if col in cols] or cols[:12]
            self.table.setColumnCount(len(chosen))
            self.table.setHorizontalHeaderLabels(chosen)

            selected_rows = rows[-50:]
            self.table.setRowCount(len(selected_rows))
            for row_idx, row in enumerate(selected_rows):
                for col_idx, col in enumerate(chosen):
                    self.table.setItem(row_idx, col_idx, QTableWidgetItem(str(row.get(col, ""))))
            self.table.resizeColumnsToContents()

        plots = pick_plot_files(rd)
        if not plots:
            self.img_layout.addWidget(
                QLabel("(找不到 results.png / PR_curve.png / confusion_matrix.png 等輸出圖)", self._img_content)
            )
            self.img_layout.addStretch(1)
            return

        for name, path in plots.items():
            self._add_thumb(name, path)
        self.img_layout.addStretch(1)

    def _add_thumb(self, name: str, path: Path) -> None:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            return
        thumb = pixmap.scaled(320, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._thumb_refs.append(thumb)

        card = QFrame(self._img_content)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(4, 4, 4, 4)
        card_layout.setSpacing(4)

        title = QLabel(name, card)
        title.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(title)

        btn = QPushButton(card)
        btn.setIconSize(thumb.size())
        btn.setFixedSize(thumb.width() + 8, thumb.height() + 8)
        btn.setStyleSheet("text-align:center;")
        btn.setIcon(QIcon(thumb))
        btn.clicked.connect(lambda _=False, n=name, p=path: self._open_full(n, p))
        card_layout.addWidget(btn, alignment=Qt.AlignCenter)

        self.img_layout.addWidget(card)

    def _open_full(self, name: str, path: Path) -> None:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            return
        dialog = _ImageDialog(name, pixmap, self)
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        dialog.destroyed.connect(lambda _=None, d=dialog: self._active_dialogs.remove(d) if d in self._active_dialogs else None)
        self._active_dialogs.append(dialog)
        dialog.showMaximized()
