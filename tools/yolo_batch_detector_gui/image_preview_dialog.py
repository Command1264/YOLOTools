from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, QPoint, QTimer, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

try:
    from image_preview_controls import ImagePreviewScrollArea, ZoomableImagePreviewLabel
except ImportError:  # pragma: no cover - package import path
    from .image_preview_controls import ImagePreviewScrollArea, ZoomableImagePreviewLabel


class ImagePreviewDialog(QWidget):
    """Top-level window for detailed full-image inspection."""

    _MIN_ZOOM_FACTOR = 0.05
    _MAX_ZOOM_FACTOR = 5.0
    _WHEEL_ZOOM_STEP = 1.25

    def __init__(self, image_path: Path, parent: QWidget | None = None) -> None:
        """Initialize the full-image preview window.

        Args:
            image_path (Path): Image path to display.
            parent (QWidget | None): Parent widget.
        """
        super().__init__(parent, Qt.Window)
        self.setWindowTitle(f"圖片檢視：{image_path.name}")
        self.resize(1280, 900)
        self._fit_zoom_factor = 1.0
        self._fit_mode_enabled = True
        self._relative_zoom_factor = 1.0

        layout = QVBoxLayout(self)
        self.scroll_area = ImagePreviewScrollArea(self)
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.scroll_area.zoom_requested.connect(self.zoom_by_wheel_delta)
        self.preview_label = ZoomableImagePreviewLabel(self.scroll_area)
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self.preview_label.clear_preview("無法載入圖片")
        else:
            self.preview_label.set_source_pixmap(pixmap)
        self.scroll_area.setWidget(self.preview_label)
        layout.addWidget(self.scroll_area, 1)

        footer_layout = QHBoxLayout()
        footer_layout.addStretch(1)
        self.zoom_factor_label = QLabel("倍率：--", self)
        self.zoom_factor_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        footer_layout.addWidget(self.zoom_factor_label)
        self.reset_zoom_button = QPushButton("重設縮放", self)
        self.reset_zoom_button.setEnabled(False)
        self.reset_zoom_button.clicked.connect(self.fit_to_window)
        footer_layout.addWidget(self.reset_zoom_button)
        layout.addLayout(footer_layout)
        QTimer.singleShot(0, self.fit_to_window)

    def fit_to_window(self) -> None:
        """Scale the preview image to fit inside the visible viewport."""
        if not self.preview_label.has_source_image:
            return
        source_size = self.preview_label.source_size()
        viewport_size = self.scroll_area.viewport().size()
        if (
            source_size.width() <= 0
            or source_size.height() <= 0
            or viewport_size.width() <= 0
            or viewport_size.height() <= 0
        ):
            return
        zoom_factor = min(
            viewport_size.width() / source_size.width(),
            viewport_size.height() / source_size.height(),
        )
        self._fit_zoom_factor = zoom_factor
        self._fit_mode_enabled = True
        self._relative_zoom_factor = 1.0
        self.preview_label.set_zoom_factor(self._absolute_zoom_factor())
        self._update_zoom_controls()

    def zoom_by_wheel_delta(self, delta_y: int, anchor_position: QPoint | None = None) -> None:
        """Zoom the preview image in or out from a mouse-wheel delta.

        Args:
            delta_y (int): Vertical mouse-wheel delta.
            anchor_position (QPoint | None): Mouse position inside the viewport.
        """
        if delta_y == 0 or not self.preview_label.has_source_image:
            return
        if anchor_position is None:
            viewport_size = self.scroll_area.viewport().size()
            anchor_position = QPoint(viewport_size.width() // 2, viewport_size.height() // 2)

        old_label_size = self.preview_label.size()
        old_label_position = self.preview_label.pos()
        anchor_in_label = anchor_position - old_label_position
        anchor_ratio_x = self._anchor_ratio(anchor_in_label.x(), old_label_size.width())
        anchor_ratio_y = self._anchor_ratio(anchor_in_label.y(), old_label_size.height())

        self._fit_mode_enabled = False
        zoom_step = self._WHEEL_ZOOM_STEP if delta_y > 0 else 1 / self._WHEEL_ZOOM_STEP
        self._relative_zoom_factor *= zoom_step
        self._relative_zoom_factor = min(
            self._MAX_ZOOM_FACTOR,
            max(self._MIN_ZOOM_FACTOR, self._relative_zoom_factor),
        )
        self.preview_label.set_zoom_factor(self._absolute_zoom_factor())
        self._scroll_to_anchor(anchor_position, anchor_ratio_x, anchor_ratio_y)
        self._update_zoom_controls()

    def resizeEvent(self, event: QEvent) -> None:  # noqa: N802
        """Keep the image fit to the window while fit mode is active."""
        super().resizeEvent(event)
        if self._fit_mode_enabled:
            QTimer.singleShot(0, self.fit_to_window)

    def _scroll_to_anchor(self, anchor_position: QPoint, anchor_ratio_x: float, anchor_ratio_y: float) -> None:
        new_label_size = self.preview_label.size()
        target_horizontal = round(new_label_size.width() * anchor_ratio_x - anchor_position.x())
        target_vertical = round(new_label_size.height() * anchor_ratio_y - anchor_position.y())
        self.scroll_area.horizontalScrollBar().setValue(target_horizontal)
        self.scroll_area.verticalScrollBar().setValue(target_vertical)

    def _absolute_zoom_factor(self) -> float:
        return self._fit_zoom_factor * self._relative_zoom_factor

    def _update_zoom_controls(self) -> None:
        self.zoom_factor_label.setText(f"倍率：{self._relative_zoom_factor:.2f}x")
        self.reset_zoom_button.setEnabled(self.preview_label.has_source_image and not self._fit_mode_enabled)

    @staticmethod
    def _anchor_ratio(anchor_coordinate: int, length: int) -> float:
        if length <= 0:
            return 0.5
        return max(0.0, min(1.0, anchor_coordinate / length))
