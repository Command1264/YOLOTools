from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QMouseEvent, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QWidget

try:
    from image_preview_dialog import ImagePreviewDialog
except ImportError:  # pragma: no cover - package import path
    from .image_preview_dialog import ImagePreviewDialog


class ImagePreviewLabel(QLabel):
    """Preview label that scales a source pixmap to the available bounds."""

    open_requested = Signal(Path)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the preview label.

        Args:
            parent (QWidget | None): Parent widget.
        """
        super().__init__(parent)
        self._source_pixmap: QPixmap | None = None
        self._source_path: Path | None = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(420, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setText("尚未選擇圖片")

    def set_source_pixmap(self, pixmap: QPixmap, source_path: Path) -> None:
        """Set the image to preview and scale it immediately.

        Args:
            pixmap (QPixmap): Source pixmap.
            source_path (Path): Image path associated with this pixmap.
        """
        self._source_pixmap = QPixmap(pixmap)
        self._source_path = source_path
        self._refresh_scaled_pixmap()

    def clear_preview(self, message: str = "尚未選擇圖片") -> None:
        """Clear the image preview and show a placeholder message.

        Args:
            message (str): Placeholder text to display.
        """
        self._source_pixmap = None
        self._source_path = None
        self.setPixmap(QPixmap())
        self.setText(message)

    def resizeEvent(self, event) -> None:  # noqa: N802
        """Rescale the source pixmap whenever the preview bounds change."""
        super().resizeEvent(event)
        self._refresh_scaled_pixmap()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """Emit an open request when the current image is double-clicked."""
        if event.button() == Qt.LeftButton and self._source_path is not None:
            self.open_requested.emit(self._source_path)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def _refresh_scaled_pixmap(self) -> None:
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return
        target_size = self.contentsRect().size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            return
        scaled = self._source_pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setText("")
        self.setPixmap(scaled)


__all__ = ["ImagePreviewDialog", "ImagePreviewLabel"]
