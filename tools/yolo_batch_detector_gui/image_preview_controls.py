from __future__ import annotations

from PySide6.QtCore import QEvent, QPoint, QSize, Signal, Qt
from PySide6.QtGui import QMouseEvent, QPixmap, QWheelEvent
from PySide6.QtWidgets import QLabel, QScrollArea, QSizePolicy, QWidget


class ZoomableImagePreviewLabel(QLabel):
    """Label that renders a source pixmap with an explicit zoom factor."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the zoomable preview label.

        Args:
            parent (QWidget | None): Parent widget.
        """
        super().__init__(parent)
        self._source_pixmap: QPixmap | None = None
        self._zoom_factor = 1.0
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setText("尚未選擇圖片")

    @property
    def zoom_factor(self) -> float:
        """Return the current zoom factor."""
        return self._zoom_factor

    @property
    def has_source_image(self) -> bool:
        """Return whether a non-null source image is loaded."""
        return self._source_pixmap is not None and not self._source_pixmap.isNull()

    def source_size(self) -> QSize:
        """Return the original source pixmap size.

        Returns:
            QSize: Original image size, or an empty size when no image is loaded.
        """
        if self._source_pixmap is None:
            return QSize()
        return self._source_pixmap.size()

    def set_source_pixmap(self, pixmap: QPixmap) -> None:
        """Set the source pixmap and render it at the current zoom factor.

        Args:
            pixmap (QPixmap): Source pixmap.
        """
        self._source_pixmap = QPixmap(pixmap)
        self._refresh_zoomed_pixmap()

    def clear_preview(self, message: str = "尚未選擇圖片") -> None:
        """Clear the image preview and show a placeholder message.

        Args:
            message (str): Placeholder text to display.
        """
        self._source_pixmap = None
        self.setPixmap(QPixmap())
        self.setText(message)

    def set_zoom_factor(self, zoom_factor: float) -> None:
        """Set the zoom factor and redraw the preview image.

        Args:
            zoom_factor (float): New zoom factor.
        """
        self._zoom_factor = max(0.01, zoom_factor)
        self._refresh_zoomed_pixmap()

    def _refresh_zoomed_pixmap(self) -> None:
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return
        source_size = self._source_pixmap.size()
        target_size = QSize(
            max(1, round(source_size.width() * self._zoom_factor)),
            max(1, round(source_size.height() * self._zoom_factor)),
        )
        scaled = self._source_pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setText("")
        self.setPixmap(scaled)
        self.resize(scaled.size())


class ImagePreviewScrollArea(QScrollArea):
    """Scroll area that converts mouse-wheel movement into zoom requests."""

    zoom_requested = Signal(int, QPoint)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the image preview scroll area.

        Args:
            parent (QWidget | None): Parent widget.
        """
        super().__init__(parent)
        self._drag_start_position: QPoint | None = None
        self._drag_start_horizontal_value = 0
        self._drag_start_vertical_value = 0
        self.viewport().installEventFilter(self)
        self.viewport().setCursor(Qt.OpenHandCursor)

    def eventFilter(self, watched: object, event: QEvent) -> bool:  # noqa: N802
        """Handle viewport mouse events for drag panning."""
        if watched is self.viewport() and isinstance(event, QMouseEvent):
            if event.type() == QEvent.Type.MouseButtonPress:
                return self._handle_mouse_press(event)
            if event.type() == QEvent.Type.MouseMove:
                return self._handle_mouse_move(event)
            if event.type() == QEvent.Type.MouseButtonRelease:
                return self._handle_mouse_release(event)
        return super().eventFilter(watched, event)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        """Emit a zoom request when the mouse wheel moves over the preview."""
        delta_y = event.angleDelta().y()
        if delta_y != 0:
            self.zoom_requested.emit(delta_y, event.position().toPoint())
            event.accept()
            return
        super().wheelEvent(event)

    def _handle_mouse_press(self, event: QMouseEvent) -> bool:
        if event.button() != Qt.LeftButton:
            return False
        self._drag_start_position = event.position().toPoint()
        self._drag_start_horizontal_value = self.horizontalScrollBar().value()
        self._drag_start_vertical_value = self.verticalScrollBar().value()
        self.viewport().setCursor(Qt.ClosedHandCursor)
        event.accept()
        return True

    def _handle_mouse_move(self, event: QMouseEvent) -> bool:
        if self._drag_start_position is None or not event.buttons() & Qt.LeftButton:
            return False
        delta = event.position().toPoint() - self._drag_start_position
        self.horizontalScrollBar().setValue(self._drag_start_horizontal_value - delta.x())
        self.verticalScrollBar().setValue(self._drag_start_vertical_value - delta.y())
        event.accept()
        return True

    def _handle_mouse_release(self, event: QMouseEvent) -> bool:
        if event.button() != Qt.LeftButton or self._drag_start_position is None:
            return False
        self._drag_start_position = None
        self.viewport().setCursor(Qt.OpenHandCursor)
        event.accept()
        return True
