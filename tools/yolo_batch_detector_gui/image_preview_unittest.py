from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QPixmap, QWheelEvent
from PySide6.QtWidgets import QApplication

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GUI_DIR = PROJECT_ROOT / "tools" / "yolo_batch_detector_gui"
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))

from image_preview import ImagePreviewDialog, ImagePreviewLabel


QT_APP = QApplication.instance() or QApplication([])


class _FakeDoubleClickEvent:
    def __init__(self) -> None:
        self.accepted = False

    def button(self) -> Qt.MouseButton:
        return Qt.LeftButton

    def accept(self) -> None:
        self.accepted = True


class ImagePreviewLabelTests(unittest.TestCase):
    """Cover image preview scaling and open request behavior."""

    def test_preview_scales_pixmap_up_to_fit_label_bounds(self) -> None:
        """Preview image should fill the available bounds while preserving aspect ratio."""
        label = ImagePreviewLabel()
        label.resize(800, 600)
        pixmap = QPixmap(100, 50)
        pixmap.fill(Qt.red)

        label.set_source_pixmap(pixmap, Path("sample.jpg"))
        rendered = label.pixmap()

        self.assertIsNotNone(rendered)
        self.assertEqual(rendered.size().width(), 800)
        self.assertEqual(rendered.size().height(), 400)

    def test_left_double_click_emits_open_request_for_current_image(self) -> None:
        """Double-clicking the preview should request a full-image window."""
        label = ImagePreviewLabel()
        label.resize(400, 300)
        pixmap = QPixmap(100, 50)
        pixmap.fill(Qt.red)
        requested_paths: list[Path] = []
        label.open_requested.connect(lambda path: requested_paths.append(path))
        label.set_source_pixmap(pixmap, Path("sample.jpg"))

        event = _FakeDoubleClickEvent()
        label.mouseDoubleClickEvent(event)

        self.assertEqual(requested_paths, [Path("sample.jpg")])
        self.assertTrue(event.accepted)


class ImagePreviewDialogTests(unittest.TestCase):
    """Cover full-image preview dialog scaling behavior."""

    def _send_wheel_event(self, dialog: ImagePreviewDialog, position: QPoint, delta_y: int) -> QWheelEvent:
        event = QWheelEvent(
            QPointF(position),
            QPointF(position),
            QPoint(0, 0),
            QPoint(0, delta_y),
            Qt.NoButton,
            Qt.NoModifier,
            Qt.ScrollUpdate,
            False,
        )
        QApplication.sendEvent(dialog.scroll_area.viewport(), event)
        return event

    def test_dialog_fits_image_to_available_window(self) -> None:
        """Full-image dialog should fit the image to the visible viewport."""
        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.png"
            pixmap = QPixmap(100, 50)
            pixmap.fill(Qt.red)
            self.assertTrue(pixmap.save(str(image_path)))

            dialog = ImagePreviewDialog(image_path)
            dialog.resize(800, 600)
            dialog.show()
            QT_APP.processEvents()

            rendered = dialog.preview_label.pixmap()
            viewport_size = dialog.scroll_area.viewport().size()

            self.assertIsNotNone(rendered)
            self.assertLessEqual(rendered.size().width(), viewport_size.width())
            self.assertLessEqual(rendered.size().height(), viewport_size.height())
            self.assertGreater(rendered.size().width(), 100)
            self.assertAlmostEqual(rendered.size().width() / rendered.size().height(), 2.0, places=1)
            dialog.close()

    def test_dialog_wheel_zooms_image(self) -> None:
        """Wheel scrolling over the full-image dialog should zoom the image."""
        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.png"
            pixmap = QPixmap(100, 50)
            pixmap.fill(Qt.red)
            self.assertTrue(pixmap.save(str(image_path)))

            dialog = ImagePreviewDialog(image_path)
            dialog.resize(800, 600)
            dialog.show()
            QT_APP.processEvents()
            initial_size = dialog.preview_label.pixmap().size()

            event = self._send_wheel_event(dialog, QPoint(20, 20), 120)
            QT_APP.processEvents()
            zoomed_size = dialog.preview_label.pixmap().size()

            self.assertGreater(zoomed_size.width(), initial_size.width())
            self.assertGreater(zoomed_size.height(), initial_size.height())
            self.assertTrue(event.isAccepted())
            dialog.close()

    def test_dialog_wheel_zoom_keeps_cursor_anchor_visible(self) -> None:
        """Zooming near the right side should scroll toward that cursor anchor."""
        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.png"
            pixmap = QPixmap(100, 50)
            pixmap.fill(Qt.red)
            self.assertTrue(pixmap.save(str(image_path)))

            dialog = ImagePreviewDialog(image_path)
            dialog.resize(800, 600)
            dialog.show()
            QT_APP.processEvents()

            self._send_wheel_event(dialog, QPoint(600, 200), 120)
            QT_APP.processEvents()

            self.assertGreater(dialog.scroll_area.horizontalScrollBar().value(), 0)
            dialog.close()

    def test_dialog_reset_button_is_persistent_and_returns_to_fit_mode(self) -> None:
        """Reset button should stay visible and only enable after manual zooming."""
        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.png"
            pixmap = QPixmap(100, 50)
            pixmap.fill(Qt.red)
            self.assertTrue(pixmap.save(str(image_path)))

            dialog = ImagePreviewDialog(image_path)
            dialog.resize(800, 600)
            dialog.show()
            QT_APP.processEvents()
            self.assertTrue(dialog.reset_zoom_button.isVisible())
            self.assertFalse(dialog.reset_zoom_button.isEnabled())

            dialog.zoom_by_wheel_delta(120, QPoint(600, 200))
            QT_APP.processEvents()
            self.assertTrue(dialog.reset_zoom_button.isEnabled())

            dialog.reset_zoom_button.click()
            QT_APP.processEvents()
            rendered = dialog.preview_label.pixmap()
            viewport_size = dialog.scroll_area.viewport().size()

            self.assertTrue(dialog.reset_zoom_button.isVisible())
            self.assertFalse(dialog.reset_zoom_button.isEnabled())
            self.assertLessEqual(rendered.size().width(), viewport_size.width())
            self.assertLessEqual(rendered.size().height(), viewport_size.height())
            dialog.close()

    def test_dialog_shows_current_zoom_factor(self) -> None:
        """Zoom factor label should reflect the current preview zoom."""
        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.png"
            pixmap = QPixmap(100, 50)
            pixmap.fill(Qt.red)
            self.assertTrue(pixmap.save(str(image_path)))

            dialog = ImagePreviewDialog(image_path)
            dialog.resize(800, 600)
            dialog.show()
            QT_APP.processEvents()

            self.assertEqual(dialog.zoom_factor_label.text(), "倍率：1.00x")

            dialog.zoom_by_wheel_delta(120, QPoint(600, 200))
            QT_APP.processEvents()

            self.assertEqual(dialog.zoom_factor_label.text(), "倍率：1.25x")
            dialog.close()

    def test_dialog_limits_relative_zoom_to_five_times(self) -> None:
        """Wheel zoom should not exceed five times the current fit size."""
        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.png"
            pixmap = QPixmap(100, 50)
            pixmap.fill(Qt.red)
            self.assertTrue(pixmap.save(str(image_path)))

            dialog = ImagePreviewDialog(image_path)
            dialog.resize(800, 600)
            dialog.show()
            QT_APP.processEvents()
            fit_zoom_factor = dialog.preview_label.zoom_factor

            for _ in range(20):
                dialog.zoom_by_wheel_delta(120, QPoint(600, 200))
            QT_APP.processEvents()

            self.assertAlmostEqual(dialog.preview_label.zoom_factor, fit_zoom_factor * 5.0)
            self.assertEqual(dialog.zoom_factor_label.text(), "倍率：5.00x")
            dialog.close()

    def test_dialog_fit_baseline_updates_with_window_size(self) -> None:
        """Fit mode should keep relative zoom at one while resizing its baseline."""
        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.png"
            pixmap = QPixmap(100, 50)
            pixmap.fill(Qt.red)
            self.assertTrue(pixmap.save(str(image_path)))

            dialog = ImagePreviewDialog(image_path)
            dialog.resize(500, 400)
            dialog.show()
            QT_APP.processEvents()
            initial_width = dialog.preview_label.pixmap().size().width()

            dialog.resize(900, 700)
            QT_APP.processEvents()
            QT_APP.processEvents()
            resized_width = dialog.preview_label.pixmap().size().width()

            self.assertGreater(resized_width, initial_width)
            self.assertEqual(dialog.zoom_factor_label.text(), "倍率：1.00x")
            self.assertFalse(dialog.reset_zoom_button.isEnabled())
            dialog.close()

    def test_dialog_left_drag_pans_in_both_directions(self) -> None:
        """Holding left mouse button and dragging should pan the zoomed image."""
        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.png"
            pixmap = QPixmap(1000, 1000)
            pixmap.fill(Qt.red)
            self.assertTrue(pixmap.save(str(image_path)))

            dialog = ImagePreviewDialog(image_path)
            dialog.resize(800, 600)
            dialog.show()
            QT_APP.processEvents()
            for _ in range(4):
                dialog.zoom_by_wheel_delta(120, QPoint(600, 400))
            QT_APP.processEvents()

            horizontal_scrollbar = dialog.scroll_area.horizontalScrollBar()
            vertical_scrollbar = dialog.scroll_area.verticalScrollBar()
            horizontal_scrollbar.setValue(100)
            vertical_scrollbar.setValue(100)
            initial_horizontal = horizontal_scrollbar.value()
            initial_vertical = vertical_scrollbar.value()

            press_event = QMouseEvent(
                QEvent.MouseButtonPress,
                QPointF(400, 300),
                QPointF(400, 300),
                Qt.LeftButton,
                Qt.LeftButton,
                Qt.NoModifier,
            )
            move_event = QMouseEvent(
                QEvent.MouseMove,
                QPointF(350, 250),
                QPointF(350, 250),
                Qt.NoButton,
                Qt.LeftButton,
                Qt.NoModifier,
            )
            release_event = QMouseEvent(
                QEvent.MouseButtonRelease,
                QPointF(350, 250),
                QPointF(350, 250),
                Qt.LeftButton,
                Qt.NoButton,
                Qt.NoModifier,
            )

            QApplication.sendEvent(dialog.scroll_area.viewport(), press_event)
            QApplication.sendEvent(dialog.scroll_area.viewport(), move_event)
            QApplication.sendEvent(dialog.scroll_area.viewport(), release_event)

            self.assertGreater(horizontal_scrollbar.value(), initial_horizontal)
            self.assertGreater(vertical_scrollbar.value(), initial_vertical)
            dialog.close()


if __name__ == "__main__":
    unittest.main()
