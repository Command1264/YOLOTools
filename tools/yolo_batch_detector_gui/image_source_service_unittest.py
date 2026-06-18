from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.yolo_batch_detector_gui.image_source_service import collect_image_paths
from tools.yolo_batch_detector_gui.models import ImageSourceError


class ImageSourceServiceTests(unittest.TestCase):
    """Cover image source path discovery rules."""

    def test_collect_image_paths_recurses_and_filters_supported_extensions(self) -> None:
        """Directories should be scanned recursively and only supported images returned."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nested = root / "nested"
            nested.mkdir()
            expected = [
                root / "a.jpg",
                root / "b.JPG",
                nested / "c.webp",
                nested / "d.PNG",
            ]
            for path in expected:
                path.write_bytes(b"fake")
            (root / "note.txt").write_text("skip", encoding="utf-8")
            (nested / "movie.mp4").write_bytes(b"skip")

            result = collect_image_paths(root)

        self.assertEqual(result, sorted(expected, key=lambda item: str(item).casefold()))

    def test_collect_image_paths_returns_single_supported_file(self) -> None:
        """A supported image file should be accepted as a one-item input."""
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.jpeg"
            image_path.write_bytes(b"fake")

            result = collect_image_paths(image_path)

        self.assertEqual(result, [image_path])

    def test_collect_image_paths_rejects_missing_path(self) -> None:
        """Missing sources should raise a project error with context."""
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "missing"

            with self.assertRaises(ImageSourceError) as context:
                collect_image_paths(missing_path)

        self.assertIn("輸入路徑不存在", str(context.exception))


if __name__ == "__main__":
    unittest.main()
