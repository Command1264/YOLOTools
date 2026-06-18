from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QFileDialog

if __package__ in {None, ""}:
    from path_helpers import text_path_dir
else:
    from .path_helpers import text_path_dir


class MainWindowBrowseMixin:
    """Handle browse dialogs and independent last-directory memory."""

    def _pick_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "選擇 YOLO 模型",
            str(self._dialog_start_dir("model")),
            "YOLO weights (*.pt);;All files (*.*)",
        )
        if path:
            self.ent_model.setText(path)
            self._last_model_dir = Path(path).parent
            self._save_config()

    def _pick_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "選擇圖片",
            str(self._dialog_start_dir("input")),
            "Images (*.jpg *.jpeg *.png *.bmp *.webp);;All files (*.*)",
        )
        if path:
            self.ent_input.setText(path)
            self._last_input_dir = Path(path).parent
            self._save_config()

    def _pick_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "選擇圖片資料夾", str(self._dialog_start_dir("input")))
        if path:
            self.ent_input.setText(path)
            self._last_input_dir = Path(path)
            self._save_config()

    def _pick_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "選擇輸出資料夾", str(self._dialog_start_dir("output")))
        if path:
            self.ent_output.setText(path)
            self._last_output_dir = Path(path)
            self._save_config()

    def _dialog_start_dir(self, category: str) -> Path:
        """Return the initial directory for one browse category."""
        candidates: list[Path | None] = []
        if category == "model":
            candidates.extend([self._last_model_dir, text_path_dir(self.ent_model.text())])
        elif category == "input":
            candidates.extend([self._last_input_dir, text_path_dir(self.ent_input.text())])
        elif category == "output":
            candidates.extend([self._last_output_dir, text_path_dir(self.ent_output.text())])
        candidates.append(Path.cwd())
        for candidate in candidates:
            if candidate is not None and candidate.exists() and candidate.is_dir():
                return candidate
        return Path.cwd()
