from __future__ import annotations

if __package__ in {None, ""}:
    from config_store import DetectorConfig, save_config
    from path_helpers import text_path_dir
else:
    from .config_store import DetectorConfig, save_config
    from .path_helpers import text_path_dir


class MainWindowConfigMixin:
    """Apply and persist GUI configuration values."""

    def _connect_config_signals(self) -> None:
        """Connect UI changes to config persistence."""
        self.ent_model.editingFinished.connect(self._save_config)
        self.ent_input.editingFinished.connect(self._save_config)
        self.ent_output.editingFinished.connect(self._save_config)
        self.ent_device.editingFinished.connect(self._save_config)
        self.chk_csv.toggled.connect(self._save_config)
        self.chk_json.toggled.connect(self._save_config)
        self.chk_annotated.toggled.connect(self._save_config)
        self.sp_conf.valueChanged.connect(self._save_config)
        self.sp_iou.valueChanged.connect(self._save_config)

    def _apply_config(self, config: DetectorConfig) -> None:
        """Apply saved config values to the UI."""
        self._loading_config = True
        try:
            self.ent_model.setText(config.model_path)
            self.ent_input.setText(config.input_path)
            self.ent_output.setText(config.output_dir)
            self.chk_csv.setChecked(config.export_csv)
            self.chk_json.setChecked(config.export_json)
            self.chk_annotated.setChecked(config.export_annotated_images)
            self.sp_conf.setValue(config.conf)
            self.sp_iou.setValue(config.iou)
            self.ent_device.setText(config.device)
            self._last_model_dir = text_path_dir(config.model_path)
            self._last_input_dir = text_path_dir(config.input_path)
            self._last_output_dir = text_path_dir(config.output_dir)
        finally:
            self._loading_config = False

    def _current_config(self) -> DetectorConfig:
        """Build a config model from the current UI state."""
        return DetectorConfig(
            model_path=self.ent_model.text().strip(),
            input_path=self.ent_input.text().strip(),
            output_dir=self.ent_output.text().strip(),
            export_csv=self.chk_csv.isChecked(),
            export_json=self.chk_json.isChecked(),
            export_annotated_images=self.chk_annotated.isChecked(),
            conf=float(self.sp_conf.value()),
            iou=float(self.sp_iou.value()),
            device=self.ent_device.text().strip(),
        )

    def _save_config(self) -> None:
        """Persist current GUI settings."""
        if self._loading_config:
            return
        save_config(self._config_path, self._current_config())
