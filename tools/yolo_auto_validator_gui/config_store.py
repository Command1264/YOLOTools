from __future__ import annotations

from pathlib import Path

if __package__ in {None, ""}:
    from config_model import ValidatorAppConfig
else:
    from .config_model import ValidatorAppConfig


class ConfigStore:
    """設定檔存取服務。"""

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        """回傳設定檔路徑。"""
        return self._path

    def load(self) -> ValidatorAppConfig:
        """讀取設定。"""
        if not self._path.exists():
            return ValidatorAppConfig()
        try:
            return ValidatorAppConfig.from_file(self._path)
        except Exception:
            return ValidatorAppConfig()

    def save(self, config: ValidatorAppConfig) -> None:
        """保存設定。"""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        config.save(self._path)
