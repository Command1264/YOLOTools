from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from config_service import _coerce_bool, _coerce_int, normalize_path

CLOSE_LABELS: dict[str, str] = {
    "ask": "詢問",
    "minimize": "縮到工具列",
    "exit": "直接關閉",
}


@dataclass
class AppConfig:
    """儲存 GUI 與啟動相關設定。"""

    model_path: str = ""
    host: str = "127.0.0.1"
    port: int = 60922
    auto_start_server: bool = False
    launch_on_startup: bool = False
    close_behavior: str = "ask"

    @classmethod
    def from_dict(cls, data: Any) -> "AppConfig":
        """Deserialize config from dict payload."""
        cfg = cls()
        if not isinstance(data, dict):
            return cfg
        cfg.model_path = normalize_path(str(data.get("model_path", cfg.model_path)))
        cfg.host = str(data.get("host", cfg.host)).strip() or cfg.host
        cfg.port = _coerce_int(data.get("port", cfg.port), cfg.port)
        cfg.auto_start_server = _coerce_bool(
            data.get("auto_start_server", cfg.auto_start_server), cfg.auto_start_server
        )
        cfg.launch_on_startup = _coerce_bool(
            data.get("launch_on_startup", cfg.launch_on_startup), cfg.launch_on_startup
        )
        behavior = str(data.get("close_behavior", cfg.close_behavior)).strip()
        cfg.close_behavior = behavior if behavior in CLOSE_LABELS else cfg.close_behavior
        return cfg

    @classmethod
    def load(cls, path: Path) -> "AppConfig":
        """Load YAML config file."""
        if not path.exists():
            return cls()
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            return cls()
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to dict."""
        return {
            "model_path": normalize_path(self.model_path),
            "host": self.host,
            "port": int(self.port),
            "auto_start_server": bool(self.auto_start_server),
            "launch_on_startup": bool(self.launch_on_startup),
            "close_behavior": self.close_behavior,
        }

    def save(self, path: Path) -> None:
        """Write YAML config file."""
        path.write_text(
            yaml.safe_dump(self.to_dict(), allow_unicode=False, sort_keys=False),
            encoding="utf-8",
        )
