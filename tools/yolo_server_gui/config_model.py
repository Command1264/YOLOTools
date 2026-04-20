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
LOG_LEVELS: tuple[str, ...] = ("debug", "info", "warning", "error", "critical")
HTTP_PROFILE_LABELS: dict[str, str] = {
    "default": "預設",
    "taichung_fire": "TaichungFire",
}


def _normalize_log_level(value: Any, default: str = "info") -> str:
    level = str(value).strip().lower()
    return level if level in LOG_LEVELS else default


def _normalize_http_profile(value: Any, default: str = "default") -> str:
    profile = str(value).strip().lower()
    return profile if profile in HTTP_PROFILE_LABELS else default


@dataclass
class AppConfig:
    """儲存 GUI 與啟動相關設定。"""

    model_path: str = ""
    host: str = "127.0.0.1"
    port: int = 60922
    auto_start_server: bool = False
    launch_on_startup: bool = False
    close_behavior: str = "ask"
    log_level: str = "info"
    log_base_dir: str = ""
    gpu_replica_count: int = 1
    decode_worker_count: int = 1
    http_profile: str = "default"

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
        cfg.log_level = _normalize_log_level(data.get("log_level", cfg.log_level), cfg.log_level)
        cfg.log_base_dir = normalize_path(str(data.get("log_base_dir", cfg.log_base_dir)).strip())
        gpu_replica_raw = data.get("gpu_replica_count", data.get("worker_count", cfg.gpu_replica_count))
        cfg.gpu_replica_count = max(1, _coerce_int(gpu_replica_raw, cfg.gpu_replica_count))
        decode_worker_raw = data.get("decode_worker_count", cfg.gpu_replica_count)
        cfg.decode_worker_count = max(1, _coerce_int(decode_worker_raw, cfg.gpu_replica_count))
        cfg.http_profile = _normalize_http_profile(data.get("http_profile", cfg.http_profile), cfg.http_profile)
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
            "log_level": _normalize_log_level(self.log_level),
            "log_base_dir": normalize_path(self.log_base_dir),
            "gpu_replica_count": max(1, int(self.gpu_replica_count)),
            "decode_worker_count": max(1, int(self.decode_worker_count)),
            "http_profile": _normalize_http_profile(self.http_profile),
        }

    def save(self, path: Path) -> None:
        """Write YAML config file."""
        path.write_text(
            yaml.safe_dump(self.to_dict(), allow_unicode=False, sort_keys=False),
            encoding="utf-8",
        )
