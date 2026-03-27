from __future__ import annotations

import ipaddress
import os
import sys
from pathlib import Path
from typing import Any, Optional


STARTUP_FILE: str = "yolo_server_gui_startup.cmd"


def normalize_path(path_str: str) -> str:
    """Normalize path text to POSIX style separators."""
    if not path_str:
        return ""
    try:
        return Path(path_str).as_posix()
    except Exception:
        return path_str.replace("\\", "/")


def _coerce_bool(value: Any, default: bool) -> bool:
    """Convert generic input to bool with fallback."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def _coerce_int(value: Any, default: int) -> int:
    """Convert generic input to int with fallback."""
    try:
        return int(str(value).strip())
    except Exception:
        return default


def validate_host(value: str) -> tuple[bool, str]:
    """Validate host input and normalize localhost."""
    val = (value or "").strip()
    if not val:
        return False, "IP 不能空白"
    if val.lower() == "localhost":
        return True, "localhost"
    try:
        ipaddress.ip_address(val)
        return True, val
    except Exception:
        return False, f"無效 IP：{val}"


def validate_port(value: str) -> tuple[bool, Optional[int], str]:
    """Validate TCP port range."""
    try:
        port = int(str(value).strip())
    except Exception:
        return False, None, "Port 必須是整數"
    if port < 1 or port > 65535:
        return False, None, "Port 必須在 1~65535 之間"
    return True, port, ""


def validate_worker_count(value: str) -> tuple[bool, Optional[int], str]:
    """Validate worker count as a positive integer."""
    try:
        worker_count = int(str(value).strip())
    except Exception:
        return False, None, "Worker 必須是整數"
    if worker_count < 1:
        return False, None, "Worker 必須大於等於 1"
    if worker_count > 32:
        return False, None, "Worker 目前限制在 32 以下"
    return True, worker_count, ""


def validate_http_profile(value: str) -> tuple[bool, str, str]:
    """Validate HTTP profile option."""
    profile = str(value).strip().lower()
    if profile in {"default", "taichung_fire"}:
        return True, profile, ""
    return False, "default", "HTTP Profile 無效"


def startup_cmd_path() -> Optional[Path]:
    """Resolve Windows startup script path."""
    if os.name != "nt":
        return None
    appdata = os.environ.get("APPDATA", "")
    if not appdata:
        return None
    return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / STARTUP_FILE


def build_startup_command() -> str:
    """Build startup command for frozen/script mode."""
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}"'
    return f'"{sys.executable}" "{Path(__file__).resolve().with_name("app.py")}"'
