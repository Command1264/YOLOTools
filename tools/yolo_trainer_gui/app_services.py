from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from history_model import HistoryRecord

APP_DIR = Path(__file__).resolve().parent
CACHE_WEIGHTS = APP_DIR / "weights_cache.json"
HISTORY_PATH = APP_DIR / "train_history.jsonl"
CONFIG_PATH = APP_DIR / "trainer_config.json"
ALL_OPTION = "全部"


def now_str() -> str:
    """Return current local timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_path(path_str: str) -> str:
    """Normalize path text to POSIX separators."""
    if not path_str:
        return ""
    try:
        return Path(path_str).as_posix()
    except Exception:
        return path_str.replace("\\", "/")


def default_dialog_dir(last_path: str, current_path: str = "") -> str:
    """Resolve best initial directory for a file dialog.

    Args:
        last_path (str): Last selected path.
        current_path (str): Current form value.

    Returns:
        str: Existing directory path.
    """
    for candidate in [last_path, current_path]:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return str(path if path.is_dir() else path.parent)
        if path.parent.exists():
            return str(path.parent)
    return str(APP_DIR)


def append_history(record: Dict[str, Any]) -> None:
    """Append one training result record."""
    with HISTORY_PATH.open("a", encoding="utf-8") as file:
        file.write(HistoryRecord(data=dict(record)).to_json_line() + "\n")


def read_history(limit: int = 500) -> list[Dict[str, Any]]:
    """Read recent training history records.

    Args:
        limit (int): Maximum number of records from the tail.

    Returns:
        list[Dict[str, Any]]: Parsed history records.
    """
    if not HISTORY_PATH.exists():
        return []
    out: list[Dict[str, Any]] = []
    with HISTORY_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(HistoryRecord.from_json_line(line).data)
            except Exception:
                pass
    return out[-limit:]
