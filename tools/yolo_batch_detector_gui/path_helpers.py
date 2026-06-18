from __future__ import annotations

from pathlib import Path


def text_path_dir(path_text: str) -> Path | None:
    """Resolve a text path into an existing directory when possible.

    Args:
        path_text (str): File or directory path text.

    Returns:
        Path | None: Existing directory, or None when unavailable.
    """
    raw = path_text.strip()
    if not raw:
        return None
    path = Path(raw)
    if path.exists() and path.is_dir():
        return path
    if path.parent.exists() and path.parent.is_dir():
        return path.parent
    return None
