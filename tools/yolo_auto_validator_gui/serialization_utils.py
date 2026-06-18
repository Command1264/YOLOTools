from __future__ import annotations

from pathlib import Path


def to_builtin_jsonable(value: object) -> object:
    """Convert nested values into JSON-serializable builtins.

    Args:
        value (object): Arbitrary value.

    Returns:
        object: Builtin JSON-serializable value.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_builtin_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_builtin_jsonable(item) for item in value]

    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return to_builtin_jsonable(item_method())
        except Exception:
            pass

    tolist_method = getattr(value, "tolist", None)
    if callable(tolist_method):
        try:
            return to_builtin_jsonable(tolist_method())
        except Exception:
            pass

    return str(value)
