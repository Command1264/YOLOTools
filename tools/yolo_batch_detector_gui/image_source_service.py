from __future__ import annotations

from pathlib import Path

if __package__ in {None, ""}:
    from models import ImageSourceError
else:
    from .models import ImageSourceError

SUPPORTED_IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def collect_image_paths(source_path: Path | str) -> list[Path]:
    """Collect supported image files from a file or directory.

    Args:
        source_path (Path | str): Image file or directory path.

    Returns:
        list[Path]: Sorted supported image paths.

    Raises:
        ImageSourceError: If the path is missing, unsupported, or contains no images.
    """
    path = Path(source_path)
    if not path.exists():
        raise ImageSourceError(f"輸入路徑不存在：{path}")
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ImageSourceError(f"不支援的圖片格式：{path.suffix}")
        return [path]
    if not path.is_dir():
        raise ImageSourceError(f"輸入路徑不是檔案或資料夾：{path}")

    image_paths = [
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    image_paths.sort(key=lambda item: str(item).casefold())
    if not image_paths:
        raise ImageSourceError(f"資料夾內找不到可用圖片：{path}")
    return image_paths
