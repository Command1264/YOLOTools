# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from typing import Callable, Optional

try:
    import pystray
    from PIL import Image, ImageDraw
except Exception:
    pystray = None
    Image = None
    ImageDraw = None

from tray_base import TrayBase
from tray_win import TrayIcon as _WinTrayIcon


class _NoopTray(TrayBase):
    def __init__(
        self,
        tooltip: str,
        on_exit: Optional[Callable[[], None]] = None,
        on_show: Optional[Callable[[], None]] = None,
        icon_path: Optional[str] = None,
    ) -> None:
        super().__init__(tooltip, on_exit=on_exit, on_show=on_show, icon_path=icon_path)

    def start(self) -> None:
        return

    def stop(self) -> None:
        return


class _PystrayTray(TrayBase):
    def __init__(
        self,
        tooltip: str,
        on_exit: Optional[Callable[[], None]] = None,
        on_show: Optional[Callable[[], None]] = None,
        icon_path: Optional[str] = None,
    ) -> None:
        super().__init__(tooltip, on_exit=on_exit, on_show=on_show, icon_path=icon_path)
        self._icon: Optional[object] = None

    def start(self) -> None:
        if self._icon is not None:
            return
        if pystray is None:
            return
        image: Optional[object] = self._load_image()
        menu: Optional[object] = (
            pystray.Menu(
                pystray.MenuItem("顯示", self._handle_show, default=True),
                pystray.MenuItem("關閉", self._handle_exit),
            )
            if self.on_show or self.on_exit
            else None
        )
        self._icon = pystray.Icon("yolo_server_gui", image, self.tooltip, menu)
        self._icon.run_detached()

    def stop(self) -> None:
        if self._icon is None:
            return
        try:
            self._icon.stop()
        finally:
            self._icon = None

    def _handle_show(self, _icon: Optional[object] = None, _item: Optional[object] = None) -> None:
        if self.on_show:
            self.on_show()

    def _handle_exit(self, _icon: Optional[object] = None, _item: Optional[object] = None) -> None:
        if self.on_exit:
            self.on_exit()

    def _load_image(self) -> Optional[object]:
        if self.icon_path and os.path.exists(self.icon_path):
            try:
                return Image.open(self.icon_path)
            except Exception:
                pass
        return self._default_image()

    @staticmethod
    def _default_image() -> Optional[object]:
        if Image is None:
            return None
        img: object = Image.new("RGB", (64, 64), color=(48, 52, 65))
        draw: object = ImageDraw.Draw(img)
        draw.rectangle((12, 12, 52, 52), outline=(235, 235, 235), width=4)
        return img


def create_tray_icon(
    tooltip: str,
    on_exit: Optional[Callable[[], None]] = None,
    on_show: Optional[Callable[[], None]] = None,
    icon_path: Optional[str] = None,
) -> TrayBase:
    if pystray is None:
        if os.name == "nt":
            return _WinTrayIcon(tooltip=tooltip, on_exit=on_exit, on_show=on_show, icon_path=icon_path)
        return _NoopTray(tooltip=tooltip, on_exit=on_exit, on_show=on_show, icon_path=icon_path)
    return _PystrayTray(tooltip=tooltip, on_exit=on_exit, on_show=on_show, icon_path=icon_path)
