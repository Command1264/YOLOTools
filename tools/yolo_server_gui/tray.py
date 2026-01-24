# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from typing import Callable, Optional

try:
    if os.name != "nt":
        import pystray
        from PIL import Image, ImageDraw
    else:
        pystray = None
        Image = None
        ImageDraw = None
except Exception:
    pystray = None
    Image = None
    ImageDraw = None

from tray_win import TrayIcon as _WinTrayIcon


class _NoopTray:
    def __init__(
        self,
        tooltip: str,
        on_exit: Optional[Callable[[], None]] = None,
        on_show: Optional[Callable[[], None]] = None,
        icon_path: Optional[str] = None,
    ):
        self.tooltip = tooltip
        self.on_exit = on_exit
        self.on_show = on_show
        self.icon_path = icon_path

    def start(self) -> None:
        return

    def stop(self) -> None:
        return


class _PystrayTray:
    def __init__(
        self,
        tooltip: str,
        on_exit: Optional[Callable[[], None]] = None,
        on_show: Optional[Callable[[], None]] = None,
        icon_path: Optional[str] = None,
    ):
        self.tooltip = tooltip
        self.on_exit = on_exit
        self.on_show = on_show
        self.icon_path = icon_path
        self._icon = None

    def start(self) -> None:
        if self._icon is not None:
            return
        if pystray is None:
            return
        image = self._load_image()
        menu = (
            pystray.Menu(
                pystray.MenuItem("顯示", self._handle_show),
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

    def _handle_show(self, _icon=None, _item=None):
        if self.on_show:
            self.on_show()

    def _handle_exit(self, _icon=None, _item=None):
        if self.on_exit:
            self.on_exit()

    def _load_image(self):
        if self.icon_path and os.path.exists(self.icon_path):
            try:
                return Image.open(self.icon_path)
            except Exception:
                pass
        return self._default_image()

    @staticmethod
    def _default_image():
        if Image is None:
            return None
        img = Image.new("RGB", (64, 64), color=(48, 52, 65))
        draw = ImageDraw.Draw(img)
        draw.rectangle((12, 12, 52, 52), outline=(235, 235, 235), width=4)
        return img


def TrayIcon(
    tooltip: str,
    on_exit: Optional[Callable[[], None]] = None,
    on_show: Optional[Callable[[], None]] = None,
    icon_path: Optional[str] = None,
):
    if pystray is None:
        if os.name == "nt":
            return _WinTrayIcon(tooltip=tooltip, on_exit=on_exit, on_show=on_show, icon_path=icon_path)
        return _NoopTray(tooltip=tooltip, on_exit=on_exit, on_show=on_show, icon_path=icon_path)
    return _PystrayTray(tooltip=tooltip, on_exit=on_exit, on_show=on_show, icon_path=icon_path)
