# -*- coding: utf-8 -*-
from __future__ import annotations

import ctypes
import os
import threading
from ctypes import wintypes
from typing import Callable, Optional

from tray import TrayBase

PTR_SIZE: int = ctypes.sizeof(ctypes.c_void_p)
ULONG_PTR = wintypes.ULONG if PTR_SIZE == 4 else ctypes.c_uint64
HWND = getattr(wintypes, "HWND", ctypes.c_void_p)
HICON = getattr(wintypes, "HICON", ctypes.c_void_p)
HCURSOR = getattr(wintypes, "HCURSOR", ctypes.c_void_p)
HBRUSH = getattr(wintypes, "HBRUSH", ctypes.c_void_p)
HINSTANCE = getattr(wintypes, "HINSTANCE", ctypes.c_void_p)
LPCWSTR = getattr(wintypes, "LPCWSTR", ctypes.c_wchar_p)
UINT = getattr(wintypes, "UINT", ctypes.c_uint)
DWORD = getattr(wintypes, "DWORD", ctypes.c_ulong)
WCHAR = getattr(wintypes, "WCHAR", ctypes.c_wchar)
LPARAM = getattr(wintypes, "LPARAM", ctypes.c_longlong if PTR_SIZE == 8 else ctypes.c_long)
WPARAM = getattr(wintypes, "WPARAM", ULONG_PTR)
LRESULT = getattr(wintypes, "LRESULT", ctypes.c_longlong if PTR_SIZE == 8 else ctypes.c_long)
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, HWND, UINT, WPARAM, LPARAM)


class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


POINT = getattr(wintypes, "POINT", _POINT)


class _MSG(ctypes.Structure):
    _fields_ = [
        ("hwnd", HWND),
        ("message", UINT),
        ("wParam", WPARAM),
        ("lParam", LPARAM),
        ("time", DWORD),
        ("pt", POINT),
    ]


MSG = getattr(wintypes, "MSG", _MSG)

class TrayIcon(TrayBase):
    def __init__(
        self,
        tooltip: str,
        on_exit: Optional[Callable[[], None]] = None,
        on_show: Optional[Callable[[], None]] = None,
        icon_path: Optional[str] = None,
    ) -> None:
        super().__init__(tooltip, on_exit=on_exit, on_show=on_show, icon_path=icon_path)
        self._thread: Optional[threading.Thread] = None
        self._ready: threading.Event = threading.Event()
        self._hwnd: Optional[int] = None
        self._wndproc: Optional[Callable[..., int]] = None

    def start(self) -> None:
        if os.name != "nt":
            return
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(2)

    def stop(self) -> None:
        if os.name != "nt":
            return
        if self._hwnd:
            ctypes.windll.user32.PostMessageW(self._hwnd, 0x0010, 0, 0)  # WM_CLOSE
        if self._thread and self._thread.is_alive() and threading.current_thread() != self._thread:
            self._thread.join(timeout=2)
        self._thread = None
        self._hwnd = None
        self._ready.clear()

    def _run(self) -> None:
        user32 = ctypes.windll.user32
        shell32 = ctypes.windll.shell32
        kernel32 = ctypes.windll.kernel32
        user32.DefWindowProcW.argtypes = [HWND, UINT, WPARAM, LPARAM]
        user32.DefWindowProcW.restype = LRESULT
        user32.GetMessageW.argtypes = [ctypes.POINTER(MSG), HWND, UINT, UINT]
        user32.GetMessageW.restype = ctypes.c_int
        user32.TranslateMessage.argtypes = [ctypes.POINTER(MSG)]
        user32.TranslateMessage.restype = ctypes.c_bool
        user32.DispatchMessageW.argtypes = [ctypes.POINTER(MSG)]
        user32.DispatchMessageW.restype = LRESULT

        WM_USER: int = 0x0400
        WM_COMMAND: int = 0x0111
        WM_DESTROY: int = 0x0002
        WM_LBUTTONDBLCLK: int = 0x0203
        WM_RBUTTONUP: int = 0x0205

        NIF_MESSAGE: int = 0x0001
        NIF_ICON: int = 0x0002
        NIF_TIP: int = 0x0004
        NIM_ADD: int = 0x0000
        NIM_DELETE: int = 0x0002

        TPM_LEFTALIGN: int = 0x0000
        TPM_BOTTOMALIGN: int = 0x0020

        ID_TRAY_EXIT: int = 1001
        CALLBACK_MESSAGE: int = WM_USER + 20

        class WNDCLASS(ctypes.Structure):
            _fields_ = [
                ("style", UINT),
                ("lpfnWndProc", WNDPROC),
                ("cbClsExtra", ctypes.c_int),
                ("cbWndExtra", ctypes.c_int),
                ("hInstance", HINSTANCE),
                ("hIcon", HICON),
                ("hCursor", HCURSOR),
                ("hbrBackground", HBRUSH),
                ("lpszMenuName", LPCWSTR),
                ("lpszClassName", LPCWSTR),
            ]

        class NOTIFYICONDATA(ctypes.Structure):
            _fields_ = [
                ("cbSize", DWORD),
                ("hWnd", HWND),
                ("uID", UINT),
                ("uFlags", UINT),
                ("uCallbackMessage", UINT),
                ("hIcon", HICON),
                ("szTip", WCHAR * 128),
            ]

        def _load_icon() -> int:
            if self.icon_path and os.path.exists(self.icon_path):
                return user32.LoadImageW(0, self.icon_path, 1, 0, 0, 0x00000010)
            return user32.LoadIconW(0, 0x7F00)

        @WNDPROC
        def _wndproc(hwnd: int, msg: int, wparam: int, lparam: int) -> int:
            if msg == CALLBACK_MESSAGE:
                if lparam == WM_RBUTTONUP:
                    self._show_menu(hwnd, ID_TRAY_EXIT)
                elif lparam == WM_LBUTTONDBLCLK and self.on_show:
                    self.on_show()
                return 0
            if msg == WM_COMMAND:
                cmd = wparam & 0xFFFF
                if cmd == ID_TRAY_EXIT:
                    if self.on_exit:
                        self.on_exit()
                return 0
            if msg == WM_DESTROY:
                nid: NOTIFYICONDATA = NOTIFYICONDATA()
                nid.cbSize = ctypes.sizeof(NOTIFYICONDATA)
                nid.hWnd = hwnd
                nid.uID = 1
                shell32.Shell_NotifyIconW(NIM_DELETE, ctypes.byref(nid))
                user32.PostQuitMessage(0)
                return 0
            return user32.DefWindowProcW(HWND(hwnd), UINT(msg), WPARAM(wparam), LPARAM(lparam))

        self._wndproc = _wndproc
        hinst: int = kernel32.GetModuleHandleW(None)
        class_name: str = "YoloServerTrayWindow"
        wndclass: WNDCLASS = WNDCLASS()
        wndclass.lpfnWndProc = _wndproc
        wndclass.hInstance = hinst
        wndclass.lpszClassName = class_name
        user32.RegisterClassW(ctypes.byref(wndclass))

        hwnd: int = user32.CreateWindowExW(
            0,
            class_name,
            class_name,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            hinst,
            None,
        )
        self._hwnd = hwnd

        nid: NOTIFYICONDATA = NOTIFYICONDATA()
        nid.cbSize = ctypes.sizeof(NOTIFYICONDATA)
        nid.hWnd = hwnd
        nid.uID = 1
        nid.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP
        nid.uCallbackMessage = CALLBACK_MESSAGE
        nid.hIcon = _load_icon()
        nid.szTip = self.tooltip
        shell32.Shell_NotifyIconW(NIM_ADD, ctypes.byref(nid))

        self._ready.set()

        msg: MSG = MSG()
        while user32.GetMessageW(ctypes.byref(msg), 0, 0, 0) != 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
        self._hwnd = None
        self._ready.clear()

    def _show_menu(self, hwnd, exit_id: int) -> None:
        user32 = ctypes.windll.user32
        menu: int = user32.CreatePopupMenu()
        user32.AppendMenuW(menu, 0x0000, exit_id, "關閉")
        pt: POINT = POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        user32.SetForegroundWindow(hwnd)
        user32.TrackPopupMenu(
            menu,
            0x0000 | 0x0020,
            pt.x,
            pt.y,
            0,
            hwnd,
            None,
        )
        user32.PostMessageW(hwnd, 0x0000, 0, 0)
