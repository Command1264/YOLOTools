from __future__ import annotations

import ipaddress
import os
import queue
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import yaml

from server import YoloServer
from tray import TrayIcon

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "yolo_server_gui_config.yml"
STARTUP_FILE = "yolo_server_gui_startup.cmd"

CLOSE_LABELS = {
    "ask": "詢問",
    "minimize": "縮到工具列",
    "exit": "直接關閉",
}


def normalize_path(path_str: str) -> str:
    if not path_str:
        return ""
    try:
        return Path(path_str).as_posix()
    except Exception:
        return path_str.replace("\\", "/")


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


@dataclass
class AppConfig:
    model_path: str = ""
    host: str = "127.0.0.1"
    port: int = 60922
    auto_start_server: bool = False
    launch_on_startup: bool = False
    close_behavior: str = "ask"

    @classmethod
    def from_dict(cls, data: Any) -> "AppConfig":
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
        if not path.exists():
            return cls()
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            return cls()
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": normalize_path(self.model_path),
            "host": self.host,
            "port": int(self.port),
            "auto_start_server": bool(self.auto_start_server),
            "launch_on_startup": bool(self.launch_on_startup),
            "close_behavior": self.close_behavior,
        }

    def save(self, path: Path) -> None:
        try:
            path.write_text(
                yaml.safe_dump(self.to_dict(), allow_unicode=False, sort_keys=False),
                encoding="utf-8",
            )
        except Exception:
            pass


def validate_host(value: str) -> Tuple[bool, str]:
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


def validate_port(value: str) -> Tuple[bool, Optional[int], str]:
    try:
        port = int(str(value).strip())
    except Exception:
        return False, None, "Port 必須是整數"
    if port < 1 or port > 65535:
        return False, None, "Port 必須在 1~65535 之間"
    return True, port, ""


def startup_cmd_path() -> Optional[Path]:
    if os.name != "nt":
        return None
    appdata = os.environ.get("APPDATA", "")
    if not appdata:
        return None
    return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / STARTUP_FILE


def build_startup_command() -> str:
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}"'
    return f'"{sys.executable}" "{Path(__file__).resolve()}"'


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("YOLO Server")
        self.geometry("720x360")
        self.minsize(640, 320)

        self.cfg = AppConfig.load(CONFIG_PATH)
        self.server: Optional[YoloServer] = None
        self._tray_queue: queue.Queue[str] = queue.Queue()
        self.tray = TrayIcon(
            tooltip="YOLO Server",
            on_exit=self._enqueue_tray_exit,
            on_show=self._enqueue_tray_show,
        )

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close_request)

        self._apply_config_to_ui()
        self._apply_startup_setting()
        self.after(200, self._poll_tray_queue)
        self.tray.start()
        if self.cfg.auto_start_server:
            self.after(200, self._start_server)

    def _build_ui(self):
        self._build_menu()

        wrap = ttk.Frame(self, padding=16)
        wrap.pack(fill="both", expand=True)

        r = 0
        ttk.Label(wrap, text="模型路徑").grid(row=r, column=0, sticky="w", pady=6)
        self.var_model_path = tk.StringVar()
        self.ent_model = ttk.Entry(wrap, textvariable=self.var_model_path)
        self.ent_model.grid(row=r, column=1, sticky="we", padx=8)
        self.btn_pick_model = ttk.Button(wrap, text="選擇...", command=self._pick_model)
        self.btn_pick_model.grid(row=r, column=2, sticky="e")

        r += 1
        ttk.Label(wrap, text="IP").grid(row=r, column=0, sticky="w", pady=6)
        self.var_host = tk.StringVar()
        self.var_port = tk.StringVar()
        ip_row = ttk.Frame(wrap)
        ip_row.grid(row=r, column=1, sticky="we", padx=8)
        self.ent_host = ttk.Entry(ip_row, textvariable=self.var_host)
        self.ent_host.pack(side="left", fill="x", expand=True)
        ttk.Label(ip_row, text="Port").pack(side="left", padx=(12, 6))
        self.ent_port = ttk.Entry(ip_row, textvariable=self.var_port, width=10)
        self.ent_port.pack(side="left")

        r += 1
        self.btn_toggle = ttk.Button(wrap, text="啟動伺服器", command=self._toggle_server)
        self.btn_toggle.grid(row=r, column=0, sticky="w", pady=12)
        self.var_status = tk.StringVar(value="狀態：未啟動")
        ttk.Label(wrap, textvariable=self.var_status).grid(row=r, column=1, sticky="w", padx=8)

        wrap.grid_columnconfigure(1, weight=1)

        self.ent_host.bind("<FocusOut>", lambda _e: self._apply_quick_settings(False))
        self.ent_port.bind("<FocusOut>", lambda _e: self._apply_quick_settings(False))

    def _build_menu(self):
        menubar = tk.Menu(self)
        menu_settings = tk.Menu(menubar, tearoff=0)
        menu_settings.add_command(label="設定...", command=self._open_settings)
        menubar.add_cascade(label="設定", menu=menu_settings)
        self.config(menu=menubar)

    def _apply_config_to_ui(self):
        self.var_model_path.set(self.cfg.model_path)
        self.var_host.set(self.cfg.host)
        self.var_port.set(str(self.cfg.port))

    def _pick_model(self):
        path = filedialog.askopenfilename(
            title="選擇模型",
            initialdir=APP_DIR,
            filetypes=[("YOLO weights", "*.pt"), ("All files", "*.*")],
        )
        if path:
            self.var_model_path.set(normalize_path(path))
            self._apply_quick_settings(False)

    def _apply_quick_settings(self, require_model: bool) -> bool:
        model_path = normalize_path(self.var_model_path.get().strip())
        if require_model:
            if not model_path:
                messagebox.showerror("設定錯誤", "請選擇模型路徑")
                self.var_model_path.set(self.cfg.model_path)
                return False
            if not Path(model_path).exists():
                messagebox.showerror("設定錯誤", "模型路徑不存在")
                self.var_model_path.set(self.cfg.model_path)
                return False
        elif model_path and not Path(model_path).exists():
            messagebox.showerror("設定錯誤", "模型路徑不存在")
            self.var_model_path.set(self.cfg.model_path)
            return False

        host_ok, host_val = validate_host(self.var_host.get())
        if not host_ok:
            messagebox.showerror("設定錯誤", host_val)
            self.var_host.set(self.cfg.host)
            return False

        port_ok, port_val, port_msg = validate_port(self.var_port.get())
        if not port_ok:
            messagebox.showerror("設定錯誤", port_msg)
            self.var_port.set(str(self.cfg.port))
            return False

        self.cfg.model_path = model_path
        self.cfg.host = host_val
        self.cfg.port = port_val
        self.cfg.save(CONFIG_PATH)
        self.var_model_path.set(model_path)
        self.var_host.set(host_val)
        self.var_port.set(str(port_val))
        return True

    def _toggle_server(self):
        if self.server and self.server.is_running():
            self._stop_server()
        else:
            self._start_server()

    def _start_server(self):
        if not self._apply_quick_settings(True):
            return
        host = self.cfg.host
        port = int(self.cfg.port)
        model_path = self.cfg.model_path
        try:
            self.server = YoloServer(model_path, host, port)
            self.server.start()
        except Exception as e:
            self.server = None
            messagebox.showerror("啟動失敗", f"無法啟動伺服器：\n{e}")
            return
        self._set_running_state(True)
        self.var_status.set(f"狀態：執行中 http://{host}:{port}")

    def _stop_server(self):
        try:
            if self.server:
                self.server.stop()
        finally:
            self.server = None
        self._set_running_state(False)
        self.var_status.set("狀態：未啟動")

    def _set_running_state(self, running: bool):
        state = "disabled" if running else "normal"
        self.ent_model.configure(state=state)
        self.ent_host.configure(state=state)
        self.ent_port.configure(state=state)
        self.btn_pick_model.configure(state=state)
        self.btn_toggle.configure(text="停止伺服器" if running else "啟動伺服器")

    def _open_settings(self):
        if getattr(self, "_settings_win", None) is not None:
            try:
                self._settings_win.lift()
                return
            except Exception:
                pass
        top = tk.Toplevel(self)
        self._settings_win = top
        top.title("設定")
        top.resizable(False, False)
        top.transient(self)
        top.grab_set()

        wrap = ttk.Frame(top, padding=16)
        wrap.pack(fill="both", expand=True)

        var_auto_start = tk.BooleanVar(value=bool(self.cfg.auto_start_server))
        var_launch_startup = tk.BooleanVar(value=bool(self.cfg.launch_on_startup))
        close_key = self.cfg.close_behavior
        var_close_behavior = tk.StringVar(value=CLOSE_LABELS.get(close_key, "詢問"))

        ttk.Checkbutton(
            wrap, text="啟動時自動開啟伺服器", variable=var_auto_start
        ).grid(row=0, column=0, sticky="w", pady=6)
        ttk.Checkbutton(
            wrap, text="開機自動啟動應用程式", variable=var_launch_startup
        ).grid(row=1, column=0, sticky="w", pady=6)

        ttk.Label(wrap, text="關閉按鈕行為").grid(row=3, column=0, sticky="w", pady=6)
        cmb_close = ttk.Combobox(
            wrap,
            textvariable=var_close_behavior,
            state="readonly",
            values=list(CLOSE_LABELS.values()),
            width=16,
        )
        cmb_close.grid(row=3, column=1, sticky="w", padx=6)

        btns = ttk.Frame(wrap)
        btns.grid(row=4, column=0, columnspan=3, sticky="e", pady=(10, 0))
        ttk.Button(btns, text="取消", command=lambda: _close(False)).pack(side="right")
        ttk.Button(btns, text="儲存", command=lambda: _close(True)).pack(side="right", padx=6)

        def _close(save_ok: bool):
            if save_ok:
                self.cfg.auto_start_server = bool(var_auto_start.get())
                self.cfg.launch_on_startup = bool(var_launch_startup.get())
                label_to_key = {v: k for k, v in CLOSE_LABELS.items()}
                self.cfg.close_behavior = label_to_key.get(var_close_behavior.get(), "ask")
                self.cfg.save(CONFIG_PATH)
                self._apply_startup_setting()
                self.tray.start()
            self._settings_win = None
            top.destroy()

        top.protocol("WM_DELETE_WINDOW", lambda: _close(False))
        self._center_dialog(top)

    def _enqueue_tray_show(self):
        try:
            self._tray_queue.put_nowait("show")
        except Exception:
            pass

    def _enqueue_tray_exit(self):
        try:
            self._tray_queue.put_nowait("exit")
        except Exception:
            pass

    def _poll_tray_queue(self):
        try:
            while True:
                action = self._tray_queue.get_nowait()
                if action == "show":
                    self._restore_window()
                elif action == "exit":
                    self._exit_app()
        except queue.Empty:
            pass
        self.after(200, self._poll_tray_queue)

    def _center_dialog(self, top: tk.Toplevel):
        top.update_idletasks()
        w = top.winfo_reqwidth()
        h = top.winfo_reqheight()
        x = self.winfo_rootx() + (self.winfo_width() - w) // 2
        y = self.winfo_rooty() + (self.winfo_height() - h) // 2
        top.geometry(f"{w}x{h}+{x}+{y}")

    def _apply_startup_setting(self):
        if os.name != "nt":
            return
        enabled = bool(self.cfg.launch_on_startup)
        path = startup_cmd_path()
        if not path:
            return
        try:
            if enabled:
                cmd = build_startup_command()
                path.write_text(f"@echo off\n{cmd}\n", encoding="utf-8")
            else:
                if path.exists():
                    path.unlink()
        except Exception:
            messagebox.showwarning("設定提醒", "無法更新開機啟動設定")

    def _on_close_request(self):
        behavior = self.cfg.close_behavior
        if behavior == "ask":
            res = messagebox.askyesnocancel(
                "關閉", "要縮到工具列嗎？\n是：縮到工具列\n否：直接關閉\n取消：不動作"
            )
            if res is None:
                return
            if res:
                self.cfg.close_behavior = "minimize"
                self.cfg.save(CONFIG_PATH)
                self._minimize_to_tray()
            else:
                self.cfg.close_behavior = "exit"
                self.cfg.save(CONFIG_PATH)
                self._exit_app()
            return
        if behavior == "minimize":
            self._minimize_to_tray()
        else:
            self._exit_app()

    def _minimize_to_tray(self):
        if os.name != "nt":
            self._exit_app()
            return
        self.withdraw()
        self.tray.start()

    def _restore_window(self):
        try:
            self.deiconify()
            self.lift()
            self.focus_force()
        except Exception:
            pass

    def _exit_app(self):
        try:
            self._stop_server()
        except Exception:
            pass
        try:
            self.tray.stop()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
