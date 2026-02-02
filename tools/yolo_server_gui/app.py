from __future__ import annotations

import ipaddress
import os
import queue
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Any, Optional, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import portalocker
import yaml

from log_manager import LogContext, LogController, setup_logging
from log_viewer import LogViewer
from server import YoloServer
from tray import create_tray_icon
from tray_base import TrayBase

APP_DIR: Path = Path(__file__).resolve().parent
CONFIG_PATH: Path = APP_DIR / "yolo_server_gui_config.yml"
STARTUP_FILE: str = "yolo_server_gui_startup.cmd"
ICON_PATH: Path = APP_DIR / "yolo_server_icon.png"

CLOSE_LABELS: dict[str, str] = {
    "ask": "詢問",
    "minimize": "縮到工具列",
    "exit": "直接關閉",
}

APP_LOG_CTRL: LogController = LogController()
LOCK_FILE_NAME: str = "yolo_server_gui.lock"


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
            APP_LOG_CTRL.exception("儲存設定失敗。path=%s", str(path))


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


class SingleInstanceLock:
    def __init__(self, lock_path: Path, log_ctrl: LogController) -> None:
        self._lock_path: Path = lock_path
        self._log_ctrl: LogController = log_ctrl
        self._lock = threading.Lock()
        self._file: Optional[object] = None
        self._stop_event = threading.Event()
        self._retainer: Optional[threading.Thread] = None

    def try_acquire(self) -> bool:
        with self._lock:
            if self._file is not None:
                return True
            try:
                handle = open(self._lock_path, "a+")
                portalocker.lock(handle, portalocker.LOCK_EX | portalocker.LOCK_NB)
                self._file = handle
                return True
            except portalocker.exceptions.LockException:
                try:
                    handle.close()
                except Exception:
                    self._log_ctrl.exception("關閉鎖檔失敗。path=%s", str(self._lock_path))
                return False
            except Exception:
                self._log_ctrl.exception("取得鎖失敗。path=%s", str(self._lock_path))
                try:
                    handle.close()
                except Exception:
                    self._log_ctrl.exception("關閉鎖檔失敗。path=%s", str(self._lock_path))
                return False

    def start_retainer(self, interval_sec: float = 3.0) -> None:
        if self._retainer and self._retainer.is_alive():
            return

        def _worker() -> None:
            while not self._stop_event.is_set():
                if self._file is None:
                    self.try_acquire()
                time.sleep(interval_sec)

        self._retainer = threading.Thread(target=_worker, daemon=True)
        self._retainer.start()

    def release(self) -> None:
        with self._lock:
            if self._file is None:
                return
            try:
                portalocker.unlock(self._file)
            except Exception:
                self._log_ctrl.exception("釋放鎖失敗。path=%s", str(self._lock_path))
            try:
                self._file.close()
            except Exception:
                self._log_ctrl.exception("關閉鎖檔失敗。path=%s", str(self._lock_path))
            self._file = None

    def stop(self) -> None:
        self._stop_event.set()
        self.release()


class App(tk.Tk):
    def __init__(self, instance_lock: Optional[SingleInstanceLock] = None) -> None:
        super().__init__()
        self.title("YOLO Server")
        self.geometry("720x360")
        self.minsize(640, 320)
        self._app_icon: tk.PhotoImage = tk.PhotoImage(file=str(ICON_PATH))
        self.iconphoto(True, self._app_icon)

        self.log_context: LogContext = setup_logging(APP_DIR)
        self.logger: Logger = self.log_context.logger
        self.log_ctrl: LogController = LogController(self.logger)
        self.log_ctrl.info("GUI 啟動")

        self.cfg: AppConfig = AppConfig.load(CONFIG_PATH)
        self.server: Optional[YoloServer] = None
        self._tray_queue: queue.Queue[str] = queue.Queue()
        self._log_viewer: Optional[LogViewer] = None
        self._settings_win: Optional[tk.Toplevel] = None
        self._instance_lock: Optional[SingleInstanceLock] = instance_lock
        self.tray: TrayBase = create_tray_icon(
            tooltip="YOLO Server",
            on_exit=self._enqueue_tray_exit,
            on_show=self._enqueue_tray_show,
            icon_path=str(ICON_PATH),
        )

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close_request)

        self._apply_config_to_ui()
        self._apply_startup_setting()
        self.after(200, self._poll_tray_queue)
        self.tray.start()
        if self.cfg.auto_start_server:
            self.after(200, self._start_server)

    def _build_ui(self) -> None:
        self._build_menu()

        wrap: ttk.Frame = ttk.Frame(self, padding=16)
        wrap.pack(fill="both", expand=True)

        r: int = 0
        ttk.Label(wrap, text="模型路徑").grid(row=r, column=0, sticky="w", pady=6)
        self.var_model_path: tk.StringVar = tk.StringVar()
        self.ent_model: ttk.Entry = ttk.Entry(wrap, textvariable=self.var_model_path)
        self.ent_model.grid(row=r, column=1, sticky="we", padx=8)
        self.btn_pick_model: ttk.Button = ttk.Button(wrap, text="選擇...", command=self._pick_model)
        self.btn_pick_model.grid(row=r, column=2, sticky="e")

        r += 1
        ttk.Label(wrap, text="IP").grid(row=r, column=0, sticky="w", pady=6)
        self.var_host: tk.StringVar = tk.StringVar()
        self.var_port: tk.StringVar = tk.StringVar()
        ip_row: ttk.Frame = ttk.Frame(wrap)
        ip_row.grid(row=r, column=1, sticky="we", padx=8)
        self.ent_host: ttk.Entry = ttk.Entry(ip_row, textvariable=self.var_host)
        self.ent_host.pack(side="left", fill="x", expand=True)
        ttk.Label(ip_row, text="Port").pack(side="left", padx=(12, 6))
        self.ent_port: ttk.Entry = ttk.Entry(ip_row, textvariable=self.var_port, width=10)
        self.ent_port.pack(side="left")

        r += 1
        self.btn_toggle: ttk.Button = ttk.Button(wrap, text="啟動伺服器", command=self._toggle_server)
        self.btn_toggle.grid(row=r, column=0, sticky="w", pady=12)
        self.var_status: tk.StringVar = tk.StringVar(value="狀態：未啟動")
        ttk.Label(wrap, textvariable=self.var_status).grid(row=r, column=1, sticky="w", padx=8)

        r += 1
        self.var_device: tk.StringVar = tk.StringVar(value="裝置：未啟動")
        ttk.Label(wrap, textvariable=self.var_device).grid(row=r, column=1, sticky="w", padx=8)

        wrap.grid_columnconfigure(1, weight=1)

        self.ent_host.bind("<FocusOut>", lambda _e: self._apply_quick_settings(False))
        self.ent_port.bind("<FocusOut>", lambda _e: self._apply_quick_settings(False))

    def _build_menu(self) -> None:
        menubar: tk.Menu = tk.Menu(self)
        menu_more: tk.Menu = tk.Menu(menubar, tearoff=0)
        menu_more.add_command(label="瀏覽運行日誌", command=self._open_log_viewer)
        menu_more.add_command(label="設定...", command=self._open_settings)
        menubar.add_cascade(label="更多功能", menu=menu_more)
        self.config(menu=menubar)

    def _apply_config_to_ui(self) -> None:
        self.var_model_path.set(self.cfg.model_path)
        self.var_host.set(self.cfg.host)
        self.var_port.set(str(self.cfg.port))

    def _pick_model(self) -> None:
        path: str = filedialog.askopenfilename(
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

    def _toggle_server(self) -> None:
        if self.server and self.server.is_running():
            self._stop_server()
        else:
            self._start_server()

    def _start_server(self) -> None:
        if not self._apply_quick_settings(True):
            return
        host: str = self.cfg.host
        port: int = int(self.cfg.port)
        model_path: str = self.cfg.model_path
        try:
            self.server = YoloServer(model_path, host, port, logger=self.logger, icon_path=str(ICON_PATH))
            self.server.start()
        except Exception as e:
            self.server = None
            self.log_ctrl.exception("啟動伺服器失敗。")
            messagebox.showerror("啟動失敗", f"無法啟動伺服器：\n{e}")
            return
        self._set_running_state(True)
        self.var_status.set(f"狀態：執行中 http://{host}:{port}")
        self.var_device.set("裝置：載入中...")
        self._load_device_async()
        self.log_ctrl.info("伺服器已啟動。host=%s port=%s", host, port)

    def _stop_server(self) -> None:
        try:
            if self.server:
                self.server.stop()
        finally:
            self.server = None
        self._set_running_state(False)
        self.var_status.set("狀態：未啟動")
        self.var_device.set("裝置：未啟動")
        self.log_ctrl.info("伺服器已停止。")

    def _set_running_state(self, running: bool) -> None:
        state: str = "disabled" if running else "normal"
        self.ent_model.configure(state=state)
        self.ent_host.configure(state=state)
        self.ent_port.configure(state=state)
        self.btn_pick_model.configure(state=state)
        self.btn_toggle.configure(text="停止伺服器" if running else "啟動伺服器")

    def _open_log_viewer(self) -> None:
        if self._log_viewer is not None:
            try:
                self._log_viewer.lift()
                return
            except Exception:
                self.log_ctrl.exception("無法提升日誌視窗。")

        def _on_close() -> None:
            self._log_viewer = None

        self.log_ctrl.info("開啟運行日誌視窗。")
        self._log_viewer = LogViewer(self, on_close=_on_close)

    def _open_settings(self) -> None:
        if getattr(self, "_settings_win", None) is not None:
            try:
                self._settings_win.lift()
                return
            except Exception:
                self.log_ctrl.exception("無法提升設定視窗。")
        top: tk.Toplevel = tk.Toplevel(self)
        self._settings_win = top
        top.title("設定")
        top.resizable(False, False)
        top.transient(self)
        top.grab_set()

        wrap: ttk.Frame = ttk.Frame(top, padding=16)
        wrap.pack(fill="both", expand=True)

        var_auto_start: tk.BooleanVar = tk.BooleanVar(value=bool(self.cfg.auto_start_server))
        var_launch_startup: tk.BooleanVar = tk.BooleanVar(value=bool(self.cfg.launch_on_startup))
        close_key: str = self.cfg.close_behavior
        var_close_behavior: tk.StringVar = tk.StringVar(value=CLOSE_LABELS.get(close_key, "詢問"))

        ttk.Checkbutton(
            wrap, text="啟動時自動開啟伺服器", variable=var_auto_start
        ).grid(row=0, column=0, sticky="w", pady=6)
        ttk.Checkbutton(
            wrap, text="開機自動啟動應用程式", variable=var_launch_startup
        ).grid(row=1, column=0, sticky="w", pady=6)

        ttk.Label(wrap, text="關閉按鈕行為").grid(row=3, column=0, sticky="w", pady=6)
        cmb_close: ttk.Combobox = ttk.Combobox(
            wrap,
            textvariable=var_close_behavior,
            state="readonly",
            values=list(CLOSE_LABELS.values()),
            width=16,
        )
        cmb_close.grid(row=3, column=1, sticky="w", padx=6)

        btns: ttk.Frame = ttk.Frame(wrap)
        btns.grid(row=4, column=0, columnspan=3, sticky="e", pady=(10, 0))
        ttk.Button(btns, text="取消", command=lambda: _close(False)).pack(side="right")
        ttk.Button(btns, text="儲存", command=lambda: _close(True)).pack(side="right", padx=6)

        def _close(save_ok: bool) -> None:
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

    def _enqueue_tray_show(self) -> None:
        try:
            self._tray_queue.put_nowait("show")
        except Exception:
            self.log_ctrl.exception("加入顯示事件到 tray 佇列失敗。")

    def _enqueue_tray_exit(self) -> None:
        try:
            self._tray_queue.put_nowait("exit")
        except Exception:
            self.log_ctrl.exception("加入退出事件到 tray 佇列失敗。")

    def _poll_tray_queue(self) -> None:
        try:
            while True:
                action: str = self._tray_queue.get_nowait()
                if action == "show":
                    self._restore_window()
                elif action == "exit":
                    self._exit_app()
        except queue.Empty:
            pass
        self.after(200, self._poll_tray_queue)

    def _center_dialog(self, top: tk.Toplevel) -> None:
        top.update_idletasks()
        w: int = top.winfo_reqwidth()
        h: int = top.winfo_reqheight()
        x: int = self.winfo_rootx() + (self.winfo_width() - w) // 2
        y: int = self.winfo_rooty() + (self.winfo_height() - h) // 2
        top.geometry(f"{w}x{h}+{x}+{y}")

    def _apply_startup_setting(self) -> None:
        if os.name != "nt":
            return
        enabled: bool = bool(self.cfg.launch_on_startup)
        path: Optional[Path] = startup_cmd_path()
        if not path:
            return
        try:
            if enabled:
                cmd: str = build_startup_command()
                path.write_text(f"@echo off\n{cmd}\n", encoding="utf-8")
            else:
                if path.exists():
                    path.unlink()
        except Exception:
            messagebox.showwarning("設定提醒", "無法更新開機啟動設定")

    def _on_close_request(self) -> None:
        behavior: str = self.cfg.close_behavior
        if behavior == "ask":
            res: Optional[bool] = messagebox.askyesnocancel(
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

    def _minimize_to_tray(self) -> None:
        if os.name != "nt":
            self._exit_app()
            return
        self.withdraw()
        self.tray.start()

    def _restore_window(self) -> None:
        try:
            self.deiconify()
            self.lift()
            self.focus_force()
        except Exception:
            self.log_ctrl.exception("還原視窗失敗。")

    def _exit_app(self) -> None:
        try:
            self._stop_server()
        except Exception:
            self.log_ctrl.exception("停止伺服器時發生錯誤。")
        try:
            self.tray.stop()
        except Exception:
            self.log_ctrl.exception("停止 tray 時發生錯誤。")
        try:
            if self._instance_lock is not None:
                self._instance_lock.stop()
        except Exception:
            self.log_ctrl.exception("釋放執行鎖失敗。")
        self.log_ctrl.info("GUI 結束。")
        self.destroy()

    def _load_device_async(self) -> None:
        def _worker() -> None:
            device_name: str = "unknown"
            try:
                if self.server:
                    device_name = self.server.get_device_name()
            except Exception:
                device_name = "unknown"
            self.after(0, lambda: self._apply_device_name(device_name))

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

    def _apply_device_name(self, device_name: str) -> None:
        if not self.server:
            return
        self.var_device.set(f"裝置：{device_name}")
        self.log_ctrl.info("使用裝置：%s", device_name)


if __name__ == "__main__":
    lock_path = Path(tempfile.gettempdir()) / LOCK_FILE_NAME
    instance_lock = SingleInstanceLock(lock_path, APP_LOG_CTRL)
    allow_start = True
    if not instance_lock.try_acquire():
        temp_root = tk.Tk()
        temp_root.withdraw()
        allow_start = messagebox.askyesno("已在執行", "偵測到已有程式在執行。\n仍要開啟新的視窗嗎？")
        temp_root.destroy()
    if not allow_start:
        sys.exit(0)
    instance_lock.start_retainer()
    app = App(instance_lock=instance_lock)
    app.mainloop()
