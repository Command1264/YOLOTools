from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from config_service import normalize_path, validate_host, validate_port, validate_worker_count
from server import YoloServer
from yolo_engine import YoloEngine


class ServerController:
    """Handle server lifecycle and dependency/device status updates."""

    def __init__(
        self,
        app: Any,
        config_path: Path,
        exec_dir: Path,
        select_icon_path: Callable[[], Optional[Path]],
    ) -> None:
        self.app = app
        self.config_path = config_path
        self.exec_dir = exec_dir
        self.select_icon_path = select_icon_path

    def apply_quick_settings(self, require_model: bool) -> bool:
        """Validate and persist quick settings from main form."""
        model_path = normalize_path(self.app.ent_model.text().strip())
        if require_model:
            if not model_path:
                self.app._show_error("設定錯誤", "請選擇模型路徑")
                self.app.ent_model.setText(self.app.cfg.model_path)
                return False
            if not Path(model_path).exists():
                self.app._show_error("設定錯誤", "模型路徑不存在")
                self.app.ent_model.setText(self.app.cfg.model_path)
                return False
        elif model_path and not Path(model_path).exists():
            self.app._show_error("設定錯誤", "模型路徑不存在")
            self.app.ent_model.setText(self.app.cfg.model_path)
            return False

        host_ok, host_val = validate_host(self.app.ent_host.text())
        if not host_ok:
            self.app._show_error("設定錯誤", host_val)
            self.app.ent_host.setText(self.app.cfg.host)
            return False

        port_ok, port_val, port_msg = validate_port(self.app.ent_port.text())
        if not port_ok:
            self.app._show_error("設定錯誤", port_msg)
            self.app.ent_port.setText(str(self.app.cfg.port))
            return False

        worker_ok, worker_val, worker_msg = validate_worker_count(self.app.ent_worker.text())
        if not worker_ok:
            self.app._show_error("設定錯誤", worker_msg)
            self.app.ent_worker.setText(str(self.app.cfg.worker_count))
            return False

        self.app.cfg.model_path = model_path
        self.app.cfg.host = host_val
        self.app.cfg.port = int(port_val)
        self.app.cfg.worker_count = int(worker_val)
        self.app.cfg.save(self.config_path)
        self.app.ent_model.setText(model_path)
        self.app.ent_host.setText(host_val)
        self.app.ent_port.setText(str(port_val))
        self.app.ent_worker.setText(str(worker_val))
        return True

    def toggle_server(self) -> None:
        """Toggle between start and stop."""
        if self.app.server and self.app.server.is_running():
            self.stop_server()
        else:
            self.start_server()

    def start_server(self) -> None:
        """Start YOLO server with current settings."""
        if not self.apply_quick_settings(True):
            return
        host: str = self.app.cfg.host
        port: int = int(self.app.cfg.port)
        model_path: str = self.app.cfg.model_path
        worker_count: int = int(self.app.cfg.worker_count)
        try:
            server_icon_path = self.select_icon_path()
            icon_path = str(server_icon_path) if server_icon_path else None
            if self.app.server is None:
                self.app.server = YoloServer(
                    model_path,
                    host,
                    port,
                    logger=self.app.logger,
                    icon_path=icon_path,
                    worker_count=worker_count,
                )
            else:
                self.app.server.update_settings(
                    model_path=model_path,
                    host=host,
                    port=port,
                    icon_path=icon_path,
                    worker_count=worker_count,
                )
            self.app.server.start()
        except Exception as exc:
            self.app.server = None
            self.app.log_ctrl.exception("啟動伺服器失敗。")
            self.app._show_error("啟動失敗", f"無法啟動伺服器：\n{exc}")
            return
        self.set_running_state(True)
        self.app.lbl_status.setText(f"狀態：執行中 http://{host}:{port}")
        self.app.lbl_device.setText("裝置：載入中...")
        self.load_device_async()
        self.app.log_ctrl.info("伺服器已啟動。host=%s port=%s worker_count=%s", host, port, worker_count)

    def stop_server(self) -> None:
        """Stop running server."""
        try:
            if self.app.server:
                self.app.server.stop()
        finally:
            self.app.server = None
        self.set_running_state(False)
        self.app.lbl_status.setText("狀態：未啟動")
        self.app.lbl_device.setText("裝置：未啟動")
        self.app.log_ctrl.info("伺服器已停止。")

    def set_running_state(self, running: bool) -> None:
        """Update form controls based on server running state."""
        self.app.ent_model.setEnabled(not running)
        self.app.ent_host.setEnabled(not running)
        self.app.ent_port.setEnabled(not running)
        self.app.ent_worker.setEnabled(not running)
        self.app.btn_pick_model.setEnabled(not running)
        self.app.btn_toggle.setText("停止伺服器" if running else "啟動伺服器")
        self.update_toggle_state()

    def try_auto_start(self) -> None:
        """Auto-start server after dependencies are ready."""
        if self.app.server and self.app.server.is_running():
            return
        if self.app._deps_ready:
            self.start_server()
            return
        if self.app._deps_loading:
            self.app._timer_single_shot(300, self.try_auto_start)
            return
        self.app.log_ctrl.warning("自動啟動失敗：核心套件未就緒。")

    def init_dep_status(self) -> None:
        """Initialize dependency status checking."""
        self.start_dep_preload()

    def set_dep_status(self, text: str) -> None:
        """Set dependency status text."""
        self.app.lbl_dep_status.setText(text)
        self.update_toggle_state()

    def update_toggle_state(self) -> None:
        """Enable toggle button only when available."""
        if self.app.server and self.app.server.is_running():
            self.app.btn_toggle.setEnabled(True)
            return
        self.app.btn_toggle.setEnabled(self.app._deps_ready)

    def start_dep_preload(self) -> None:
        """Start dependency preload worker."""
        if self.app._deps_loading:
            return
        self.app._deps_token += 1
        token = self.app._deps_token
        self.app._deps_loading = True
        self.app._deps_ready = False
        self.set_dep_status("核心套件：載入中...")

        def _worker() -> None:
            error_msg: Optional[str] = None
            try:
                YoloEngine.preload_dependencies()
            except Exception as exc:
                error_msg = str(exc) or "unknown error"
            self.app._dep_result_queue.put((token, error_msg))

        threading.Thread(target=_worker, daemon=True).start()

    def on_dep_preload_done(self, token: int, error_msg: Optional[str]) -> None:
        """Apply dependency preload result from background queue."""
        if token != self.app._deps_token:
            return
        self.app._deps_loading = False
        if error_msg:
            self.app._deps_ready = False
            self.set_dep_status("核心套件：載入失敗")
            self.app.log_ctrl.warning("核心套件載入失敗：%s", error_msg)
            return
        self.app._deps_ready = True
        self.set_dep_status("核心套件：就緒")

    def load_device_async(self) -> None:
        """Query server device status on background thread without touching model init."""

        def _worker() -> None:
            device_name: str = "unknown"
            try:
                if self.app.server:
                    for _ in range(300):
                        if not self.app.server:
                            device_name = "unknown"
                            break
                        if self.app.server.server_state == "warmup_failed":
                            device_name = "warmup failed"
                            break
                        if self.app.server.is_ready:
                            device_name = self.app.server.get_device_name()
                            break
                        device_name = "loading"
                        threading.Event().wait(0.2)
            except Exception:
                device_name = "unknown"
            self.app._device_queue.put(device_name)

        threading.Thread(target=_worker, daemon=True).start()

    def apply_device_name(self, device_name: str) -> None:
        """Apply queried device name to UI."""
        if not self.app.server:
            return
        if device_name == "loading":
            self.app.lbl_device.setText("裝置：載入中...")
            return
        if device_name == "warmup failed":
            self.app.lbl_device.setText("裝置：載入失敗")
            self.app.log_ctrl.warning("模型 warmup 失敗。")
            return
        self.app.lbl_device.setText(f"裝置：{device_name}")
        self.app.log_ctrl.info("使用裝置：%s", device_name)

    def poll_dep_queue(self) -> None:
        """Poll dependency preload queue."""
        try:
            while True:
                token, error_msg = self.app._dep_result_queue.get_nowait()
                self.on_dep_preload_done(token, error_msg)
        except queue.Empty:
            return

    def poll_device_queue(self) -> None:
        """Poll device info queue."""
        try:
            while True:
                device_name = self.app._device_queue.get_nowait()
                self.apply_device_name(device_name)
        except queue.Empty:
            return
