from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import uuid

from PySide6.QtCore import QObject, Signal

if __package__ in {None, ""}:
    from engine_protocol import EngineEvent, EngineRequest
    from exceptions import EngineClientError
    from logging_utils import configure_logging, get_logger
else:
    from .engine_protocol import EngineEvent, EngineRequest
    from .exceptions import EngineClientError
    from .logging_utils import configure_logging, get_logger

LOGGER = get_logger(__name__)


def _engine_entrypoint(request_queue, event_queue) -> None:
    """子行程入口點。"""
    if __package__ in {None, ""}:
        from engine_process import run_engine_loop
    else:
        from .engine_process import run_engine_loop
    run_engine_loop(request_queue, event_queue)


class ValidationEngineClient(QObject):
    """背景驗證引擎管理器。"""

    event_received = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        configure_logging("yolo_auto_validator.engine_client")
        self._ctx = mp.get_context("spawn")
        self._request_queue = None
        self._event_queue = None
        self._process = None
        self._poll_thread: threading.Thread | None = None
        self._closing = False
        self._poll_generation = 0
        LOGGER.info("Initializing validation engine client.")
        self.start()

    def start(self) -> None:
        """啟動背景引擎。"""
        if self._process is not None and self._process.is_alive():
            LOGGER.debug("Validation engine process already running. pid=%s", self._process.pid)
            return
        self._request_queue = self._ctx.Queue()
        self._event_queue = self._ctx.Queue()
        self._process = self._ctx.Process(
            target=_engine_entrypoint,
            args=(self._request_queue, self._event_queue),
            daemon=True,
        )
        self._process.start()
        self._closing = False
        self._poll_generation += 1
        generation = self._poll_generation
        self._poll_thread = threading.Thread(target=self._poll_events, args=(generation,), daemon=True)
        self._poll_thread.start()
        LOGGER.info("Validation engine process started. pid=%s generation=%s", self._process.pid, generation)

    def shutdown(self) -> None:
        """正常關閉背景引擎。"""
        LOGGER.info("Shutting down validation engine client.")
        self._closing = True
        if self._request_queue is None:
            return
        try:
            self._request_queue.put(EngineRequest(request_id=self._new_request_id(), action="shutdown"))
        except Exception:
            LOGGER.exception("Failed to send shutdown request to validation engine.")
        if self._process is not None:
            self._process.join(timeout=3)
            if self._process.is_alive():
                LOGGER.warning("Validation engine did not stop gracefully; terminating. pid=%s", self._process.pid)
                self._process.terminate()
                self._process.join(timeout=3)
        self._request_queue = None
        self._event_queue = None
        self._process = None

    def terminate_and_restart(self) -> None:
        """強制終止目前引擎並重建。"""
        LOGGER.warning("Force-restarting validation engine client.")
        self._poll_generation += 1
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=3)
        self._process = None
        self._request_queue = None
        self._event_queue = None
        self.start()

    def warmup_runtime(self) -> str:
        """要求背景載入 ultralytics。"""
        return self._send("warmup_runtime", {})

    def load_model_info(self, model_path: str) -> str:
        """讀取模型資訊。"""
        return self._send("load_model_info", {"model_path": model_path})

    def validate_dataset(self, payload: dict[str, object]) -> str:
        """執行單次驗證。"""
        return self._send("validate_dataset", payload)

    def _send(self, action: str, payload: dict[str, object]) -> str:
        if self._request_queue is None:
            raise EngineClientError("背景引擎尚未啟動。")
        if self._process is None or not self._process.is_alive():
            exit_code = None if self._process is None else self._process.exitcode
            message = f"背景引擎未存活，無法送出請求。action={action}, exit_code={exit_code}"
            LOGGER.error(message)
            raise EngineClientError(message)
        request_id = self._new_request_id()
        LOGGER.debug("Sending engine request. request_id=%s action=%s payload_keys=%s", request_id, action, sorted(payload.keys()))
        try:
            self._request_queue.put(EngineRequest(request_id=request_id, action=action, payload=payload))
        except Exception as exc:
            LOGGER.exception("Failed to send engine request. request_id=%s action=%s", request_id, action)
            raise EngineClientError(f"送出背景引擎請求失敗。action={action}") from exc
        return request_id

    def _poll_events(self, generation: int) -> None:
        if self._event_queue is None:
            return
        LOGGER.debug("Validation engine event poll thread started. generation=%s", generation)
        while not self._closing and generation == self._poll_generation:
            try:
                event: EngineEvent = self._event_queue.get(timeout=0.2)
            except queue.Empty:
                if self._process is None:
                    LOGGER.debug("Validation engine poll thread exiting because process reference is missing.")
                    return
                if self._process.is_alive():
                    continue
                exit_code = self._process.exitcode
                LOGGER.error("Validation engine process exited unexpectedly. exit_code=%s generation=%s", exit_code, generation)
                self.event_received.emit(
                    EngineEvent(
                        request_id="",
                        kind="error",
                        payload={
                            "message": f"背景驗證引擎已異常結束。exit_code={exit_code}",
                            "action": "engine_process",
                            "stage": "poll_events",
                            "exit_code": exit_code,
                        },
                    )
                )
                return
            LOGGER.debug("Engine event received from subprocess. request_id=%s kind=%s", event.request_id, event.kind)
            self.event_received.emit(event)

    @staticmethod
    def _new_request_id() -> str:
        return uuid.uuid4().hex
