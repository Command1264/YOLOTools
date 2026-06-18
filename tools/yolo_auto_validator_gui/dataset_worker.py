from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QObject, Signal

if __package__ in {None, ""}:
    from logging_utils import get_logger
else:
    from .logging_utils import get_logger

LOGGER = get_logger(__name__)


class DatasetTaskWorker(QObject):
    """背景資料處理 worker。"""

    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()
    log = Signal(str)
    progress_known = Signal(int, int)
    progress_unknown = Signal(str)

    def __init__(self, task: Callable[..., object]) -> None:
        super().__init__()
        self._task = task
        self._cancelled = False

    def request_cancel(self) -> None:
        """要求取消作業。"""
        self._cancelled = True

    def is_cancelled(self) -> bool:
        """回傳是否已取消。"""
        return self._cancelled

    def run(self) -> None:
        """執行背景作業。"""
        task_name = getattr(self._task, "__name__", repr(self._task))
        LOGGER.debug("Dataset worker started. task=%s", task_name)
        try:
            result = self._task(
                log_cb=self.log.emit,
                progress_known_cb=self.progress_known.emit,
                progress_unknown_cb=self.progress_unknown.emit,
                cancel_cb=self.is_cancelled,
            )
        except Exception as exc:
            if self._cancelled:
                LOGGER.info("Dataset worker cancelled after exception. task=%s message=%s", task_name, exc)
                self.cancelled.emit()
                return
            LOGGER.exception("Dataset worker failed. task=%s", task_name)
            self.failed.emit(str(exc))
            return
        if self._cancelled:
            LOGGER.info("Dataset worker cancelled. task=%s", task_name)
            self.cancelled.emit()
            return
        LOGGER.debug("Dataset worker finished successfully. task=%s", task_name)
        self.finished.emit(result)
