from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import portalocker

from log_manager import LogController


class SingleInstanceLock:
    """以 lock file 確保單一實例執行。"""

    def __init__(self, lock_path: Path, log_ctrl: LogController) -> None:
        self._lock_path: Path = lock_path
        self._log_ctrl: LogController = log_ctrl
        self._lock = threading.Lock()
        self._file: Optional[object] = None
        self._stop_event = threading.Event()
        self._retainer: Optional[threading.Thread] = None

    def try_acquire(self) -> bool:
        """Try to acquire an exclusive non-blocking lock."""
        with self._lock:
            if self._file is not None:
                return True
            handle: Optional[object] = None
            try:
                handle = open(self._lock_path, "a+")
                portalocker.lock(handle, portalocker.LOCK_EX | portalocker.LOCK_NB)
                self._file = handle
                return True
            except portalocker.exceptions.LockException:
                if handle is not None:
                    try:
                        handle.close()
                    except Exception:
                        self._log_ctrl.exception("關閉鎖檔失敗。path=%s", str(self._lock_path))
                return False
            except Exception:
                self._log_ctrl.exception("取得鎖失敗。path=%s", str(self._lock_path))
                if handle is not None:
                    try:
                        handle.close()
                    except Exception:
                        self._log_ctrl.exception("關閉鎖檔失敗。path=%s", str(self._lock_path))
                return False

    def start_retainer(self, interval_sec: float = 3.0) -> None:
        """Start a background thread to retain lock ownership."""
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
        """Release held lock."""
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
        """Stop lock retainer and release resources."""
        self._stop_event.set()
        self.release()
