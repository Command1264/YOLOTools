from __future__ import annotations

import queue
import threading
import time
from typing import Optional

from http_schema import DetectResult
from inference_job import InferenceTask, InferenceTaskResult, TaskQueueItem
from inference_worker import InferenceWorker
from log_manager import LogController, get_logger


class InferenceDispatcher:
    """Queue inference work and dispatch it to one or more workers."""

    def __init__(
        self,
        model_path: str,
        conf: float,
        iou: float,
        worker_count: int = 1,
        queue_size: int = 0,
    ) -> None:
        self._model_path = model_path
        self._conf = conf
        self._iou = iou
        self._worker_count = max(1, int(worker_count))
        self._task_queue: queue.Queue[TaskQueueItem] = queue.Queue(maxsize=max(0, int(queue_size)))
        self._workers: list[InferenceWorker] = []
        self._started = False
        self._stopped = False
        self._start_lock = threading.RLock()
        self._log_ctrl = LogController(get_logger())

    @property
    def is_ready(self) -> bool:
        return bool(self._workers) and all(worker.is_ready for worker in self._workers)

    @property
    def is_warming_up(self) -> bool:
        return bool(self._workers) and any(worker.is_warming_up for worker in self._workers) and not self.is_ready

    @property
    def warmup_error(self) -> Optional[str]:
        for worker in self._workers:
            if worker.warmup_error:
                return worker.warmup_error
        return None

    @property
    def device_name(self) -> str:
        for worker in self._workers:
            if worker.device_name != "unknown":
                return worker.device_name
        return "unknown"

    @property
    def queue_size(self) -> int:
        return self._task_queue.qsize()

    @property
    def worker_count(self) -> int:
        return self._worker_count

    def start(self) -> None:
        """Start worker threads once."""
        with self._start_lock:
            if self._started:
                return
            self._workers = [
                InferenceWorker(
                    worker_id=index,
                    model_path=self._model_path,
                    conf=self._conf,
                    iou=self._iou,
                    task_queue=self._task_queue,
                )
                for index in range(self._worker_count)
            ]
            for worker in self._workers:
                worker.start()
            self._started = True
            self._stopped = False
            self._log_ctrl.info(
                "Inference dispatcher started. worker_count=%s queue_size=%s",
                self._worker_count,
                self._task_queue.maxsize,
            )

    def stop(self) -> None:
        """Stop workers and fail remaining queued tasks."""
        with self._start_lock:
            if not self._started or self._stopped:
                return
            self._stopped = True
            self._fail_pending_tasks()
            for _ in self._workers:
                self._task_queue.put(None)
            for worker in self._workers:
                worker.join(timeout=2.0)
            self._workers = []
            self._started = False
            self._log_ctrl.info("Inference dispatcher stopped.")

    def wait_until_ready(self, timeout_sec: float | None = None) -> bool:
        """Wait until all workers are warmed up or one fails."""
        deadline = None if timeout_sec is None else time.perf_counter() + timeout_sec
        while True:
            if self.is_ready:
                return True
            if self.warmup_error:
                return False
            if deadline is not None and time.perf_counter() >= deadline:
                return False
            time.sleep(0.05)

    def submit(self, thread_name: str, image_b64: str, conf: float, iou: float | None) -> DetectResult:
        """Submit one inference task and block until the worker finishes it."""
        if not self._started or self._stopped:
            raise RuntimeError("Inference dispatcher is not running.")
        task = InferenceTask(
            thread_name=thread_name,
            image_b64=image_b64,
            conf=conf,
            iou=iou,
        )
        self._task_queue.put(task)
        outcome = task.result_queue.get()
        if outcome.error is not None:
            raise outcome.error
        if outcome.result is None:
            raise RuntimeError("Inference dispatcher returned an empty result.")
        return outcome.result

    def _fail_pending_tasks(self) -> None:
        pending: list[InferenceTask] = []
        while True:
            try:
                item = self._task_queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                self._task_queue.task_done()
                continue
            pending.append(item)
            self._task_queue.task_done()

        for task in pending:
            task.result_queue.put(
                InferenceTaskResult(
                    error=RuntimeError(
                        f"Inference dispatcher stopped before task execution. thread_name={task.thread_name}"
                    )
                )
            )
