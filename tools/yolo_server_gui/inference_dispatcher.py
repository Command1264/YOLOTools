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
        gpu_replica_count: int = 1,
        queue_size: int = 0,
    ) -> None:
        self._model_path = model_path
        self._conf = conf
        self._iou = iou
        self._gpu_replica_count = max(1, int(gpu_replica_count))
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
    def gpu_replica_count(self) -> int:
        return self._gpu_replica_count

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
                for index in range(self._gpu_replica_count)
            ]
            for worker in self._workers:
                worker.start()
            self._started = True
            self._stopped = False
            self._log_ctrl.info(
                "Inference dispatcher started. gpu_replica_count=%s queue_size=%s",
                self._gpu_replica_count,
                self._task_queue.maxsize,
            )

    def stop(self) -> bool:
        """Stop workers and fail remaining queued tasks.

        Returns:
            bool: True if all workers stopped within the join timeout.
        """
        with self._start_lock:
            if not self._started or self._stopped:
                return True
            self._stopped = True
            self._fail_pending_tasks()
            for _ in self._workers:
                self._task_queue.put(None)
            alive_worker_names: list[str] = []
            for worker in self._workers:
                worker.join(timeout=2.0)
                if worker.is_alive():
                    alive_worker_names.append(worker.name)
            all_workers_stopped = not alive_worker_names
            if alive_worker_names:
                self._log_ctrl.warning(
                    "Inference dispatcher workers did not stop within timeout. workers=%s",
                    ", ".join(alive_worker_names),
                )
            self._workers = []
            self._started = False
            self._log_ctrl.info("Inference dispatcher stopped.")
            return all_workers_stopped

    def wait_until_ready(
        self,
        timeout_sec: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> bool:
        """Wait until all workers are warmed up or one fails."""
        deadline = None if timeout_sec is None else time.perf_counter() + timeout_sec
        while True:
            if cancel_event is not None and cancel_event.is_set():
                return False
            if self.is_ready:
                return True
            if self.warmup_error:
                return False
            if deadline is not None and time.perf_counter() >= deadline:
                return False
            time.sleep(0.05)

    def submit(self, thread_name: str, image_b64: str, conf: float, iou: float | None) -> DetectResult:
        """Submit one inference task and block until the worker finishes it."""
        results = self.submit_many(thread_name=thread_name, images_b64=[image_b64], conf=conf, iou=iou)
        if not results:
            raise RuntimeError("Inference dispatcher returned an empty batch.")
        return results[0]

    def submit_many(
        self,
        thread_name: str,
        images_b64: list[str],
        conf: float,
        iou: float | None,
    ) -> list[DetectResult]:
        """Submit a batch inference task and block until the worker finishes it."""
        if not self._started or self._stopped:
            raise RuntimeError("Inference dispatcher is not running.")
        if not images_b64:
            return []
        task = InferenceTask(
            thread_name=thread_name,
            images_b64=list(images_b64),
            conf=conf,
            iou=iou,
        )
        self._task_queue.put(task)
        outcome = task.result_queue.get()
        if outcome.error is not None:
            raise outcome.error
        if outcome.results is None:
            raise RuntimeError("Inference dispatcher returned an empty result.")
        return outcome.results

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
