from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
import time
from typing import Optional

from decode_worker import DecodeWorker
from http_schema import DetectResult
from inference_job import DecodedTaskQueueItem, InferenceTask, InferenceTaskResult, TaskQueueItem
from inference_worker import InferenceWorker
from log_manager import LogController, get_logger


@dataclass
class WorkerPipeline:
    """Hold one decode worker and one paired GPU worker."""

    decode_queue: queue.Queue[TaskQueueItem]
    gpu_queue: queue.Queue[DecodedTaskQueueItem]
    decode_worker: DecodeWorker
    gpu_worker: InferenceWorker


class InferenceDispatcher:
    """Queue inference work and dispatch it to one or more workers."""

    def __init__(
        self,
        model_path: str,
        conf: float,
        iou: float,
        gpu_replica_count: int = 1,
        decode_worker_count: Optional[int] = None,
        queue_size: int = 0,
    ) -> None:
        self._model_path = model_path
        self._conf = conf
        self._iou = iou
        self._gpu_replica_count = max(1, int(gpu_replica_count))
        self._decode_worker_count = self._gpu_replica_count if decode_worker_count is None else max(
            1, int(decode_worker_count)
        )
        if self._decode_worker_count != self._gpu_replica_count:
            raise ValueError("decode_worker_count must equal gpu_replica_count for 1:1 worker mapping.")
        self._queue_size = max(0, int(queue_size))
        self._pipelines: list[WorkerPipeline] = []
        self._started = False
        self._stopped = False
        self._start_lock = threading.RLock()
        self._submit_lock = threading.Lock()
        self._next_pipeline_index = 0
        self._log_ctrl = LogController(get_logger())

    @property
    def is_ready(self) -> bool:
        return bool(self._pipelines) and all(pipeline.gpu_worker.is_ready for pipeline in self._pipelines)

    @property
    def is_warming_up(self) -> bool:
        return (
            bool(self._pipelines)
            and any(pipeline.gpu_worker.is_warming_up for pipeline in self._pipelines)
            and not self.is_ready
        )

    @property
    def warmup_error(self) -> Optional[str]:
        for pipeline in self._pipelines:
            if pipeline.gpu_worker.warmup_error:
                return pipeline.gpu_worker.warmup_error
        return None

    @property
    def device_name(self) -> str:
        for pipeline in self._pipelines:
            if pipeline.gpu_worker.device_name != "unknown":
                return pipeline.gpu_worker.device_name
        return "unknown"

    @property
    def queue_size(self) -> int:
        return sum(pipeline.decode_queue.qsize() + pipeline.gpu_queue.qsize() for pipeline in self._pipelines)

    @property
    def gpu_replica_count(self) -> int:
        return self._gpu_replica_count

    @property
    def decode_worker_count(self) -> int:
        return self._decode_worker_count

    def start(self) -> None:
        """Start worker threads once."""
        with self._start_lock:
            if self._started:
                return
            self._pipelines = []
            for index in range(self._gpu_replica_count):
                decode_queue: queue.Queue[TaskQueueItem] = queue.Queue(maxsize=self._queue_size)
                gpu_queue: queue.Queue[DecodedTaskQueueItem] = queue.Queue(maxsize=self._queue_size)
                decode_worker = DecodeWorker(
                    worker_id=index,
                    task_queue=decode_queue,
                    decoded_task_queue=gpu_queue,
                )
                gpu_worker = InferenceWorker(
                    worker_id=index,
                    model_path=self._model_path,
                    conf=self._conf,
                    iou=self._iou,
                    task_queue=gpu_queue,
                )
                gpu_worker.start()
                decode_worker.start()
                self._pipelines.append(
                    WorkerPipeline(
                        decode_queue=decode_queue,
                        gpu_queue=gpu_queue,
                        decode_worker=decode_worker,
                        gpu_worker=gpu_worker,
                    )
                )
            self._started = True
            self._stopped = False
            self._next_pipeline_index = 0
            self._log_ctrl.info(
                "Inference dispatcher started. gpu_replica_count=%s decode_worker_count=%s queue_size=%s",
                self._gpu_replica_count,
                self._decode_worker_count,
                self._queue_size,
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
            for pipeline in self._pipelines:
                pipeline.decode_queue.put(None)
            alive_worker_names: list[str] = []
            for pipeline in self._pipelines:
                pipeline.decode_worker.join(timeout=2.0)
                if pipeline.decode_worker.is_alive():
                    alive_worker_names.append(pipeline.decode_worker.name)
            for pipeline in self._pipelines:
                pipeline.gpu_queue.put(None)
            for pipeline in self._pipelines:
                pipeline.gpu_worker.join(timeout=2.0)
                if pipeline.gpu_worker.is_alive():
                    alive_worker_names.append(pipeline.gpu_worker.name)
            all_workers_stopped = not alive_worker_names
            if alive_worker_names:
                self._log_ctrl.warning(
                    "Inference dispatcher workers did not stop within timeout. workers=%s",
                    ", ".join(alive_worker_names),
                )
            self._pipelines = []
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
        pipeline = self._select_pipeline()
        pipeline.decode_queue.put(task)
        outcome = task.result_queue.get()
        if outcome.error is not None:
            raise outcome.error
        if outcome.results is None:
            raise RuntimeError("Inference dispatcher returned an empty result.")
        return outcome.results

    def _select_pipeline(self) -> WorkerPipeline:
        with self._submit_lock:
            if not self._pipelines:
                raise RuntimeError("Inference dispatcher has no active worker pipelines.")
            pipeline_index = self._next_pipeline_index
            self._next_pipeline_index = (self._next_pipeline_index + 1) % len(self._pipelines)
            return self._pipelines[pipeline_index]

    def _fail_pending_tasks(self) -> None:
        pending: list[InferenceTask] = []
        for pipeline in self._pipelines:
            while True:
                try:
                    item = pipeline.decode_queue.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    pipeline.decode_queue.task_done()
                    continue
                pending.append(item)
                pipeline.decode_queue.task_done()
            while True:
                try:
                    decoded_item = pipeline.gpu_queue.get_nowait()
                except queue.Empty:
                    break
                if decoded_item is None:
                    pipeline.gpu_queue.task_done()
                    continue
                pending.append(decoded_item.task)
                pipeline.gpu_queue.task_done()

        for task in pending:
            task.result_queue.put(
                InferenceTaskResult(
                    error=RuntimeError(
                        f"Inference dispatcher stopped before task execution. thread_name={task.thread_name}"
                    )
                )
            )
