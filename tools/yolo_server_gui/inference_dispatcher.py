from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import queue
import threading
import time
from typing import Optional

from decode_worker import DecodeWorker
from http_schema import DetectResult
from inference_job import DecodedTaskQueueItem, InferenceTask, InferenceTaskResult, TaskQueueItem
from inference_worker import InferenceWorker, MAX_BATCH_IMAGES_PER_TASK
from log_manager import LogController, get_logger

MAX_PIPELINES_PER_REQUEST = 4


@dataclass
class WorkerPipeline:
    """Hold one decode worker and one paired GPU worker."""

    decode_queue: queue.Queue[TaskQueueItem]
    gpu_queue: queue.Queue[DecodedTaskQueueItem]
    decode_worker: DecodeWorker
    gpu_worker: InferenceWorker


@dataclass(frozen=True)
class ChunkAssignment:
    """Track one chunk task and the original indices it represents."""

    pipeline_index: int
    pipeline: WorkerPipeline
    task: InferenceTask
    original_indices: list[int]


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
        self._active_request_count = 0
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
            self._active_request_count = 0
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
            self._active_request_count = 0
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
        """Submit a batch inference task and block until all chunks finish."""
        if not self._started or self._stopped:
            raise RuntimeError("Inference dispatcher is not running.")
        if not images_b64:
            return []
        chunk_specs = deque(self._build_chunk_specs(images_b64))
        merged_results: list[DetectResult | None] = [None] * len(images_b64)
        active_assignments: list[ChunkAssignment] = []
        self._enter_active_request()
        try:
            while chunk_specs or active_assignments:
                request_window = self._calculate_request_window(
                    remaining_chunk_count=len(chunk_specs) + len(active_assignments)
                )
                while chunk_specs and len(active_assignments) < request_window:
                    original_indices, chunk_images = chunk_specs.popleft()
                    task = self._build_task(
                        thread_name=thread_name,
                        images_b64=chunk_images,
                        conf=conf,
                        iou=iou,
                    )
                    active_pipeline_indices = {assignment.pipeline_index for assignment in active_assignments}
                    pipeline_index, pipeline = self._select_pipeline(excluded_indices=active_pipeline_indices)
                    pipeline.decode_queue.put(task)
                    active_assignments.append(
                        ChunkAssignment(
                            pipeline_index=pipeline_index,
                            pipeline=pipeline,
                            task=task,
                            original_indices=original_indices,
                        )
                    )
                if not active_assignments:
                    continue
                assignment, outcome = self._wait_for_next_completed_assignment(active_assignments)
                active_assignments.remove(assignment)
                chunk_results = self._extract_task_results(outcome)
                self._merge_chunk_results(
                    merged_results=merged_results,
                    original_indices=assignment.original_indices,
                    chunk_results=chunk_results,
                )
        finally:
            self._leave_active_request()

        if any(result is None for result in merged_results):
            raise RuntimeError("Inference dispatcher fanout merge produced incomplete results.")
        return [result for result in merged_results if result is not None]

    def _build_task(
        self,
        thread_name: str,
        images_b64: list[str],
        conf: float,
        iou: float | None,
    ) -> InferenceTask:
        return InferenceTask(
            thread_name=thread_name,
            images_b64=list(images_b64),
            conf=conf,
            iou=iou,
        )

    def _extract_task_results(self, outcome: InferenceTaskResult) -> list[DetectResult]:
        if outcome.error is not None:
            raise outcome.error
        if outcome.results is None:
            raise RuntimeError("Inference dispatcher returned an empty result.")
        return outcome.results

    def _build_chunk_specs(self, images_b64: list[str]) -> list[tuple[list[int], list[str]]]:
        chunk_specs: list[tuple[list[int], list[str]]] = []
        for start in range(0, len(images_b64), MAX_BATCH_IMAGES_PER_TASK):
            end = min(len(images_b64), start + MAX_BATCH_IMAGES_PER_TASK)
            chunk_indices = list(range(start, end))
            chunk_images = list(images_b64[start:end])
            chunk_specs.append((chunk_indices, chunk_images))
        return chunk_specs

    def _merge_chunk_results(
        self,
        merged_results: list[DetectResult | None],
        original_indices: list[int],
        chunk_results: list[DetectResult],
    ) -> None:
        if len(chunk_results) != len(original_indices):
            raise RuntimeError(
                "Inference dispatcher chunk returned mismatched result count. "
                f"expected={len(original_indices)} actual={len(chunk_results)}"
            )
        for offset, result in enumerate(chunk_results):
            merged_results[original_indices[offset]] = result

    def _wait_for_next_completed_assignment(
        self,
        active_assignments: list[ChunkAssignment],
    ) -> tuple[ChunkAssignment, InferenceTaskResult]:
        while True:
            for assignment in list(active_assignments):
                try:
                    outcome = assignment.task.result_queue.get_nowait()
                except queue.Empty:
                    continue
                return assignment, outcome
            time.sleep(0.005)

    def _enter_active_request(self) -> None:
        with self._submit_lock:
            self._active_request_count += 1

    def _leave_active_request(self) -> None:
        with self._submit_lock:
            self._active_request_count = max(0, self._active_request_count - 1)

    def _calculate_request_window(self, remaining_chunk_count: int) -> int:
        with self._submit_lock:
            if not self._pipelines:
                raise RuntimeError("Inference dispatcher has no active worker pipelines.")
            effective_capacity = min(len(self._pipelines), MAX_PIPELINES_PER_REQUEST)
            active_request_count = max(1, self._active_request_count)
            fair_share = max(1, effective_capacity // active_request_count)
        return max(1, min(remaining_chunk_count, effective_capacity, fair_share))

    def _select_pipeline(
        self,
        excluded_indices: set[int] | None = None,
    ) -> tuple[int, WorkerPipeline]:
        with self._submit_lock:
            if not self._pipelines:
                raise RuntimeError("Inference dispatcher has no active worker pipelines.")
            blocked_indices = excluded_indices or set()
            start_index = self._next_pipeline_index
            ranked: list[tuple[int, int, int, WorkerPipeline]] = []
            for offset in range(len(self._pipelines)):
                pipeline_index = (start_index + offset) % len(self._pipelines)
                if pipeline_index in blocked_indices and len(blocked_indices) < len(self._pipelines):
                    continue
                pipeline = self._pipelines[pipeline_index]
                load = pipeline.decode_queue.qsize() + pipeline.gpu_queue.qsize()
                ranked.append((load, offset, pipeline_index, pipeline))
            ranked.sort(key=lambda item: (item[0], item[1]))
            if not ranked:
                raise RuntimeError("Inference dispatcher has no selectable worker pipeline.")
            selected = ranked[0]
            self._next_pipeline_index = (selected[2] + 1) % len(self._pipelines)
            return selected[2], selected[3]

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
