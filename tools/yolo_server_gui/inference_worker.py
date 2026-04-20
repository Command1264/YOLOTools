from __future__ import annotations

import queue
import threading
import time
from typing import Optional

from http_schema import DetectResult, DetectionItem
from inference_job import DecodedInferenceTask, DecodedTaskQueueItem, InferenceTaskResult
from log_manager import LogController, get_logger
from yolo_engine import Detection, YoloEngine

MAX_BATCH_IMAGES_PER_TASK = 8


def _pick_top1(dets: list[Detection]) -> tuple[str, float]:
    if not dets:
        return "none", 0.0
    best = max(dets, key=lambda det: det.conf)
    return best.class_name, float(best.conf)


def _dets_to_payload(dets: list[Detection]) -> list[DetectionItem]:
    payload: list[DetectionItem] = []
    for det in dets:
        payload.append(
            DetectionItem(
                class_id=int(det.class_id),
                class_name=det.class_name,
                conf=float(det.conf),
                xyxy=[int(value) for value in det.xyxy],
            )
        )
    return payload


class InferenceWorker(threading.Thread):
    """Consume decoded inference tasks with one dedicated model instance."""

    def __init__(
        self,
        worker_id: int,
        model_path: str,
        conf: float,
        iou: float,
        task_queue: queue.Queue[DecodedTaskQueueItem],
    ) -> None:
        super().__init__(daemon=True, name=f"inference-worker-{worker_id}")
        self.worker_id = worker_id
        self._task_queue = task_queue
        self._engine = YoloEngine(model_path=model_path, conf=conf, iou=iou)
        self._log_ctrl = LogController(get_logger())
        self._ready = False
        self._warming_up = False
        self._warmup_error: Optional[str] = None

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def is_warming_up(self) -> bool:
        return self._warming_up

    @property
    def warmup_error(self) -> Optional[str]:
        return self._warmup_error or self._engine.warmup_error

    @property
    def device_name(self) -> str:
        return self._engine.device_name

    def run(self) -> None:
        self._warming_up = True
        try:
            self._engine.warmup()
            self._ready = True
        except Exception as exc:
            self._warmup_error = str(exc) or exc.__class__.__name__
            self._log_ctrl.exception("Inference worker warmup failed. worker_id=%s", self.worker_id)
        finally:
            self._warming_up = False

        while True:
            decoded_task = self._task_queue.get()
            try:
                if decoded_task is None:
                    return
                self._log_ctrl.debug(
                    "Inference worker task started. worker_id=%s thread_name=%s",
                    self.worker_id,
                    decoded_task.task.thread_name,
                )
                decoded_task.task.result_queue.put(self._process_task(decoded_task))
            finally:
                self._task_queue.task_done()

    def _process_task(self, decoded_task: DecodedInferenceTask) -> InferenceTaskResult:
        results = list(decoded_task.empty_results)
        task = decoded_task.task
        if not self._ready:
            return InferenceTaskResult(
                results=results,
                error=RuntimeError(f"Inference worker not ready. worker_id={self.worker_id}"),
            )

        if not decoded_task.decoded_images:
            return InferenceTaskResult(
                results=results,
            )

        infer_start = time.perf_counter()
        try:
            for start in range(0, len(decoded_task.decoded_images), MAX_BATCH_IMAGES_PER_TASK):
                end = start + MAX_BATCH_IMAGES_PER_TASK
                chunk_images = decoded_task.decoded_images[start:end]
                chunk_indices = decoded_task.decoded_indices[start:end]
                chunk_outputs = self._engine.infer_many(chunk_images, conf=task.conf, iou=task.iou)
                if len(chunk_outputs) != len(chunk_indices):
                    raise RuntimeError(
                        "Batched inference returned mismatched result count. "
                        f"worker_id={self.worker_id} expected={len(chunk_indices)} actual={len(chunk_outputs)}"
                    )
                for output_index, (_, detections) in enumerate(chunk_outputs):
                    classify_type, percentage = _pick_top1(detections)
                    results[chunk_indices[output_index]] = DetectResult(
                        classify_type=classify_type,
                        percentage=percentage,
                        detections=_dets_to_payload(detections),
                    )
        except Exception as exc:
            self._log_ctrl.exception(
                "Inference worker task failed. worker_id=%s thread_name=%s image_count=%s",
                self.worker_id,
                task.thread_name,
                len(task.images_b64),
            )
            error = RuntimeError(
                f"Inference worker failed. worker_id={self.worker_id} thread_name={task.thread_name}"
            )
            error.__cause__ = exc
            return InferenceTaskResult(
                results=results,
                error=error,
            )

        infer_elapsed_ms = max(0.0, (time.perf_counter() - infer_start) * 1000.0)
        self._log_ctrl.debug(
            "Inference worker task completed. worker_id=%s thread_name=%s image_count=%s decode_ms=%.2f infer_ms=%.2f",
            self.worker_id,
            task.thread_name,
            len(task.images_b64),
            decoded_task.decode_elapsed_ms,
            infer_elapsed_ms,
        )
        return InferenceTaskResult(
            results=results
        )
