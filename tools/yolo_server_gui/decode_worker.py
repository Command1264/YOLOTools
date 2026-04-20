from __future__ import annotations

import base64
import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np

from http_schema import DetectResult
from inference_job import DecodedInferenceTask, InferenceTask, InferenceTaskResult, DecodedTaskQueueItem, TaskQueueItem
from log_manager import LogController, get_logger


def _strip_data_url(data: str) -> str:
    if not data:
        return ""
    if "," in data:
        return data.split(",", 1)[1]
    return data


def _decode_base64_image(b64_str: str) -> Optional[np.ndarray]:
    if not b64_str:
        return None
    try:
        raw = base64.b64decode(_strip_data_url(b64_str), validate=False)
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def build_empty_detect_result() -> DetectResult:
    """Create a stable empty result payload for decode/infer failures."""
    return DetectResult(classify_type="none", percentage=0.0, detections=[])


class DecodeWorker(threading.Thread):
    """Decode base64 images and forward them to the paired GPU worker."""

    def __init__(
        self,
        worker_id: int,
        task_queue: queue.Queue[TaskQueueItem],
        decoded_task_queue: queue.Queue[DecodedTaskQueueItem],
    ) -> None:
        super().__init__(daemon=True, name=f"decode-worker-{worker_id}")
        self.worker_id = worker_id
        self._task_queue = task_queue
        self._decoded_task_queue = decoded_task_queue
        self._log_ctrl = LogController(get_logger())

    def run(self) -> None:
        while True:
            task = self._task_queue.get()
            try:
                if task is None:
                    return
                decoded_task = self._process_task(task)
                if decoded_task is not None:
                    self._decoded_task_queue.put(decoded_task)
            finally:
                self._task_queue.task_done()

    def _process_task(self, task: InferenceTask) -> Optional[DecodedInferenceTask]:
        empty_results = [build_empty_detect_result() for _ in task.images_b64]
        decoded_images: list[object] = []
        decoded_indices: list[int] = []
        decode_start = time.perf_counter()
        for index, image_b64 in enumerate(task.images_b64):
            image = _decode_base64_image(image_b64)
            if image is None:
                self._log_ctrl.debug(
                    "Decode worker decode failed. worker_id=%s thread_name=%s image_index=%s",
                    self.worker_id,
                    task.thread_name,
                    index,
                )
                continue
            decoded_indices.append(index)
            decoded_images.append(image)

        if not decoded_images:
            task.result_queue.put(InferenceTaskResult(results=empty_results))
            return None

        decode_elapsed_ms = max(0.0, (time.perf_counter() - decode_start) * 1000.0)
        self._log_ctrl.debug(
            "Decode worker task completed. worker_id=%s thread_name=%s image_count=%s decoded_count=%s decode_ms=%.2f",
            self.worker_id,
            task.thread_name,
            len(task.images_b64),
            len(decoded_images),
            decode_elapsed_ms,
        )
        return DecodedInferenceTask(
            task=task,
            decoded_images=decoded_images,
            decoded_indices=decoded_indices,
            empty_results=empty_results,
            decode_elapsed_ms=decode_elapsed_ms,
        )
