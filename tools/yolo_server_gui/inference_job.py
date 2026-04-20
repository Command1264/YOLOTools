from __future__ import annotations

import queue
from dataclasses import dataclass, field
from typing import Optional

from http_schema import DetectResult


@dataclass
class InferenceTaskResult:
    """Carry one inference result or one execution error."""

    results: Optional[list[DetectResult]] = None
    error: Optional[Exception] = None


@dataclass
class InferenceTask:
    """Represent one queued inference request."""

    thread_name: str
    images_b64: list[str]
    conf: float
    iou: float | None
    result_queue: queue.Queue[InferenceTaskResult] = field(
        default_factory=lambda: queue.Queue(maxsize=1)
    )


@dataclass
class DecodedInferenceTask:
    """Carry decoded images for one inference request."""

    task: InferenceTask
    decoded_images: list[object]
    decoded_indices: list[int]
    empty_results: list[DetectResult]
    decode_elapsed_ms: float = 0.0


TaskQueueItem = InferenceTask | None
DecodedTaskQueueItem = DecodedInferenceTask | None
