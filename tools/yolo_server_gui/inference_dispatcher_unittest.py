from __future__ import annotations

import queue
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GUI_DIR = PROJECT_ROOT / "tools" / "yolo_server_gui"
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))

from inference_dispatcher import InferenceDispatcher, WorkerPipeline
from yolo_engine import Detection


class FakeYoloEngine:
    """Lightweight inference engine stub used by dispatcher tests."""

    _call_lock = threading.Lock()
    _call_threads: list[str] = []
    _call_batches: list[list[object]] = []

    def __init__(self, model_path: str, conf: float = 0.25, iou: float = 0.45) -> None:
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.device_name = "cuda:0 (Fake GPU)"
        self.warmup_error = None

    @classmethod
    def reset_calls(cls) -> None:
        with cls._call_lock:
            cls._call_threads = []
            cls._call_batches = []

    @classmethod
    def call_threads(cls) -> list[str]:
        with cls._call_lock:
            return list(cls._call_threads)

    @classmethod
    def call_batches(cls) -> list[list[object]]:
        with cls._call_lock:
            return [list(batch) for batch in cls._call_batches]

    def warmup(self) -> None:
        return

    def infer_many(self, frames_bgr: list[object], conf: float | None = None, iou: float | None = None):
        with self._call_lock:
            self._call_threads.append(threading.current_thread().name)
            self._call_batches.append(list(frames_bgr))
        outputs = []
        for frame in frames_bgr:
            outputs.append(
                (
                    None,
                    [
                        Detection(
                            class_id=0,
                            class_name=str(frame),
                            conf=0.95,
                            xyxy=(1, 2, 3, 4),
                        )
                    ],
                )
            )
        return outputs


class InferenceDispatcherTests(unittest.TestCase):
    """Cover the decode -> GPU worker pipeline."""

    def setUp(self) -> None:
        FakeYoloEngine.reset_calls()

    def test_submit_many_runs_through_decode_and_gpu_workers(self) -> None:
        """Decoded images should flow through the worker pair and keep result order."""
        with (
            patch("decode_worker._decode_base64_image", side_effect=lambda payload: f"decoded:{payload}"),
            patch("inference_worker.YoloEngine", FakeYoloEngine),
        ):
            dispatcher = InferenceDispatcher(
                model_path="dummy.pt",
                conf=0.25,
                iou=0.45,
                gpu_replica_count=1,
                decode_worker_count=1,
            )
            dispatcher.start()
            try:
                results = dispatcher.submit_many(
                    thread_name="unit-test",
                    images_b64=["img-a", "img-b"],
                    conf=0.3,
                    iou=0.5,
                )
            finally:
                dispatcher.stop()

        self.assertEqual([result.classify_type for result in results], ["decoded:img-a", "decoded:img-b"])
        self.assertEqual(results[0].detections[0].xyxy, [1, 2, 3, 4])
        self.assertEqual(FakeYoloEngine.call_batches(), [["decoded:img-a", "decoded:img-b"]])
        self.assertEqual(len(set(FakeYoloEngine.call_threads())), 1)

    def test_submit_many_keeps_small_batch_on_one_pipeline(self) -> None:
        """Batches within one GPU batch size should stay on a single pipeline."""
        with (
            patch("decode_worker._decode_base64_image", side_effect=lambda payload: f"decoded:{payload}"),
            patch("inference_worker.YoloEngine", FakeYoloEngine),
        ):
            dispatcher = InferenceDispatcher(
                model_path="dummy.pt",
                conf=0.25,
                iou=0.45,
                gpu_replica_count=2,
                decode_worker_count=2,
            )
            dispatcher.start()
            try:
                results = dispatcher.submit_many(
                    thread_name="small-batch",
                    images_b64=[f"img-{index}" for index in range(4)],
                    conf=0.3,
                    iou=0.5,
                )
            finally:
                dispatcher.stop()

        self.assertEqual(
            [result.classify_type for result in results],
            [f"decoded:img-{index}" for index in range(4)],
        )
        self.assertEqual(len(FakeYoloEngine.call_batches()), 1)
        self.assertEqual(FakeYoloEngine.call_batches()[0], [f"decoded:img-{index}" for index in range(4)])
        self.assertEqual(len(set(FakeYoloEngine.call_threads())), 1)

    def test_submit_many_fans_out_large_batch_and_keeps_global_order(self) -> None:
        """Large batches should fan out across pipelines and merge back in input order."""
        with (
            patch("decode_worker._decode_base64_image", side_effect=lambda payload: f"decoded:{payload}"),
            patch("inference_worker.YoloEngine", FakeYoloEngine),
        ):
            dispatcher = InferenceDispatcher(
                model_path="dummy.pt",
                conf=0.25,
                iou=0.45,
                gpu_replica_count=2,
                decode_worker_count=2,
            )
            dispatcher.start()
            try:
                images = [f"img-{index}" for index in range(10)]
                results = dispatcher.submit_many(
                    thread_name="large-batch",
                    images_b64=images,
                    conf=0.3,
                    iou=0.5,
                )
            finally:
                dispatcher.stop()

        self.assertEqual(
            [result.classify_type for result in results],
            [f"decoded:img-{index}" for index in range(10)],
        )
        self.assertEqual(len(FakeYoloEngine.call_batches()), 2)
        self.assertEqual(
            sorted(len(batch) for batch in FakeYoloEngine.call_batches()),
            [2, 8],
        )
        self.assertGreaterEqual(len(set(FakeYoloEngine.call_threads())), 2)

    def test_submit_many_caps_initial_request_window_to_four_pipelines(self) -> None:
        """One large request should keep the active fanout window at four chunks or fewer."""
        with (
            patch("decode_worker._decode_base64_image", side_effect=lambda payload: f"decoded:{payload}"),
            patch("inference_worker.YoloEngine", FakeYoloEngine),
        ):
            dispatcher = InferenceDispatcher(
                model_path="dummy.pt",
                conf=0.25,
                iou=0.45,
                gpu_replica_count=6,
                decode_worker_count=6,
            )
            dispatcher.start()
            try:
                results = dispatcher.submit_many(
                    thread_name="fanout-cap",
                    images_b64=[f"img-{index}" for index in range(40)],
                    conf=0.3,
                    iou=0.5,
                )
            finally:
                dispatcher.stop()

        self.assertEqual(len(results), 40)
        self.assertEqual(len(FakeYoloEngine.call_batches()), 5)
        self.assertGreaterEqual(len(set(FakeYoloEngine.call_threads())), 4)

    def test_calculate_request_window_shrinks_when_multiple_requests_are_active(self) -> None:
        """Concurrent requests should reduce the per-request fanout window."""
        dispatcher = InferenceDispatcher(
            model_path="dummy.pt",
            conf=0.25,
            iou=0.45,
            gpu_replica_count=4,
            decode_worker_count=4,
        )
        dispatcher._pipelines = [object(), object(), object(), object()]

        dispatcher._active_request_count = 1
        self.assertEqual(dispatcher._calculate_request_window(remaining_chunk_count=8), 4)

        dispatcher._active_request_count = 2
        self.assertEqual(dispatcher._calculate_request_window(remaining_chunk_count=8), 2)

        dispatcher._active_request_count = 5
        self.assertEqual(dispatcher._calculate_request_window(remaining_chunk_count=8), 1)

    def test_select_pipeline_avoids_pipeline_already_used_by_same_request_when_possible(self) -> None:
        """A request with multiple in-flight chunks should spread across different pipelines first."""
        dispatcher = InferenceDispatcher(
            model_path="dummy.pt",
            conf=0.25,
            iou=0.45,
            gpu_replica_count=2,
            decode_worker_count=2,
        )
        dispatcher._pipelines = [
            WorkerPipeline(
                decode_queue=queue.Queue(),
                gpu_queue=queue.Queue(),
                decode_worker=None,
                gpu_worker=None,
            ),
            WorkerPipeline(
                decode_queue=queue.Queue(),
                gpu_queue=queue.Queue(),
                decode_worker=None,
                gpu_worker=None,
            ),
        ]
        dispatcher._next_pipeline_index = 0

        pipeline_index, _ = dispatcher._select_pipeline(excluded_indices={0})

        self.assertEqual(pipeline_index, 1)

    def test_submit_many_keeps_decode_failures_at_original_indices_after_fanout(self) -> None:
        """Decode failures should remain aligned to their original positions after merge."""
        def fake_decode(payload: str) -> object | None:
            if payload == "bad-image":
                return None
            return f"decoded:{payload}"

        with (
            patch("decode_worker._decode_base64_image", side_effect=fake_decode),
            patch("inference_worker.YoloEngine", FakeYoloEngine),
        ):
            dispatcher = InferenceDispatcher(
                model_path="dummy.pt",
                conf=0.25,
                iou=0.45,
                gpu_replica_count=2,
                decode_worker_count=2,
            )
            dispatcher.start()
            try:
                images = [f"img-{index}" for index in range(10)]
                images[8] = "bad-image"
                results = dispatcher.submit_many(
                    thread_name="decode-failure",
                    images_b64=images,
                    conf=0.3,
                    iou=0.5,
                )
            finally:
                dispatcher.stop()

        self.assertEqual(len(results), 10)
        self.assertEqual(results[7].classify_type, "decoded:img-7")
        self.assertEqual(results[8].classify_type, "none")
        self.assertEqual(results[9].classify_type, "decoded:img-9")

    def test_requires_decode_and_gpu_worker_counts_to_match(self) -> None:
        """Current pipeline uses strict 1:1 decode-to-GPU worker mapping."""
        with self.assertRaises(ValueError):
            InferenceDispatcher(
                model_path="dummy.pt",
                conf=0.25,
                iou=0.45,
                gpu_replica_count=2,
                decode_worker_count=1,
            )


if __name__ == "__main__":
    unittest.main()
