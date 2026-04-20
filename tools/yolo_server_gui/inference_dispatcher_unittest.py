from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GUI_DIR = PROJECT_ROOT / "tools" / "yolo_server_gui"
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))

from inference_dispatcher import InferenceDispatcher
from yolo_engine import Detection


class FakeYoloEngine:
    """Lightweight inference engine stub used by dispatcher tests."""

    def __init__(self, model_path: str, conf: float = 0.25, iou: float = 0.45) -> None:
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.device_name = "cuda:0 (Fake GPU)"
        self.warmup_error = None

    def warmup(self) -> None:
        return

    def infer_many(self, frames_bgr: list[object], conf: float | None = None, iou: float | None = None):
        outputs = []
        for index, _ in enumerate(frames_bgr):
            outputs.append(
                (
                    None,
                    [
                        Detection(
                            class_id=0,
                            class_name=f"fake-{index}",
                            conf=0.95,
                            xyxy=(1, 2, 3, 4),
                        )
                    ],
                )
            )
        return outputs


class InferenceDispatcherTests(unittest.TestCase):
    """Cover the decode -> GPU worker pipeline."""

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

        self.assertEqual([result.classify_type for result in results], ["fake-0", "fake-1"])
        self.assertEqual(results[0].detections[0].xyxy, [1, 2, 3, 4])

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
