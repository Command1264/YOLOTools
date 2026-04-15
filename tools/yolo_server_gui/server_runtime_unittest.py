from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from flask import g

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GUI_DIR = PROJECT_ROOT / "tools" / "yolo_server_gui"
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))

import server as server_module
from config_model import AppConfig
from http_schema import DetectResult
from server import YoloServer


class RecordingDispatcher:
    """Collect inference submissions for batch-path tests."""

    def __init__(self) -> None:
        self.queue_size = 0
        self.gpu_replica_count = 1
        self.is_ready = True
        self.is_warming_up = False
        self.warmup_error = None
        self.device_name = "cpu"
        self.calls: list[tuple[str, list[str], float, float | None]] = []

    def start(self) -> None:
        return

    def stop(self) -> bool:
        return True

    def wait_until_ready(self, timeout_sec: float | None = None, cancel_event=None) -> bool:
        return True

    def submit_many(
        self,
        thread_name: str,
        images_b64: list[str],
        conf: float,
        iou: float | None,
    ) -> list[DetectResult]:
        self.calls.append((thread_name, list(images_b64), conf, iou))
        return [
            DetectResult(classify_type=f"image-{index}", percentage=0.9, detections=[])
            for index in range(len(images_b64))
        ]


class BlockingDispatcher:
    """Simulate a warmup wait that only ends when shutdown is requested."""

    def __init__(self) -> None:
        self.queue_size = 0
        self.gpu_replica_count = 1
        self.is_ready = False
        self.is_warming_up = True
        self.warmup_error = None
        self.device_name = "unknown"
        self.wait_started = threading.Event()
        self.cancel_event = None

    def start(self) -> None:
        return

    def stop(self) -> bool:
        return True

    def wait_until_ready(self, timeout_sec: float | None = None, cancel_event=None) -> bool:
        self.cancel_event = cancel_event
        self.wait_started.set()
        if cancel_event is not None:
            cancel_event.wait(timeout=1.0)
        return False


class DummyHttpServer:
    """Minimal HTTP server stub for server lifecycle tests."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.stopped = threading.Event()

    def serve_forever(self) -> None:
        self.started.set()
        self.stopped.wait(timeout=1.0)

    def shutdown(self) -> None:
        self.stopped.set()

    def server_close(self) -> None:
        return


class ServerRuntimeTests(unittest.TestCase):
    """Cover batch inference and shutdown runtime behavior."""

    def test_app_config_reads_legacy_worker_count_field(self) -> None:
        """Older configs should still hydrate the renamed GPU replica setting."""
        cfg = AppConfig.from_dict({"worker_count": 3})
        self.assertEqual(cfg.gpu_replica_count, 3)

    def test_infer_many_submits_whole_batch_once(self) -> None:
        """Batch requests should be forwarded to the dispatcher in one call."""
        dispatcher = RecordingDispatcher()

        with patch.object(YoloServer, "_create_dispatcher", lambda self, model_path: dispatcher):
            server = YoloServer(model_path="dummy.pt", host="127.0.0.1", port=60922)

        with server._app.test_request_context("/detect", method="POST"):
            g._thread_name = "tray-review"
            results = server._infer_many(images_b64=["img-a", "img-b"], conf=0.3, iou=0.5)

        self.assertEqual(len(dispatcher.calls), 1)
        self.assertEqual(dispatcher.calls[0], ("tray-review", ["img-a", "img-b"], 0.3, 0.5))
        self.assertEqual([result.classify_type for result in results], ["image-0", "image-1"])

    def test_stop_cancels_warmup_wait_during_shutdown(self) -> None:
        """Server stop should cancel the warmup thread instead of timing out later."""
        dispatcher = BlockingDispatcher()
        http_server = DummyHttpServer()

        with (
            patch.object(YoloServer, "_create_dispatcher", lambda self, model_path: dispatcher),
            patch.object(YoloServer, "_validate_bind_target", lambda self: None),
            patch.object(server_module, "make_server", return_value=http_server),
        ):
            server = YoloServer(model_path="dummy.pt", host="127.0.0.1", port=60922)
            server.start()
            self.assertTrue(http_server.started.wait(timeout=1.0))
            self.assertTrue(dispatcher.wait_started.wait(timeout=1.0))
            server.stop()

        self.assertIsNotNone(dispatcher.cancel_event)
        self.assertTrue(dispatcher.cancel_event.is_set())
        self.assertEqual(server.server_state, "stopped")


if __name__ == "__main__":
    unittest.main()
