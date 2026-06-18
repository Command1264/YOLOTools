from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

from PIL import Image
from PySide6.QtWidgets import QApplication

from tools.yolo_auto_validator_gui.engine_client import ValidationEngineClient
from tools.yolo_auto_validator_gui.exceptions import EngineClientError

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class EngineClientTests(unittest.TestCase):
    """驗證背景引擎基本功能。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_engine_can_load_model_info_and_validate_tiny_dataset(self) -> None:
        model_paths = list(Path("models").glob("*.pt"))
        if not model_paths:
            self.skipTest("找不到可用的模型檔。")
        engine = ValidationEngineClient()
        events: list[object] = []
        engine.event_received.connect(events.append)

        warmup_id = engine.warmup_runtime()
        self._wait_for_event(events, warmup_id, "completed", timeout_sec=30)

        model_info_id = engine.load_model_info(str(model_paths[0].resolve()))
        model_info_event = self._wait_for_event(events, model_info_id, "model_info", timeout_sec=60)
        names = list(model_info_event.payload.get("names", []))
        self.assertTrue(names)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images" / "val"
            label_dir = root / "labels" / "val"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            Image.new("RGB", (32, 32), color="white").save(image_dir / "sample.jpg")
            (label_dir / "sample.txt").write_text("", encoding="utf-8")
            (root / "data.yaml").write_text(
                f"path: {root.resolve().as_posix()}\ntrain: images/val\nval: images/val\nnames: [Fire, Smoke]\n",
                encoding="utf-8",
            )
            validate_id = engine.validate_dataset(
                {
                    "title": "tiny",
                    "dataset_key": "tiny",
                    "model_path": str(model_paths[0].resolve()),
                    "data_yaml_path": str((root / "data.yaml").resolve()),
                    "project_dir": str((root / "runs").resolve()),
                    "run_name": "tiny",
                    "device": "cpu",
                    "conf_threshold": 0.25,
                    "iou_threshold": 0.45,
                    "contains_unlabeled_images": True,
                    "empty_label_count": 1,
                }
            )
            done_event = self._wait_for_event(events, validate_id, "validation_done", timeout_sec=120)
            labels = list(done_event.payload.get("labels", []))
            self.assertTrue(labels)
            self.assertEqual(labels[-1], "background")

        engine.shutdown()

    def test_engine_rejects_requests_after_process_dies(self) -> None:
        engine = ValidationEngineClient()
        try:
            self.assertIsNotNone(engine._process)
            engine._process.terminate()
            engine._process.join(timeout=3)
            with self.assertRaises(EngineClientError):
                engine.warmup_runtime()
        finally:
            engine.shutdown()

    def _wait_for_event(self, events: list[object], request_id: str, kind: str, timeout_sec: float):
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            self._app.processEvents()
            for event in list(events):
                if getattr(event, "request_id", "") == request_id and getattr(event, "kind", "") == kind:
                    return event
                if getattr(event, "request_id", "") == request_id and getattr(event, "kind", "") == "error":
                    raise AssertionError(f"背景引擎錯誤：{event.payload}")
            time.sleep(0.02)
        raise AssertionError(f"逾時未收到事件 kind={kind}")


if __name__ == "__main__":
    unittest.main()
