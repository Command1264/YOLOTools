from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.yolo_auto_validator_gui.engine_process import (
    ValidationExecutionPlan,
    _count_true_negative_background_images,
    _run_model_validation_with_retry,
    _select_validation_plan,
)


class _FakeResult:
    def __init__(self, prediction_count: int) -> None:
        self.boxes = [object()] * prediction_count


class _FakeModel:
    def __init__(
        self,
        prediction_counts: list[int] | dict[str, int],
        *,
        fail_first_cuda_call: bool = False,
    ) -> None:
        self._prediction_counts = prediction_counts
        self._fail_first_cuda_call = fail_first_cuda_call
        self._has_failed_cuda = False
        self.calls: list[dict[str, object]] = []

    def predict(self, **kwargs):
        self.calls.append(dict(kwargs))
        device = str(kwargs.get("device", ""))
        if self._fail_first_cuda_call and device.startswith("cuda") and not self._has_failed_cuda:
            self._has_failed_cuda = True
            raise RuntimeError("CUDA out of memory. Fake test failure.")
        source_items = [Path(item).name for item in list(kwargs.get("source", []))]
        if isinstance(self._prediction_counts, dict):
            for item in source_items:
                yield _FakeResult(self._prediction_counts[item])
            return
        for count in self._prediction_counts:
            yield _FakeResult(count)


class _FakeValidationModel:
    def __init__(self, *, fail_cuda_batches: set[int] | None = None) -> None:
        self._fail_cuda_batches = set(fail_cuda_batches or set())
        self.calls: list[dict[str, object]] = []

    def val(self, **kwargs):
        self.calls.append(dict(kwargs))
        device = str(kwargs.get("device", ""))
        batch = int(kwargs.get("batch", 0))
        if device.startswith("cuda") and batch in self._fail_cuda_batches:
            self._fail_cuda_batches.remove(batch)
            raise RuntimeError("CUDA out of memory. Fake validation failure.")
        return {"status": "ok", "batch": batch, "device": device}


class _FakeCudaModule:
    def __init__(self, *, free_bytes: int, total_bytes: int, available: bool = True) -> None:
        self._free_bytes = free_bytes
        self._total_bytes = total_bytes
        self._available = available
        self.empty_cache_calls = 0
        self.ipc_collect_calls = 0

    def is_available(self) -> bool:
        return self._available

    def mem_get_info(self, device_index: int) -> tuple[int, int]:
        return self._free_bytes, self._total_bytes

    def get_device_properties(self, device_index: int):
        class _Props:
            total_memory = self._total_bytes

        return _Props()

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1

    def ipc_collect(self) -> None:
        self.ipc_collect_calls += 1


class _FakeTorchModule:
    def __init__(self, cuda_module: _FakeCudaModule) -> None:
        self.cuda = cuda_module


class EngineProcessTests(unittest.TestCase):
    """驗證 engine process 的補充背景統計。"""

    def test_count_true_negative_background_images_only_counts_empty_gt_and_empty_pred(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images" / "val"
            label_dir = root / "labels" / "val"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            for name in ("a.jpg", "b.jpg", "c.jpg"):
                Image.new("RGB", (16, 16), color="white").save(image_dir / name)
            (label_dir / "a.txt").write_text("", encoding="utf-8")
            (label_dir / "b.txt").write_text("", encoding="utf-8")
            (label_dir / "c.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            (root / "data.yaml").write_text(
                f"path: {root.resolve().as_posix()}\nval: images/val\nnames: [Fire]\n",
                encoding="utf-8",
            )

            count = _count_true_negative_background_images(
                model=_FakeModel([0, 2]),
                data_yaml_path=root / "data.yaml",
                conf_threshold=0.25,
                iou_threshold=0.45,
                device="",
            )

        self.assertEqual(count, 1)

    def test_select_validation_plan_uses_gpu_batch4_for_6g_card_with_5g_free(self) -> None:
        torch_module = _FakeTorchModule(
            _FakeCudaModule(
                free_bytes=int(5.02 * 1024**3),
                total_bytes=6 * 1024**3,
            )
        )

        plan = _select_validation_plan(
            requested_device="",
            torch_module=torch_module,
        )

        self.assertEqual(plan.execution_device, "cuda:0")
        self.assertEqual(plan.batch_size, 4)
        self.assertEqual(plan.strategy, "validation_gpu_batch4")

    def test_run_model_validation_with_retry_reduces_batch_after_cuda_oom(self) -> None:
        model = _FakeValidationModel(fail_cuda_batches={4})
        torch_module = _FakeTorchModule(
            _FakeCudaModule(
                free_bytes=6 * 1024**3,
                total_bytes=8 * 1024**3,
            )
        )
        result, plan = _run_model_validation_with_retry(
            model=model,
            base_kwargs={"data": "demo.yaml", "split": "val"},
            initial_plan=ValidationExecutionPlan(
                execution_device="cuda:0",
                batch_size=4,
                strategy="validation_gpu_batch4",
                free_bytes=6 * 1024**3,
                total_bytes=8 * 1024**3,
            ),
            torch_module=torch_module,
            title="demo",
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(plan.execution_device, "cuda:0")
        self.assertEqual(plan.batch_size, 2)
        self.assertEqual(model.calls[0]["batch"], 4)
        self.assertEqual(model.calls[1]["batch"], 2)

    def test_run_model_validation_with_retry_falls_back_to_cpu_after_repeated_cuda_oom(self) -> None:
        model = _FakeValidationModel(fail_cuda_batches={4, 2, 1})
        torch_module = _FakeTorchModule(
            _FakeCudaModule(
                free_bytes=6 * 1024**3,
                total_bytes=8 * 1024**3,
            )
        )
        result, plan = _run_model_validation_with_retry(
            model=model,
            base_kwargs={"data": "demo.yaml", "split": "val"},
            initial_plan=ValidationExecutionPlan(
                execution_device="cuda:0",
                batch_size=4,
                strategy="validation_gpu_batch4",
                free_bytes=6 * 1024**3,
                total_bytes=8 * 1024**3,
            ),
            torch_module=torch_module,
            title="demo",
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(plan.execution_device, "cpu")
        self.assertEqual(plan.batch_size, 1)
        self.assertEqual([call["device"] for call in model.calls], ["cuda:0", "cuda:0", "cuda:0", "cpu"])

    def test_count_true_negative_background_images_uses_cpu_when_free_vram_is_low(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images" / "val"
            label_dir = root / "labels" / "val"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            for name in ("a.jpg", "b.jpg"):
                Image.new("RGB", (16, 16), color="white").save(image_dir / name)
                (label_dir / f"{Path(name).stem}.txt").write_text("", encoding="utf-8")
            (root / "data.yaml").write_text(
                f"path: {root.resolve().as_posix()}\nval: images/val\nnames: [Fire]\n",
                encoding="utf-8",
            )

            model = _FakeModel({"a.jpg": 0, "b.jpg": 1})
            torch_module = _FakeTorchModule(
                _FakeCudaModule(
                    free_bytes=512 * 1024**2,
                    total_bytes=6 * 1024**3,
                )
            )
            count = _count_true_negative_background_images(
                model=model,
                data_yaml_path=root / "data.yaml",
                conf_threshold=0.25,
                iou_threshold=0.45,
                device="0",
                torch_module=torch_module,
            )

        self.assertEqual(count, 1)
        self.assertTrue(model.calls)
        self.assertTrue(all(call.get("device") == "cpu" for call in model.calls))

    def test_count_true_negative_background_images_falls_back_to_cpu_after_cuda_oom(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images" / "val"
            label_dir = root / "labels" / "val"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            for name in ("a.jpg", "b.jpg"):
                Image.new("RGB", (16, 16), color="white").save(image_dir / name)
                (label_dir / f"{Path(name).stem}.txt").write_text("", encoding="utf-8")
            (root / "data.yaml").write_text(
                f"path: {root.resolve().as_posix()}\nval: images/val\nnames: [Fire]\n",
                encoding="utf-8",
            )

            model = _FakeModel(
                {"a.jpg": 0, "b.jpg": 0},
                fail_first_cuda_call=True,
            )
            torch_module = _FakeTorchModule(
                _FakeCudaModule(
                    free_bytes=9 * 1024**3,
                    total_bytes=12 * 1024**3,
                )
            )
            count = _count_true_negative_background_images(
                model=model,
                data_yaml_path=root / "data.yaml",
                conf_threshold=0.25,
                iou_threshold=0.45,
                device="0",
                torch_module=torch_module,
            )

        self.assertEqual(count, 2)
        self.assertGreaterEqual(len(model.calls), 2)
        self.assertEqual(model.calls[0].get("device"), "cuda:0")
        self.assertTrue(any(call.get("device") == "cpu" for call in model.calls[1:]))


if __name__ == "__main__":
    unittest.main()
