from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.yolo_auto_validator_gui.results_store import (
    ConfusionMatrixExport,
    DatasetResultRecord,
    ValidationRunResult,
    load_saved_run,
    save_validation_run_result,
)


class ResultsStoreTests(unittest.TestCase):
    """驗證結果保存與載入。"""

    def test_save_validation_run_result_handles_numpy_scalar_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir) / "run"
            dataset_run_dir = run_root / "datasets" / "one"
            aggregate_run_dir = run_root / "aggregate"
            dataset_record = DatasetResultRecord(
                dataset_key="one",
                title="dataset one",
                run_dir=str(dataset_run_dir),
                metrics={"map50": 0.5},
                per_class=[
                    {
                        "class_id": np.int64(1),
                        "instances": np.int64(10),
                        "precision": np.float32(0.25),
                    }
                ],
                confusion=ConfusionMatrixExport(
                    labels=["fire", "smoke", "background"],
                    raw_matrix=[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
                    normalized_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                ),
                plot_files={},
                contains_unlabeled_images=True,
                empty_label_count=3,
                background_true_negative_count=3,
            )
            aggregate_record = DatasetResultRecord(
                dataset_key="aggregate",
                title="aggregate",
                run_dir=str(aggregate_run_dir),
                metrics={"map50": 0.6},
                per_class=[
                    {
                        "class_id": np.int64(2),
                        "instances": np.int64(20),
                        "precision": np.float64(0.75),
                    }
                ],
                confusion=dataset_record.confusion,
                plot_files={},
                contains_unlabeled_images=True,
                empty_label_count=3,
                background_true_negative_count=4,
            )
            run_result = ValidationRunResult(
                run_dir=str(run_root),
                dataset_results=[dataset_record],
                aggregate_result=aggregate_record,
                warnings=[],
            )

            index_path = save_validation_run_result(run_result)
            raw_json = json.loads(index_path.read_text(encoding="utf-8"))
            loaded_run = load_saved_run(index_path)

        self.assertIsInstance(raw_json["dataset_results"][0]["per_class"][0]["class_id"], int)
        self.assertIsInstance(raw_json["dataset_results"][0]["per_class"][0]["precision"], float)
        self.assertEqual(loaded_run.dataset_results[0].per_class[0]["class_id"], 1)
        self.assertEqual(loaded_run.aggregate_result.per_class[0]["instances"], 20)
        self.assertEqual(raw_json["dataset_results"][0]["background_true_negative_count"], 3)
        self.assertEqual(loaded_run.aggregate_result.background_true_negative_count, 4)


if __name__ == "__main__":
    unittest.main()
