from __future__ import annotations

import os
import unittest

from PySide6.QtWidgets import QApplication

from tools.yolo_auto_validator_gui.results_store import (
    ConfusionMatrixExport,
    DatasetResultRecord,
    ValidationRunResult,
)
from tools.yolo_auto_validator_gui.results_view import ResultsView

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class ResultsViewTests(unittest.TestCase):
    """驗證結果視圖行為。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_results_view_shows_background_label(self) -> None:
        view = ResultsView()
        record = DatasetResultRecord(
            dataset_key="aggregate",
            title="全部資料集彙總",
            run_dir="D:/tmp/run",
            metrics={"metrics/mAP50(B)": 0.1},
            per_class=[],
            confusion=ConfusionMatrixExport(
                labels=["Fire", "Smoke", "background"],
                raw_matrix=[[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
                normalized_matrix=[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.5, 0.0, 0.0]],
            ),
            plot_files={},
            contains_unlabeled_images=True,
            empty_label_count=2,
            background_true_negative_count=5,
        )
        view.set_run_result(
            ValidationRunResult(
                run_dir="D:/tmp/run",
                dataset_results=[],
                aggregate_result=record,
                warnings=[],
            )
        )
        self.assertIn("background", view.txt_summary.toPlainText())
        self.assertIn("真正 Background -> Background：5", view.txt_summary.toPlainText())


if __name__ == "__main__":
    unittest.main()
