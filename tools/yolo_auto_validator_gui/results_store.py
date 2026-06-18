from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

if __package__ in {None, ""}:
    from logging_utils import get_logger
    from serialization_utils import to_builtin_jsonable
else:
    from .logging_utils import get_logger
    from .serialization_utils import to_builtin_jsonable

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class ConfusionMatrixExport:
    """混淆矩陣匯出資料。"""

    labels: list[str]
    raw_matrix: list[list[float]]
    normalized_matrix: list[list[float]]


@dataclass(frozen=True)
class DatasetResultRecord:
    """單一資料集驗證結果。"""

    dataset_key: str
    title: str
    run_dir: str
    metrics: dict[str, float]
    per_class: list[dict[str, object]]
    confusion: ConfusionMatrixExport
    plot_files: dict[str, str]
    contains_unlabeled_images: bool
    empty_label_count: int
    background_true_negative_count: int


@dataclass(frozen=True)
class ValidationRunResult:
    """整輪驗證結果。"""

    run_dir: str
    dataset_results: list[DatasetResultRecord]
    aggregate_result: DatasetResultRecord
    warnings: list[str]


def record_from_engine_payload(payload: dict[str, object]) -> DatasetResultRecord:
    """由背景引擎 payload 建立結果紀錄。"""
    per_class_value = to_builtin_jsonable(payload.get("per_class", []))
    return DatasetResultRecord(
        dataset_key=str(payload.get("dataset_key", "")),
        title=str(payload.get("title", "")),
        run_dir=str(payload.get("run_dir", "")),
        metrics={str(key): float(value) for key, value in dict(payload.get("metrics", {})).items()},
        per_class=per_class_value if isinstance(per_class_value, list) else [],
        confusion=ConfusionMatrixExport(
            labels=[str(item) for item in list(payload.get("labels", []))],
            raw_matrix=[[float(cell) for cell in row] for row in list(payload.get("raw_matrix", []))],
            normalized_matrix=[[float(cell) for cell in row] for row in list(payload.get("normalized_matrix", []))],
        ),
        plot_files={str(key): str(value) for key, value in dict(payload.get("plot_files", {})).items()},
        contains_unlabeled_images=bool(payload.get("contains_unlabeled_images", False)),
        empty_label_count=int(payload.get("empty_label_count", 0)),
        background_true_negative_count=int(payload.get("background_true_negative_count", 0)),
    )


def save_validation_run_result(run_result: ValidationRunResult) -> Path:
    """保存整輪驗證結果索引。"""
    run_dir = Path(run_result.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Saving validation run result. run_dir=%s dataset_count=%s", run_dir, len(run_result.dataset_results))
    for record in [*run_result.dataset_results, run_result.aggregate_result]:
        _write_confusion_exports(Path(record.run_dir), record.confusion)
    index_path = run_dir / "run_index.json"
    index_path.write_text(
        json.dumps(to_builtin_jsonable(asdict(run_result)), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Validation run result saved. index_path=%s", index_path)
    return index_path


def load_saved_run(index_path: Path) -> ValidationRunResult:
    """由 index.json 載入驗證結果。"""
    obj = json.loads(index_path.read_text(encoding="utf-8"))
    dataset_results = [_record_from_dict(item) for item in list(obj.get("dataset_results", []))]
    aggregate_result = _record_from_dict(dict(obj.get("aggregate_result", {})))
    return ValidationRunResult(
        run_dir=str(obj.get("run_dir", "")),
        dataset_results=dataset_results,
        aggregate_result=aggregate_result,
        warnings=[str(item) for item in list(obj.get("warnings", []))],
    )


def list_saved_run_indexes(runs_root: Path) -> list[Path]:
    """列出所有已保存的 run index。"""
    return sorted(runs_root.glob("*/run_index.json"), reverse=True)


def _record_from_dict(obj: dict[str, object]) -> DatasetResultRecord:
    confusion_dict = dict(obj.get("confusion", {}))
    per_class_value = to_builtin_jsonable(obj.get("per_class", []))
    return DatasetResultRecord(
        dataset_key=str(obj.get("dataset_key", "")),
        title=str(obj.get("title", "")),
        run_dir=str(obj.get("run_dir", "")),
        metrics={str(key): float(value) for key, value in dict(obj.get("metrics", {})).items()},
        per_class=per_class_value if isinstance(per_class_value, list) else [],
        confusion=ConfusionMatrixExport(
            labels=[str(item) for item in list(confusion_dict.get("labels", []))],
            raw_matrix=[[float(cell) for cell in row] for row in list(confusion_dict.get("raw_matrix", []))],
            normalized_matrix=[[float(cell) for cell in row] for row in list(confusion_dict.get("normalized_matrix", []))],
        ),
        plot_files={str(key): str(value) for key, value in dict(obj.get("plot_files", {})).items()},
        contains_unlabeled_images=bool(obj.get("contains_unlabeled_images", False)),
        empty_label_count=int(obj.get("empty_label_count", 0)),
        background_true_negative_count=int(obj.get("background_true_negative_count", 0)),
    )


def _write_confusion_exports(run_dir: Path, confusion: ConfusionMatrixExport) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.debug("Writing confusion exports. run_dir=%s label_count=%s", run_dir, len(confusion.labels))
    raw_json_path = run_dir / "confusion_matrix_raw.json"
    normalized_json_path = run_dir / "confusion_matrix_normalized.json"
    raw_csv_path = run_dir / "confusion_matrix_raw.csv"
    normalized_csv_path = run_dir / "confusion_matrix_normalized.csv"
    raw_json_path.write_text(json.dumps(asdict(confusion), ensure_ascii=False, indent=2), encoding="utf-8")
    normalized_json_path.write_text(
        json.dumps(
            {
                "labels": confusion.labels,
                "matrix": confusion.normalized_matrix,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_matrix_csv(raw_csv_path, confusion.labels, confusion.raw_matrix)
    _write_matrix_csv(normalized_csv_path, confusion.labels, confusion.normalized_matrix)
    _write_confusion_plot(run_dir / "confusion_matrix.png", confusion.labels, confusion.raw_matrix, normalize=False)
    _write_confusion_plot(
        run_dir / "confusion_matrix_normalized.png",
        confusion.labels,
        confusion.normalized_matrix,
        normalize=True,
    )


def _write_matrix_csv(path: Path, labels: list[str], matrix: list[list[float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["predicted/true", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row])


def _write_confusion_plot(path: Path, labels: list[str], matrix: list[list[float]], normalize: bool) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        LOGGER.exception("Failed to import matplotlib for confusion plot rendering. path=%s", path)
        return

    try:
        array = np.asarray(matrix, dtype=float)
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        display_array = array.copy()
        display_array[display_array < 0.005] = np.nan
        image = ax.imshow(display_array, cmap="Blues", vmin=0.0, interpolation="none")
        color_threshold = 0.45 * (1.0 if normalize else np.nanmax(array) if array.size else 1.0)
        for row_index, row in enumerate(array):
            for col_index, value in enumerate(row):
                if np.isnan(display_array[row_index, col_index]):
                    continue
                text = f"{value:.2f}" if normalize else f"{int(round(value))}"
                ax.text(
                    col_index,
                    row_index,
                    text,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if value > color_threshold else "black",
                )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.05)
        ax.set_xlabel("True", labelpad=10)
        ax.set_ylabel("Predicted", labelpad=10)
        ax.set_title("Confusion Matrix Normalized" if normalize else "Confusion Matrix", pad=20)
        ticks = list(range(len(labels)))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(labels, rotation=90, ha="center")
        ax.set_yticklabels(labels)
        fig.subplots_adjust(left=0.12, right=0.88, top=0.92, bottom=0.22)
        fig.savefig(path, dpi=250)
        plt.close(fig)
    except Exception:
        LOGGER.exception("Failed to render confusion plot. path=%s", path)
