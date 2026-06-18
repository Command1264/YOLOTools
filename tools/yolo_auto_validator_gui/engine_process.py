from __future__ import annotations

import gc
import traceback
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    from engine_protocol import EngineEvent, EngineRequest
    from logging_utils import configure_logging, get_logger
    from serialization_utils import to_builtin_jsonable
    from yaml_mapping_model import YamlMappingModel
else:
    from .engine_protocol import EngineEvent, EngineRequest
    from .logging_utils import configure_logging, get_logger
    from .serialization_utils import to_builtin_jsonable
    from .yaml_mapping_model import YamlMappingModel

LOGGER = get_logger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
BYTES_PER_GIB = 1024**3
VALIDATION_DIRECT_CPU_FREE_BYTES = int(1.0 * BYTES_PER_GIB)
VALIDATION_LOW_FREE_BYTES = int(2.5 * BYTES_PER_GIB)
VALIDATION_MEDIUM_FREE_BYTES = int(4.5 * BYTES_PER_GIB)
VALIDATION_HIGH_FREE_BYTES = int(8.0 * BYTES_PER_GIB)
DIRECT_CPU_FREE_BYTES = int(1.25 * BYTES_PER_GIB)
LOW_FREE_BYTES = int(2.5 * BYTES_PER_GIB)
MEDIUM_FREE_BYTES = int(4.0 * BYTES_PER_GIB)
HIGH_FREE_BYTES = int(8.0 * BYTES_PER_GIB)


@dataclass(frozen=True)
class ValidationExecutionPlan:
    """模型驗證執行策略。"""

    execution_device: str
    batch_size: int
    strategy: str
    free_bytes: int | None = None
    total_bytes: int | None = None


@dataclass(frozen=True)
class BackgroundCountPlan:
    """Background 真陰性補算執行策略。"""

    execution_device: str
    batch_size: int
    strategy: str
    free_bytes: int | None = None
    total_bytes: int | None = None


def run_engine_loop(request_queue, event_queue) -> None:
    """背景驗證引擎主迴圈。"""
    configure_logging("yolo_auto_validator.engine_process")
    yolo_class = None
    loaded_model_path = ""
    loaded_model = None
    LOGGER.info("Validation engine subprocess started.")

    def emit(request_id: str, kind: str, payload: dict[str, object] | None = None) -> None:
        LOGGER.debug("Emitting engine event. request_id=%s kind=%s payload_keys=%s", request_id, kind, sorted((payload or {}).keys()))
        event_queue.put(EngineEvent(request_id=request_id, kind=kind, payload=payload or {}))

    while True:
        request: EngineRequest = request_queue.get()
        LOGGER.info("Engine request received. request_id=%s action=%s", request.request_id, request.action)
        if request.action == "shutdown":
            emit(request.request_id, "stopped", {})
            LOGGER.info("Validation engine subprocess shutting down.")
            return
        stage = "received_request"
        try:
            if request.action == "warmup_runtime":
                stage = "import_ultralytics"
                emit(request.request_id, "progress_unknown", {"message": "背景載入 ultralytics 中..."})
                yolo_class = _import_yolo_class()
                emit(request.request_id, "log", {"message": "ultralytics 背景載入完成。"})
                emit(request.request_id, "completed", {})
                continue

            if request.action == "load_model_info":
                stage = "load_model_info"
                emit(request.request_id, "progress_unknown", {"message": "背景讀取模型資訊中..."})
                stage = "import_ultralytics"
                yolo_class = yolo_class or _import_yolo_class()
                model_path = str(request.payload.get("model_path", ""))
                stage = "ensure_model"
                loaded_model_path, loaded_model = _ensure_model(yolo_class, model_path, loaded_model_path, loaded_model)
                stage = "extract_model_names"
                names = _normalize_names(getattr(loaded_model, "names", {}))
                emit(
                    request.request_id,
                    "model_info",
                    {
                        "model_path": model_path,
                        "names": names,
                    },
                )
                emit(request.request_id, "completed", {})
                continue

            if request.action == "validate_dataset":
                stage = "validate_dataset"
                emit(request.request_id, "progress_unknown", {"message": "背景驗證中..."})
                stage = "import_ultralytics"
                yolo_class = yolo_class or _import_yolo_class()
                model_path = str(request.payload.get("model_path", ""))
                stage = "ensure_model"
                loaded_model_path, loaded_model = _ensure_model(yolo_class, model_path, loaded_model_path, loaded_model)
                emit(request.request_id, "log", {"message": f"開始驗證：{request.payload.get('title', '')}"})
                stage = "run_validation"
                result_payload = _run_validation(loaded_model, request.payload)
                emit(request.request_id, "validation_done", result_payload)
                emit(request.request_id, "completed", {})
                continue

            raise RuntimeError(f"未知引擎動作：{request.action}")
        except Exception as exc:
            traceback_text = traceback.format_exc()
            LOGGER.error(
                "Engine request failed. request_id=%s action=%s stage=%s error=%s\n%s",
                request.request_id,
                request.action,
                stage,
                exc,
                traceback_text.rstrip(),
            )
            emit(
                request.request_id,
                "error",
                {
                    "message": str(exc),
                    "action": request.action,
                    "stage": stage,
                    "traceback": traceback_text,
                },
            )


def _import_yolo_class():
    LOGGER.info("Importing ultralytics.YOLO in validation subprocess.")
    from ultralytics import YOLO

    return YOLO


def _ensure_model(yolo_class, model_path: str, loaded_model_path: str, loaded_model):
    if not model_path:
        raise ValueError("模型路徑不可為空。")
    if loaded_model is not None and loaded_model_path == model_path:
        LOGGER.debug("Reusing cached YOLO model. model_path=%s", model_path)
        return loaded_model_path, loaded_model
    LOGGER.info("Loading YOLO model in validation subprocess. model_path=%s", model_path)
    return model_path, yolo_class(model_path)


def _normalize_names(raw_names: object) -> list[str]:
    if isinstance(raw_names, dict):
        return [str(value) for _, value in sorted(raw_names.items(), key=lambda item: int(item[0]))]
    if isinstance(raw_names, list):
        return [str(item) for item in raw_names]
    return []


def _run_validation(model, payload: dict[str, object]) -> dict[str, object]:
    data_yaml_path = str(payload.get("data_yaml_path", ""))
    project_dir = Path(str(payload.get("project_dir", ""))).resolve()
    run_name = str(payload.get("run_name", ""))
    conf_threshold = float(payload.get("conf_threshold", 0.25))
    iou_threshold = float(payload.get("iou_threshold", 0.45))
    kwargs: dict[str, object] = {
        "data": data_yaml_path,
        "split": "val",
        "plots": True,
        "project": str(project_dir),
        "name": run_name,
        "exist_ok": True,
        "verbose": False,
        "workers": 0,
        "conf": conf_threshold,
        "iou": iou_threshold,
    }
    device = str(payload.get("device", "")).strip()
    torch_module = _import_torch_module()
    validation_plan = _select_validation_plan(
        requested_device=device,
        torch_module=torch_module,
    )
    kwargs = _apply_validation_plan(kwargs, validation_plan)
    LOGGER.info(
        "Starting model validation. title=%s dataset_key=%s data_yaml=%s project_dir=%s run_name=%s strategy=%s execution_device=%s batch_size=%s free_gib=%s total_gib=%s",
        payload.get("title", ""),
        payload.get("dataset_key", ""),
        data_yaml_path,
        project_dir,
        run_name,
        validation_plan.strategy,
        validation_plan.execution_device,
        validation_plan.batch_size,
        _format_gib(validation_plan.free_bytes),
        _format_gib(validation_plan.total_bytes),
    )
    LOGGER.debug("Validation kwargs prepared. kwargs=%s", kwargs)
    results, completed_validation_plan = _run_model_validation_with_retry(
        model=model,
        base_kwargs=kwargs,
        initial_plan=validation_plan,
        torch_module=torch_module,
        title=str(payload.get("title", "")),
    )
    LOGGER.info(
        "model.val completed. title=%s save_dir=%s execution_device=%s batch_size=%s strategy=%s",
        payload.get("title", ""),
        getattr(results, "save_dir", ""),
        completed_validation_plan.execution_device,
        completed_validation_plan.batch_size,
        completed_validation_plan.strategy,
    )
    confusion_matrix = getattr(results, "confusion_matrix", None)
    if confusion_matrix is None:
        raise RuntimeError("驗證結果缺少 confusion_matrix。")
    raw_matrix = confusion_matrix.matrix.tolist()
    _clear_torch_device_cache(completed_validation_plan.execution_device, torch_module)
    background_true_negative_count = _count_true_negative_background_images(
        model=model,
        data_yaml_path=Path(data_yaml_path),
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        device=completed_validation_plan.execution_device,
        torch_module=torch_module,
    )
    raw_matrix = _inject_background_true_negatives(raw_matrix, background_true_negative_count)
    normalized_matrix = _normalize_confusion_matrix(raw_matrix)
    labels = _confusion_labels(confusion_matrix)
    save_dir = Path(getattr(results, "save_dir"))
    plot_files = _collect_plot_files(save_dir)
    LOGGER.info(
        "Validation result serialized. title=%s labels=%s plot_count=%s background_true_negative_count=%s",
        payload.get("title", ""),
        len(labels),
        len(plot_files),
        background_true_negative_count,
    )
    return {
        "title": str(payload.get("title", "")),
        "dataset_key": str(payload.get("dataset_key", "")),
        "run_dir": str(save_dir),
        "metrics": {key: float(value) for key, value in getattr(results, "results_dict", {}).items()},
        "per_class": to_builtin_jsonable(getattr(results, "summary", lambda: [])()),
        "labels": labels,
        "raw_matrix": raw_matrix,
        "normalized_matrix": normalized_matrix,
        "plot_files": plot_files,
        "contains_unlabeled_images": bool(payload.get("contains_unlabeled_images", False)),
        "empty_label_count": int(payload.get("empty_label_count", 0)),
        "background_true_negative_count": background_true_negative_count,
    }


def _apply_validation_plan(kwargs: dict[str, object], plan: ValidationExecutionPlan) -> dict[str, object]:
    out = dict(kwargs)
    out["batch"] = int(plan.batch_size)
    if plan.execution_device:
        out["device"] = plan.execution_device
    else:
        out.pop("device", None)
    return out


def _run_model_validation_with_retry(
    *,
    model,
    base_kwargs: dict[str, object],
    initial_plan: ValidationExecutionPlan,
    torch_module,
    title: str,
):
    current_plan = initial_plan
    while True:
        current_kwargs = _apply_validation_plan(base_kwargs, current_plan)
        _clear_torch_device_cache(current_plan.execution_device, torch_module)
        try:
            return model.val(**current_kwargs), current_plan
        except RuntimeError as exc:
            next_plan = _next_validation_retry_plan(exc, current_plan)
            if next_plan is None:
                raise
            LOGGER.warning(
                "Validation OOM detected. title=%s execution_device=%s batch_size=%s strategy=%s Retrying with execution_device=%s batch_size=%s strategy=%s.",
                title,
                current_plan.execution_device,
                current_plan.batch_size,
                current_plan.strategy,
                next_plan.execution_device,
                next_plan.batch_size,
                next_plan.strategy,
            )
            _clear_torch_device_cache(current_plan.execution_device, torch_module)
            current_plan = next_plan


def _next_validation_retry_plan(
    exc: RuntimeError,
    current_plan: ValidationExecutionPlan,
) -> ValidationExecutionPlan | None:
    if not _should_retry_validation_on_lower_batch(exc, current_plan.execution_device):
        return None
    if _is_cuda_device(current_plan.execution_device) and current_plan.batch_size > 1:
        return ValidationExecutionPlan(
            execution_device=current_plan.execution_device,
            batch_size=max(1, current_plan.batch_size // 2),
            strategy=f"{current_plan.strategy}_retry_lower_batch",
            free_bytes=current_plan.free_bytes,
            total_bytes=current_plan.total_bytes,
        )
    if _is_cuda_device(current_plan.execution_device):
        return ValidationExecutionPlan(
            execution_device="cpu",
            batch_size=1,
            strategy="validation_cpu_fallback_after_cuda_oom",
            free_bytes=current_plan.free_bytes,
            total_bytes=current_plan.total_bytes,
        )
    return None


def _select_validation_plan(
    *,
    requested_device: str,
    torch_module,
) -> ValidationExecutionPlan:
    normalized_device = _normalize_requested_device(requested_device, torch_module)
    if normalized_device == "cpu":
        return ValidationExecutionPlan(
            execution_device="cpu",
            batch_size=1,
            strategy="validation_cpu_requested_or_cuda_unavailable",
        )
    if not _is_cuda_device(normalized_device):
        return ValidationExecutionPlan(
            execution_device=normalized_device,
            batch_size=1,
            strategy="validation_non_cuda_device",
        )

    device_index = _extract_cuda_device_index(normalized_device)
    free_bytes, total_bytes = _read_cuda_memory_info(torch_module, device_index)
    if total_bytes is None:
        return ValidationExecutionPlan(
            execution_device=normalized_device,
            batch_size=1,
            strategy="validation_gpu_unknown_memory_batch1",
        )
    if free_bytes is None:
        free_bytes = total_bytes

    if free_bytes < VALIDATION_DIRECT_CPU_FREE_BYTES:
        return ValidationExecutionPlan(
            execution_device="cpu",
            batch_size=1,
            strategy="validation_cpu_low_free_vram",
            free_bytes=free_bytes,
            total_bytes=total_bytes,
        )
    if free_bytes < VALIDATION_LOW_FREE_BYTES:
        batch_size = 1
        strategy = "validation_gpu_batch1"
    elif free_bytes < VALIDATION_MEDIUM_FREE_BYTES:
        batch_size = 2
        strategy = "validation_gpu_batch2"
    elif free_bytes < VALIDATION_HIGH_FREE_BYTES:
        batch_size = 4
        strategy = "validation_gpu_batch4"
    else:
        batch_size = 8
        strategy = "validation_gpu_batch8"
    return ValidationExecutionPlan(
        execution_device=normalized_device,
        batch_size=batch_size,
        strategy=strategy,
        free_bytes=free_bytes,
        total_bytes=total_bytes,
    )


def _confusion_labels(confusion_matrix) -> list[str]:
    names = getattr(confusion_matrix, "names", {})
    label_names = _normalize_names(names)
    return [*label_names, "background"]


def _normalize_confusion_matrix(matrix) -> list[list[float]]:
    import numpy as np

    array = np.asarray(matrix, dtype=float)
    normalized = array / (array.sum(0).reshape(1, -1) + 1e-9)
    return normalized.tolist()


def _inject_background_true_negatives(raw_matrix: list[list[float]], count: int) -> list[list[float]]:
    if not raw_matrix or not raw_matrix[-1]:
        return raw_matrix
    adjusted = [[float(cell) for cell in row] for row in raw_matrix]
    adjusted[-1][-1] += float(count)
    return adjusted


def _count_true_negative_background_images(
    *,
    model,
    data_yaml_path: Path,
    conf_threshold: float,
    iou_threshold: float,
    device: str,
    torch_module=None,
) -> int:
    negative_images = _collect_negative_validation_images(data_yaml_path)
    if not negative_images:
        LOGGER.info("No negative validation images found for background/background counting. data_yaml=%s", data_yaml_path)
        return 0

    torch_module = torch_module or _import_torch_module()
    plan = _select_background_count_plan(
        requested_device=device,
        negative_image_count=len(negative_images),
        torch_module=torch_module,
    )
    LOGGER.info(
        "Counting true negative background images. data_yaml=%s negative_image_count=%s strategy=%s execution_device=%s batch_size=%s free_gib=%s total_gib=%s",
        data_yaml_path,
        len(negative_images),
        plan.strategy,
        plan.execution_device,
        plan.batch_size,
        _format_gib(plan.free_bytes),
        _format_gib(plan.total_bytes),
    )
    _clear_torch_device_cache(plan.execution_device, torch_module)
    try:
        true_negative_count = _predict_true_negative_background_images(
            model=model,
            negative_images=negative_images,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            execution_device=plan.execution_device,
            batch_size=plan.batch_size,
        )
    except RuntimeError as exc:
        if not _should_retry_background_count_on_cpu(exc, plan.execution_device):
            raise
        LOGGER.warning(
            "CUDA OOM while counting true negatives. data_yaml=%s execution_device=%s batch_size=%s Falling back to CPU.",
            data_yaml_path,
            plan.execution_device,
            plan.batch_size,
        )
        _clear_torch_device_cache(plan.execution_device, torch_module)
        cpu_plan = BackgroundCountPlan(
            execution_device="cpu",
            batch_size=1,
            strategy="cpu_fallback_after_cuda_oom",
            free_bytes=plan.free_bytes,
            total_bytes=plan.total_bytes,
        )
        true_negative_count = _predict_true_negative_background_images(
            model=model,
            negative_images=negative_images,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            execution_device=cpu_plan.execution_device,
            batch_size=cpu_plan.batch_size,
        )
    LOGGER.info(
        "True negative background counting completed. data_yaml=%s true_negative_count=%s negative_image_count=%s",
        data_yaml_path,
        true_negative_count,
        len(negative_images),
    )
    return true_negative_count


def _predict_true_negative_background_images(
    *,
    model,
    negative_images: list[Path],
    conf_threshold: float,
    iou_threshold: float,
    execution_device: str,
    batch_size: int,
) -> int:
    true_negative_count = 0
    total_batches = max(1, (len(negative_images) + batch_size - 1) // batch_size)
    for batch_index, start in enumerate(range(0, len(negative_images), batch_size), start=1):
        batch_paths = negative_images[start : start + batch_size]
        predict_kwargs: dict[str, object] = {
            "source": [str(path) for path in batch_paths],
            "conf": conf_threshold,
            "iou": iou_threshold,
            "verbose": False,
            "save": False,
            "stream": True,
            "batch": len(batch_paths),
        }
        if execution_device:
            predict_kwargs["device"] = execution_device
        LOGGER.debug(
            "Running background counting prediction batch. execution_device=%s batch_index=%s total_batches=%s batch_size=%s",
            execution_device,
            batch_index,
            total_batches,
            len(batch_paths),
        )
        for result in model.predict(**predict_kwargs):
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                true_negative_count += 1
    return true_negative_count


def _select_background_count_plan(
    *,
    requested_device: str,
    negative_image_count: int,
    torch_module,
) -> BackgroundCountPlan:
    normalized_device = _normalize_requested_device(requested_device, torch_module)
    if normalized_device == "cpu":
        return BackgroundCountPlan(
            execution_device="cpu",
            batch_size=1,
            strategy="cpu_requested_or_cuda_unavailable",
        )
    if not _is_cuda_device(normalized_device):
        return BackgroundCountPlan(
            execution_device=normalized_device,
            batch_size=min(2, max(1, negative_image_count)),
            strategy="non_cuda_device",
        )

    device_index = _extract_cuda_device_index(normalized_device)
    free_bytes, total_bytes = _read_cuda_memory_info(torch_module, device_index)
    if total_bytes is None:
        return BackgroundCountPlan(
            execution_device=normalized_device,
            batch_size=1,
            strategy="gpu_unknown_memory_batch1",
        )
    if free_bytes is None:
        free_bytes = total_bytes

    if free_bytes < DIRECT_CPU_FREE_BYTES:
        return BackgroundCountPlan(
            execution_device="cpu",
            batch_size=1,
            strategy="cpu_low_free_vram",
            free_bytes=free_bytes,
            total_bytes=total_bytes,
        )
    if total_bytes <= 6 * BYTES_PER_GIB and free_bytes < LOW_FREE_BYTES:
        return BackgroundCountPlan(
            execution_device="cpu",
            batch_size=1,
            strategy="cpu_small_gpu_low_free_vram",
            free_bytes=free_bytes,
            total_bytes=total_bytes,
        )
    if free_bytes < LOW_FREE_BYTES:
        batch_size = 1
        strategy = "gpu_batch1"
    elif free_bytes < MEDIUM_FREE_BYTES:
        batch_size = 2
        strategy = "gpu_batch2"
    elif free_bytes < HIGH_FREE_BYTES:
        batch_size = 4
        strategy = "gpu_batch4"
    else:
        batch_size = 8
        strategy = "gpu_batch8"
    return BackgroundCountPlan(
        execution_device=normalized_device,
        batch_size=min(batch_size, max(1, negative_image_count)),
        strategy=strategy,
        free_bytes=free_bytes,
        total_bytes=total_bytes,
    )


def _normalize_requested_device(requested_device: str, torch_module) -> str:
    raw_device = requested_device.strip()
    primary_device = raw_device.split(",")[0].strip()
    if not primary_device:
        return "cuda:0" if _torch_cuda_available(torch_module) else "cpu"
    lowered = primary_device.lower()
    if lowered == "cpu":
        return "cpu"
    if lowered.isdigit():
        return f"cuda:{lowered}"
    if lowered == "cuda":
        return "cuda:0"
    return lowered


def _torch_cuda_available(torch_module) -> bool:
    cuda_module = getattr(torch_module, "cuda", None)
    return bool(cuda_module is not None and cuda_module.is_available())


def _is_cuda_device(device: str) -> bool:
    return device.startswith("cuda")


def _extract_cuda_device_index(device: str) -> int:
    if ":" not in device:
        return 0
    _, _, suffix = device.partition(":")
    return int(suffix or 0)


def _read_cuda_memory_info(torch_module, device_index: int) -> tuple[int | None, int | None]:
    cuda_module = getattr(torch_module, "cuda", None)
    if cuda_module is None or not cuda_module.is_available():
        return None, None
    free_bytes: int | None = None
    total_bytes: int | None = None
    try:
        free_bytes, total_bytes = cuda_module.mem_get_info(device_index)
    except Exception:
        LOGGER.debug("torch.cuda.mem_get_info unavailable for device_index=%s", device_index, exc_info=True)
    if total_bytes is None:
        try:
            total_bytes = int(cuda_module.get_device_properties(device_index).total_memory)
        except Exception:
            LOGGER.debug("torch.cuda.get_device_properties failed for device_index=%s", device_index, exc_info=True)
    return free_bytes, total_bytes


def _clear_torch_device_cache(device: str, torch_module=None) -> None:
    torch_module = torch_module or _import_torch_module()
    gc.collect()
    normalized_device = _normalize_requested_device(device, torch_module)
    if not _is_cuda_device(normalized_device):
        return
    cuda_module = getattr(torch_module, "cuda", None)
    if cuda_module is None or not cuda_module.is_available():
        return
    try:
        cuda_module.empty_cache()
        if hasattr(cuda_module, "ipc_collect"):
            cuda_module.ipc_collect()
        LOGGER.debug("Cleared CUDA cache. execution_device=%s", normalized_device)
    except Exception:
        LOGGER.warning("Failed to clear CUDA cache. execution_device=%s", normalized_device, exc_info=True)


def _should_retry_background_count_on_cpu(exc: RuntimeError, execution_device: str) -> bool:
    if not _is_cuda_device(execution_device):
        return False
    message = str(exc).lower()
    return "out of memory" in message or "cuda out of memory" in message


def _should_retry_validation_on_lower_batch(exc: RuntimeError, execution_device: str) -> bool:
    if not _is_cuda_device(execution_device):
        return False
    message = str(exc).lower()
    return "out of memory" in message or "cuda out of memory" in message


def _format_gib(value: int | None) -> str:
    if value is None:
        return "unknown"
    return f"{value / BYTES_PER_GIB:.2f}"


def _import_torch_module():
    import torch

    return torch


def _collect_negative_validation_images(data_yaml_path: Path) -> list[Path]:
    mapping = YamlMappingModel.from_file(data_yaml_path).data
    dataset_root = _resolve_dataset_root(data_yaml_path, str(mapping.get("path", ".")))
    val_entries = _normalize_split_entries(mapping.get("val"))
    images = _collect_split_images(val_entries, dataset_root, data_yaml_path.parent)
    negative_images: list[Path] = []
    for image_path in images:
        label_path = _find_label_path(image_path)
        if label_path is None:
            negative_images.append(image_path)
            continue
        if not label_path.read_text(encoding="utf-8", errors="ignore").strip():
            negative_images.append(image_path)
    return negative_images


def _resolve_dataset_root(data_yaml_path: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (data_yaml_path.parent / path).resolve()


def _normalize_split_entries(raw_value: object) -> list[str]:
    if isinstance(raw_value, list):
        return [str(item) for item in raw_value]
    return [str(raw_value)]


def _collect_split_images(entries: list[str], dataset_root: Path, yaml_dir: Path) -> list[Path]:
    images: list[Path] = []
    for raw_entry in entries:
        entry_path = _resolve_entry_path(raw_entry, dataset_root, yaml_dir)
        if entry_path.is_dir():
            images.extend([path for path in sorted(entry_path.rglob("*")) if path.suffix.lower() in IMAGE_EXTENSIONS])
            continue
        if entry_path.is_file() and entry_path.suffix.lower() == ".txt":
            text = entry_path.read_text(encoding="utf-8", errors="ignore")
            for line in text.splitlines():
                item = line.strip()
                if not item or item.startswith("#"):
                    continue
                images.append(_resolve_entry_path(item, dataset_root, entry_path.parent))
            continue
        if entry_path.is_file() and entry_path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(entry_path)
    return images


def _resolve_entry_path(raw_entry: str, dataset_root: Path, yaml_dir: Path) -> Path:
    path = Path(raw_entry)
    if path.is_absolute() and path.exists():
        return path.resolve()
    candidate_root = (dataset_root / raw_entry).resolve()
    if candidate_root.exists():
        return candidate_root
    candidate_yaml = (yaml_dir / raw_entry).resolve()
    if candidate_yaml.exists():
        return candidate_yaml
    return candidate_root


def _find_label_path(image_path: Path) -> Path | None:
    candidates = [image_path.with_suffix(".txt")]
    parts = list(image_path.parts)
    for index, part in enumerate(parts):
        if part.lower() != "images":
            continue
        replaced = parts[:]
        replaced[index] = "labels"
        candidates.insert(0, Path(*replaced).with_suffix(".txt"))
        break
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _collect_plot_files(save_dir: Path) -> dict[str, str]:
    known_names = [
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "P_curve.png",
        "R_curve.png",
        "F1_curve.png",
    ]
    out: dict[str, str] = {}
    for name in known_names:
        path = save_dir / name
        if path.exists():
            out[name] = str(path)
        else:
            LOGGER.debug("Validation plot not found. save_dir=%s file=%s", save_dir, name)
    return out
