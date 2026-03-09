import importlib
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import List, Optional, Tuple, TYPE_CHECKING

import cv2
import numpy as np

from log_manager import LogController, get_logger

if TYPE_CHECKING:
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.plotting import Colors

@dataclass
class Detection:
    class_id: int
    class_name: str
    conf: float
    xyxy: Tuple[int, int, int, int]


class YoloEngine:
    @staticmethod
    def preload_dependencies() -> None:
        importlib.import_module("torch")
        importlib.import_module("ultralytics")
        importlib.import_module("ultralytics.utils.plotting")

    def __init__(self, model_path: str, conf: float = 0.25, iou: float = 0.45, device: str = "") -> None:
        self.model_path: str = model_path
        self.conf: float = conf
        self.iou: float = iou
        self.device: str = device
        self._model: Optional["YOLO"] = None
        self._names: Optional[dict[int, str]] = None
        self._cuda_available: bool = False
        self._device_name: str = "unknown"
        self._colors: Optional["Colors"] = None
        self._torch_mod: Optional[ModuleType] = None
        self._log_ctrl: LogController = LogController(get_logger())
        self._load_lock = threading.RLock()
        self._predict_lock = threading.RLock()
        self._is_warmed_up: bool = False
        self._is_warming_up: bool = False
        self._warmup_error: Optional[str] = None

    def update_model_path(self, model_path: str) -> None:
        with self._load_lock:
            self.model_path = model_path
            self._model = None
            self._names = None
            self._device_name = "unknown"
            self._is_warmed_up = False
            self._is_warming_up = False
            self._warmup_error = None
        self._log_ctrl.info("模型路徑已更新。model_path=%s", model_path)

    def _ensure_torch(self) -> ModuleType:
        if self._torch_mod is None:
            self._torch_mod = importlib.import_module("torch")
        return self._torch_mod

    def _ensure_colors(self) -> "Colors":
        if self._colors is None:
            colors_mod = importlib.import_module("ultralytics.utils.plotting")
            self._colors = colors_mod.Colors()
        return self._colors

    @staticmethod
    def _format_model_size(size_bytes: int) -> str:
        units = ["B", "KB", "MB", "GB", "TB", "PB"]
        value = float(max(0, size_bytes))
        unit_index = 0
        while unit_index < len(units) - 1:
            next_value = value / 1000.0
            if next_value < 0.5:
                break
            value = next_value
            unit_index += 1
        if unit_index == 0:
            return f"{int(value)}{units[unit_index]}"
        return f"{value:.2f}{units[unit_index]}"

    def _log_model_loading_start(self) -> None:
        model_path = Path(self.model_path)
        model_name = model_path.name or self.model_path
        try:
            size_bytes = model_path.stat().st_size
            size_text = self._format_model_size(size_bytes)
        except Exception:
            size_text = "unknown"
        self._log_ctrl.info(
            "開始載入模型。name=%s path=%s size=%s",
            model_name,
            str(model_path),
            size_text,
        )

    def _log_model_loading_done(self, elapsed_sec: float) -> None:
        model_path = Path(self.model_path)
        model_name = model_path.name or self.model_path
        self._log_ctrl.info(
            "模型載入完成。name=%s device=%s elapsed=%.3fs",
            model_name,
            self._device_name,
            elapsed_sec,
        )

    def load(self) -> None:
        with self._load_lock:
            torch_mod = self._ensure_torch()
            self._cuda_available = torch_mod.cuda.is_available()
            if not self.device:
                self.device = "cuda:0" if self._cuda_available else "cpu"
            if self._model is None:
                start_time = time.perf_counter()
                self._log_model_loading_start()
                yolo_mod = importlib.import_module("ultralytics")
                self._model = yolo_mod.YOLO(self.model_path)
                self._names = self._model.model.names
                if self.device:
                    try:
                        self._model.to(self.device)
                    except Exception:
                        self._log_ctrl.exception("模型載入後切換裝置失敗。device=%s", self.device)
            # Resolve actual device after model is loaded.
            self._device_name = self._resolve_device_name()
            if self._model is not None and "start_time" in locals():
                self._log_model_loading_done(time.perf_counter() - start_time)

    @property
    def names(self) -> dict[int, str]:
        return self._names or {}

    @property
    def device_name(self) -> str:
        return self._device_name

    @property
    def cuda_available(self) -> bool:
        return self._cuda_available

    @property
    def is_ready(self) -> bool:
        return self._is_warmed_up

    @property
    def is_warming_up(self) -> bool:
        return self._is_warming_up

    @property
    def warmup_error(self) -> Optional[str]:
        return self._warmup_error

    def infer(
        self, frame_bgr: object, conf: Optional[float] = None, iou: Optional[float] = None
    ) -> Tuple[Optional[object], List[Detection]]:
        self.load()
        conf = self.conf if conf is None else conf
        iou = self.iou if iou is None else iou
        with self._predict_lock:
            results = self._model.predict(
                source=frame_bgr,
                conf=conf,
                iou=iou,
                device=self.device,
                verbose=False,
            )
            self._device_name = self._resolve_device_name()
        if not results:
            return None, []
        r: object = results[0]
        dets: List[Detection] = []
        if r.boxes is not None:
            for b in r.boxes:
                cls_id: int = int(b.cls.item())
                conf = float(b.conf.item())
                xyxy: list[int] = b.xyxy[0].cpu().numpy().astype(int).tolist()
                dets.append(
                    Detection(
                        class_id=cls_id,
                        class_name=self.names.get(cls_id, str(cls_id)),
                        conf=conf,
                        xyxy=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                    )
                )
        return r, dets

    def warmup(self) -> None:
        """Load the model and run a dummy inference once."""
        with self._predict_lock:
            if self._is_warmed_up:
                return
            if self._is_warming_up:
                return
            self._is_warming_up = True
            self._warmup_error = None
            self.load()
            try:
                dummy = np.zeros((64, 64, 3), dtype=np.uint8)
                self._model.predict(
                    source=dummy,
                    conf=self.conf,
                    iou=self.iou,
                    device=self.device,
                    verbose=False,
                )
                self._device_name = self._resolve_device_name()
                self._is_warmed_up = True
            except Exception as exc:
                self._warmup_error = str(exc) or exc.__class__.__name__
                raise
            finally:
                self._is_warming_up = False

    def _resolve_device_name(self) -> str:
        try:
            torch_mod = self._ensure_torch()
            dev: Optional[str] = None
            if self._model is not None and getattr(self._model, "model", None) is not None:
                try:
                    param = next(self._model.model.parameters())
                    dev = str(param.device)
                except Exception:
                    dev = None
            if not dev and self._model is not None:
                dev = str(self._model.device)
            if not dev:
                return "unknown"
            if dev.startswith("cuda"):
                try:
                    idx: int = 0
                    if ":" in dev:
                        idx = int(dev.split(":")[1])
                    gpu_name: str = torch_mod.cuda.get_device_name(idx)
                    return f"{dev} ({gpu_name})"
                except Exception:
                    return dev
            return dev
        except Exception:
            return "unknown"

    def draw_fallback(self, frame_bgr: object, dets: List[Detection]) -> object:
        if frame_bgr is None or not hasattr(frame_bgr, "shape"):
            return frame_bgr
        try:
            h: int
            w: int
            h, w = frame_bgr.shape[:2]
        except Exception:
            return frame_bgr
        if not h or not w:
            scale: float = 0.5
        else:
            scale = max(0.5, min(h, w) / 640)
        thickness: int = max(1, int(round(scale)))
        font_scale: float = 0.5 * scale
        colors = self._ensure_colors()
        for d in dets:
            x1: int
            y1: int
            x2: int
            y2: int
            x1, y1, x2, y2 = d.xyxy
            label: str = f"{d.class_name} {d.conf:.2f}"
            color: tuple[int, int, int] = colors(d.class_id, bgr=True)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                frame_bgr,
                label,
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
                lineType=cv2.LINE_AA,
            )
        return frame_bgr
