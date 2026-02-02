from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
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
    def __init__(self, model_path: str, conf: float = 0.25, iou: float = 0.45, device: str = "") -> None:
        self.model_path: str = model_path
        self.conf: float = conf
        self.iou: float = iou
        self.device: str = device
        self._model: Optional[YOLO] = None
        self._names: Optional[dict[int, str]] = None
        self._cuda_available: bool = False
        self._device_name: str = "unknown"
        self._colors: Colors = Colors()

    def load(self) -> None:
        self._cuda_available = torch.cuda.is_available()
        if not self.device:
            self.device = "cuda:0" if self._cuda_available else "cpu"
        if self._model is None:
            self._model = YOLO(self.model_path)
            self._names = self._model.model.names
            if self.device:
                try:
                    self._model.to(self.device)
                except Exception:
                    pass
        # Resolve actual device after model is loaded.
        self._device_name = self._resolve_device_name()

    @property
    def names(self) -> dict[int, str]:
        return self._names or {}

    @property
    def device_name(self) -> str:
        return self._device_name

    @property
    def cuda_available(self) -> bool:
        return self._cuda_available

    def infer(
        self, frame_bgr: object, conf: Optional[float] = None, iou: Optional[float] = None
    ) -> Tuple[Optional[object], List[Detection]]:
        self.load()
        conf = self.conf if conf is None else conf
        iou = self.iou if iou is None else iou
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

    def _resolve_device_name(self) -> str:
        try:
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
                    gpu_name: str = torch.cuda.get_device_name(idx)
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
        for d in dets:
            x1: int
            y1: int
            x2: int
            y2: int
            x1, y1, x2, y2 = d.xyxy
            label: str = f"{d.class_name} {d.conf:.2f}"
            color: tuple[int, int, int] = self._colors(d.class_id, bgr=True)
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
