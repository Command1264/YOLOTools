from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import torch
from ultralytics import YOLO


@dataclass
class Detection:
    class_id: int
    class_name: str
    conf: float
    xyxy: Tuple[int, int, int, int]


class YoloEngine:
    def __init__(self, model_path: str, conf: float = 0.25, iou: float = 0.45, device: str = ""):
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.device = device
        self._model: Optional[YOLO] = None
        self._names = None
        self._cuda_available = False

    def load(self):
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
    def names(self):
        return self._names or {}

    @property
    def device_name(self):
        return getattr(self, "_device_name", "unknown")

    @property
    def cuda_available(self):
        return self._cuda_available

    def infer(self, frame_bgr, conf: Optional[float] = None, iou: Optional[float] = None) -> Tuple[Optional[object], List[Detection]]:
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
        r = results[0]
        dets: List[Detection] = []
        if r.boxes is not None:
            for b in r.boxes:
                cls_id = int(b.cls.item())
                conf = float(b.conf.item())
                xyxy = b.xyxy[0].cpu().numpy().astype(int).tolist()
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
            dev = None
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
                    idx = 0
                    if ":" in dev:
                        idx = int(dev.split(":")[1])
                    gpu_name = torch.cuda.get_device_name(idx)
                    return f"{dev} ({gpu_name})"
                except Exception:
                    return dev
            return dev
        except Exception:
            return "unknown"

    @staticmethod
    def draw_fallback(frame_bgr, dets: List[Detection]):
        for d in dets:
            x1, y1, x2, y2 = d.xyxy
            label = f"{d.class_name} {d.conf:.2f}"
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame_bgr,
                label,
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                lineType=cv2.LINE_AA,
            )
        return frame_bgr
