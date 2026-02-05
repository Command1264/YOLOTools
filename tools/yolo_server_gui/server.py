from __future__ import annotations

import base64
import json
import os
import threading
from http import HTTPStatus
from logging import Logger
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from flask import Flask, Response, request
from werkzeug.serving import make_server

from log_manager import LogController, get_logger
from yolo_engine import Detection, YoloEngine


def _strip_data_url(data: str) -> str:
    if not data:
        return ""
    if "," in data:
        return data.split(",", 1)[1]
    return data


def _decode_base64_image(b64_str: str) -> Optional[np.ndarray]:
    if not b64_str:
        return None
    try:
        raw = base64.b64decode(_strip_data_url(b64_str), validate=False)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def _pick_top1(dets: list[Detection]) -> Tuple[str, float]:
    if not dets:
        return "none", 0.0
    best: Detection = max(dets, key=lambda d: d.conf)
    return best.class_name, float(best.conf)

def _dets_to_payload(dets: list[Detection]) -> list[Dict[str, Any]]:
    payload: list[Dict[str, Any]] = []
    for d in dets or []:
        payload.append(
            {
                "classId": int(d.class_id),
                "className": d.class_name,
                "conf": float(d.conf),
                "xyxy": [int(v) for v in d.xyxy],
            }
        )
    return payload


class YoloServer:
    """YOLO inference HTTP server based on Flask."""

    def __init__(
        self,
        model_path: str,
        host: str,
        port: int,
        conf: float = 0.25,
        logger: Optional[Logger] = None,
        icon_path: Optional[str] = None,
    ) -> None:
        self.model_path: str = model_path
        self.host: str = host
        self.port: int = port
        self.conf: float = conf
        self._logger: Logger = logger or get_logger()
        self._log_ctrl: LogController = LogController(self._logger)
        self._icon_path: Optional[str] = icon_path
        self._engine: YoloEngine = YoloEngine(model_path)
        self._app: Flask = Flask(__name__)
        self._app.add_url_rule("/", "index", self._handle_index, methods=["GET"])
        self._app.add_url_rule("/favicon.ico", "favicon", self._handle_favicon, methods=["GET"])
        self._app.add_url_rule("/detect", "detect", self._handle_detect, methods=["POST"])
        self._server: Optional[object] = None
        self._thread: Optional[threading.Thread] = None

    def is_running(self) -> bool:
        return self._server is not None

    def start(self) -> None:
        if self._server is not None:
            return
        self._server = self._create_server()
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self._log_ctrl.info("Server started. host=%s port=%s", self.host, self.port)

    def stop(self) -> None:
        if self._server is None:
            return
        try:
            self._server.shutdown()
            self._server.server_close()
        finally:
            self._server = None
            self._thread = None
        self._log_ctrl.info("Server stopped.")

    def update_model(self, model_path: str) -> None:
        self.model_path = model_path
        self._engine.update_model_path(model_path)
        self._log_ctrl.info("Model updated. model_path=%s", model_path)

    def update_settings(
        self,
        model_path: str,
        host: str,
        port: int,
        icon_path: Optional[str] = None,
    ) -> None:
        self.model_path = model_path
        self.host = host
        self.port = port
        if icon_path is not None:
            self._icon_path = icon_path
        self._engine.update_model_path(model_path)
        self._log_ctrl.info(
            "Settings updated. model_path=%s host=%s port=%s", model_path, host, port
        )

    def get_device_name(self) -> str:
        try:
            self._engine.load()
            return self._engine.device_name
        except Exception:
            return "unknown"

    def _create_server(self):
        try:
            return make_server(self.host, self.port, self._app, threaded=True)
        except TypeError:
            return make_server(self.host, self.port, self._app)

    def _text_response(self, status: HTTPStatus, text: str) -> Response:
        return Response(text.encode("utf-8"), status=status.value, content_type="text/plain; charset=utf-8")

    def _json_response(self, status: HTTPStatus, payload: Dict[str, Any]) -> Response:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return Response(data, status=status.value, content_type="application/json; charset=utf-8")

    def _handle_index(self) -> Response:
        return self._text_response(HTTPStatus.OK, "YOLO Server is running...")

    def _handle_favicon(self) -> Response:
        if self._icon_path and os.path.exists(self._icon_path):
            try:
                with open(self._icon_path, "rb") as f:
                    data = f.read()
                content_type = "image/x-icon" if self._icon_path.lower().endswith(".ico") else "image/png"
                return Response(data, status=HTTPStatus.OK.value, content_type=content_type)
            except Exception:
                self._log_ctrl.exception("favicon 載入失敗。path=%s", self._icon_path)
                return self._text_response(HTTPStatus.INTERNAL_SERVER_ERROR, "favicon load error")
        if self._icon_path:
            self._log_ctrl.warning("favicon 檔案不存在。path=%s", self._icon_path)
        return self._text_response(HTTPStatus.NOT_FOUND, "favicon not found")

    def _handle_detect(self) -> Response:
        payload: Any = request.get_json(silent=True)
        if not isinstance(payload, dict):
            self._log_ctrl.warning("Detect request with invalid JSON.")
            return self._json_response(HTTPStatus.BAD_REQUEST, {"error": "invalid json"})

        thread_name: str = payload.get("threadName", "")
        if "image" in payload:
            result: Dict[str, Any] = self._infer_single(payload.get("image", ""), self.conf)
            return self._json_response(HTTPStatus.OK, {"threadName": thread_name, "result": result})

        if "images" in payload:
            images: Any = payload.get("images", [])
            if not isinstance(images, list):
                self._log_ctrl.warning("Detect request with invalid images list.")
                return self._json_response(HTTPStatus.BAD_REQUEST, {"error": "images must be list"})
            results: list[Dict[str, Any]] = [self._infer_single(img, self.conf) for img in images]
            return self._json_response(HTTPStatus.OK, {"threadName": thread_name, "result": results})

        return self._json_response(HTTPStatus.BAD_REQUEST, {"error": "missing image or images"})

    def _infer_single(self, image_b64: str, conf: float) -> Dict[str, Any]:
        img: Optional[np.ndarray] = _decode_base64_image(image_b64)
        if img is None:
            return {"classifyType": "none", "percentage": 0.0, "detections": []}
        try:
            _, dets = self._engine.infer(img, conf=conf)
        except Exception:
            self._log_ctrl.exception("Inference failed.")
            return {"classifyType": "none", "percentage": 0.0, "detections": []}
        cls_name, score = _pick_top1(dets)
        return {
            "classifyType": cls_name,
            "percentage": score,
            "detections": _dets_to_payload(dets),
        }
