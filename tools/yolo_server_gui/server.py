from __future__ import annotations

import base64
import json
import os
import threading
import time
from http import HTTPStatus
from logging import Logger
from typing import Any, Optional, Tuple

import cv2
import numpy as np
from flask import Flask, Response, g, request
from werkzeug.serving import WSGIRequestHandler, make_server

from http_codec import RequestPayloadError, encode_detect_response, encode_error, parse_detect_request
from http_schema import DetectResponse, DetectResult, DetectionItem
from log_manager import LogController, get_logger
from yolo_engine import Detection, YoloEngine


class _SilentRequestHandler(WSGIRequestHandler):
    """Disable default werkzeug request log lines."""

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        return


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

def _dets_to_payload(dets: list[Detection]) -> list[DetectionItem]:
    payload: list[DetectionItem] = []
    for d in dets or []:
        payload.append(
            DetectionItem(
                class_id=int(d.class_id),
                class_name=d.class_name,
                conf=float(d.conf),
                xyxy=[int(v) for v in d.xyxy],
            )
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
        self._app.before_request(self._before_request)
        self._app.after_request(self._after_request)
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
            return make_server(
                self.host,
                self.port,
                self._app,
                threaded=True,
                request_handler=_SilentRequestHandler,
            )
        except TypeError:
            return make_server(
                self.host,
                self.port,
                self._app,
                request_handler=_SilentRequestHandler,
            )

    def _before_request(self) -> None:
        g._request_start_time = time.perf_counter()

    def _after_request(self, response: Response) -> Response:
        try:
            start = getattr(g, "_request_start_time", None)
            elapsed_sec = 0.0 if start is None else max(0.0, time.perf_counter() - float(start))
            protocol = str(request.environ.get("SERVER_PROTOCOL", "HTTP/1.1"))
            path = request.full_path[:-1] if request.full_path.endswith("?") else request.full_path
            remote_addr = request.remote_addr or "-"
            self._log_ctrl.info(
                '%s - - "%s %s %s" %s %.3fs',
                remote_addr,
                request.method,
                path,
                protocol,
                response.status_code,
                elapsed_sec,
            )
        except Exception:
            self._log_ctrl.exception("HTTP request logging failed.")
        return response

    def _text_response(self, status: HTTPStatus, text: str) -> Response:
        return Response(text.encode("utf-8"), status=status.value, content_type="text/plain; charset=utf-8")

    def _json_response(self, status: HTTPStatus, payload: dict[str, object]) -> Response:
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
        try:
            parsed_request = parse_detect_request(payload)
        except RequestPayloadError as exc:
            self._log_ctrl.warning("Detect request invalid: %s", exc)
            return self._json_response(HTTPStatus.BAD_REQUEST, encode_error(str(exc)))

        conf = self.conf if parsed_request.conf is None else parsed_request.conf
        iou = parsed_request.iou
        results = [self._infer_single(image_b64=img, conf=conf, iou=iou) for img in parsed_request.images]
        result_payload: DetectResult | list[DetectResult]
        if parsed_request.is_batch:
            result_payload = results
        else:
            result_payload = results[0]
        response_payload = DetectResponse(
            thread_name=parsed_request.thread_name,
            result=result_payload,
        )
        return self._json_response(HTTPStatus.OK, encode_detect_response(response_payload))

    def _infer_single(self, image_b64: str, conf: float, iou: float | None) -> DetectResult:
        img: Optional[np.ndarray] = _decode_base64_image(image_b64)
        if img is None:
            return DetectResult(classify_type="none", percentage=0.0, detections=[])
        try:
            _, dets = self._engine.infer(img, conf=conf, iou=iou)
        except Exception:
            self._log_ctrl.exception("Inference failed.")
            return DetectResult(classify_type="none", percentage=0.0, detections=[])
        cls_name, score = _pick_top1(dets)
        return DetectResult(classify_type=cls_name, percentage=score, detections=_dets_to_payload(dets))
