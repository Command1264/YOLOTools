from __future__ import annotations

import base64
import json
import os
import socket
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

    def setup(self) -> None:
        super().setup()
        owner = getattr(self.server, "http_owner", None)
        if owner is not None:
            owner._register_connection(self.connection)

    def finish(self) -> None:
        try:
            owner = getattr(self.server, "http_owner", None)
            if owner is not None:
                owner._unregister_connection(self.connection)
        finally:
            super().finish()

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


def _extract_thread_name(payload: Any) -> str:
    if not isinstance(payload, dict):
        return "null"
    thread_name = payload.get("threadName")
    if thread_name is None or thread_name == "":
        return "null"
    return str(thread_name)

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
        self._warmup_thread: Optional[threading.Thread] = None
        self._state_lock = threading.RLock()
        self._active_requests = 0
        self._server_state = "stopped"
        self._connection_lock = threading.RLock()
        self._active_connections: set[socket.socket] = set()
        self._rejected_requests = 0

    def is_running(self) -> bool:
        with self._state_lock:
            return self._server is not None and self._server_state != "stopped"

    @property
    def is_ready(self) -> bool:
        return self._engine.is_ready

    @property
    def is_warming_up(self) -> bool:
        return self._engine.is_warming_up

    @property
    def server_state(self) -> str:
        with self._state_lock:
            return self._server_state

    def start(self) -> None:
        if self._server is not None:
            return
        self._server = self._create_server()
        with self._state_lock:
            self._server_state = "warming_up"
            self._active_requests = 0
            self._rejected_requests = 0
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()
        self._warmup_thread = threading.Thread(target=self._run_warmup, daemon=True)
        self._warmup_thread.start()
        self._log_ctrl.info("Server started. host=%s port=%s", self.host, self.port)

    def stop(self) -> None:
        if self._server is None:
            return
        try:
            with self._state_lock:
                self._server_state = "shutting_down"
                active_requests = self._active_requests
            closed_connections = self._close_active_connections()
            self._log_ctrl.warning(
                "Server shutdown requested. active_requests=%s closed_connections=%s rejected_requests=%s",
                active_requests,
                closed_connections,
                self._rejected_requests,
            )
            self._server.shutdown()
            self._server.server_close()
        finally:
            self._server = None
            self._thread = None
            self._warmup_thread = None
            with self._state_lock:
                self._server_state = "stopped"
                self._active_requests = 0
                self._rejected_requests = 0
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
        if self._engine.is_ready:
            return self._engine.device_name
        if self._engine.is_warming_up:
            return "loading"
        return self._engine.device_name

    def _create_server(self):
        try:
            server = make_server(
                self.host,
                self.port,
                self._app,
                threaded=True,
                request_handler=_SilentRequestHandler,
            )
            setattr(server, "http_owner", self)
            setattr(server, "daemon_threads", True)
            setattr(server, "block_on_close", False)
            return server
        except TypeError:
            server = make_server(
                self.host,
                self.port,
                self._app,
                request_handler=_SilentRequestHandler,
            )
            setattr(server, "http_owner", self)
            setattr(server, "daemon_threads", True)
            setattr(server, "block_on_close", False)
            return server

    def _run_server(self) -> None:
        if self._server is None:
            return
        self._server.serve_forever()

    def _run_warmup(self) -> None:
        try:
            self._engine.warmup()
            with self._state_lock:
                if self._server_state != "stopped":
                    self._server_state = "running"
            self._log_ctrl.info("Model warmup completed.")
        except Exception:
            with self._state_lock:
                if self._server_state != "stopped":
                    self._server_state = "warmup_failed"
            self._log_ctrl.exception("Model warmup failed.")

    def _before_request(self) -> Optional[Response]:
        g._request_start_time = time.perf_counter()
        g._counted_request = False
        g._thread_name = "null"
        if request.method == "POST":
            try:
                g._thread_name = _extract_thread_name(request.get_json(silent=True))
            except Exception:
                g._thread_name = "null"
        self._log_ctrl.debug(
            "HTTP request entered. method=%s path=%s threadName=%s state=%s",
            request.method,
            request.path,
            g._thread_name,
            self.server_state,
        )
        with self._state_lock:
            self._active_requests += 1
        g._counted_request = True
        try:
            protocol = str(request.environ.get("SERVER_PROTOCOL", "HTTP/1.1"))
            path = request.full_path[:-1] if request.full_path.endswith("?") else request.full_path
            remote_addr = request.remote_addr or "-"
            self._log_ctrl.info(
                '%s - - "%s %s %s" threadName=%s started',
                remote_addr,
                request.method,
                path,
                protocol,
                g._thread_name,
            )
        except Exception:
            self._log_ctrl.exception("HTTP request start logging failed.")
        if self._server_state == "shutting_down":
            self._log_ctrl.debug("HTTP request rejected during shutdown. path=%s", request.path)
            return self._service_unavailable_response(HTTPStatus.SERVICE_UNAVAILABLE)
        if self._engine.is_ready:
            self._log_ctrl.debug("HTTP request accepted. path=%s ready=%s", request.path, self._engine.is_ready)
            return None
        self._log_ctrl.debug(
            "HTTP request rejected while model unavailable. path=%s warming_up=%s warmup_error=%s",
            request.path,
            self._engine.is_warming_up,
            self._engine.warmup_error,
        )
        return self._service_unavailable_response(HTTPStatus.SERVICE_UNAVAILABLE)

    def _after_request(self, response: Response) -> Response:
        try:
            response.headers["Connection"] = "close"
            start = getattr(g, "_request_start_time", None)
            elapsed_sec = 0.0 if start is None else max(0.0, time.perf_counter() - float(start))
            protocol = str(request.environ.get("SERVER_PROTOCOL", "HTTP/1.1"))
            path = request.full_path[:-1] if request.full_path.endswith("?") else request.full_path
            remote_addr = request.remote_addr or "-"
            self._log_ctrl.info(
                '%s - - "%s %s %s" threadName=%s %s %.3fs',
                remote_addr,
                request.method,
                path,
                protocol,
                getattr(g, "_thread_name", "null"),
                response.status_code,
                elapsed_sec,
            )
        except Exception:
            self._log_ctrl.exception("HTTP request logging failed.")
        finally:
            if getattr(g, "_counted_request", False):
                with self._state_lock:
                    self._active_requests = max(0, self._active_requests - 1)
        return response

    def _service_unavailable_response(self, status: HTTPStatus) -> Response:
        state = self.server_state
        if state == "shutting_down":
            error_detail = "server is shutting down"
        elif state == "warmup_failed":
            error_detail = self._engine.warmup_error or "model warmup failed"
        else:
            error_detail = self._engine.warmup_error or "model is warming up"
        if request.path == "/detect":
            return self._json_response(
                status,
                {"error": f"service unavailable: {error_detail}"},
            )
        return self._text_response(
            status,
            f"Service unavailable: {error_detail}",
        )

    def _register_connection(self, connection: socket.socket) -> None:
        with self._connection_lock:
            self._active_connections.add(connection)

    def _unregister_connection(self, connection: socket.socket) -> None:
        with self._connection_lock:
            self._active_connections.discard(connection)

    def _close_active_connections(self) -> int:
        with self._connection_lock:
            connections = list(self._active_connections)
            self._active_connections.clear()
        closed_count = 0
        for conn in connections:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                conn.close()
                closed_count += 1
            except Exception:
                pass
        return closed_count

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
        if self.server_state == "shutting_down":
            self._log_ctrl.debug("Detect request rejected because server is shutting down.")
            return self._service_unavailable_response(HTTPStatus.SERVICE_UNAVAILABLE)
        self._log_ctrl.debug("Detect request parsing started. threadName=%s", getattr(g, "_thread_name", "null"))
        payload: Any = request.get_json(silent=True)
        try:
            parsed_request = parse_detect_request(payload)
        except RequestPayloadError as exc:
            self._log_ctrl.warning("Detect request invalid: %s", exc)
            return self._json_response(HTTPStatus.BAD_REQUEST, encode_error(str(exc)))
        self._log_ctrl.debug(
            "Detect request parsed. threadName=%s image_count=%s is_batch=%s conf=%s iou=%s",
            parsed_request.thread_name or "null",
            len(parsed_request.images),
            parsed_request.is_batch,
            parsed_request.conf,
            parsed_request.iou,
        )
        if self.server_state == "shutting_down":
            self._log_ctrl.debug("Detect request aborted after parse because server is shutting down.")
            return self._service_unavailable_response(HTTPStatus.SERVICE_UNAVAILABLE)
        conf = self.conf if parsed_request.conf is None else parsed_request.conf
        iou = parsed_request.iou
        self._log_ctrl.debug(
            "Detect inference batch started. threadName=%s image_count=%s conf=%s iou=%s",
            parsed_request.thread_name or "null",
            len(parsed_request.images),
            conf,
            iou,
        )
        results = [self._infer_single(image_b64=img, conf=conf, iou=iou) for img in parsed_request.images]
        result_payload: DetectResult | list[DetectResult]
        if parsed_request.is_batch:
            result_payload = results
        else:
            result_payload = results[0]
        self._log_ctrl.debug(
            "Detect inference batch completed. threadName=%s result_count=%s",
            parsed_request.thread_name or "null",
            len(results),
        )
        response_payload = DetectResponse(
            thread_name=parsed_request.thread_name,
            result=result_payload,
        )
        self._log_ctrl.debug(
            "Detect response encoded. threadName=%s is_batch=%s",
            parsed_request.thread_name or "null",
            parsed_request.is_batch,
        )
        return self._json_response(HTTPStatus.OK, encode_detect_response(response_payload))

    def _infer_single(self, image_b64: str, conf: float, iou: float | None) -> DetectResult:
        if self.server_state == "shutting_down":
            self._log_ctrl.debug("Single-image inference skipped because server is shutting down.")
            return DetectResult(classify_type="none", percentage=0.0, detections=[])
        self._log_ctrl.debug("Single-image decode started. threadName=%s", getattr(g, "_thread_name", "null"))
        img: Optional[np.ndarray] = _decode_base64_image(image_b64)
        if img is None:
            self._log_ctrl.debug("Single-image decode failed. threadName=%s", getattr(g, "_thread_name", "null"))
            return DetectResult(classify_type="none", percentage=0.0, detections=[])
        self._log_ctrl.debug(
            "Single-image decode completed. threadName=%s shape=%s",
            getattr(g, "_thread_name", "null"),
            getattr(img, "shape", None),
        )
        try:
            self._log_ctrl.debug("Single-image inference started. threadName=%s", getattr(g, "_thread_name", "null"))
            _, dets = self._engine.infer(img, conf=conf, iou=iou)
        except Exception:
            self._log_ctrl.exception("Inference failed.")
            return DetectResult(classify_type="none", percentage=0.0, detections=[])
        self._log_ctrl.debug(
            "Single-image inference completed. threadName=%s detection_count=%s",
            getattr(g, "_thread_name", "null"),
            len(dets),
        )
        cls_name, score = _pick_top1(dets)
        self._log_ctrl.debug(
            "Single-image inference result prepared. threadName=%s classify_type=%s percentage=%s",
            getattr(g, "_thread_name", "null"),
            cls_name,
            score,
        )
        return DetectResult(classify_type=cls_name, percentage=score, detections=_dets_to_payload(dets))
