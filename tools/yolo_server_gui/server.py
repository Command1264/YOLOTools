from __future__ import annotations

import base64
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

import sys

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
VALIDATOR_DIR = ROOT_DIR / "yolo_validator_gui"
if str(VALIDATOR_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATOR_DIR))

from yolo_engine import YoloEngine


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


def _pick_top1(dets) -> Tuple[str, float]:
    if not dets:
        return "none", 0.0
    best = max(dets, key=lambda d: d.conf)
    return best.class_name, float(best.conf)


class YoloServer:
    def __init__(self, model_path: str, host: str, port: int, conf: float = 0.25):
        self.model_path = model_path
        self.host = host
        self.port = port
        self.conf = conf
        self._engine = YoloEngine(model_path)
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def is_running(self) -> bool:
        return self._server is not None

    def start(self) -> None:
        if self._server is not None:
            return
        handler_cls = self._make_handler()
        self._server = ThreadingHTTPServer((self.host, self.port), handler_cls)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is None:
            return
        try:
            self._server.shutdown()
            self._server.server_close()
        finally:
            self._server = None
            self._thread = None

    def update_model(self, model_path: str) -> None:
        self.model_path = model_path
        self._engine = YoloEngine(model_path, conf=self.conf)

    def _make_handler(self):
        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            def _send_json(self, status: HTTPStatus, payload: Dict[str, Any]) -> None:
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status.value)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _send_text(self, status: HTTPStatus, text: str) -> None:
                data = text.encode("utf-8")
                self.send_response(status.value)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _read_json(self) -> Optional[Dict[str, Any]]:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except Exception:
                    length = 0
                if length <= 0:
                    return None
                try:
                    raw = self.rfile.read(length)
                    return json.loads(raw.decode("utf-8"))
                except Exception:
                    return None

            def do_GET(self):
                path = self.path.rstrip("/") or "/"
                if path == "/":
                    self._send_text(HTTPStatus.OK, "YOLO Server is running...")
                else:
                    self._send_text(HTTPStatus.NOT_FOUND, "Not Found")

            def do_POST(self):
                path = self.path.rstrip("/") or "/"
                if path != "/detect":
                    self._send_text(HTTPStatus.NOT_FOUND, "Not Found")
                    return
                payload = self._read_json()
                if not isinstance(payload, dict):
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid json"})
                    return
                thread_name = payload.get("threadName", "")
                if "image" in payload:
                    result = self._infer_single(payload.get("image", ""), server_ref.conf)
                    self._send_json(HTTPStatus.OK, {"threadName": thread_name, "result": result})
                    return
                if "images" in payload:
                    images = payload.get("images", [])
                    if not isinstance(images, list):
                        self._send_json(HTTPStatus.BAD_REQUEST, {"error": "images must be list"})
                        return
                    results = [self._infer_single(img, server_ref.conf) for img in images]
                    self._send_json(HTTPStatus.OK, {"threadName": thread_name, "result": results})
                    return
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "missing image or images"})

            def _infer_single(self, image_b64: str, conf: float) -> Dict[str, Any]:
                img = _decode_base64_image(image_b64)
                if img is None:
                    return {"classifyType": "none", "percentage": 0.0}
                try:
                    _, dets = server_ref._engine.infer(img, conf=conf)
                except Exception:
                    return {"classifyType": "none", "percentage": 0.0}
                cls_name, score = _pick_top1(dets)
                return {"classifyType": cls_name, "percentage": score}

            def log_message(self, format, *args):
                return

        return Handler
