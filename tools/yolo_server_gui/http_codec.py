from __future__ import annotations

from typing import Any, Dict, List

from http_schema import DetectRequest, DetectResponse


class RequestPayloadError(Exception):
    """Raised when request payload is invalid."""


def parse_detect_request(payload: Any) -> DetectRequest:
    """Parse request JSON payload into a normalized request model.

    Args:
        payload (Any): Raw JSON payload from HTTP request.

    Returns:
        DetectRequest: Parsed request model.

    Raises:
        RequestPayloadError: If payload format is invalid.
    """
    if not isinstance(payload, dict):
        raise RequestPayloadError("invalid json object")

    thread_name = str(payload.get("threadName", ""))
    conf = _optional_float(payload.get("conf"), "conf")
    iou = _optional_float(payload.get("iou"), "iou")
    _validate_threshold(conf, "conf")
    _validate_threshold(iou, "iou")

    has_image = "image" in payload
    has_images = "images" in payload
    if has_image == has_images:
        raise RequestPayloadError("request must contain exactly one of image or images")

    if has_image:
        image = payload.get("image")
        if not isinstance(image, str) or not image.strip():
            raise RequestPayloadError("image must be a non-empty string")
        return DetectRequest(
            thread_name=thread_name,
            images=[image],
            is_batch=False,
            conf=conf,
            iou=iou,
        )

    images_raw = payload.get("images")
    if not isinstance(images_raw, list) or not images_raw:
        raise RequestPayloadError("images must be a non-empty list")
    images: List[str] = []
    for item in images_raw:
        if not isinstance(item, str) or not item.strip():
            raise RequestPayloadError("each image in images must be a non-empty string")
        images.append(item)

    return DetectRequest(
        thread_name=thread_name,
        images=images,
        is_batch=True,
        conf=conf,
        iou=iou,
    )


def encode_detect_response(response: DetectResponse) -> Dict[str, object]:
    """Encode response model into JSON payload.

    Args:
        response (DetectResponse): Response model.

    Returns:
        Dict[str, object]: JSON-ready response payload.
    """
    return response.to_dict()


def encode_error(message: str) -> Dict[str, str]:
    """Encode error response payload.

    Args:
        message (str): Error message.

    Returns:
        Dict[str, str]: JSON-ready error payload.
    """
    return {"error": message}


def _optional_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception as exc:
        raise RequestPayloadError(f"{field_name} must be a number") from exc


def _validate_threshold(value: float | None, field_name: str) -> None:
    if value is None:
        return
    if value < 0.0 or value > 1.0:
        raise RequestPayloadError(f"{field_name} must be in range [0.0, 1.0]")
