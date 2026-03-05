from __future__ import annotations

from typing import Any, Dict, List, Tuple

try:
    from .http_schema import (
        DetectRequest,
        DetectResponse,
        DetectResult,
        DetectionResponse,
        DetectionRoi,
    )
except ImportError:
    from http_schema import (
        DetectRequest,
        DetectResponse,
        DetectResult,
        DetectionResponse,
        DetectionRoi,
    )


class RequestPayloadError(Exception):
    """Raised when /Detection request payload is invalid."""


def parse_detect_request(payload: Any) -> DetectRequest:
    """Parse request payload into the server-compatible request model.

    Input rules are aligned with the legacy `/Detection` endpoint.

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
    conf = _optional_float(payload.get("conf"))
    iou = _optional_float(payload.get("iou"))

    if "images" in payload:
        images = payload["images"]
        if not isinstance(images, list):
            raise RequestPayloadError("`images` must be a list")
        return DetectRequest(
            thread_name=thread_name,
            images=images,
            is_batch=True,
            conf=conf,
            iou=iou,
        )

    if "image" in payload:
        return DetectRequest(
            thread_name=thread_name,
            images=[payload["image"]],
            is_batch=False,
            conf=conf,
            iou=iou,
        )

    raise RequestPayloadError("Missing `image` or `images`")


def parse_detection_request(payload: Any) -> DetectRequest:
    """Backward-compatible alias for parse_detect_request.

    Args:
        payload (Any): Raw JSON payload from HTTP request.

    Returns:
        DetectRequest: Parsed request model.
    """
    return parse_detect_request(payload)


def encode_detect_response(response: DetectResponse) -> Dict[str, object]:
    """Encode server detect response into `/Detection` payload shape.

    Args:
        response (DetectResponse): Response model from server flow.

    Returns:
        Dict[str, object]: JSON-ready `/Detection` payload.
    """
    results: List[DetectResult]
    if isinstance(response.result, list):
        results = response.result
    else:
        results = [response.result]

    roi_items: List[DetectionRoi] = []
    for result_item in results:
        cvt_json = []
        for det in result_item.detections:
            rect = _to_roi_rect(det)
            if rect is not None:
                cvt_json.append(rect)
        roi_items.append(
            build_roi(
                thread_name=response.thread_name,
                classify_type=result_item.classify_type,
                prob=float(result_item.percentage),
                cvt_json=cvt_json,
            )
        )

    return DetectionResponse(thread_name=response.thread_name, cvt_jsons=roi_items).to_dict()


def encode_detection_response(response: DetectResponse | DetectionResponse) -> Dict[str, object]:
    """Encode `/Detection` response model into JSON payload.

    Args:
        response (DetectResponse | DetectionResponse): Response model.

    Returns:
        Dict[str, object]: JSON-ready response payload.
    """
    if isinstance(response, DetectResponse):
        return encode_detect_response(response)
    return response.to_dict()


def encode_error(message: str) -> Dict[str, str]:
    """Encode error payload.

    Args:
        message (str): Error message.

    Returns:
        Dict[str, str]: JSON-ready error payload.
    """
    return {"error": message}


def normalize_classify_type(classify_type: str) -> str:
    """Normalize class labels according to project mapping rules.

    Args:
        classify_type (str): Original class label.

    Returns:
        str: Normalized class label.
    """
    raw_value = str(classify_type).strip()
    normalized_key = raw_value.lower()
    class_mapping = {
        "s": "Smoke",
        "smoke": "Smoke",
        "f": "Fire",
        "fire": "Fire",
    }
    return class_mapping.get(normalized_key, "Smokeless")


def derive_fire_smoke_scores(
    classify_type: str,
    prob: float,
) -> Tuple[float, float, float, float]:
    """Derive F/S max/min scores based on replacement rules.

    Args:
        classify_type (str): Normalized class label.
        prob (float): Classification probability.

    Returns:
        Tuple[float, float, float, float]: (fmax, fmin, smax, smin).
    """
    if classify_type == "Fire":
        return prob, prob, 0.0, 1.0
    if classify_type == "Smoke":
        return 0.0, 1.0, prob, prob
    return 0.0, 1.0, 0.0, 1.0


def build_roi(
    thread_name: str,
    classify_type: str,
    prob: float,
    cvt_json: List[Dict[str, object]],
) -> DetectionRoi:
    """Build a single ROI object according to `/Detection` schema.

    Args:
        thread_name (str): Thread name from request payload.
        classify_type (str): Class label.
        prob (float): Class probability.
        cvt_json (List[Dict[str, object]]): Detection items payload.

    Returns:
        DetectionRoi: ROI response object.
    """
    normalized_type = normalize_classify_type(classify_type)
    fmax = 0.0
    fmin = 1.0
    smax = 0.0
    smin = 1.0

    for rect in cvt_json:
        rect_type = normalize_classify_type(
            str(rect.get("Type", rect.get("type", "")))
        )
        try:
            rect_prob = float(rect.get("Prob", rect.get("prob", 0.0)))
        except Exception:
            rect_prob = 0.0

        if rect_type == "Fire":
            fmax = max(fmax, rect_prob)
            fmin = min(fmin, rect_prob)
        elif rect_type == "Smoke":
            smax = max(smax, rect_prob)
            smin = min(smin, rect_prob)

    # Fallback to classification score when there is no per-rect score of that class.
    base_fmax, base_fmin, base_smax, base_smin = derive_fire_smoke_scores(normalized_type, prob)
    if fmin == 1.0 and fmax == 0.0:
        fmax, fmin = base_fmax, base_fmin
    if smin == 1.0 and smax == 0.0:
        smax, smin = base_smax, base_smin

    return DetectionRoi(
        thread_name=thread_name,
        classify_type=normalized_type,
        prob=prob,
        fmax=fmax,
        fmin=fmin,
        smax=smax,
        smin=smin,
        cvt_json=cvt_json,
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _to_json_item(item: Any) -> Dict[str, object]:
    if hasattr(item, "to_dict"):
        data = item.to_dict()
        if isinstance(data, dict):
            return data
    if isinstance(item, dict):
        return item
    return {"raw": str(item)}


def _to_roi_rect(item: Any) -> Dict[str, object] | None:
    data = _to_json_item(item)
    class_name = str(data.get("className", data.get("type", "")))
    normalized_type = normalize_classify_type(class_name)

    xyxy_raw = data.get("xyxy")
    x1 = y1 = x2 = y2 = 0
    if isinstance(xyxy_raw, list) and len(xyxy_raw) >= 4:
        try:
            x1 = int(xyxy_raw[0])
            y1 = int(xyxy_raw[1])
            x2 = int(xyxy_raw[2])
            y2 = int(xyxy_raw[3])
        except Exception:
            x1 = y1 = x2 = y2 = 0

    width = max(0, x2 - x1)
    height = max(0, y2 - y1)

    try:
        prob = float(data.get("conf", data.get("prob", 0.0)))
    except Exception:
        prob = 0.0

    if normalized_type not in {"Fire", "Smoke"}:
        return None

    # Align with legacy Python payload shape expected by FireSmokeUI:
    # {"Type": "...", "Rect": "x, y, width, height", "Prob": ...}
    return {
        "Type": normalized_type,
        "Rect": f"{x1}, {y1}, {width}, {height}",
        "Prob": prob,
    }
