from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Union


@dataclass(frozen=True)
class DetectionItem:
    """Serialized detection item."""

    class_id: int
    class_name: str
    conf: float
    xyxy: List[int]

    def to_dict(self) -> Dict[str, object]:
        """Serialize detection item as JSON-ready dict.

        Returns:
            Dict[str, object]: JSON-ready payload.
        """
        return {
            "classId": self.class_id,
            "className": self.class_name,
            "conf": self.conf,
            "xyxy": self.xyxy,
        }


@dataclass(frozen=True)
class DetectResult:
    """Serialized detect result."""

    classify_type: str
    percentage: float
    detections: List[DetectionItem]

    def to_dict(self) -> Dict[str, object]:
        """Serialize detect result as JSON-ready dict.

        Returns:
            Dict[str, object]: JSON-ready payload.
        """
        return {
            "classifyType": self.classify_type,
            "percentage": self.percentage,
            "detections": [item.to_dict() for item in self.detections],
        }


@dataclass(frozen=True)
class DetectRequest:
    """Decoded detect request payload."""

    thread_name: str
    images: List[str]
    is_batch: bool
    conf: float | None
    iou: float | None


@dataclass(frozen=True)
class DetectResponse:
    """Encoded detect response payload."""

    thread_name: str
    result: Union[DetectResult, List[DetectResult]]

    def to_dict(self) -> Dict[str, object]:
        """Serialize response as JSON-ready dict.

        Returns:
            Dict[str, object]: JSON-ready payload.
        """
        if isinstance(self.result, list):
            result_payload: object = [item.to_dict() for item in self.result]
        else:
            result_payload = self.result.to_dict()
        return {
            "threadName": self.thread_name,
            "result": result_payload,
        }


@dataclass(frozen=True)
class DetectionRoi:
    """Per-image ROI payload for /Detection response."""

    thread_name: str
    classify_type: str
    prob: float
    fmax: float
    fmin: float
    smax: float
    smin: float
    cvt_json: List[Dict[str, object]]

    def to_dict(self) -> Dict[str, object]:
        """Serialize ROI payload as JSON-ready dict.

        Returns:
            Dict[str, object]: JSON-ready payload.
        """
        return {
            "threadName": self.thread_name,
            "Classifytype": self.classify_type,
            "Prob": self.prob,
            "Fmax": self.fmax,
            "Fmin": self.fmin,
            "Smax": self.smax,
            "Smin": self.smin,
            "cvtJson": self.cvt_json,
        }


@dataclass(frozen=True)
class DetectionResponse:
    """Encoded /Detection response payload."""

    thread_name: str
    cvt_jsons: List[DetectionRoi]

    def to_dict(self) -> Dict[str, object]:
        """Serialize response payload as JSON-ready dict.

        Returns:
            Dict[str, object]: JSON-ready payload.
        """
        return {
            "threadName": self.thread_name,
            "cvtJsons": [item.to_dict() for item in self.cvt_jsons],
        }


# Compatibility aliases for /Detection naming.
DetectionRequest = DetectRequest
ROIJson = DetectionRoi
ROIsJson = DetectionResponse
