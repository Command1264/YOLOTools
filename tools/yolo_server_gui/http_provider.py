from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable


HTTP_PROFILES: tuple[str, ...] = ("default", "taichung_fire")


@dataclass(frozen=True)
class HttpProvider:
    """Resolved HTTP codec/schema provider."""

    profile_name: str
    request_payload_error: type[Exception]
    parse_detect_request: Callable[[Any], Any]
    encode_detect_response: Callable[[Any], dict[str, object]]
    encode_error: Callable[[str], dict[str, str]]
    detect_response_cls: type[Any]
    detect_result_cls: type[Any]


def normalize_http_profile(profile: Any, default: str = "default") -> str:
    """Normalize configured HTTP profile value."""
    normalized = str(profile).strip().lower()
    return normalized if normalized in HTTP_PROFILES else default


def resolve_http_provider(profile: Any) -> HttpProvider:
    """Resolve HTTP codec/schema implementation for the selected profile."""
    normalized_profile = normalize_http_profile(profile)
    if normalized_profile == "taichung_fire":
        codec_module = import_module("TaichungFire.http_codec")
        schema_module = import_module("TaichungFire.http_schema")
    else:
        codec_module = import_module("http_codec")
        schema_module = import_module("http_schema")

    return HttpProvider(
        profile_name=normalized_profile,
        request_payload_error=codec_module.RequestPayloadError,
        parse_detect_request=codec_module.parse_detect_request,
        encode_detect_response=codec_module.encode_detect_response,
        encode_error=codec_module.encode_error,
        detect_response_cls=schema_module.DetectResponse,
        detect_result_cls=schema_module.DetectResult,
    )
