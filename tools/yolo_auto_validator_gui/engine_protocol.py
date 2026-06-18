from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EngineRequest:
    """送往背景引擎的請求。"""

    request_id: str
    action: str
    payload: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EngineEvent:
    """背景引擎事件。"""

    request_id: str
    kind: str
    payload: dict[str, object] = field(default_factory=dict)
