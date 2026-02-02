from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional


class TrayBase(ABC):
    def __init__(
        self,
        tooltip: str,
        on_exit: Optional[Callable[[], None]] = None,
        on_show: Optional[Callable[[], None]] = None,
        icon_path: Optional[str] = None,
    ) -> None:
        self.tooltip: str = tooltip
        self.on_exit: Optional[Callable[[], None]] = on_exit
        self.on_show: Optional[Callable[[], None]] = on_show
        self.icon_path: Optional[str] = icon_path

    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError
