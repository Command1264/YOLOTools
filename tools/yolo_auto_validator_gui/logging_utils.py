from __future__ import annotations

import logging
import os
import sys

_HANDLER_NAME = "yolo_auto_validator_terminal"
_DEFAULT_LOG_LEVEL = os.environ.get("YOLO_AUTO_VALIDATOR_LOG_LEVEL", "DEBUG").upper()
_LOGGER_NAMESPACES = ("yolo_auto_validator", "tools.yolo_auto_validator_gui")


def configure_logging(component_name: str) -> logging.Logger:
    """Configure shared terminal logging for the auto validator.

    Args:
        component_name (str): Logger name for the caller.

    Returns:
        logging.Logger: Configured logger.
    """
    log_level = _parse_log_level(_DEFAULT_LOG_LEVEL)
    _ensure_logging_configured(log_level)
    logger = logging.getLogger(_normalize_logger_name(component_name))
    logger.debug("Terminal logging configured. level=%s", logging.getLevelName(log_level))
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a logger by name.

    Args:
        name (str): Logger name.

    Returns:
        logging.Logger: Logger instance.
    """
    _ensure_logging_configured(_parse_log_level(_DEFAULT_LOG_LEVEL))
    return logging.getLogger(_normalize_logger_name(name))


def _parse_log_level(raw_level: str) -> int:
    level = getattr(logging, raw_level.upper(), None)
    if isinstance(level, int):
        return level
    return logging.DEBUG


def _normalize_logger_name(name: str) -> str:
    if name.startswith(_LOGGER_NAMESPACES):
        return name
    if name == "__main__":
        return "yolo_auto_validator.__main__"
    return f"yolo_auto_validator.{name}"


def _ensure_logging_configured(log_level: int) -> None:
    for namespace in _LOGGER_NAMESPACES:
        namespace_logger = logging.getLogger(namespace)
        handler_names = {getattr(handler, "name", "") for handler in namespace_logger.handlers}
        if _HANDLER_NAME not in handler_names:
            handler = logging.StreamHandler(sys.stdout)
            handler.name = _HANDLER_NAME
            handler.setFormatter(
                logging.Formatter(
                    fmt="%(asctime)s | %(levelname)-8s | %(processName)s/%(threadName)s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            namespace_logger.addHandler(handler)
        namespace_logger.setLevel(log_level)
        namespace_logger.propagate = False
    logging.getLogger().setLevel(logging.WARNING)
    logging.captureWarnings(True)
