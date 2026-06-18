from __future__ import annotations


class AutoValidatorError(Exception):
    """自動驗證器基底例外。"""


class PreviewError(AutoValidatorError):
    """資料集預檢失敗。"""


class ArchiveBackendError(AutoValidatorError):
    """壓縮檔後端不可用。"""


class CacheError(AutoValidatorError):
    """快取處理失敗。"""


class DatasetPrepareError(AutoValidatorError):
    """資料集準備失敗。"""


class EngineClientError(AutoValidatorError):
    """驗證引擎失敗。"""


class OperationCancelledError(AutoValidatorError):
    """使用者取消作業。"""

