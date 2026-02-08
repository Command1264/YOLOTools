Behavior rules: see `workflow/SYSTEM.md`.
行為規則請見 `workflow/SYSTEM.md`。

# Project Overview (English)
This project trains and validates YOLO models to detect fire and smoke.

## Tech Stack
- Python 3.12.10
- Python environment path: `./YOLOToolsEnv/Scripts/python`
- ultralytics 8.4.7
- torch 2.9.1+cu126
- See `requirements.txt` for the rest

## Disallowed Packages
- None

## Directory Layout
- `/tools/` - utilities for dataset validation and training support
- `/models/` - trained weights and model artifacts
- `/trainingData/` - datasets (keep naming consistent; if renamed, update configs)
- `/dist/` - build outputs

## Dataset Notes
- Use Ultralytics YAML format.
- Keep class ordering stable across train/val/test.
- Store dataset configs and split lists in a reproducible way.

## Outputs and Tracking
- Save metrics, logs, and plots under a clear experiment folder.
- Use consistent naming for weights (best/last) and export formats.

# Coding Standards

## Naming Conventions
- Files: snake_case (e.g., yolo_server.py)
- Classes: PascalCase (e.g., UserController)
- Functions: snake_case (e.g., get_user_by_id)
- Variables: snake_case (e.g., user_count)
- Constants: UPPER_SNAKE_CASE (e.g., MAX_RETRY_COUNT)
- Private members: prefix "_" (e.g., _validate_input)

## Typing & Static Analysis
- Public APIs must be type-annotated.
- Use `typing` standard types first.
- Prefer `@dataclass` for data structures.
- Avoid `Any`; if required, document the reason.

## Docstrings
- All public functions/classes must have Google-style docstrings.
- Include purpose, args, returns, and raised exceptions.

Example:
```
def get_user_by_id(user_id: int) -> User:
    """
    Retrieve user information by user ID.

    Args:
        user_id (int): Unique identifier of the user.

    Returns:
        User: User data object.

    Raises:
        UserNotFoundError: If the user does not exist.
    """
```

## Error Handling
- Use custom exceptions for business logic errors.
- Inherit from `Exception` or a project-specific base error.

Example:
```
class ApplicationError(Exception):
    """Base class for application-level errors."""
class UserNotFoundError(ApplicationError):
    def __init__(self, user_id: int):
        super().__init__(f"User not found. user_id={user_id}")
        self.user_id = user_id
```

## try / except Rules
- All async functions must use try/except.
- Error messages must include context.
- Do not silently swallow exceptions.

Example:
```
async def get_user_profile(user_id: int) -> User:
    try:
        return await fetch_user(user_id)
    except UserNotFoundError as exc:
        raise
    except Exception as exc:
        raise ApplicationError(
            f"Failed to get user profile. user_id={user_id}"
        ) from exc
```


---

# 專案概述（繁體中文）
這是一個 YOLO 訓練與驗證專案，用於偵測火焰與煙霧。

## 技術棧
- Python 3.12.10
- Python 環境路徑：`./YOLOToolsEnv/Scripts/python`
- ultralytics 8.4.7
- torch 2.9.1+cu126
- 其餘請見 `requirements.txt`

## 禁止使用的套件
- 無

## 目錄結構
- `/tools/` - 資料集驗證與訓練輔助工具
- `/models/` - 訓練權重與模型產物
- `/trainingData/` - 資料集（若更名需同步更新設定檔）
- `/dist/` - 打包輸出

## 資料集規範
- 使用 Ultralytics YAML 格式。
- class 順序必須在 train/val/test 一致。
- 建議保留可重現的 split 與設定檔。

## 輸出與紀錄
- 以清楚的實驗資料夾存放 metrics、logs、plots。
- 權重命名規則一致（best/last）並標註匯出格式。

# 程式碼規範

## 命名慣例
- 檔案名稱：snake_case（例：yolo_server.py）
- 類別名稱：PascalCase（例：UserController）
- 函數名稱：snake_case（例：get_user_by_id）
- 變數名稱：snake_case（例：user_count）
- 常數：UPPER_SNAKE_CASE（例：MAX_RETRY_COUNT）
- 私有成員：前綴底線 _（例：_validate_input）

## 型別與靜態檢查（Typing & Static Analysis）
- 禁止使用未標註型別的公開 API。
- 所有公開函式與方法必須標註型別。
- 優先使用 `typing` 標準型別。
- 資料結構優先使用 `@dataclass`。
- 避免使用 `Any`，若必要需明確註解原因。

## 文件註解（Docstring）
- 所有公開函式 / 類別都必須有 Docstring。
- 採用 Google Style Docstring。
- Docstring 需說明功能、參數、回傳值、可能拋出的例外。

範例：
```
def get_user_by_id(user_id: int) -> User:
    """
    Retrieve user information by user ID.

    Args:
        user_id (int): Unique identifier of the user.

    Returns:
        User: User data object.

    Raises:
        UserNotFoundError: If the user does not exist.
    """
```

## 錯誤處理（Error Handling）
- 所有業務錯誤必須使用 `自訂例外`。
- 統一繼承 `Exception` 或專用基底錯誤類。

範例：
```
class ApplicationError(Exception):
    """Base class for application-level errors."""
class UserNotFoundError(ApplicationError):
    def __init__(self, user_id: int):
        super().__init__(f"User not found. user_id={user_id}")
        self.user_id = user_id
```

## try / except 規範
- 所有 async 函式必須使用 try / except。
- 錯誤訊息必須包含上下文資訊。
- 不允許裸捕捉 `Exception` 後什麼都不做。

範例：
```
async def get_user_profile(user_id: int) -> User:
    try:
        return await fetch_user(user_id)
    except UserNotFoundError as exc:
        raise
    except Exception as exc:
        raise ApplicationError(
            f"Failed to get user profile. user_id={user_id}"
        ) from exc
```
