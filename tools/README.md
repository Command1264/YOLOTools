# YOLOTools 工具總覽

本資料夾收錄資料集處理、模型訓練、模型驗證、推論服務、批次辨識與打包輔助工具。多數 GUI 工具以 PySide6 建構，建議在專案根目錄使用既有虛擬環境啟動：

```powershell
.\YOLOToolsEnv\Scripts\python <工具路徑>
```

## 快速選擇

| 需求 | 建議工具 | 說明 |
| --- | --- | --- |
| 調整圖片尺寸、補邊、轉格式 | `change_image_resolution.py` | 批次圖片解析度處理，適合訓練前整理影像。 |
| 從影片抽圖 | `video_frame_extractor_gui.py` | 依秒數或間隔擷取 frame，建立標註素材。 |
| 合併多個 YOLO 資料集 | `yolo_dataset_merge_gui.py` | 讀取資料夾或 zip 的 `data.yaml`，合併 train/val/test。 |
| 將 YOLOv4/Darknet 資料集轉為 Ultralytics YAML 格式 | `yolo_v4_to_v12_gui.py` | 轉換圖片/labels、重建 split 與 `data.yaml`。 |
| 修改 zip 內 YOLO 類別順序或名稱 | `yolo_zip_relabel_gui_v3.py` | 對 zip dataset 重新排序 class id、改 label name 並輸出新 zip。 |
| 本機訓練 YOLO | [yolo_trainer_gui](yolo_trainer_gui/README.md) | GUI 設定模型、資料集、epochs、device 並啟動訓練。 |
| Google Colab 訓練 YOLO | [yolo_trainer_colab](yolo_trainer_colab/README.md) | Colab notebook，適合使用雲端 GPU。 |
| 手動驗證模型推論效果 | [yolo_validator_gui](yolo_validator_gui/README.md) | 對圖片、資料夾或影片逐張/逐幀推論與預覽。 |
| 自動驗證多個資料集 | [yolo_auto_validator_gui](yolo_auto_validator_gui/README.md) | 批次驗證資料集 zip/rar/folder，輸出指標與混淆矩陣。 |
| 開 HTTP 推論服務 | [yolo_server_gui](yolo_server_gui/README.md) | 啟動 Flask YOLO server，供其他程式呼叫。 |
| 批次辨識圖片並匯出結果 | [yolo_batch_detector_gui](yolo_batch_detector_gui/README.md) | 載入模型與圖片/資料夾，輸出 CSV/JSON/標註圖。 |
| 打包 Python 工具 | [python_packager_gui](python_packager_gui/README.md) | PyInstaller/Nuitka GUI 打包工具。 |
| 偵測打包提示 | `packaging_detect_cli.py` | 掃描 Python 入口，產生打包所需 hidden imports / data hints。 |
| 下載 YOLO26 官方模型 | `download_yolo_26_models.py` | 透過 Ultralytics 下載 `yolo26n/s/m/l/x` 權重。 |

## 常見資料流

1. 從影片建立素材：`video_frame_extractor_gui.py`
2. 整理圖片尺寸：`change_image_resolution.py`
3. 轉換或合併資料集：`yolo_v4_to_v12_gui.py` / `yolo_dataset_merge_gui.py`
4. 修正 labels：`yolo_zip_relabel_gui_v3.py`
5. 訓練模型：[yolo_trainer_gui](yolo_trainer_gui/README.md) 或 [yolo_trainer_colab](yolo_trainer_colab/README.md)
6. 驗證模型：[yolo_validator_gui](yolo_validator_gui/README.md) 或 [yolo_auto_validator_gui](yolo_auto_validator_gui/README.md)
7. 部署推論：[yolo_server_gui](yolo_server_gui/README.md)
8. 批次檢查圖片：[yolo_batch_detector_gui](yolo_batch_detector_gui/README.md)

## 單檔 GUI 工具

### `change_image_resolution.py`

用途：批次調整圖片解析度，支援等比例縮放、letterbox 補邊、背景色設定與 Unicode 路徑讀寫。

啟動：

```powershell
.\YOLOToolsEnv\Scripts\python tools\change_image_resolution.py
```

操作流程：

1. 選擇輸入圖片或資料夾。
2. 選擇輸出資料夾。
3. 設定目標寬高、補邊顏色與輸出格式。
4. 開始處理，等待進度完成。
5. 到輸出資料夾檢查結果。

適用情境：訓練前將資料集圖片統一成相同尺寸，或保留原比例並補邊避免物件變形。

### `video_frame_extractor_gui.py`

用途：從影片擷取圖片 frame，可用於建立訓練資料、抽樣檢查影片內容。

啟動：

```powershell
.\YOLOToolsEnv\Scripts\python tools\video_frame_extractor_gui.py
```

操作流程：

1. 選擇影片檔。
2. 選擇輸出資料夾。
3. 設定擷取間隔、命名方式與輸出格式。
4. 開始擷取。
5. 將輸出的圖片拿去標註或後續資料集整理。

注意：輸出張數會隨影片長度與擷取間隔快速增加，建議先用較大間隔測試。

### `yolo_dataset_merge_gui.py`

用途：合併多個 YOLO dataset，支援從資料夾或 zip 讀取 `data.yaml`，協助統一 class names 與 split。

啟動：

```powershell
.\YOLOToolsEnv\Scripts\python tools\yolo_dataset_merge_gui.py
```

操作流程：

1. 加入多個資料集來源。
2. 檢查每個來源的 class names 與 split。
3. 選擇輸出資料夾。
4. 執行合併。
5. 檢查輸出的 `data.yaml`、images、labels。

注意：class 順序必須穩定。若來源資料集 class names 不一致，應先確認對應關係再合併。

### `yolo_v4_to_v12_gui.py`

用途：將 YOLOv4/Darknet 常見資料格式轉成 Ultralytics 可用的 YOLO YAML 資料集格式。

啟動：

```powershell
.\YOLOToolsEnv\Scripts\python tools\yolo_v4_to_v12_gui.py
```

操作流程：

1. 選擇 YOLOv4/Darknet 資料集來源。
2. 選擇輸出資料夾。
3. 設定 train/val/test split 比例與 seed。
4. 執行轉換。
5. 用輸出的 `data.yaml` 進行 Ultralytics 訓練。

注意：轉換後請抽查 labels 的 class id 是否與新 `names` 順序一致。

### `yolo_zip_relabel_gui_v3.py`

用途：直接處理 zip 裡的 YOLO dataset，重新命名 label、刪除不需要的 class、調整 class id 順序並輸出新 zip。

啟動：

```powershell
.\YOLOToolsEnv\Scripts\python tools\yolo_zip_relabel_gui_v3.py
```

操作流程：

1. 選擇來源 dataset zip。
2. 選擇輸出資料夾。
3. 檢查工具讀出的 label 清單。
4. 重新排序、刪除或改名 labels。
5. 按「一鍵處理並輸出 Zip」。
6. 用新 zip 進行訓練或驗證。

注意：刪除 class 會改變後續 class id，提交訓練前應檢查 `data.yaml`。

## CLI 與輔助工具

### `packaging_detect_cli.py`

用途：分析 Python 入口檔的 import 與 runtime 行為，輸出 PyInstaller/Nuitka 打包提示。

啟動範例：

```powershell
.\YOLOToolsEnv\Scripts\python tools\packaging_detect_cli.py path\to\entry.py
```

適用情境：打包後缺少 hidden import、資料檔或 Qt plugin 時，先用此工具產生提示，再放入 [python_packager_gui](python_packager_gui/README.md) 的進階參數。

### `download_yolo_26_models.py`

用途：透過 Ultralytics 觸發下載 YOLO26 官方模型權重。

啟動：

```powershell
.\YOLOToolsEnv\Scripts\python tools\download_yolo_26_models.py
```

注意：需要網路連線，且會依 Ultralytics 行為下載或快取模型。
