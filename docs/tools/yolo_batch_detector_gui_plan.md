# YOLO 批次辨識檢視器計畫

## Summary

- 新增獨立 PySide6 GUI 工具：`tools/yolo_batch_detector_gui/`，名稱暫定「YOLO 批次辨識檢視器」。
- 工具支援載入 `.pt` YOLO 模型、讀取並顯示 labels、遞迴載入單張圖片或資料夾圖片、背景推論、結果檢視、bbox 表格、統計，以及 CSV / JSON / 標註圖片匯出選項。
- GUI 文字使用繁體中文，核心邏輯拆出可測模組，避免耦合既有 `yolo_server_gui` 或 `yolo_validator_gui`。

## Key Changes

- 新增 `tools/yolo_batch_detector_gui/`，包含 GUI 入口、主視窗、背景 worker、YOLO 服務、圖片來源服務、統計服務、匯出服務與資料模型。
- 顯示模型 labels、所有圖片結果表、單張圖片 bbox、單張圖片預覽、統計摘要。
- 提供 `conf`、`iou`、`device` 基本設定，預設 `conf=0.25`、`iou=0.45`、`device=""`。
- 推論在背景執行，可停止，停止後保留已完成結果。
- 匯出選項可個別勾選 CSV、JSON、標註圖片。

## Interfaces And Rules

- `DetectionBox(class_id, label, confidence, x1, y1, x2, y2)`
- `ImagePrediction(image_path, detections, labels, annotated_image_path=None, error_message=None)`
- `DetectionSummary(total_images, no_detection_count, single_label_counts, multi_label_count, per_label_image_counts)`
- `ExportOptions(output_dir, write_csv, write_json, write_annotated_images)`

統計規則：

- 沒有 bbox：歸入「沒有」。
- 單張圖片只有一種 unique label：歸入該 label；同 label 多個 bbox 仍只算該 label 類別一次。
- 單張圖片有兩種以上 unique label：歸入「多個 label」。
- 「各自 label 數量」使用每張圖片的 unique labels 統計；同圖出現 `label1 + label2` 時兩者各 `+1`。

匯出規則：

- CSV：一列一個 bbox；無偵測圖片保留一列 `status=no_detection`。
- JSON：包含模型 labels、每張圖片 detections、summary、推論設定。
- 標註圖片：輸出到使用者選擇資料夾下的 `annotated/`，遞迴輸入時保留相對子路徑避免同名覆蓋。

## Test Plan

- `image_source_service_unittest.py`：遞迴掃描、大小寫副檔名、非圖片略過、路徑不存在錯誤。
- `statistics_service_unittest.py`：無偵測、同 label 多 bbox、不同 label 多分類、per-label 圖片數。
- `export_service_unittest.py`：CSV/JSON 內容與標註圖片輸出開關。
- `yolo_model_service_unittest.py`：用 mock YOLO 測 labels dict/list 正規化與 bbox 解析，不依賴真 GPU。

驗證指令：

```powershell
.\YOLOToolsEnv\Scripts\python -m unittest tools.yolo_batch_detector_gui.image_source_service_unittest tools.yolo_batch_detector_gui.statistics_service_unittest tools.yolo_batch_detector_gui.export_service_unittest tools.yolo_batch_detector_gui.yolo_model_service_unittest
.\YOLOToolsEnv\Scripts\python -m compileall tools\yolo_batch_detector_gui
```

