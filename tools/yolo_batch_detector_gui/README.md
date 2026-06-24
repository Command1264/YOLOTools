# YOLO Batch Detector GUI

## 用途

本工具用於載入 YOLO 模型，批次辨識單張圖片或整個資料夾，並檢視 bbox、統計與匯出結果。

## 可以做什麼

- 載入 `.pt` YOLO 模型。
- 讀取並顯示模型 labels。
- 選擇單張圖片或資料夾，遞迴收集圖片。
- 設定 `conf`、`iou`、`device`。
- 背景執行批次推論，UI 不凍結。
- 顯示所有圖片結果、單張圖片 bbox 與預覽圖。
- 雙擊圖片開啟可縮放、可拖曳的詳細檢視視窗。
- 統計無偵測、單一 label、多 label、各 label 圖片數。
- 匯出 CSV、JSON 與標註圖片。

## 啟動方式

```powershell
.\YOLOToolsEnv\Scripts\python tools\yolo_batch_detector_gui\app.py
```

## 前置需求

- YOLO `.pt` 模型。
- 圖片或圖片資料夾。
- 支援圖片副檔名包含 `.jpg`、`.jpeg`、`.png`、`.bmp`、`.webp`。

## 操作流程

1. 在「模型」區按「瀏覽...」選擇 `.pt`。
2. 按「載入 labels」確認模型類別顯示。
3. 在「輸入」區選擇「圖片...」或「資料夾...」。
4. 選擇輸出資料夾。
5. 勾選需要的匯出格式：CSV、JSON、標註圖片。
6. 設定 `conf`、`iou`、`device`。
7. 按「開始辨識」。
8. 在結果表選取圖片，查看 bbox 明細與預覽。
9. 雙擊預覽圖可開啟詳細檢視，使用滾輪縮放、左鍵拖曳、重設縮放。
10. 按「匯出結果」輸出檔案。

## 統計規則

- 沒有 bbox：歸入「沒有」。
- 單張圖片只有一種 unique label：歸入該 label；同 label 多個 bbox 仍只算一次。
- 單張圖片有兩種以上 unique label：歸入「多個 label」。
- 各 label 圖片數量以每張圖片的 unique labels 計算。

## 匯出內容

- CSV：一列一個 bbox；無偵測圖片保留 `status=no_detection`。
- JSON：模型 labels、每張圖片 detections、summary 與推論設定。
- 標註圖片：輸出到 `annotated/`，資料夾輸入會保留相對子路徑避免同名覆蓋。

## 注意事項

- `device` 留空代表交給 Ultralytics 自動判斷。
- 批次資料夾很大時，推論時間與輸出標註圖容量都會增加。
- 錯誤列可雙擊查看可複製的錯誤訊息。

## 相關工具

- 需要手動查看影片或 HTTP 推論請使用 [yolo_validator_gui](../yolo_validator_gui/README.md)。
- 需要驗證整份 YOLO dataset 指標請使用 [yolo_auto_validator_gui](../yolo_auto_validator_gui/README.md)。
