# YOLO Validator GUI

## 用途

本工具是互動式 YOLO 推論檢視器，可載入模型並對圖片、資料夾或影片進行推論預覽。它適合人工快速檢查模型效果。

## 可以做什麼

- 載入本機 `.pt` YOLO 模型。
- 選擇單張圖片、圖片資料夾或影片。
- 設定 `conf`、`iou`、`device`。
- 本機推論或透過 YOLO HTTP Server 推論。
- 預覽 bbox、類別與推論結果。
- 對影片依間隔逐幀播放與推論。

## 啟動方式

```powershell
.\YOLOToolsEnv\Scripts\python tools\yolo_validator_gui\app.py
```

## 前置需求

- YOLO `.pt` 模型。
- 測試圖片、資料夾或影片。
- 若使用 HTTP 模式，需先啟動 [yolo_server_gui](../yolo_server_gui/README.md)。

## 操作流程

1. 選擇模型檔。
2. 選擇輸入檔案或資料夾。
3. 設定 `conf`、`iou`、`device`。
4. 如需 HTTP 推論，設定 HTTP URL。
5. 按「開始」。
6. 查看預覽圖與 bbox。
7. 若輸入是影片，可調整播放間隔與順序。
8. 需要中止時按「停止」。

## 使用建議

- 快速看模型是否能抓到目標：用單張圖片。
- 檢查資料夾整體效果：用圖片資料夾。
- 檢查監視器或錄影資料：用影片模式。
- 想測 server API 效能與整合：先啟動 YOLO Server，再使用 HTTP URL。

## 注意事項

- `conf` 太高會漏檢，太低會增加誤檢。
- `iou` 影響 NMS 合併行為。
- 影片推論可能較耗 GPU，建議先用較長間隔測試。
- HTTP 模式的結果取決於 server 端載入的模型與設定。

## 相關工具

- 需要批次統計與匯出 CSV/JSON 時，使用 [yolo_batch_detector_gui](../yolo_batch_detector_gui/README.md)。
- 需要大量資料集驗證與混淆矩陣時，使用 [yolo_auto_validator_gui](../yolo_auto_validator_gui/README.md)。
