# YOLO Auto Validator GUI

## 用途

本工具用於批次驗證 YOLO 模型與多個資料集來源，適合比較模型在不同資料集上的 precision、recall、confusion matrix 與背景負樣本表現。

## 可以做什麼

- 載入 YOLO `.pt` 模型並讀取 class names。
- 加入多個資料集資料夾、zip 或 rar。
- 對資料集來源進行預檢，讀取 `data.yaml`、split 與 class names。
- 校正模型 labels 與資料集 labels 的對應。
- 設定 `conf`、`iou`、`device`。
- 使用背景 engine process 執行驗證，避免 UI 卡住。
- 輸出驗證結果、混淆矩陣、圖表與結果索引。

## 啟動方式

```powershell
.\YOLOToolsEnv\Scripts\python tools\yolo_auto_validator_gui\app.py
```

## 前置需求

- YOLO `.pt` 模型。
- 一個或多個 Ultralytics YOLO 格式資料集。
- 資料集來源可為資料夾、zip 或 rar。
- rar 來源需要 `rarfile` 套件與可用的 rar 後端。

## 操作流程

1. 在「模型」區選擇 `.pt` 模型。
2. 設定 `conf`、`iou`、`device`。
3. 用「加入資料夾」或「加入 zip / rar」加入資料集來源。
4. 等待預檢完成；必要時按「重新預檢」。
5. 檢查資料集 class names 與模型 labels 的對應。
6. 設定暫存資料夾。
7. 按「開始驗證」。
8. 在結果頁查看每個資料集與整體結果。
9. 到輸出 run 目錄查看圖表與匯出檔。

## 輸出內容

- 每個資料集的驗證結果。
- aggregate 統計。
- confusion matrix 與相關圖表。
- 背景 true negative 統計。
- run index 與序列化結果檔。

## 注意事項

- 資料集的 `data.yaml` 必須能解析出 `names` 與 split。
- 大型 zip/rar 會先解壓或快取到暫存位置，請準備足夠磁碟空間。
- 類別順序不一致時，不應直接驗證，應先校正 mapping。
- GPU OOM 時工具會嘗試降低 batch 或 fallback，但仍建議先用小資料集測試。

## 相關工具

- 手動逐張檢查推論效果請使用 [yolo_validator_gui](../yolo_validator_gui/README.md)。
- 需要整理資料集 class 順序請使用 `../yolo_zip_relabel_gui_v3.py` 或 `../yolo_dataset_merge_gui.py`。
