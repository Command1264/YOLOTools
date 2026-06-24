# YOLO Trainer GUI

## 用途

本工具是本機 Ultralytics YOLO 訓練 GUI，用於從 zip 資料集啟動訓練、追蹤進度、保存訓練歷史與查看常見訓練輸出圖。

## 可以做什麼

- 選擇官方 YOLO 模型或自訂 `.pt` 權重。
- 選擇資料集 zip，並指定解壓/訓練工作資料夾。
- 設定 `epochs`、`imgsz`、`batch`、`device` 等訓練參數。
- 在 GUI 內查看 epoch/batch 進度、裝置資訊與日誌。
- 保存訓練歷史，並在「歷史」與「硬核視覺化」頁面查看 run 結果。
- 將訓練結果打包成 zip 輸出。

## 啟動方式

在專案根目錄執行：

```powershell
.\YOLOToolsEnv\Scripts\python tools\yolo_trainer_gui\app.py
```

## 前置需求

- 已安裝 `requirements.txt` 內的套件。
- 資料集建議為 zip，內含 Ultralytics YOLO 格式的 `data.yaml`、images、labels。
- 若使用自訂權重，請準備 `.pt` 檔。
- GPU 訓練請確認 CUDA、PyTorch 與顯示卡驅動可用。

## 操作流程

1. 在「模型」區選擇任務、版本、大小，或用「自訂權重」選擇 `.pt`。
2. 選擇「資料集（zip）」。
3. 選擇「訓練工作資料夾」作為解壓與訓練中間輸出位置。
4. 選擇「輸出 zip 存放資料夾」。
5. 選擇「模型存放資料夾」，讓工具快取下載或選取模型。
6. 設定 `epochs`、`imgsz`、`batch`、`device`。
7. 按「開始訓練」。
8. 觀察進度與日誌；需要中止時按「停止」。
9. 訓練完成後，到輸出資料夾取得結果 zip，或在「歷史」頁查看過去 run。

## 輸出內容

- Ultralytics run directory。
- 訓練結果 zip。
- 訓練歷史紀錄。
- 常見圖表，例如 results、PR curve、confusion matrix。

## 注意事項

- 工作資料夾與輸出資料夾建議不要放在同步雲端資料夾，避免大量小檔造成 I/O 變慢。
- `batch` 過大可能造成 CUDA OOM，可先用較小 batch 測試。
- `device` 可填空白讓 Ultralytics 自動判斷，也可填 `cpu`、`0`、`0,1`。
- 訓練資料 class 順序必須和 `data.yaml` 的 `names` 一致。

## 相關工具

- 需要在 Colab 訓練時，使用 [yolo_trainer_colab](../yolo_trainer_colab/README.md)。
- 需要驗證訓練結果時，使用 [yolo_validator_gui](../yolo_validator_gui/README.md) 或 [yolo_auto_validator_gui](../yolo_auto_validator_gui/README.md)。
