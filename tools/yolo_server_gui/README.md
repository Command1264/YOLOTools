# YOLO Server GUI

## 用途

本工具用 GUI 啟動 YOLO HTTP 推論服務，供其他程式或工具以 HTTP 方式送圖片並取得辨識結果。

## 可以做什麼

- 載入 `.pt` YOLO 模型。
- 設定 host、port、GPU worker 數量、decode worker 數量與 HTTP worker 數量。
- 啟動/停止 Flask 推論服務。
- 顯示伺服器狀態與目前推論裝置。
- 在系統托盤背景執行。
- 管理 log 儲存位置與關閉按鈕行為。
- 支援批次推論，並可將大型請求切分到多個 worker pipeline。

## 啟動方式

```powershell
.\YOLOToolsEnv\Scripts\python tools\yolo_server_gui\app.py
```

也可以使用：

```powershell
tools\yolo_server_gui\start server.bat
```

## 前置需求

- YOLO `.pt` 模型。
- 若要 GPU 推論，需確認 CUDA/PyTorch 可用。
- 若要讓其他電腦存取，需確認防火牆與 host 綁定設定。

## 操作流程

1. 選擇模型檔。
2. 設定 host 與 port。
3. 設定 GPU、Decode、HTTP worker 數量。
4. 按「啟動伺服器」。
5. 觀察狀態文字與裝置資訊。
6. 由外部程式送 HTTP request 到 server。
7. 完成後按「停止」或依托盤設定關閉。

## HTTP 使用情境

- 給 [yolo_validator_gui](../yolo_validator_gui/README.md) 以 HTTP 模式測試模型。
- 給其他桌面程式、批次程式或服務共用同一個 YOLO 推論程序。
- 避免每個工具各自載入模型造成 GPU 記憶體浪費。

## `/detect` API 範例

單張圖片 request：

```json
{
  "threadName": "demo-single",
  "image": "<base64-image>",
  "conf": 0.25,
  "iou": 0.45
}
```

批次圖片 request：

```json
{
  "threadName": "demo-batch",
  "images": ["<base64-image-1>", "<base64-image-2>"],
  "conf": 0.25,
  "iou": 0.45
}
```

Response schema：

```json
{
  "threadName": "demo-single",
  "result": {
    "classifyType": "object_a",
    "percentage": 0.93,
    "detections": [
      {
        "classId": 0,
        "className": "object_a",
        "conf": 0.93,
        "xyxy": [120, 80, 360, 260]
      }
    ]
  }
}
```

欄位說明：

- `threadName`：呼叫端自訂識別值，方便 log 追蹤。
- `image` / `images`：base64 image；兩者擇一。
- `conf` / `iou`：可選門檻值，未提供時使用 server 預設。
- `detections`：每個 bounding box 的 class、confidence 與 `xyxy` 座標。
- 批次 request 會以 list 回傳 `result`，順序與輸入圖片一致。

## 設定建議

- 單 GPU 通常先從 `GPU=1` 開始。
- Decode worker 可依 CPU 核心數調整。
- 若單次 request 包含大量圖片，server 會切分批次並保序合併結果。
- port 被占用時，請換另一個 port 或關閉占用程序。

## 注意事項

- 服務啟動後會佔用模型與 GPU 記憶體。
- 若 host 綁定 `0.0.0.0`，代表可能允許區網存取，請注意網路環境。
- log 可用於追蹤啟動失敗、推論錯誤與 worker 狀態。

## 相關工具

- 手動測試 HTTP 推論請使用 [yolo_validator_gui](../yolo_validator_gui/README.md)。
- 需要打包成 exe 時，使用 [python_packager_gui](../python_packager_gui/README.md)。
