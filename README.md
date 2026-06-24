# YOLOTools

## 專案簡介 / Overview
YOLOTools 是一套用於訓練與驗證 YOLO 模型的工具集合，專注於火焰與煙霧偵測。  
本專案提供資料集整理與驗證、訓練與推論輔助、以及多個 GUI 小工具，方便快速建立與迭代模型。

## 功能概覽 / Features
- 資料集驗證與處理（解析、重標註、影像轉換）
- YOLO 訓練與驗證輔助
- GUI 工具（訓練/驗證/推論/轉換等）
- 模型下載與版本轉換
- PyInstaller 打包設定（spec）

## 技術棧 / Tech Stack
- Python 3.12.10
- ultralytics 8.4.7
- torch 2.9.1+cu126
- 其他依賴請見 `requirements.txt`

## 目錄結構 / Directory Layout
- `tools/`：資料集驗證與訓練輔助工具
- `models/`：訓練權重與模型產物
- `trainingData/`：資料集（若更名需同步更新設定檔）
- `dist/`：打包輸出
- `workflow/`：行為規範與工作流程
- `*.spec`：PyInstaller 打包設定

## 資料集規範 / Dataset Notes
- 使用 Ultralytics YAML 格式
- class 順序必須在 train/val/test 一致
- 建議保留可重現的 split 與設定檔

## 快速開始 / Quick Start
以下為一般使用流程（Windows PowerShell）：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 工具清單 / Tools
完整工具說明、啟動指令與操作流程請見 [`tools/README.md`](tools/README.md)。

常用分類包含：
- 資料集整理：影像解析度調整、影片抽幀、YOLO zip 重標註、資料集合併、YOLO 版本轉換。
- 模型訓練：本機訓練 GUI、Google Colab 訓練流程。
- 模型驗證與推論：單次驗證 GUI、自動批次驗證 GUI、批次圖片辨識檢視器、YOLO HTTP server GUI。
- 打包與部署：Python 打包 GUI、打包需求偵測 CLI。

## 打包 / Packaging
根目錄下的 `*.spec` 為 PyInstaller 設定檔，打包輸出會放在 `dist/`。

## 輸出與紀錄 / Outputs & Tracking
- metrics / logs / plots 請放在清楚的實驗資料夾
- 權重命名維持一致（best/last），並標註匯出格式

## 注意事項 / Notes
- 檔名與類別命名請遵循專案規範（見 `AGENTS.md`）
- 公開函式需具備型別註記與 Google-style docstring
