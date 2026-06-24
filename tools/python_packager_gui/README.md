# Python Packager GUI

## 用途
`python_packager_gui` 是 Python 打包圖形介面工具，用來把專案中的 Python 入口檔打包成可執行程式。它適合用在 YOLOTools 內的 GUI 工具，或其他需要透過 PyInstaller / Nuitka 打包的 Python 程式。

## 可以做什麼
- 選擇 Python 解譯器與入口 `.py` 檔案。
- 設定輸出名稱、icon、build 目錄、dist 目錄與 spec 目錄。
- 選擇 PyInstaller 或 Nuitka 打包模式。
- 設定 GUI 程式常用的無 console 視窗模式。
- 加入額外資料檔、hidden imports 與進階打包參數。
- 儲存、載入、更新與刪除打包設定。
- 執行打包、停止打包，並在介面中查看輸出日誌。

## 啟動方式
在專案根目錄執行：

```powershell
.\YOLOToolsEnv\Scripts\python tools\python_packager_gui\app.py
```

## 使用前準備
- 確認要打包的 Python 程式可在目前環境正常啟動。
- 確認依賴套件已安裝在選用的 Python 環境中。
- 若要產生 GUI 程式，建議先手動執行入口檔確認 PySide6 視窗可正常開啟。
- 若要使用 icon，請先準備 `.ico` 檔案。

## 操作流程
1. 開啟工具後，選擇既有設定，或新增一組設定。
2. 選擇 Python 解譯器路徑。
3. 選擇要打包的入口 Python 檔案。
4. 設定輸出程式名稱。
5. 選擇打包工具：PyInstaller 或 Nuitka。
6. 依需求設定 icon、build 目錄、dist 目錄與 spec 目錄。
7. 若是 GUI 程式，勾選無 console 視窗模式。
8. 加入額外資料檔或 hidden imports。
9. 如有特殊需求，填寫進階參數。
10. 儲存設定後開始打包。
11. 從輸出日誌確認打包是否成功，並到 dist 目錄檢查產物。

## 常見搭配
- 可先使用 `tools/packaging_detect_cli.py` 掃描入口檔，取得 hidden imports 與資料檔建議。
- 打包 PySide6 GUI 工具時，通常會需要確認圖片、設定檔、模型或其他資料檔是否有一併加入。
- 若產物啟動後缺檔，回到本工具補上額外資料檔後重新打包。

## 注意事項
- PyInstaller 與 Nuitka 對資料檔參數格式不同，切換打包工具後請重新檢查資料檔設定。
- 若打包結果看起來使用到舊檔案，可清除 build / dist 後再重新打包。
- 打包 YOLO 相關工具時，模型檔通常體積較大，建議依部署情境決定是否放入執行檔旁邊，而不是直接包進單一檔案。
