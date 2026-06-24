# YOLO Trainer Colab

## 用途

本工具提供 Google Colab 訓練流程，適合本機 GPU 不足或希望使用雲端 GPU 訓練 YOLO 模型時使用。

## 可以做什麼

- 在 Colab 掛載 Google Drive。
- 安裝 Colab 需要的 YOLO 訓練套件。
- 從 Google Drive 讀取資料集 zip 與模型權重。
- 將資料集解壓到 Colab 本機或 Drive 工作目錄。
- 修正 `data.yaml` 路徑。
- 執行 Ultralytics YOLO 訓練。
- 將訓練結果打包回 Google Drive。

## 主要檔案

- `train_colab.ipynb`：建議直接上傳或從 Colab 開啟的 notebook。
- `train_colab.py`：notebook 的 Python 來源版本，便於維護與比較。

## 啟動方式

此工具主要在 Google Colab 使用，請開啟 `tools/yolo_trainer_colab/train_colab.ipynb`。`train_colab.py` 主要作為 notebook 的維護來源，不是一般本機訓練入口。

## 操作流程

1. 將 `tools/yolo_trainer_colab/train_colab.ipynb` 上傳到 Google Colab。
2. 將資料集 zip 放到 Google Drive，例如：

   ```text
   /content/drive/MyDrive/YOLOTools/trainingData/dataset.zip
   ```

3. 將模型權重放到 Google Drive，或使用 Ultralytics 支援的模型名稱。
4. 依 notebook 內「設定訓練參數」區塊修改：
   - `dataset_zip`
   - `model`
   - `epochs`
   - `imgsz`
   - `batch`
   - `device`
   - 輸出 zip 位置
5. 依序執行 notebook cells。
6. 訓練完成後，到 Google Drive 的 `output_zips` 取得訓練結果。

## storage mode

- `fast`：資料從 Drive 讀取，但解壓與訓練放在 Colab 本機 `/content`。速度較快，建議預設使用。
- `persistent`：工作資料放在 Google Drive，可跨 runtime 保留，但大量小檔 I/O 較慢。

## 注意事項

- Colab runtime 重啟後，本機 `/content` 內容會消失，重要結果要寫回 Drive。
- Colab 通常已內建 PyTorch 與 CUDA，不建議在 notebook 中任意重裝 torch。
- 資料集 zip 內仍需符合 Ultralytics YOLO YAML 格式。
- 如果使用 Drive 內 `.pt` 權重，請確認路徑正確且權限可讀。

## 相關工具

- 本機訓練請使用 [yolo_trainer_gui](../yolo_trainer_gui/README.md)。
- 訓練後驗證請使用 [yolo_auto_validator_gui](../yolo_auto_validator_gui/README.md)。
