# 銲錫接點疲勞壽命預測 - 混合PINN-LSTM模型

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

本項目實現了結合物理信息神經網絡(PINN)和長短期記憶網絡(LSTM)的混合模型，用於預測電子封裝中銲錫接點的疲勞壽命。該混合架構充分利用靜態結構參數與時間序列資料，同時整合物理約束以提供更準確、可靠的預測結果。

## 專案特色

- **雙分支架構設計**：PINN處理靜態結構參數，LSTM處理時間序列資料
- **物理知識嵌入**：將疲勞壽命物理模型 (Nf=55.83⋅(ΔW)^(-2.259)) 作為約束條件
- **時序特徵提取**：使用雙向LSTM捕捉非線性塑性應變功的時間變化特徵
- **注意力機制**：結合物理知識與時序特徵的融合注意力機制
- **小樣本資料集優化**：針對僅81筆的CAE資料集特別優化模型結構與訓練策略

## 專案架構

```
solder_fatigue_prediction/
│
├── data/
│   ├── raw/                # 原始資料
│   └── processed/          # 預處理後的資料
│
├── src/
│   ├── data/               # 資料處理模組
│   ├── models/             # 模型實現
│   ├── training/           # 訓練相關功能
│   └── utils/              # 工具函數
│
├── notebooks/              # Jupyter筆記本
│
├── configs/                # 配置文件
│
├── scripts/                # 執行腳本
│
└── outputs/                # 輸出結果
```

## 安裝說明

### 前置要求

- Python 3.7 或更高版本
- PyTorch 1.10 或更高版本
- CUDA（可選，用於GPU加速）

### 安裝步驟

1. 複製專案程式碼
```bash
git clone https://github.com/yourusername/solder_fatigue_prediction.git
cd solder_fatigue_prediction
```

2. 創建並啟用虛擬環境（可選）
```bash
python -m venv venv
source venv/bin/activate  # 在Linux/macOS上
# 或
venv\Scripts\activate     # 在Windows上
```

3. 安裝依賴項
```bash
pip install -r requirements.txt
```

4. 安裝專案（開發模式）
```bash
pip install -e .
```

## 使用方法

### 訓練模型

```bash
# 使用默認配置訓練模型
python scripts/train.py

# 自定義配置訓練模型
python scripts/train.py --config configs/model_config.yaml --train-config configs/train_config.yaml --data path/to/data.csv --output-dir outputs/my_model
```

### 預測

```bash
# 使用已訓練模型進行預測
python scripts/predict.py --model-path outputs/models/best_model.pt --input path/to/test_data.csv
```

### 評估

```bash
# 評估模型性能
python scripts/evaluate.py --model-path outputs/models/best_model.pt --data path/to/test_data.csv
```

## 資料格式

模型接受包含以下資料的CSV文件：

1. **結構參數**
   - `Die`: 晶片高度 (μm)
   - `stud`: 銅高度 (μm)
   - `mold`: 環氧樹脂厚度 (μm)
   - `PCB`: 基板厚度 (mm)
   - `Unit_warpage`: 翹曲變形量 (μm)

2. **時間序列資料**
   - `NLPLWK_up_3600`-`NLPLWK_up_14400`: 上界面非線性塑性應變功時間序列
   - `NLPLWK_down_3600`-`NLPLWK_down_14400`: 下界面非線性塑性應變功時間序列

3. **預測目標**
   - `Nf_pred (cycles)`: 疲勞壽命循環數

## 模型結構

### PINN分支

處理靜態結構參數，計算非線性塑性應變能密度變化量(ΔW)，並應用物理約束公式：
```
Nf = 55.83 * (ΔW)^(-2.259)
```

### LSTM分支

處理時間序列特徵，使用雙向LSTM和注意力機制捕捉銲錫接點在熱循環中的動態行為。

### 特徵融合層

使用注意力門控機制融合PINN和LSTM分支的特徵，並平衡物理約束與數據驅動的貢獻。

## 參考文獻

1. SACQ Solder Board Level Reliability Evaluation and Life Prediction Model for Wafer Level Packages
2. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations
3. Long Short-Term Memory networks for time series forecasting

## 授權協議

MIT License

## 聯絡資訊

如有任何問題，請聯絡：your.email@example.com
#   s o l d e r _ f a t i g u e _ p r e d i c t i o n  
 