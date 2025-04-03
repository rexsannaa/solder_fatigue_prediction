#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
models package - 模型模組
本模組包含了銲錫接點疲勞壽命預測所需的各種神經網絡模型，
包括物理資訊神經網絡(PINN)、長短期記憶網絡(LSTM)和混合模型等。

主要組件:
- pinn: 提供物理資訊神經網絡實現，專注於靜態結構參數處理與物理約束
- lstm: 提供長短期記憶網絡實現，專注於時間序列資料處理
- hybrid_model: 提供混合PINN-LSTM模型實現，整合靜態特徵和時間序列特徵
"""

from src.models.pinn import PINNModel
from src.models.lstm import LSTMModel
from src.models.hybrid_model import HybridPINNLSTMModel

# 定義模組常數
# 模型架構相關參數
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_LSTM_LAYERS = 2
DEFAULT_DROPOUT = 0.2
DEFAULT_BIDIRECTIONAL = True

__all__ = [
    'PINNModel', 
    'LSTMModel', 
    'HybridPINNLSTMModel',
    'DEFAULT_HIDDEN_SIZE',
    'DEFAULT_LSTM_LAYERS',
    'DEFAULT_DROPOUT',
    'DEFAULT_BIDIRECTIONAL'
]