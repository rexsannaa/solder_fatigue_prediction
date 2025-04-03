#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
solder_fatigue_prediction package
銲錫接點疲勞壽命預測模型套件

此套件實現了結合物理信息神經網絡(PINN)和長短期記憶網絡(LSTM)的混合模型，
用於準確預測電子封裝中銲錫接點的疲勞壽命。

作者: 專案開發團隊
創建日期: 2025/04/02
"""

import os
import sys

# 添加套件根目錄至路徑
package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)

# 版本信息
__version__ = '0.1.0'

# 導入主要子模組
from src.data import preprocess, dataset
from src.models import pinn, lstm, hybrid_model
from src.training import losses, trainer
from src.utils import metrics, visualization, physics

# 定義套件層級的常數
RANDOM_SEED = 42