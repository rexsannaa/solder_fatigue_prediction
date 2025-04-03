#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
utils package - 工具模組
本模組包含了銲錫接點疲勞壽命預測所需的各種工具函數，
包括評估指標計算、視覺化工具和物理約束函數等。

主要組件:
- metrics: 提供模型評估指標的計算功能
- visualization: 提供資料和結果的視覺化功能
- physics: 提供物理約束方程和物理知識轉換函數
"""

from src.utils.metrics import (
    calculate_rmse,
    calculate_r2,
    calculate_mae,
    calculate_mape,
    calculate_relative_error,
    evaluate_model
)

from src.utils.visualization import (
    plot_prediction_vs_true,
    plot_training_history,
    plot_feature_importance,
    plot_attention_weights,
    create_error_histogram
)

from src.utils.physics import (
    calculate_delta_w,
    nf_from_delta_w,
    delta_w_from_nf,
    strain_energy_equation,
    validate_physical_constraints
)

# 定義模組常數
PHYSICS_CONSTANTS = {
    'a': 55.83,  # 物理模型係數 a
    'b': -2.259  # 物理模型係數 b
}

__all__ = [
    'calculate_rmse', 'calculate_r2', 'calculate_mae', 'calculate_mape',
    'calculate_relative_error', 'evaluate_model',
    'plot_prediction_vs_true', 'plot_training_history', 'plot_feature_importance',
    'plot_attention_weights', 'create_error_histogram',
    'calculate_delta_w', 'nf_from_delta_w', 'delta_w_from_nf',
    'strain_energy_equation', 'validate_physical_constraints',
    'PHYSICS_CONSTANTS'
]