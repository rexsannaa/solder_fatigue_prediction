#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
training package - 訓練模組
本模組包含了銲錫接點疲勞壽命預測模型訓練所需的功能，
包括損失函數定義、訓練器實現和回調函數等。

主要組件:
- losses: 提供各種損失函數，包括物理約束損失
- trainer: 提供模型訓練和評估的主要功能
- callbacks: 提供訓練過程中的回調函數，如早停和學習率調度
"""

from src.training.losses import (
    MSELoss,
    PhysicsConstraintLoss,
    ConsistencyLoss,
    HybridLoss
)

from src.training.trainer import (
    Trainer,
    EarlyStopping,
    LearningRateScheduler
)

from src.training.callbacks import (
    ModelCheckpoint,
    TensorBoardLogger,
    ProgressBar
)

# 定義模組常數
DEFAULT_PATIENCE = 20
DEFAULT_LAMBDA_PHYSICS = 0.1
DEFAULT_LAMBDA_CONSISTENCY = 0.1

__all__ = [
    'MSELoss', 'PhysicsConstraintLoss', 'ConsistencyLoss', 'HybridLoss',
    'Trainer', 'EarlyStopping', 'LearningRateScheduler',
    'ModelCheckpoint', 'TensorBoardLogger', 'ProgressBar',
    'DEFAULT_PATIENCE', 'DEFAULT_LAMBDA_PHYSICS', 'DEFAULT_LAMBDA_CONSISTENCY'
]