#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
data package - 資料處理模組
本模組包含了銲錫接點疲勞壽命預測所需的資料處理功能，
包括資料預處理、資料集封裝等功能。

主要組件:
- preprocess: 提供資料清洗、特徵工程和資料標準化等功能
- dataset: 提供資料集的封裝，方便模型訓練和評估
"""

from src.data.preprocess import (
    load_data,
    standardize_features,
    prepare_time_series,
    train_val_test_split,
    process_pipeline
)

from src.data.dataset import (
    SolderFatigueDataset,
    TimeSeriesDataset,
    create_dataloaders
)

# 定義模組常數
FEATURE_COLUMNS = ['Die', 'stud', 'mold', 'PCB', 'Unit_warpage']
TIMESERIES_COLUMNS = [
    'NLPLWK_up_3600', 'NLPLWK_up_7200', 'NLPLWK_up_10800', 'NLPLWK_up_14400',
    'NLPLWK_down_3600', 'NLPLWK_down_7200', 'NLPLWK_down_10800', 'NLPLWK_down_14400'
]
TARGET_COLUMN = 'Nf_pred (cycles)'

__all__ = [
    'load_data', 'standardize_features', 'prepare_time_series', 
    'train_val_test_split', 'process_pipeline', 'SolderFatigueDataset',
    'TimeSeriesDataset', 'create_dataloaders', 'FEATURE_COLUMNS',
    'TIMESERIES_COLUMNS', 'TARGET_COLUMN'
]