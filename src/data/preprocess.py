#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
preprocess.py - 資料預處理模組
本模組提供銲錫接點疲勞壽命預測所需的資料預處理功能，
包括資料加載、清洗、特徵工程、標準化和資料分割等。

主要功能:
1. 資料加載與清洗
2. 特徵工程與轉換
3. 資料標準化
4. 資料分割
5. 時間序列處理
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

# Constants
FEATURE_COLUMNS = ['Die', 'stud', 'mold', 'PCB', 'Unit_warpage']
TIMESERIES_COLUMNS = [
    'NLPLWK_up_3600', 'NLPLWK_up_7200', 'NLPLWK_up_10800', 'NLPLWK_up_14400',
    'NLPLWK_down_3600', 'NLPLWK_down_7200', 'NLPLWK_down_10800', 'NLPLWK_down_14400'
]
TARGET_COLUMN = 'Nf_pred (cycles)'


def load_data(filepath, encoding='utf-8'):
    """
    加載原始資料
    
    參數:
        filepath (str): 資料檔案路徑
        encoding (str): 檔案編碼
        
    返回:
        pandas.DataFrame: 原始資料
    """
    try:
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath, encoding=encoding)
        logger.info(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")
        
        # 檢查必要欄位是否存在
        required_cols = set(FEATURE_COLUMNS + [TARGET_COLUMN])
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
        
        # 檢查時間序列欄位是否存在
        timeseries_count = sum(col in df.columns for col in TIMESERIES_COLUMNS)
        logger.info(f"Found {timeseries_count} time series columns")
        
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def standardize_features(df, feature_cols=None, target_col=TARGET_COLUMN, scaler_type='robust'):
    """
    標準化特徵
    
    參數:
        df (pandas.DataFrame): 資料框
        feature_cols (list): 要標準化的特徵欄位
        target_col (str): 目標欄位
        scaler_type (str): 標準化方法，'standard' 或 'robust'
        
    返回:
        tuple: (標準化後的特徵, 標準化器, 目標值, 特徵欄位列表)
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS
    
    # 確保所有特徵欄位存在
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing feature columns: {missing_cols}")
        feature_cols = [col for col in feature_cols if col in df.columns]
    
    # 選擇標準化器
    if scaler_type.lower() == 'standard':
        scaler = StandardScaler()
    elif scaler_type.lower() == 'robust':
        scaler = RobustScaler()
    else:
        logger.warning(f"Unsupported scaler type: {scaler_type}, using RobustScaler")
        scaler = RobustScaler()
    
    # 標準化特徵
    X = df[feature_cols].values
    X_scaled = scaler.fit_transform(X)
    
    # 提取目標值
    if target_col in df.columns:
        y = df[target_col].values
    else:
        logger.warning(f"Target column {target_col} not found in dataframe")
        y = None
    
    logger.info(f"Standardized {len(feature_cols)} features")
    
    return X_scaled, scaler, y, feature_cols


def prepare_time_series(df, normalize=True):
    """
    準備時間序列資料
    
    參數:
        df (pandas.DataFrame): 資料框
        normalize (bool): 是否標準化時間序列
        
    返回:
        numpy.ndarray: 時間序列資料，形狀為 (樣本數, 時間步數, 特徵數)
    """
    # 找出所有時間序列欄位
    up_cols = [col for col in df.columns if col.startswith('NLPLWK_up_')]
    down_cols = [col for col in df.columns if col.startswith('NLPLWK_down_')]
    
    # 排序時間序列欄位
    up_cols.sort()
    down_cols.sort()
    
    # 檢查時間步數是否匹配
    if len(up_cols) != len(down_cols):
        logger.warning(f"Number of time steps mismatch: up={len(up_cols)}, down={len(down_cols)}")
        min_steps = min(len(up_cols), len(down_cols))
        up_cols = up_cols[:min_steps]
        down_cols = down_cols[:min_steps]
    
    time_steps = len(up_cols)
    n_samples = len(df)
    
    # 創建時間序列資料，形狀為 (樣本數, 時間步數, 特徵數)
    time_series_data = np.zeros((n_samples, time_steps, 2))
    
    # 填充時間序列資料
    for i, (up_col, down_col) in enumerate(zip(up_cols, down_cols)):
        time_series_data[:, i, 0] = df[up_col].values
        time_series_data[:, i, 1] = df[down_col].values
    
    # 標準化時間序列
    if normalize:
        for feat_idx in range(2):  # 上下界面
            # 計算均值和標準差
            feat_mean = np.mean(time_series_data[:, :, feat_idx])
            feat_std = np.std(time_series_data[:, :, feat_idx])
            
            # 確保標準差非零
            if feat_std < 1e-8:
                feat_std = 1.0
                
            # 標準化
            time_series_data[:, :, feat_idx] = (time_series_data[:, :, feat_idx] - feat_mean) / feat_std
    
    logger.info(f"Prepared time series data with shape {time_series_data.shape}")
    
    return time_series_data


def train_val_test_split(X, time_series, y, test_size=0.15, val_size=0.15, random_state=42):
    """
    將資料分割為訓練集、驗證集和測試集
    
    參數:
        X (numpy.ndarray): 特徵資料
        time_series (numpy.ndarray): 時間序列資料
        y (numpy.ndarray): 目標值
        test_size (float): 測試集比例
        val_size (float): 驗證集比例
        random_state (int): 隨機種子
        
    返回:
        dict: 包含分割後資料的字典
    """
    # 首先分割出測試集
    X_temp, X_test, time_series_temp, time_series_test, y_temp, y_test = train_test_split(
        X, time_series, y, test_size=test_size, random_state=random_state
    )
    
    # 從剩餘資料中分割出驗證集
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, time_series_train, time_series_val, y_train, y_val = train_test_split(
        X_temp, time_series_temp, y_temp, test_size=val_ratio, random_state=random_state
    )
    
    logger.info(f"Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'time_series_train': time_series_train,
        'time_series_val': time_series_val,
        'time_series_test': time_series_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }


def process_pipeline(data_path, test_size=0.15, val_size=0.15, random_state=42):
    """
    完整的資料處理管線
    
    參數:
        data_path (str): 資料檔案路徑
        test_size (float): 測試集比例
        val_size (float): 驗證集比例
        random_state (int): 隨機種子
        
    返回:
        dict: 包含處理後資料的字典
    """
    # 1. 加載資料
    df = load_data(data_path)
    
    # 2. 標準化特徵
    X, scaler, y, feature_cols = standardize_features(df)
    
    # 3. 準備時間序列資料
    time_series = prepare_time_series(df)
    
    # 4. 分割資料
    split_data = train_val_test_split(X, time_series, y, test_size, val_size, random_state)
    
    # 5. 整合所有資料
    result = {
        'X': X,
        'time_series': time_series,
        'y': y,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'df': df
    }
    result.update(split_data)
    
    logger.info("Data processing pipeline completed")
    
    return result