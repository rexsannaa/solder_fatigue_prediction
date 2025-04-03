#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
preprocess.py - 資料預處理模組
本模組提供加載和預處理銲錫接點疲勞壽命預測所需的CAE資料，
包括資料清洗、標準化、特徵工程及資料集分割等功能。

主要功能:
1. 資料加載與初步清洗
2. 特徵標準化
3. 時間序列資料準備
4. 資料集分割(訓練/驗證/測試)
5. 完整的資料處理管道
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

# 常數定義
FEATURE_COLUMNS = ['Die', 'stud', 'mold', 'PCB', 'Unit_warpage']
TIMESERIES_COLUMNS = [
    'NLPLWK_up_3600', 'NLPLWK_up_7200', 'NLPLWK_up_10800', 'NLPLWK_up_14400',
    'NLPLWK_down_3600', 'NLPLWK_down_7200', 'NLPLWK_down_10800', 'NLPLWK_down_14400'
]
TARGET_COLUMN = 'Nf_pred (cycles)'

def load_data(filepath, encoding='utf-8'):
    """
    加載並初步清洗CAE資料
    
    參數:
        filepath (str): CSV資料檔案路徑
        encoding (str): 檔案編碼，預設為'utf-8'
    
    返回:
        pandas.DataFrame: 清洗後的資料集
    """
    try:
        logger.info(f"正在讀取資料: {filepath}")
        df = pd.read_csv(filepath, encoding=encoding)
        
        # 檢查必要欄位是否存在
        required_columns = FEATURE_COLUMNS + TIMESERIES_COLUMNS + [TARGET_COLUMN]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"資料缺少必要欄位: {missing_columns}")
        
        # 移除空值行
        initial_rows = len(df)
        df = df.dropna(subset=required_columns)
        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            logger.warning(f"移除了 {removed_rows} 行含有空值的資料")
        
        # 基本檢查: 確保數值型別欄位
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning(f"欄位 {col} 不是數值型別，嘗試轉換")
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 計算統計指標，用於診斷
        stats = df[required_columns].describe()
        logger.debug(f"資料統計摘要:\n{stats}")
        
        return df
        
    except Exception as e:
        logger.error(f"資料載入失敗: {str(e)}")
        raise

def standardize_features(df, feature_cols=None, target_col=None, scaler_type='standard'):
    """
    標準化特徵資料
    
    參數:
        df (pandas.DataFrame): 原始資料集
        feature_cols (list): 要標準化的特徵欄位，預設為FEATURE_COLUMNS
        target_col (str): 目標欄位，預設為TARGET_COLUMN
        scaler_type (str): 縮放器類型，'standard'或'minmax'
    
    返回:
        tuple: (標準化後的特徵資料, 標準化器, 目標資料)
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS
    
    if target_col is None:
        target_col = TARGET_COLUMN
    
    # 選擇縮放器
    if scaler_type.lower() == 'standard':
        scaler = StandardScaler()
    elif scaler_type.lower() == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"不支援的縮放器類型: {scaler_type}")
    
    # 標準化特徵
    X = df[feature_cols].values
    X_scaled = scaler.fit_transform(X)
    
    # 獲取目標變數
    y = df[target_col].values if target_col in df.columns else None
    
    return X_scaled, scaler, y

def prepare_time_series(df, timeseries_cols=None, sequence_length=4):
    """
    準備時間序列資料
    
    參數:
        df (pandas.DataFrame): 原始資料集
        timeseries_cols (list): 時間序列欄位，預設為TIMESERIES_COLUMNS  
        sequence_length (int): 序列長度，預設為4
    
    返回:
        numpy.ndarray: 形狀為 (樣本數, 序列長度, 特徵數) 的時間序列資料
    """
    if timeseries_cols is None:
        timeseries_cols = TIMESERIES_COLUMNS
    
    # 根據資料結構將時間序列資料重組
    # 注意: 我們的時間序列資料格式為 [up_3600, up_7200, up_10800, up_14400, down_3600, ...]
    # 需要重組為 [(up_3600, down_3600), (up_7200, down_7200), ...]
    
    # 將時間序列資料分為上界面和下界面
    up_series = [col for col in timeseries_cols if 'up' in col]
    down_series = [col for col in timeseries_cols if 'down' in col]
    
    # 確保上下界面序列對應
    if len(up_series) != len(down_series):
        raise ValueError("上界面和下界面的時間序列資料數量不匹配")
    
    n_samples = len(df)
    n_timepoints = len(up_series)  # 時間點數量
    n_features = 2  # 上下界面兩個特徵
    
    # 創建 3D 時間序列資料: (樣本數, 時間點數, 特徵數)
    timeseries_data = np.zeros((n_samples, n_timepoints, n_features))
    
    for i, (up_col, down_col) in enumerate(zip(up_series, down_series)):
        timeseries_data[:, i, 0] = df[up_col].values  # 上界面
        timeseries_data[:, i, 1] = df[down_col].values  # 下界面
    
    # 標準化時間序列資料
    # 對每個特徵維度進行標準化
    for feature_idx in range(n_features):
        feature_data = timeseries_data[:, :, feature_idx]
        feature_mean = np.mean(feature_data)
        feature_std = np.std(feature_data)
        timeseries_data[:, :, feature_idx] = (feature_data - feature_mean) / feature_std
    
    logger.info(f"時間序列資料形狀: {timeseries_data.shape}")
    return timeseries_data

def train_val_test_split(X, time_series, y, test_size=0.15, val_size=0.15, random_state=42):
    """
    將資料集分割為訓練、驗證和測試集
    
    參數:
        X (numpy.ndarray): 標準化後的特徵資料
        time_series (numpy.ndarray): 時間序列資料
        y (numpy.ndarray): 目標資料
        test_size (float): 測試集比例
        val_size (float): 驗證集比例
        random_state (int): 隨機種子
    
    返回:
        tuple: (訓練特徵, 驗證特徵, 測試特徵, 訓練時間序列, 驗證時間序列, 
                測試時間序列, 訓練目標, 驗證目標, 測試目標)
    """
    # 先分割出測試集
    X_temp, X_test, time_series_temp, time_series_test, y_temp, y_test = train_test_split(
        X, time_series, y, test_size=test_size, random_state=random_state
    )
    
    # 從剩餘數據中分割出驗證集
    # 計算驗證集在剩餘數據中的比例
    val_ratio = val_size / (1 - test_size)
    
    X_train, X_val, time_series_train, time_series_val, y_train, y_val = train_test_split(
        X_temp, time_series_temp, y_temp, test_size=val_ratio, random_state=random_state
    )
    
    # 記錄分割後的資料大小
    logger.info(f"訓練集: {len(X_train)} 樣本, 驗證集: {len(X_val)} 樣本, 測試集: {len(X_test)} 樣本")
    
    return (X_train, X_val, X_test, 
            time_series_train, time_series_val, time_series_test, 
            y_train, y_val, y_test)

def process_pipeline(data_filepath, test_size=0.15, val_size=0.15, random_state=42):
    """
    完整的資料處理管道，整合上述所有步驟
    
    參數:
        data_filepath (str): 資料檔案路徑
        test_size (float): 測試集比例
        val_size (float): 驗證集比例
        random_state (int): 隨機種子
    
    返回:
        dict: 包含預處理後的所有資料和相關資訊的字典
    """
    # 1. 載入資料
    df = load_data(data_filepath)
    
    # 2. 標準化特徵
    X_scaled, feature_scaler, y = standardize_features(df)
    
    # 3. 準備時間序列資料
    time_series_data = prepare_time_series(df)
    
    # 4. 分割資料集
    (X_train, X_val, X_test, 
     time_series_train, time_series_val, time_series_test, 
     y_train, y_val, y_test) = train_val_test_split(
        X_scaled, time_series_data, y, 
        test_size=test_size, val_size=val_size, random_state=random_state
    )
    
    # 5. 計算目標變數(y)的縮放參數，用於反標準化預測結果
    y_scaler = StandardScaler()
    y_scaler.fit(y.reshape(-1, 1))
    
    # 返回包含所有處理後資料和輔助資訊的字典
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'time_series_train': time_series_train,
        'time_series_val': time_series_val,
        'time_series_test': time_series_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_scaler': feature_scaler,
        'y_scaler': y_scaler,
        'feature_columns': FEATURE_COLUMNS,
        'timeseries_columns': TIMESERIES_COLUMNS,
        'target_column': TARGET_COLUMN,
        'df_original': df
    }

if __name__ == "__main__":
    # 簡單的測試代碼
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 獲取相對於專案根目錄的資料路徑
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    data_path = os.path.join(project_root, "data/raw/Training_data_warpage_final_20250321_v1.2.csv")
    
    if os.path.exists(data_path):
        result = process_pipeline(data_path)
        logger.info(f"預處理完成，資料集大小 - 訓練集: {result['X_train'].shape[0]}, "
                   f"驗證集: {result['X_val'].shape[0]}, 測試集: {result['X_test'].shape[0]}")
    else:
        logger.error(f"找不到資料檔案: {data_path}")