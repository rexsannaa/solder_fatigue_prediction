#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
preprocess.py - 資料預處理模組
本模組提供與銲錫接點疲勞壽命預測相關的資料預處理功能，
包括資料載入、特徵標準化和時間序列處理等操作。

主要功能:
1. 資料載入與基本處理
2. 特徵標準化
3. 時間序列資料準備
4. 資料分割
5. 完整處理流程
"""

import numpy as np
import pandas as pd
import logging
import traceback
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def load_data(filepath):
    """
    載入資料檔案
    
    參數:
        filepath (str): 資料檔案路徑
        
    返回:
        pandas.DataFrame: 載入的資料框
    """
    try:
        file_path = Path(filepath)
        if not file_path.exists():
            raise FileNotFoundError(f"找不到資料檔案: {filepath}")
        
        suffix = file_path.suffix.lower()
        if suffix == '.csv':
            df = pd.read_csv(filepath)
        elif suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath)
        else:
            raise ValueError(f"不支援的檔案格式: {suffix}")
        
        logger.info(f"成功載入資料檔案: {filepath}，共 {len(df)} 筆資料，{df.shape[1]} 個欄位")
        
        missing_counts = df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]
        if not missing_cols.empty:
            logger.warning(f"存在缺失值的欄位: {missing_cols.to_dict()}")
        
        return df
    except Exception as e:
        logger.error(f"載入資料檔案時發生錯誤: {str(e)}")
        raise

def standardize_features(df, feature_cols, target_col=None):
    """
    標準化特徵
    
    參數:
        df (pandas.DataFrame): 資料框
        feature_cols (list): 要標準化的特徵欄位名稱列表
        target_col (str, optional): 目標欄位名稱
        
    返回:
        tuple: (標準化後的特徵, 標準化器, 目標值)
    """
    try:
        for col in feature_cols:
            if col not in df.columns:
                raise ValueError(f"特徵欄位 '{col}' 不存在於資料中")
        
        scaler = StandardScaler()
        
        X = df[feature_cols].values
        X_scaled = scaler.fit_transform(X)
        
        y = None
        if target_col:
            if target_col not in df.columns:
                raise ValueError(f"目標欄位 '{target_col}' 不存在於資料中")
            y = df[target_col].values
        
        logger.info(f"標準化完成，特徵數量: {len(feature_cols)}")
        logger.debug(f"標準化參數 - 均值: {scaler.mean_}, 標準差: {scaler.scale_}")
        
        return X_scaled, scaler, y
    except Exception as e:
        logger.error(f"標準化特徵時發生錯誤: {str(e)}")
        raise

def prepare_time_series(df, time_series_prefix=None, time_points=None):
    """
    準備時間序列資料
    
    參數:
        df (pandas.DataFrame): 資料框
        time_series_prefix (list, optional): 時間序列欄位前綴列表，如 ['NLPLWK_up_', 'NLPLWK_down_']
        time_points (list, optional): 時間點列表，如 [3600, 7200, 10800, 14400]
        
    返回:
        numpy.ndarray: 時間序列資料，形狀為 (樣本數, 時間步數, 特徵數)
    """
    try:
        if time_series_prefix is None:
            time_series_prefix = ['NLPLWK_up_', 'NLPLWK_down_']
        
        if time_points is None:
            time_points = [3600, 7200, 10800, 14400]
            
            all_cols = df.columns
            time_points_set = set()
            
            for prefix in time_series_prefix:
                cols = [col for col in all_cols if col.startswith(prefix)]
                for col in cols:
                    try:
                        time_point = col.replace(prefix, '')
                        time_point = int(time_point)
                        time_points_set.add(time_point)
                    except ValueError:
                        continue
            
            if time_points_set:
                time_points = sorted(list(time_points_set))
        # 添加調試輸出
        print(f"[DEBUG] 原始時間序列資料統計:")
        for prefix in time_series_prefix:
            cols = [col for col in df.columns if col.startswith(prefix)]
            for col in cols:
                data = df[col].values
                print(f"[DEBUG] {col} - 範圍: {np.min(data):.6e} - {np.max(data):.6e}, 平均值: {np.mean(data):.6e}")
        n_samples = len(df)
        n_time_steps = len(time_points)
        n_features = len(time_series_prefix)
        
        time_series_data = np.zeros((n_samples, n_time_steps, n_features))
        
        for feature_idx, prefix in enumerate(time_series_prefix):
            for time_idx, time_point in enumerate(time_points):
                col_name = f"{prefix}{time_point}"
                if col_name in df.columns:
                    time_series_data[:, time_idx, feature_idx] = df[col_name].values
                else:
                    logger.warning(f"時間序列欄位 '{col_name}' 不存在，將使用零填充")
        
        logger.info(f"時間序列資料準備完成，形狀: {time_series_data.shape}")
        logger.debug(f"時間序列特徵: {time_series_prefix}, 時間點: {time_points}")
        # 添加最終時間序列資料的調試輸出
        print(f"[DEBUG] 最終時間序列資料形狀: {time_series_data.shape}")
        print(f"[DEBUG] 時間序列上界面數據範例 (第一個樣本):")
        print(time_series_data[0, :, 0])
        print(f"[DEBUG] 時間序列下界面數據範例 (第一個樣本):")
        print(time_series_data[0, :, 1])

        return time_series_data
    except Exception as e:
        logger.error(f"準備時間序列資料時發生錯誤: {str(e)}")
        raise

def train_val_test_split(X, time_series, y, test_size=0.15, val_size=0.15, random_state=42, stratify=None):
    """
    將資料分割為訓練集、驗證集和測試集
    
    參數:
        X (numpy.ndarray): 靜態特徵
        time_series (numpy.ndarray): 時間序列特徵
        y (numpy.ndarray): 目標值
        test_size (float): 測試集比例
        val_size (float): 驗證集比例
        random_state (int): 隨機種子
        stratify (numpy.ndarray, optional): 分層抽樣依據
        
    返回:
        dict: 包含分割後資料的字典
    """
    try:
        if stratify is not None:
            if np.issubdtype(stratify.dtype, np.number) and len(np.unique(stratify)) > 10:
                y_log = np.log(stratify + 1e-8)
                n_bins = min(10, len(y) // 8)
                bins = np.linspace(y_log.min(), y_log.max(), n_bins + 1)
                stratify = np.digitize(y_log, bins)
        
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        X_temp, X_test, ts_temp, ts_test, y_temp, y_test = train_test_split(
            X, time_series, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
        
        if stratify is not None:
            stratify_temp = stratify[np.isin(np.arange(len(stratify)), np.arange(len(X))[~np.isin(np.arange(len(X)), np.where(np.isin(X, X_test))[0])])]
        else:
            stratify_temp = None
        
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, ts_train, ts_val, y_train, y_val = train_test_split(
            X_temp, ts_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=stratify_temp
        )
        
        logger.info(f"資料分割完成 - 訓練集: {len(X_train)} 樣本, 驗證集: {len(X_val)} 樣本, 測試集: {len(X_test)} 樣本")
        
        return {
            'X_train': X_train, 'ts_train': ts_train, 'y_train': y_train,
            'X_val': X_val, 'ts_val': ts_val, 'y_val': y_val,
            'X_test': X_test, 'ts_test': ts_test, 'y_test': y_test
        }
    except Exception as e:
        logger.error(f"分割資料時發生錯誤: {str(e)}")
        raise

def process_pipeline(filepath, feature_cols, target_col, time_series_prefix=None, time_points=None,
                    test_size=0.15, val_size=0.15, random_state=42):
    """
    完整資料處理流程
    
    參數:
        filepath (str): 資料檔案路徑
        feature_cols (list): 特徵欄位名稱列表
        target_col (str): 目標欄位名稱
        time_series_prefix (list, optional): 時間序列欄位前綴列表
        time_points (list, optional): 時間點列表
        test_size (float): 測試集比例
        val_size (float): 驗證集比例
        random_state (int): 隨機種子
        
    返回:
        dict: 包含處理後資料的字典
    """
    try:
        df = load_data(filepath)
        X, feature_scaler, y = standardize_features(df, feature_cols, target_col)
        time_series = prepare_time_series(df, time_series_prefix, time_points)
        split_data = train_val_test_split(X, time_series, y, test_size, val_size, random_state)
        
        split_data['feature_scaler'] = feature_scaler
        split_data['df'] = df
        
        logger.info("完整資料處理流程完成")
        
        return split_data
    except Exception as e:
        logger.error(f"詳細錯誤追蹤: {traceback.format_exc()}")
        logger.error(f"執行資料處理流程時發生錯誤: {str(e)}")
        raise

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        logger.info("測試資料預處理模組")
        
        test_data = {
            'Die': [250, 200, 150],
            'stud': [80, 70, 60],
            'mold': [75, 65, 55],
            'PCB': [1.0, 0.8, 0.6],
            'Unit_warpage': [10, 8, 6],
            'NLPLWK_up_3600': [0.001, 0.002, 0.003],
            'NLPLWK_up_7200': [0.002, 0.004, 0.006],
            'NLPLWK_down_3600': [0.0008, 0.0016, 0.0024],
            'NLPLWK_down_7200': [0.0016, 0.0032, 0.0048],
            'Nf_pred (cycles)': [2000, 1500, 1000]
        }
        
        test_df = pd.DataFrame(test_data)
        test_df.to_csv('test_data.csv', index=False)
        
        df = load_data('test_data.csv')
        logger.info(f"載入的資料頭部:\n{df.head()}")
        
        feature_cols = ['Die', 'stud', 'mold', 'PCB', 'Unit_warpage']
        X, scaler, y = standardize_features(df, feature_cols, 'Nf_pred (cycles)')
        logger.info(f"標準化後的特徵形狀: {X.shape}")
        
        time_series = prepare_time_series(df, ['NLPLWK_up_', 'NLPLWK_down_'], [3600, 7200])
        logger.info(f"時間序列資料形狀: {time_series.shape}")
        
        split_data = train_val_test_split(X, time_series, y, test_size=0.33, val_size=0.33)
        logger.info(f"分割後的訓練集形狀: {split_data['X_train'].shape}")
        
        pipeline_data = process_pipeline(
            'test_data.csv', 
            feature_cols, 
            'Nf_pred (cycles)',
            ['NLPLWK_up_', 'NLPLWK_down_'],
            [3600, 7200]
        )
        logger.info("完整流程測試通過")
        
        import os
        os.remove('test_data.csv')
        
    except Exception as e:
        logger.error(f"測試失敗: {str(e)}")