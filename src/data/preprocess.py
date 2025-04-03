#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
preprocess.py - 改進的資料預處理模組
本模組提供加載和預處理銲錫接點疲勞壽命預測所需的CAE資料，
包括資料清洗、標準化、資料增強和資料集分割等功能。

主要功能:
1. 資料加載與清洗
2. 特徵標準化與轉換
3. 物理知識驅動的資料增強
4. 時間序列資料處理
5. 資料集分割與處理
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, KFold
import logging
import sys

# 確保可以導入專案模組
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 導入資料增強模組
from src.utils.data_augmentation import perform_comprehensive_augmentation

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
        
        # 檢查異常值
        for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if len(outliers) > 0:
                logger.info(f"欄位 {col} 中檢測到 {len(outliers)} 個異常值")
                # 在小樣本數據集中，我們不移除異常值，而是記錄它們
        
        # 計算時間序列數據的基本統計
        for ts_col in TIMESERIES_COLUMNS:
            is_monotonic = df[ts_col].diff().dropna().ge(0).all()
            if not is_monotonic:
                logger.warning(f"時間序列欄位 {ts_col} 不是單調增加的，這可能違反物理規律")
        
        # 檢查目標值的分佈
        logger.info(f"目標變數 '{TARGET_COLUMN}' 統計: 最小值={df[TARGET_COLUMN].min()}, "
                   f"最大值={df[TARGET_COLUMN].max()}, 均值={df[TARGET_COLUMN].mean()}, "
                   f"標準差={df[TARGET_COLUMN].std()}")
        
        # 檢查存在的目標值分佈是否平衡和足夠多樣化
        if len(df) < 100:
            logger.warning("數據集樣本數較少，考慮使用資料增強技術")
        
        # 計算統計指標，用於診斷
        stats = df[required_columns].describe()
        logger.debug(f"資料統計摘要:\n{stats}")
        
        # 檢查特徵間的相關性
        corr_matrix = df[FEATURE_COLUMNS].corr()
        high_corr_pairs = []
        for i in range(len(FEATURE_COLUMNS)):
            for j in range(i+1, len(FEATURE_COLUMNS)):
                corr = abs(corr_matrix.iloc[i, j])
                if corr > 0.8:
                    high_corr_pairs.append((FEATURE_COLUMNS[i], FEATURE_COLUMNS[j], corr))
        
        if high_corr_pairs:
            logger.info("檢測到高度相關的特徵對:")
            for col1, col2, corr in high_corr_pairs:
                logger.info(f"  {col1} <-> {col2}: {corr:.4f}")
        
        return df
        
    except Exception as e:
        logger.error(f"資料載入失敗: {str(e)}")
        raise

def apply_feature_transformation(df, feature_cols=None, log_transform_target=True):
    """
    應用特徵轉換，包括對數變換和特徵創建
    
    參數:
        df (pandas.DataFrame): 原始資料集
        feature_cols (list): 要處理的特徵欄位
        log_transform_target (bool): 是否對目標變數應用對數變換
    
    返回:
        pandas.DataFrame: 轉換後的資料集
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS
    
    # 創建資料的副本以避免修改原始資料
    transformed_df = df.copy()
    
    # 1. 特徵比率計算
    logger.info("創建特徵比率...")
    
    # Die/PCB 比例
    transformed_df['Die_PCB_ratio'] = transformed_df['Die'] / (transformed_df['PCB'] * 1000)
    
    # stud/mold 比例
    transformed_df['stud_mold_ratio'] = transformed_df['stud'] / transformed_df['mold']
    
    # 總厚度
    transformed_df['total_thickness'] = transformed_df['Die'] + transformed_df['stud'] + transformed_df['mold']
    
    # 2. 物理知識導向特徵
    logger.info("創建物理知識導向特徵...")
    
    # 厚度不平衡因子 (Die與mold的厚度差)
    transformed_df['thickness_imbalance'] = abs(transformed_df['Die'] - transformed_df['mold'])
    
    # 翹曲因子 (翹曲變形與總厚度的比例)
    transformed_df['warpage_factor'] = transformed_df['Unit_warpage'] / transformed_df['total_thickness']
    
    # 3. 對數變換
    if log_transform_target and TARGET_COLUMN in df.columns:
        logger.info(f"對目標變數 '{TARGET_COLUMN}' 應用對數變換")
        transformed_df['log_' + TARGET_COLUMN] = np.log1p(transformed_df[TARGET_COLUMN])
    
    # 4. 時間序列統計特徵
    logger.info("從時間序列中提取統計特徵...")
    
    # 提取上界面和下界面時間序列
    up_cols = [col for col in df.columns if col.startswith('NLPLWK_up_')]
    down_cols = [col for col in df.columns if col.startswith('NLPLWK_down_')]
    
    if up_cols and down_cols:
        # 上下界面最大值
        transformed_df['up_max'] = transformed_df[up_cols].max(axis=1)
        transformed_df['down_max'] = transformed_df[down_cols].max(axis=1)
        
        # 上下界面最大值比率
        transformed_df['up_down_ratio'] = transformed_df['up_max'] / transformed_df['down_max']
        
        # 時間序列增長率
        if len(up_cols) >= 2:
            transformed_df['up_growth_rate'] = (transformed_df[up_cols[-1]] - transformed_df[up_cols[0]]) / transformed_df[up_cols[0]]
        
        if len(down_cols) >= 2:
            transformed_df['down_growth_rate'] = (transformed_df[down_cols[-1]] - transformed_df[down_cols[0]]) / transformed_df[down_cols[0]]
    
    logger.info(f"特徵轉換完成，從 {len(df.columns)} 個特徵增加到 {len(transformed_df.columns)} 個特徵")
    
    return transformed_df

def standardize_features(df, feature_cols=None, target_col=None, scaler_type='robust', log_target=True):
    """
    標準化特徵資料
    
    參數:
        df (pandas.DataFrame): 原始資料集
        feature_cols (list): 要標準化的特徵欄位，預設為FEATURE_COLUMNS
        target_col (str): 目標欄位，預設為TARGET_COLUMN
        scaler_type (str): 縮放器類型，'standard'、'minmax'或'robust'
        log_target (bool): 是否對目標變數應用對數變換
        
    返回:
        tuple: (標準化後的特徵資料, 標準化器, 目標資料, 目標標準化器)
    """
    if feature_cols is None:
        base_feature_cols = FEATURE_COLUMNS
        # 包括從apply_feature_transformation創建的額外特徵
        extra_feature_cols = [
            'Die_PCB_ratio', 'stud_mold_ratio', 'total_thickness',
            'thickness_imbalance', 'warpage_factor', 
            'up_max', 'down_max', 'up_down_ratio', 
            'up_growth_rate', 'down_growth_rate'
        ]
        # 僅包括實際存在於df中的額外特徵
        extra_feature_cols = [col for col in extra_feature_cols if col in df.columns]
        feature_cols = base_feature_cols + extra_feature_cols
    
    if target_col is None:
        target_col = 'log_' + TARGET_COLUMN if log_target and 'log_' + TARGET_COLUMN in df.columns else TARGET_COLUMN
    
    # 選擇縮放器
    if scaler_type.lower() == 'standard':
        scaler = StandardScaler()
    elif scaler_type.lower() == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type.lower() == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"不支援的縮放器類型: {scaler_type}")
    
    # 確保所有選定的特徵都存在
    existing_feature_cols = [col for col in feature_cols if col in df.columns]
    if len(existing_feature_cols) < len(feature_cols):
        missing_cols = [col for col in feature_cols if col not in df.columns]
        logger.warning(f"以下特徵列不存在于資料框中: {missing_cols}")
        feature_cols = existing_feature_cols
    
    # 標準化特徵
    X = df[feature_cols].values
    X_scaled = scaler.fit_transform(X)
    
    # 獲取和標準化目標變數
    y = df[target_col].values if target_col in df.columns else None
    y_scaler = None
    
    if y is not None:
        # 使用單變量縮放器處理目標變數
        y_scaler = StandardScaler()
        y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    else:
        y_scaled = None
    
    return X_scaled, scaler, y_scaled, y_scaler, feature_cols

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
    # 將時間序列資料分為上界面和下界面
    up_series = [col for col in timeseries_cols if 'up' in col]
    down_series = [col for col in timeseries_cols if 'down' in col]
    
    # 確保上下界面序列對應
    if len(up_series) != len(down_series):
        logger.warning("上界面和下界面的時間序列資料數量不匹配")
        # 尋找共同的時間點
        up_times = [int(col.split('_')[-1]) for col in up_series]
        down_times = [int(col.split('_')[-1]) for col in down_series]
        common_times = sorted(set(up_times).intersection(set(down_times)))
        
        up_series = [f'NLPLWK_up_{t}' for t in common_times]
        down_series = [f'NLPLWK_down_{t}' for t in common_times]
        
        logger.info(f"使用共同的時間點: {common_times}")
    
    # 確保時間序列的順序
    up_series.sort()
    down_series.sort()
    
    n_samples = len(df)
    n_timepoints = len(up_series)  # 時間點數量
    n_features = 2  # 上下界面兩個特徵
    
    logger.info(f"時間序列形狀: 樣本數={n_samples}, 時間點數={n_timepoints}, 特徵數={n_features}")
    
    # 創建 3D 時間序列資料: (樣本數, 時間點數, 特徵數)
    timeseries_data = np.zeros((n_samples, n_timepoints, n_features))
    
    for i, (up_col, down_col) in enumerate(zip(up_series, down_series)):
        if up_col in df.columns and down_col in df.columns:
            timeseries_data[:, i, 0] = df[up_col].values  # 上界面
            timeseries_data[:, i, 1] = df[down_col].values  # 下界面
        else:
            logger.warning(f"時間序列欄位不存在: {up_col} 或 {down_col}")
    
    # 標準化時間序列資料 (對每個特徵維度單獨標準化)
    for feature_idx in range(n_features):
        feature_data = timeseries_data[:, :, feature_idx]
        feature_mean = np.mean(feature_data)
        feature_std = np.std(feature_data)
        # 避免零標準差
        if feature_std < 1e-10:
            logger.warning(f"特徵 {feature_idx} 的標準差接近零，使用默認標準差 1.0")
            feature_std = 1.0
        timeseries_data[:, :, feature_idx] = (feature_data - feature_mean) / feature_std
    
    return timeseries_data

def augment_data(df, synthetic_samples=50, perturbation_variations=2, mix_factor=0.3, noise_level=0.08):
    """
    使用資料增強技術擴充小樣本資料集
    
    參數:
        df (pandas.DataFrame): 原始資料集
        synthetic_samples (int): 生成的合成樣本數量
        perturbation_variations (int): 每個原始樣本生成的變體數量
        mix_factor (float): 混合因子
        noise_level (float): 噪聲水平
        
    返回:
        pandas.DataFrame: 增強後的資料集
    """
    logger.info(f"開始資料增強，原始數據大小: {len(df)}")
    
    # 使用資料增強模組進行數據增強
    augmented_df = perform_comprehensive_augmentation(
        df,
        synthetic_samples=synthetic_samples,
        perturbation_variations=perturbation_variations,
        mix_factor=mix_factor,
        noise_level=noise_level,
        validate_samples=True
    )
    
    logger.info(f"資料增強完成，增強後數據大小: {len(augmented_df)}")
    
    return augmented_df

def train_val_test_split(X, time_series, y, test_size=0.15, val_size=0.15, 
                       random_state=42, stratify=None, k_fold=None):
    """
    將資料集分割為訓練、驗證和測試集
    
    參數:
        X (numpy.ndarray): 標準化後的特徵資料
        time_series (numpy.ndarray): 時間序列資料
        y (numpy.ndarray): 目標資料
        test_size (float): 測試集比例
        val_size (float): 驗證集比例
        random_state (int): 隨機種子
        stratify (array-like): 分層抽樣的標籤，適用於不平衡資料集
        k_fold (int): K折交叉驗證的折數，若不為None則返回K折
        
    返回:
        dict: 包含分割結果的字典或K折數據
    """
    if k_fold is not None and k_fold > 1:
        # K折交叉驗證
        logger.info(f"使用 {k_fold} 折交叉驗證")
        
        kf = KFold(n_splits=k_fold, shuffle=True, random_state=random_state)
        fold_data = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            time_series_train, time_series_val = time_series[train_idx], time_series[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            fold_data.append({
                'fold': fold,
                'X_train': X_train,
                'X_val': X_val,
                'time_series_train': time_series_train,
                'time_series_val': time_series_val,
                'y_train': y_train,
                'y_val': y_val
            })
        
        return {'k_fold_data': fold_data}
    
    # 如果不使用K折交叉驗證，進行單次分割
    # 處理分層抽樣
    if stratify is not None:
        logger.info("使用分層抽樣進行資料分割")
        
        # 首先區分目標值範圍，創建分層標籤
        if stratify is True and y is not None:
            # 自動創建分層標籤
            num_bins = min(len(y) // 8, 5)  # 確保每個bin至少有8個樣本
            bins = np.quantile(y, np.linspace(0, 1, num_bins + 1))
            stratify_labels = np.digitize(y, bins)
            logger.info(f"根據目標值範圍創建了 {num_bins} 個層")
        else:
            stratify_labels = stratify
    else:
        stratify_labels = None
    
    # 先分割出測試集
    X_temp, X_test, time_series_temp, time_series_test, y_temp, y_test = train_test_split(
        X, time_series, y, test_size=test_size, random_state=random_state,
        stratify=stratify_labels
    )
    
    # 更新分層標籤
    if stratify_labels is not None:
        stratify_temp = np.digitize(y_temp, np.quantile(y_temp, np.linspace(0, 1, 5)))
    else:
        stratify_temp = None
    
    # 從剩餘數據中分割出驗證集
    val_ratio = val_size / (1 - test_size)
    
    X_train, X_val, time_series_train, time_series_val, y_train, y_val = train_test_split(
        X_temp, time_series_temp, y_temp, test_size=val_ratio, random_state=random_state,
        stratify=stratify_temp
    )
    
    # 記錄分割後的資料大小
    logger.info(f"訓練集: {len(X_train)} 樣本, 驗證集: {len(X_val)} 樣本, 測試集: {len(X_test)} 樣本")
    
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

def process_pipeline(data_filepath, test_size=0.15, val_size=0.15, random_state=42, 
                     scaler_type='robust', log_transform=True, augmentation=True, k_fold=None,
                     synthetic_samples=50, perturbation_variations=2):
    """
    完整的資料處理管道，整合上述所有步驟
    
    參數:
        data_filepath (str): 資料檔案路徑
        test_size (float): 測試集比例
        val_size (float): 驗證集比例
        random_state (int): 隨機種子
        scaler_type (str): 標準化類型
        log_transform (bool): 是否對目標變數應用對數變換
        augmentation (bool): 是否使用資料增強
        k_fold (int): K折交叉驗證的折數，若不為None則使用K折
        synthetic_samples (int): 生成的合成樣本數量
        perturbation_variations (int): 每個原始樣本生成的變體數量
        
    返回:
        dict: 包含預處理後的所有資料和相關資訊的字典
    """
    # 1. 載入資料
    df = load_data(data_filepath)
    original_df = df.copy()
    
    # 2. 應用特徵轉換
    df = apply_feature_transformation(df, log_transform_target=log_transform)
    
    # 3. 資料增強 (如果啟用)
    if augmentation and len(df) < 200:  # 只對小樣本數據集進行增強
        logger.info("使用物理知識驅動的資料增強...")
        augmented_df = augment_data(
            df, 
            synthetic_samples=synthetic_samples,
            perturbation_variations=perturbation_variations
        )
        
        # 將原始資料標記為非合成
        df['is_synthetic'] = 0
        
        # 標記合成資料
        synthetic_mask = ~augmented_df.index.isin(df.index)
        augmented_df.loc[synthetic_mask, 'is_synthetic'] = 1
        
        # 使用增強後的資料集
        df = augmented_df
    
    # 4. 標準化特徵
    target_col = 'log_' + TARGET_COLUMN if log_transform and 'log_' + TARGET_COLUMN in df.columns else TARGET_COLUMN
    X_scaled, feature_scaler, y_scaled, y_scaler, used_features = standardize_features(
        df, 
        target_col=target_col,
        scaler_type=scaler_type,
        log_target=log_transform
    )
    
    # 5. 準備時間序列資料
    time_series_data = prepare_time_series(df)
    
    # 6. 分割資料集
    # 使用目標值的分位數進行分層抽樣
    split_results = train_val_test_split(
        X_scaled, time_series_data, y_scaled, 
        test_size=test_size, val_size=val_size, 
        random_state=random_state, stratify=True,
        k_fold=k_fold
    )
    
    # 7. 準備返回結果
    result = {
        'feature_scaler': feature_scaler,
        'y_scaler': y_scaler,
        'feature_columns': used_features,
        'timeseries_columns': TIMESERIES_COLUMNS,
        'target_column': target_col,
        'original_target_column': TARGET_COLUMN,
        'df_original': original_df,
        'df_processed': df,
        'log_transform': log_transform,
        'X_all': X_scaled,
        'time_series_all': time_series_data,
        'y_all': y_scaled,
        'y_all_original': df[TARGET_COLUMN].values if TARGET_COLUMN in df.columns else None
    }
    
    # 添加分割結果
    result.update(split_results)
    
    return result

if __name__ == "__main__":
    # 簡單的測試代碼
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 獲取相對於專案根目錄的資料路徑
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    data_path = os.path.join(project_root, "data/raw/Training_data_warpage_final_20250321_v1.2.csv")
    
    if os.path.exists(data_path):
        # 測試完整資料處理管道
        result = process_pipeline(
            data_path,
            test_size=0.15,
            val_size=0.15,
            augmentation=True,
            synthetic_samples=40,
            perturbation_variations=1
        )
        
        logger.info(f"預處理完成:")
        logger.info(f"  原始資料大小: {len(result['df_original'])}")
        logger.info(f"  處理後資料大小: {len(result['df_processed'])}")
        logger.info(f"  訓練集: {result['X_train'].shape[0]}")
        logger.info(f"  驗證集: {result['X_val'].shape[0]}")
        logger.info(f"  測試集: {result['X_test'].shape[0]}")
        logger.info(f"  使用特徵數: {len(result['feature_columns'])}")
    else:
        logger.error(f"找不到資料檔案: {data_path}")