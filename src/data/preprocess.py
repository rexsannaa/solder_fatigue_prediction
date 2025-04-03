#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
improved_preprocess.py - 改進的資料預處理模組
本模組提供銲錫接點疲勞壽命預測的高級資料預處理功能，
特別針對小樣本資料集優化，並與物理知識驅動的資料增強整合。

主要功能:
1. 資料加載與清洗
2. 特徵工程與轉換
3. 強化的物理知識驅動資料增強
4. 小樣本數據集處理優化
5. 特徵選擇與重要性分析
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import mutual_info_regression
import logging
import matplotlib.pyplot as plt

# 確保可以導入專案模組
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 嘗試導入資料增強模組
try:
    from scripts.data_augmentation import perform_comprehensive_augmentation
    data_augmentation_available = True
except ImportError:
    data_augmentation_available = False
    logging.warning("無法導入資料增強模組，將使用基本資料增強")

logger = logging.getLogger(__name__)

# 常數定義
STATIC_FEATURES = ['Die', 'stud', 'mold', 'PCB', 'Unit_warpage']
TIME_SERIES_PREFIXES = ['NLPLWK_up_', 'NLPLWK_down_']
TARGET_COLUMN = 'Nf_pred (cycles)'


def load_data(filepath, encoding='utf-8', validate=True):
    """
    加載並初步清洗CAE資料
    
    參數:
        filepath (str): CSV資料檔案路徑
        encoding (str): 檔案編碼，預設為'utf-8'
        validate (bool): 是否進行資料驗證
    
    返回:
        pandas.DataFrame: 清洗後的資料集
    """
    try:
        logger.info(f"正在讀取資料: {filepath}")
        df = pd.read_csv(filepath, encoding=encoding)
        
        initial_rows = len(df)
        logger.info(f"原始資料包含 {initial_rows} 行")
        
        # 檢查資料格式
        if validate:
            # 找出所有時間序列列
            time_series_cols = []
            for prefix in TIME_SERIES_PREFIXES:
                time_series_cols.extend([col for col in df.columns if col.startswith(prefix)])
            
            # 檢查必要欄位
            required_columns = STATIC_FEATURES + [TARGET_COLUMN]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.warning(f"資料缺少必要欄位: {missing_columns}")
                if TARGET_COLUMN in missing_columns:
                    raise ValueError(f"目標欄位 {TARGET_COLUMN} 不存在")
            
            # 檢查時間序列欄位
            if len(time_series_cols) == 0:
                logger.warning("未找到任何時間序列欄位")
        
        # 移除空值行
        df = df.dropna(subset=[col for col in required_columns if col in df.columns])
        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            logger.info(f"移除了 {removed_rows} 行含有空值的資料")
        
        # 數據類型轉換
        for col in df.columns:
            if col in STATIC_FEATURES or col.startswith(tuple(TIME_SERIES_PREFIXES)) or col == TARGET_COLUMN:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    logger.warning(f"無法將欄位 {col} 轉換為數值型態")
        
        # 若移除空值後的數據量太少，發出警告
        if len(df) < 50:
            logger.warning(f"數據集樣本數較少 ({len(df)} 筆)，可能需要使用資料增強")
        
        return df
        
    except Exception as e:
        logger.error(f"資料載入失敗: {str(e)}")
        raise


def create_derived_features(df):
    """
    創建衍生特徵，增強模型能力
    
    參數:
        df (pandas.DataFrame): 原始數據框
        
    返回:
        pandas.DataFrame: 含衍生特徵的數據框
    """
    df_new = df.copy()
    
    logger.info("創建衍生特徵...")
    
    # 1. 結構比例特徵
    df_new['Die_PCB_ratio'] = df_new['Die'] / (df_new['PCB'] * 1000)  # PCB轉為微米
    df_new['stud_mold_ratio'] = df_new['stud'] / df_new['mold']
    df_new['Die_stud_ratio'] = df_new['Die'] / df_new['stud']
    
    # 2. 物理導向特徵
    df_new['total_thickness'] = df_new['Die'] + df_new['stud'] + df_new['mold'] + (df_new['PCB'] * 1000)
    df_new['warpage_stress'] = df_new['Unit_warpage'] / df_new['total_thickness']
    
    # 3. 時間序列衍生特徵
    up_cols = [col for col in df.columns if col.startswith('NLPLWK_up_')]
    down_cols = [col for col in df.columns if col.startswith('NLPLWK_down_')]
    
    if len(up_cols) >= 2 and len(down_cols) >= 2:
        # 排序時間序列列
        up_cols.sort()
        down_cols.sort()
        
        # 計算上界面最終值
        df_new['up_final'] = df_new[up_cols[-1]]
        
        # 計算下界面最終值
        df_new['down_final'] = df_new[down_cols[-1]]
        
        # 計算上界面增長率
        df_new['up_growth_rate'] = (df_new[up_cols[-1]] - df_new[up_cols[0]]) / (df_new[up_cols[0]] + 1e-8)
        
        # 計算下界面增長率
        df_new['down_growth_rate'] = (df_new[down_cols[-1]] - df_new[down_cols[0]]) / (df_new[down_cols[0]] + 1e-8)
        
        # 計算上下界面比例
        df_new['up_down_ratio'] = df_new['up_final'] / (df_new['down_final'] + 1e-8)
        
        # 計算時間序列曲線斜率(使用前半段和後半段的差異)
        mid_point = len(up_cols) // 2
        if mid_point > 0:
            df_new['up_slope_change'] = ((df_new[up_cols[-1]] - df_new[up_cols[mid_point]]) / 
                                       (df_new[up_cols[mid_point]] - df_new[up_cols[0]] + 1e-8))
            df_new['down_slope_change'] = ((df_new[down_cols[-1]] - df_new[down_cols[mid_point]]) / 
                                          (df_new[down_cols[mid_point]] - df_new[down_cols[0]] + 1e-8))
    
    # 4. 基於物理模型的特徵
    # 估算非線性塑性應變能密度變化量(ΔW)
    # 簡化版本: 取上下界面最終值的加權平均
    if 'up_final' in df_new.columns and 'down_final' in df_new.columns:
        df_new['estimated_delta_w'] = 0.5 * df_new['up_final'] + 0.5 * df_new['down_final']
        
        # 基於物理模型(Nf=a*(ΔW)^b)計算估計壽命
        a_coefficient = 55.83
        b_coefficient = -2.259
        df_new['physics_estimated_nf'] = a_coefficient * (df_new['estimated_delta_w'] ** b_coefficient)
        
        # 計算實際與物理估計壽命的比例
        if TARGET_COLUMN in df_new.columns:
            df_new['nf_physics_ratio'] = df_new[TARGET_COLUMN] / (df_new['physics_estimated_nf'] + 1e-8)
    
    # 5. 交互項特徵
    df_new['Die_warpage_interaction'] = df_new['Die'] * df_new['Unit_warpage']
    df_new['stud_PCB_interaction'] = df_new['stud'] * df_new['PCB']
    
    # 記錄新增的特徵
    new_features = [col for col in df_new.columns if col not in df.columns]
    logger.info(f"已創建 {len(new_features)} 個衍生特徵: {new_features}")
    
    return df_new


def select_features(df, target_col=TARGET_COLUMN, top_n=None, method='mi'):
    """
    進行特徵選擇，選出最相關的特徵
    
    參數:
        df (pandas.DataFrame): 包含特徵和目標變數的數據框
        target_col (str): 目標列名
        top_n (int): 選擇前N個特徵，如果為None則返回所有特徵的重要性
        method (str): 特徵選擇方法，'mi'為互信息
        
    返回:
        tuple: (所選特徵列表, 特徵重要性字典)
    """
    if target_col not in df.columns:
        logger.warning(f"目標欄位 {target_col} 不在數據框中")
        return list(df.columns), {}
    
    # 排除時間序列列和目標列
    feature_cols = [col for col in df.columns 
                   if not col.startswith(tuple(TIME_SERIES_PREFIXES)) 
                   and col != target_col]
    
    # 計算特徵重要性
    if method == 'mi':
        # 使用互信息計算特徵重要性
        try:
            mi = mutual_info_regression(df[feature_cols], df[target_col])
            importance = dict(zip(feature_cols, mi))
            
            # 按重要性排序特徵
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            
            if top_n is not None:
                selected_features = [f[0] for f in sorted_features[:top_n]]
            else:
                selected_features = feature_cols
                
            logger.info(f"完成特徵選擇，選出 {len(selected_features)} 個特徵")
            return selected_features, importance
        except Exception as e:
            logger.error(f"特徵選擇失敗: {str(e)}")
            return feature_cols, {}
    else:
        logger.warning(f"不支援的特徵選擇方法: {method}")
        return feature_cols, {}


def apply_transformations(df, cols=None, target_col=TARGET_COLUMN, log_transform_target=True):
    """
    對特徵和目標變數應用變換
    
    參數:
        df (pandas.DataFrame): 數據框
        cols (list): 要變換的列，None表示所有數值列
        target_col (str): 目標列名
        log_transform_target (bool): 是否對目標變數應用對數變換
        
    返回:
        tuple: (變換後的數據框, 變換器字典)
    """
    df_transformed = df.copy()
    transformers = {}
    
    # 如果未指定列，選擇所有數值列
    if cols is None:
        cols = df.select_dtypes(include=['number']).columns.tolist()
        if target_col in cols:
            cols.remove(target_col)
    
    logger.info(f"對 {len(cols)} 個特徵應用變換")
    
    # 對目標變數應用對數變換
    if log_transform_target and target_col in df.columns:
        if (df[target_col] <= 0).any():
            logger.warning("目標變數含有小於等於0的值，使用log1p變換")
            df_transformed['log_' + target_col] = np.log1p(df_transformed[target_col])
        else:
            logger.info("對目標變數應用對數變換")
            df_transformed['log_' + target_col] = np.log(df_transformed[target_col])
    
    # 對特徵應用冪變換 (處理偏態分佈)
    power_cols = [col for col in cols 
                 if col not in [target_col] 
                 and not col.startswith(tuple(TIME_SERIES_PREFIXES))]
    
    if power_cols:
        try:
            power_transformer = PowerTransformer(method='yeo-johnson')
            df_transformed[power_cols] = power_transformer.fit_transform(df[power_cols])
            transformers['power_transformer'] = power_transformer
            logger.info(f"已對 {len(power_cols)} 個特徵應用冪變換")
        except Exception as e:
            logger.warning(f"應用冪變換時發生錯誤: {str(e)}")
    
    return df_transformed, transformers


def standardize_features(df, feature_cols=None, target_col=TARGET_COLUMN, scaler_type='robust'):
    """
    標準化特徵
    
    參數:
        df (pandas.DataFrame): 數據框
        feature_cols (list): 要標準化的特徵列
        target_col (str): 目標列名
        scaler_type (str): 標準化類型，'standard', 'robust'
        
    返回:
        tuple: (標準化特徵數組, 標準化器, 目標值數組)
    """
    if feature_cols is None:
        # 排除時間序列列和目標列
        feature_cols = [col for col in df.columns 
                       if not col.startswith(tuple(TIME_SERIES_PREFIXES)) 
                       and col != target_col
                       and col != 'log_' + target_col]
    
    # 選擇標準化器
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        logger.warning(f"不支援的標準化類型: {scaler_type}，使用RobustScaler")
        scaler = RobustScaler()
    
    # 標準化特徵
    X = df[feature_cols].values
    X_scaled = scaler.fit_transform(X)
    
    # 準備目標變數 (如果有對數變換版本，優先使用)
    if 'log_' + target_col in df.columns:
        y = df['log_' + target_col].values
    elif target_col in df.columns:
        y = df[target_col].values
    else:
        y = None
    
    logger.info(f"已標準化 {len(feature_cols)} 個特徵")
    
    return X_scaled, scaler, y, feature_cols


def prepare_time_series(df, normalize=True):
    """
    準備時間序列資料
    
    參數:
        df (pandas.DataFrame): 數據框
        normalize (bool): 是否標準化時間序列
        
    返回:
        numpy.ndarray: 形狀為(樣本數, 時間步數, 特徵數)的時間序列數組
    """
    # 找出所有時間序列列
    up_cols = [col for col in df.columns if col.startswith('NLPLWK_up_')]
    down_cols = [col for col in df.columns if col.startswith('NLPLWK_down_')]
    
    # 排序時間序列列
    up_cols.sort()
    down_cols.sort()
    
    # 檢查時間步數是否匹配
    if len(up_cols) != len(down_cols):
        logger.warning(f"上下界面時間步數不一致: 上={len(up_cols)}, 下={len(down_cols)}")
        # 截取相同步數
        min_steps = min(len(up_cols), len(down_cols))
        up_cols = up_cols[:min_steps]
        down_cols = down_cols[:min_steps]
    
    time_steps = len(up_cols)
    n_samples = len(df)
    
    # 創建時間序列數組 (樣本數, 時間步數, 特徵數)
    time_series_data = np.zeros((n_samples, time_steps, 2))
    
    # 填充時間序列數據
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
    
    logger.info(f"已準備時間序列資料: 形狀=({n_samples}, {time_steps}, 2)")
    
    return time_series_data


def augment_data(df, synthetic_samples=50, variations_per_sample=2, mix_factor=0.3, noise_level=0.08):
    """
    使用物理知識驅動的資料增強
    
    參數:
        df (pandas.DataFrame): 原始數據框
        synthetic_samples (int): 生成的合成樣本數量
        variations_per_sample (int): 每個原始樣本生成的變體數量
        mix_factor (float): 混合因子
        noise_level (float): 噪聲水平
        
    返回:
        pandas.DataFrame: 增強後的數據框
    """
    if data_augmentation_available:
        try:
            logger.info(f"使用專用資料增強模組，生成 {synthetic_samples} 個合成樣本")
            augmented_df = perform_comprehensive_augmentation(
                df, 
                synthetic_samples=synthetic_samples,
                perturbation_variations=variations_per_sample,
                mix_factor=mix_factor,
                noise_level=noise_level,
                validate_samples=True
            )
            return augmented_df
        except Exception as e:
            logger.error(f"使用專用資料增強模組失敗: {str(e)}，將使用基本資料增強")
            
    # 基本資料增強
    logger.info("使用基本資料增強方法")
    
    # 創建一個副本
    original_df = df.copy()
    
    # 添加標籤，識別原始資料
    original_df['is_synthetic'] = 0
    
    # 收集所有增強的樣本
    all_samples = [original_df]
    
    # 1. 參數微擾
    if variations_per_sample > 0:
        logger.info(f"生成每個樣本的 {variations_per_sample} 個微擾變體")
        for _ in range(variations_per_sample):
            # 複製原始數據
            perturbed_df = original_df.copy()
            perturbed_df['is_synthetic'] = 1
            
            # 對靜態特徵進行隨機微擾
            for col in STATIC_FEATURES:
                if col in perturbed_df.columns:
                    # 生成隨機擾動因子
                    perturbation = np.random.normal(1.0, noise_level, size=len(perturbed_df))
                    perturbed_df[col] = perturbed_df[col] * perturbation
            
            # 對時間序列特徵進行微擾
            for prefix in TIME_SERIES_PREFIXES:
                time_cols = [col for col in perturbed_df.columns if col.startswith(prefix)]
                for col in time_cols:
                    perturbation = np.random.normal(1.0, noise_level/2, size=len(perturbed_df))
                    perturbed_df[col] = perturbed_df[col] * perturbation
            
            # 調整目標變數
            if TARGET_COLUMN in perturbed_df.columns:
                # 基於物理知識的調整
                if 'estimated_delta_w' in perturbed_df.columns:
                    # 重新估計delta_w
                    up_cols = [col for col in perturbed_df.columns if col.startswith('NLPLWK_up_')]
                    down_cols = [col for col in perturbed_df.columns if col.startswith('NLPLWK_down_')]
                    if up_cols and down_cols:
                        up_cols.sort()
                        down_cols.sort()
                        perturbed_df['estimated_delta_w'] = 0.5 * perturbed_df[up_cols[-1]] + 0.5 * perturbed_df[down_cols[-1]]
                    
                    # 基於物理模型預測壽命
                    a_coefficient = 55.83
                    b_coefficient = -2.259
                    theoretical_nf = a_coefficient * (perturbed_df['estimated_delta_w'] ** b_coefficient)
                    
                    # 添加隨機變異
                    perturbed_df[TARGET_COLUMN] = theoretical_nf * np.random.lognormal(0, noise_level, size=len(perturbed_df))
                else:
                    # 簡單微擾
                    perturbed_df[TARGET_COLUMN] = perturbed_df[TARGET_COLUMN] * np.random.lognormal(0, noise_level, size=len(perturbed_df))
            
            all_samples.append(perturbed_df)
    
    # 2. 完全合成樣本 (如果需要)
    if synthetic_samples > 0:
        logger.info(f"生成 {synthetic_samples} 個完全合成樣本")
        
        # 計算特徵範圍
        feature_ranges = {}
        for col in STATIC_FEATURES:
            if col in original_df.columns:
                feature_ranges[col] = {
                    'min': original_df[col].min(),
                    'max': original_df[col].max(),
                    'mean': original_df[col].mean(),
                    'std': original_df[col].std()
                }
        
        # 生成合成樣本
        synthetic_samples_list = []
        for _ in range(synthetic_samples):
            sample = {}
            sample['is_synthetic'] = 1
            
            # 生成靜態特徵
            for col, ranges in feature_ranges.items():
                # 使用截斷正態分佈生成在有效範圍內的值
                while True:
                    value = np.random.normal(ranges['mean'], ranges['std'] * 1.2)
                    if ranges['min'] * 0.9 <= value <= ranges['max'] * 1.1:
                        break
                sample[col] = value
            
            # 生成時間序列
            # 首先找到典型的時間步數
            up_cols = [col for col in original_df.columns if col.startswith('NLPLWK_up_')]
            down_cols = [col for col in original_df.columns if col.startswith('NLPLWK_down_')]
            
            if up_cols and down_cols:
                up_cols.sort()
                down_cols.sort()
                
                # 計算初始delta_w (基於物理知識)
                # 這裡使用簡化的物理模型估計delta_w
                effective_strain = (
                    5e-6 * (sample['Unit_warpage'] / sample['PCB']) * 
                    (sample['Die'] / (sample['stud'] + sample['mold']))
                )
                
                # 估計delta_w
                delta_w = effective_strain**2 * np.random.uniform(0.5, 1.5)
                
                # 生成時間序列
                up_ratio = np.array([0.2, 0.5, 0.8, 1.0])  # 上界面累積比例
                down_ratio = np.array([0.15, 0.45, 0.75, 1.0])  # 下界面累積比例
                
                for i, (up_col, down_col) in enumerate(zip(up_cols, down_cols)):
                    # 生成上界面數據
                    up_value = delta_w * up_ratio[i % len(up_ratio)] * np.random.uniform(0.9, 1.1)
                    sample[up_col] = up_value
                    
                    # 生成下界面數據
                    down_value = delta_w * down_ratio[i % len(down_ratio)] * np.random.uniform(0.9, 1.1)
                    sample[down_col] = down_value
                
                # 估計疲勞壽命
                if i == len(up_cols) - 1:  # 使用最後時間步的值
                    a_coefficient = 55.83
                    b_coefficient = -2.259
                    estimated_delta_w = 0.5 * up_value + 0.5 * down_value
                    nf = a_coefficient * (estimated_delta_w ** b_coefficient)
                    
                    # 加入隨機變異
                    sample[TARGET_COLUMN] = nf * np.random.lognormal(0, noise_level)
            
            synthetic_samples_list.append(sample)
        
        # 創建合成樣本的數據框
        if synthetic_samples_list:
            synthetic_df = pd.DataFrame(synthetic_samples_list)
            
            # 確保所有與原始數據框相同的列
            for col in original_df.columns:
                if col not in synthetic_df.columns:
                    if col in original_df.select_dtypes(include=['number']).columns:
                        # 對於數值列，用平均值填充
                        synthetic_df[col] = original_df[col].mean()
                    else:
                        # 對於非數值列，用最常見值填充
                        synthetic_df[col] = original_df[col].mode()[0]
            
            all_samples.append(synthetic_df)
    
    # 合併所有樣本
    augmented_df = pd.concat(all_samples, ignore_index=True)
    
    # 基本清理
    # 移除明顯不合理的值
    if TARGET_COLUMN in augmented_df.columns:
        target_min = original_df[TARGET_COLUMN].min() * 0.5
        target_max = original_df[TARGET_COLUMN].max() * 2.0
        
        invalid_mask = (augmented_df[TARGET_COLUMN] < target_min) | (augmented_df[TARGET_COLUMN] > target_max)
        if sum(invalid_mask) > 0:
            logger.info(f"移除 {sum(invalid_mask)} 筆不合理的合成樣本")
            augmented_df = augmented_df[~invalid_mask]
    
    logger.info(f"資料增強完成，原始樣本數: {len(original_df)}，增強後樣本數: {len(augmented_df)}")
    
    return augmented_df


def train_val_test_split_stratified(X, time_series, y, test_size=0.15, val_size=0.15, random_state=42):
    """
    進行分層抽樣的訓練、驗證、測試集分割
    
    參數:
        X (numpy.ndarray): 特徵數組
        time_series (numpy.ndarray): 時間序列數組
        y (numpy.ndarray): 目標數組
        test_size (float): 測試集比例
        val_size (float): 驗證集比例
        random_state (int): 隨機種子
        
    返回:
        dict: 包含分割後數據的字典
    """
    # 創建分層抽樣的標籤
    # 對於連續目標變數，將其分為幾個區間
    if len(y) < 100:  # 小樣本數據集
        n_bins = min(5, len(y) // 5)  # 確保每個區間至少有5個樣本
    else:
        n_bins = 10
    
    # 使用分位數創建均勻的區間
    bins = np.percentile(y, np.linspace(0, 100, n_bins + 1))
    
    # 創建分層標籤
    strata = np.digitize(y, bins)
    
    logger.info(f"使用分層抽樣進行數據分割，創建了 {n_bins} 個層")
    
    # 首先分割出測試集
    X_temp, X_test, time_series_temp, time_series_test, y_temp, y_test, strata_temp = train_test_split(
        X, time_series, y, strata, test_size=test_size, random_state=random_state, stratify=strata
    )
    
    # 再從剩餘數據中分割出驗證集
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, time_series_train, time_series_val, y_train, y_val = train_test_split(
        X_temp, time_series_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=strata_temp
    )
    
    logger.info(f"數據分割完成: 訓練集 {len(y_train)} 樣本, 驗證集 {len(y_val)} 樣本, 測試集 {len(y_test)} 樣本")
    
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
                    log_transform=True, augmentation=True, feature_engineering=True,
                    top_n_features=None, synthetic_samples=50):
    """
    完整的數據處理管道
    
    參數:
        data_filepath (str): 數據文件路徑
        test_size (float): 測試集比例
        val_size (float): 驗證集比例
        random_state (int): 隨機種子
        log_transform (bool): 是否對目標變數應用對數變換
        augmentation (bool): 是否應用資料增強
        feature_engineering (bool): 是否進行特徵工程
        top_n_features (int): 選擇的特徵數量
        synthetic_samples (int): 合成樣本數量
        
    返回:
        dict: 包含處理後數據和元數據的字典
    """
    # 1. 載入數據
    df = load_data(data_filepath)
    original_df = df.copy()
    
    # 2. 特徵工程
    if feature_engineering:
        df = create_derived_features(df)
    
    # 3. 應用變換
    df_transformed, transformers = apply_transformations(df, target_col=TARGET_COLUMN, log_transform_target=log_transform)
    
    # 4. 資料增強
    if augmentation and len(df) < 200:  # 只對小樣本數據集進行增強
        df_augmented = augment_data(
            df_transformed, 
            synthetic_samples=synthetic_samples,
            variations_per_sample=2
        )
    else:
        df_augmented = df_transformed
    
    # 5. 特徵選擇
    target_col = 'log_' + TARGET_COLUMN if log_transform and 'log_' + TARGET_COLUMN in df_augmented.columns else TARGET_COLUMN
    selected_features, feature_importance = select_features(df_augmented, target_col=target_col, top_n=top_n_features)
    
    # 6. 標準化特徵
    X_scaled, scaler, y, feature_cols = standardize_features(
        df_augmented, 
        feature_cols=selected_features, 
        target_col=TARGET_COLUMN,
        scaler_type='robust'
    )
    
    # 7. 準備時間序列數據
    time_series_data = prepare_time_series(df_augmented, normalize=True)
    
    # 8. 分割數據集
    split_data = train_val_test_split_stratified(
        X_scaled, time_series_data, y, 
        test_size=test_size, val_size=val_size, random_state=random_state
    )
    
    # 9. 組裝返回結果
    result = {
        # 原始資料
        'original_df': original_df,
        'processed_df': df_augmented,
        
        # 特徵信息
        'feature_cols': feature_cols,
        'feature_importance': feature_importance,
        'target_column': TARGET_COLUMN,
        'log_transform': log_transform,
        
        # 變換器和縮放器
        'transformers': transformers,
        'scaler': scaler,
        
        # 完整數據
        'X_all': X_scaled,
        'time_series_all': time_series_data,
        'y_all': y,
        
        # 分割數據
        'X_train': split_data['X_train'],
        'X_val': split_data['X_val'],
        'X_test': split_data['X_test'],
        'time_series_train': split_data['time_series_train'],
        'time_series_val': split_data['time_series_val'],
        'time_series_test': split_data['time_series_test'],
        'y_train': split_data['y_train'],
        'y_val': split_data['y_val'],
        'y_test': split_data['y_test']
    }
    
    # 產生資料摘要
    logger.info("\n" + "="*50)
    logger.info("資料處理摘要:")
    logger.info(f"原始樣本數: {len(original_df)}")
    logger.info(f"處理後樣本數: {len(df_augmented)}")
    logger.info(f"所選特徵數: {len(feature_cols)}")
    logger.info(f"訓練集: {len(split_data['y_train'])} 樣本")
    logger.info(f"驗證集: {len(split_data['y_val'])} 樣本")
    logger.info(f"測試集: {len(split_data['y_test'])} 樣本")
    logger.info("="*50)
    
    return result


def analyze_feature_importance(feature_importance, save_path=None, figsize=(12, 8)):
    """
    分析並視覺化特徵重要性
    
    參數:
        feature_importance (dict): 特徵重要性字典
        save_path (str): 圖表保存路徑
        figsize (tuple): 圖表尺寸
    
    返回:
        matplotlib.figure.Figure: 圖表對象
    """
    # 將特徵重要性轉換為數據框
    importance_df = pd.DataFrame({
        'feature': list(feature_importance.keys()),
        'importance': list(feature_importance.values())
    })
    
    # 按重要性排序
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # 繪製特徵重要性條形圖
    fig, ax = plt.subplots(figsize=figsize)
    
    # 使用漸變色繪製條形圖
    bars = ax.barh(importance_df['feature'], importance_df['importance'],
                  color=plt.cm.viridis(np.linspace(0, 0.8, len(importance_df))))
    
    # 添加標題和標籤
    ax.set_title('特徵重要性分析', fontsize=15)
    ax.set_xlabel('重要性分數', fontsize=12)
    ax.set_ylabel('特徵', fontsize=12)
    
    # 添加數值標籤
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.01 * max(importance_df['importance']), 
               bar.get_y() + bar.get_height()/2, 
               f'{width:.4f}', 
               ha='left', va='center', fontsize=9)
    
    # 添加網格線
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    
    # 反轉y軸，使最重要的特徵顯示在頂部
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    # 保存圖表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"特徵重要性圖表已保存至: {save_path}")
    
    return fig


def analyze_target_distribution(df, target_col=TARGET_COLUMN, log_scale=True, save_path=None, figsize=(12, 6)):
    """
    分析目標變數分佈
    
    參數:
        df (pandas.DataFrame): 數據框
        target_col (str): 目標列名
        log_scale (bool): 是否使用對數刻度
        save_path (str): 圖表保存路徑
        figsize (tuple): 圖表尺寸
    
    返回:
        matplotlib.figure.Figure: 圖表對象
    """
    if target_col not in df.columns:
        logger.warning(f"目標列 {target_col} 不在數據框中")
        return None
    
    # 創建圖表
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 繪製直方圖
    axes[0].hist(df[target_col], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].set_title(f'{target_col} 分佈')
    axes[0].set_xlabel(target_col)
    axes[0].set_ylabel('頻率')
    
    if log_scale and (df[target_col] > 0).all():
        axes[0].set_xscale('log')
    
    # 繪製箱形圖
    axes[1].boxplot(df[target_col], vert=False, widths=0.7)
    axes[1].set_title(f'{target_col} 箱形圖')
    axes[1].set_xlabel(target_col)
    
    if log_scale and (df[target_col] > 0).all():
        axes[1].set_xscale('log')
    
    # 添加統計摘要
    stats = df[target_col].describe()
    stat_text = (
        f"統計摘要:\n"
        f"平均值: {stats['mean']:.2f}\n"
        f"標準差: {stats['std']:.2f}\n"
        f"最小值: {stats['min']:.2f}\n"
        f"25%分位: {stats['25%']:.2f}\n"
        f"中位數: {stats['50%']:.2f}\n"
        f"75%分位: {stats['75%']:.2f}\n"
        f"最大值: {stats['max']:.2f}"
    )
    
    plt.figtext(0.5, 0.01, stat_text, ha='center', 
               bbox=dict(facecolor='wheat', alpha=0.5), fontsize=10)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # 保存圖表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"目標分佈圖表已保存至: {save_path}")
    
    return fig


def create_dataloaders(X_train, X_val, X_test, 
                      time_series_train, time_series_val, time_series_test,
                      y_train, y_val, y_test,
                      batch_size=8, num_workers=0, pin_memory=False):
    """
    創建PyTorch數據載入器
    
    參數:
        X_train, X_val, X_test (numpy.ndarray): 訓練、驗證和測試特徵
        time_series_train, time_series_val, time_series_test (numpy.ndarray): 時間序列數據
        y_train, y_val, y_test (numpy.ndarray): 目標變數
        batch_size (int): 批次大小
        num_workers (int): 數據載入線程數
        pin_memory (bool): 是否鎖定內存
        
    返回:
        dict: 包含數據載入器的字典
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    
    # 轉換為PyTorch張量
    X_train_tensor = torch.FloatTensor(X_train)
    X_val_tensor = torch.FloatTensor(X_val)
    X_test_tensor = torch.FloatTensor(X_test)
    
    time_series_train_tensor = torch.FloatTensor(time_series_train)
    time_series_val_tensor = torch.FloatTensor(time_series_val)
    time_series_test_tensor = torch.FloatTensor(time_series_test)
    
    y_train_tensor = torch.FloatTensor(y_train)
    y_val_tensor = torch.FloatTensor(y_val)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # 創建數據集
    train_dataset = TensorDataset(X_train_tensor, time_series_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, time_series_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, time_series_test_tensor, y_test_tensor)
    
    # 創建數據載入器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    logger.info(f"已創建數據載入器 - 批次大小: {batch_size}")
    logger.info(f"訓練集: {len(train_dataset)} 樣本, {len(train_loader)} 批次")
    logger.info(f"驗證集: {len(val_dataset)} 樣本, {len(val_loader)} 批次")
    logger.info(f"測試集: {len(test_dataset)} 樣本, {len(test_loader)} 批次")
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader
    }


if __name__ == "__main__":
    # 測試代碼
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 資料路徑
    data_path = os.path.join(project_root, "data/raw/Training_data_warpage_final_20250321_v1.2.csv")
    
    if os.path.exists(data_path):
        # 測試數據處理管道
        logger.info("開始測試資料處理管道")
        
        result = process_pipeline(
            data_path,
            test_size=0.15,
            val_size=0.15,
            random_state=42,
            log_transform=True,
            augmentation=True,
            feature_engineering=True,
            synthetic_samples=40
        )
        
        # 測試特徵重要性分析
        if 'feature_importance' in result:
            logger.info("測試特徵重要性分析")
            fig = analyze_feature_importance(result['feature_importance'])
            plt.show()
        
        # 測試目標分佈分析
        logger.info("測試目標分佈分析")
        fig = analyze_target_distribution(result['original_df'])
        plt.show()
        
        # 測試創建數據載入器
        logger.info("測試創建數據載入器")
        try:
            dataloaders = create_dataloaders(
                result['X_train'], result['X_val'], result['X_test'],
                result['time_series_train'], result['time_series_val'], result['time_series_test'],
                result['y_train'], result['y_val'], result['y_test'],
                batch_size=8
            )
            logger.info("數據載入器創建成功")
        except Exception as e:
            logger.error(f"創建數據載入器失敗: {str(e)}")
        
        logger.info("測試完成")
    else:
        logger.error(f"找不到資料檔案: {data_path}")