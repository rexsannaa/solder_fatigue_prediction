#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
data_augmentation.py - 物理知識驅動的資料增強模組
本模組提供基於物理知識的資料增強功能，用於擴充銲錫接點疲勞壽命預測的小樣本資料集，
生成合成數據以提升模型訓練效果。

主要功能:
1. 基於物理模型生成合成樣本
2. 參數微擾生成相似樣本
3. 物理知識引導的合成時間序列生成
4. 物理約束驗證功能
"""

import numpy as np
import pandas as pd
import logging
from scipy.stats import norm, uniform, lognorm
from collections import defaultdict

logger = logging.getLogger(__name__)

# 物理模型常數
A_COEFFICIENT = 55.83
B_COEFFICIENT = -2.259

def delta_w_to_nf(delta_w, a=A_COEFFICIENT, b=B_COEFFICIENT):
    """
    根據非線性塑性應變能密度變化量(ΔW)計算疲勞壽命(Nf)
    
    參數:
        delta_w (float or array): 非線性塑性應變能密度變化量
        a (float): 係數a
        b (float): 係數b
        
    返回:
        float or array: 計算的疲勞壽命
    """
    delta_w = np.maximum(np.asarray(delta_w), 1e-10)
    nf = a * np.power(delta_w, b)
    return nf

def nf_to_delta_w(nf, a=A_COEFFICIENT, b=B_COEFFICIENT):
    """
    根據疲勞壽命(Nf)計算非線性塑性應變能密度變化量(ΔW)
    
    參數:
        nf (float or array): 疲勞壽命
        a (float): 係數a
        b (float): 係數b
        
    返回:
        float or array: 計算的非線性塑性應變能密度變化量
    """
    nf = np.maximum(np.asarray(nf), 1e-10)
    delta_w = np.power(nf / a, 1 / b)
    return delta_w

def calculate_effective_strain(structure_params):
    """
    計算有效應變
    根據結構參數估算有效應變
    
    參數:
        structure_params (dict): 結構參數，包含Die, stud, mold, PCB等
        
    返回:
        float: 估計的有效應變
    """
    # 提取結構參數
    die = structure_params.get('Die', 200)  # 晶片高度 (μm)
    stud = structure_params.get('stud', 70)  # 銅高度 (μm)
    mold = structure_params.get('mold', 65)  # 環氧樹脂 (μm)
    pcb = structure_params.get('PCB', 0.8)   # PCB厚度 (mm)
    
    # 材料的熱膨脹係數 (CTE, ppm/°C)
    cte_die = 2.8    # 矽晶片
    cte_stud = 17.0  # 銅柱
    cte_mold = 12.0  # 環氧樹脂
    cte_pcb = 16.0   # PCB基板
    
    # 將PCB厚度轉換為微米，使單位一致
    pcb_um = pcb * 1000
    
    # 溫度循環範圍（140°C到-40°C）
    temp_range = 180.0
    
    # 計算CTE失配導致的應變
    strain_die_pcb = temp_range * abs(cte_die - cte_pcb) * 1e-6
    strain_mold_pcb = temp_range * abs(cte_mold - cte_pcb) * 1e-6
    
    # 考慮結構尺寸對應變分佈的影響
    # 銲錫高度通常是限制應變的關鍵因素
    solder_height = stud * 0.5  # 假設銲錫高度約為銅柱高度的一半
    
    # 幾何因子：Die尺寸與PCB厚度比、環氧樹脂厚度與整體厚度比
    die_factor = die / (die + mold + pcb_um)
    mold_factor = mold / (die + mold + pcb_um)
    
    # 計算加權有效應變
    effective_strain = (strain_die_pcb * die_factor + strain_mold_pcb * mold_factor) * \
                      (100 / solder_height)**0.3  # 應變隨銲錫高度降低而增加
    
    # 添加翹曲變形的影響
    warpage = structure_params.get('Unit_warpage', 10.0)
    warpage_factor = 1.0 + 0.005 * warpage  # 假設每增加1單位的翹曲變形，應變增加0.5%
    
    effective_strain = effective_strain * warpage_factor
    
    return effective_strain

def generate_time_series(delta_w, noise_level=0.05, time_points=4):
    """
    根據delta_w生成時間序列數據
    
    參數:
        delta_w (float): 非線性塑性應變能密度變化量
        noise_level (float): 噪聲水平
        time_points (int): 時間點數量
        
    返回:
        tuple: (上界面時間序列, 下界面時間序列)
    """
    # 定義累積比例(基於物理模型)
    up_ratio = np.array([0.2, 0.5, 0.8, 1.0])  # 上界面累積比例
    down_ratio = np.array([0.15, 0.45, 0.75, 1.0])  # 下界面累積比例
    
    # 生成上界面時間序列
    up_series = delta_w * up_ratio
    
    # 添加噪聲
    noise_up = np.random.normal(0, noise_level * up_series, size=time_points)
    up_series = up_series + noise_up
    
    # 確保單調增加
    up_series = np.maximum.accumulate(up_series)
    
    # 生成下界面時間序列
    down_series = delta_w * down_ratio
    
    # 添加噪聲
    noise_down = np.random.normal(0, noise_level * down_series, size=time_points)
    down_series = down_series + noise_down
    
    # 確保單調增加
    down_series = np.maximum.accumulate(down_series)
    
    return up_series, down_series

def generate_synthetic_samples(original_df, n_samples=100, noise_level=0.1, validate_samples=True):
    """
    生成合成樣本
    
    參數:
        original_df (pd.DataFrame): 原始數據
        n_samples (int): 生成的樣本數量
        noise_level (float): 噪聲水平
        validate_samples (bool): 是否驗證生成的樣本
        
    返回:
        pd.DataFrame: 包含生成樣本的數據框
    """
    # 提取原始數據的結構參數範圍
    param_ranges = {}
    for col in ['Die', 'stud', 'mold', 'PCB', 'Unit_warpage']:
        if col in original_df.columns:
            param_ranges[col] = {
                'min': original_df[col].min(),
                'max': original_df[col].max(),
                'mean': original_df[col].mean(),
                'std': original_df[col].std()
            }
    
    # 提取原始數據的目標值範圍
    target_col = 'Nf_pred (cycles)'
    target_range = {
        'min': original_df[target_col].min(),
        'max': original_df[target_col].max(),
        'mean': original_df[target_col].mean(),
        'std': original_df[target_col].std(),
        'log_mean': np.log(original_df[target_col] + 1e-10).mean(),
        'log_std': np.log(original_df[target_col] + 1e-10).std()
    }
    
    # 初始化合成數據
    synthetic_data = []
    valid_count = 0
    
    # 生成合成樣本
    attempts = 0
    max_attempts = n_samples * 10  # 限制嘗試次數
    
    while valid_count < n_samples and attempts < max_attempts:
        attempts += 1
        
        # 1. 生成結構參數
        sample = {}
        for param, ranges in param_ranges.items():
            # 使用截斷正態分佈生成參數
            mean = ranges['mean']
            std = ranges['std'] * (1 + noise_level)  # 稍微增加標準差以提高多樣性
            min_val = ranges['min'] * 0.95  # 允許比原始範圍稍小
            max_val = ranges['max'] * 1.05  # 允許比原始範圍稍大
            
            # 生成參數值直到它在有效範圍內
            while True:
                value = np.random.normal(mean, std)
                if min_val <= value <= max_val:
                    break
            
            sample[param] = value
        
        # 2. 計算有效應變
        effective_strain = calculate_effective_strain(sample)
        
        # 3. 計算delta_w (基於物理模型與一些隨機變異)
        # 添加隨機變異模擬實際情況的複雜性
        strain_variation = np.random.uniform(0.8, 1.2)
        delta_w = effective_strain**2 * strain_variation * np.random.uniform(1.0, 2.0)
        
        # 4. 根據物理模型計算疲勞壽命
        nf = delta_w_to_nf(delta_w)
        
        # 添加一些隨機變異以反映模型不確定性
        nf_variation = np.random.lognormal(0, noise_level)
        nf = nf * nf_variation
        
        # 5. 生成時間序列數據
        up_series, down_series = generate_time_series(delta_w, noise_level)
        
        # 6. 驗證生成的樣本是否合理
        is_valid = True
        if validate_samples:
            # 檢查nf是否在合理範圍內
            if nf < target_range['min'] * 0.5 or nf > target_range['max'] * 2.0:
                is_valid = False
            
            # 檢查delta_w和nf是否遵守物理關係
            nf_from_physics = delta_w_to_nf(delta_w)
            relative_error = abs((nf - nf_from_physics) / nf_from_physics)
            if relative_error > 0.3:  # 允許30%的相對誤差
                is_valid = False
        
        # 7. 如果樣本有效，添加到合成數據中
        if is_valid:
            sample_data = sample.copy()
            sample_data[target_col] = nf
            
            # 添加時間序列數據
            for i, (up_val, down_val) in enumerate(zip(up_series, down_series)):
                time_point = (i + 1) * 3600
                sample_data[f'NLPLWK_up_{time_point}'] = up_val
                sample_data[f'NLPLWK_down_{time_point}'] = down_val
            
            synthetic_data.append(sample_data)
            valid_count += 1
            
            if valid_count % 10 == 0:
                logger.debug(f"已生成 {valid_count}/{n_samples} 個有效樣本")
    
    if valid_count < n_samples:
        logger.warning(f"僅生成了 {valid_count}/{n_samples} 個有效樣本，達到最大嘗試次數 {max_attempts}")
    
    # 將合成數據轉換為DataFrame
    synthetic_df = pd.DataFrame(synthetic_data)
    
    # 確保列順序與原始數據相同
    if len(original_df.columns) == len(synthetic_df.columns):
        synthetic_df = synthetic_df[original_df.columns]
    
    return synthetic_df

def apply_parameter_perturbation(df, perturbation_factor=0.05, n_variations=2):
    """
    對原始數據進行參數微擾以生成更多樣本
    
    參數:
        df (pd.DataFrame): 原始數據
        perturbation_factor (float): 微擾因子
        n_variations (int): 每個原始樣本生成的變體數量
        
    返回:
        pd.DataFrame: 包含原始和變體樣本的數據框
    """
    # 提取結構參數列
    structure_cols = ['Die', 'stud', 'mold', 'PCB', 'Unit_warpage']
    
    # 提取時間序列列
    time_series_cols = [col for col in df.columns if 'NLPLWK' in col]
    
    # 目標列
    target_col = 'Nf_pred (cycles)'
    
    # 創建新的數據框以保存原始和變體樣本
    all_samples = [df.copy()]
    
    # 對每個原始樣本生成變體
    for i in range(n_variations):
        variation_df = df.copy()
        
        # 對結構參數進行微擾
        for col in structure_cols:
            if col in df.columns:
                # 生成微擾因子
                perturbation = np.random.normal(1.0, perturbation_factor, size=len(df))
                variation_df[col] = df[col] * perturbation
        
        # 根據微擾後的結構參數重新計算物理量
        for idx, row in variation_df.iterrows():
            # 提取結構參數
            structure_params = {col: row[col] for col in structure_cols if col in df.columns}
            
            # 計算有效應變
            effective_strain = calculate_effective_strain(structure_params)
            
            # 計算delta_w
            strain_variation = np.random.uniform(0.95, 1.05)
            delta_w = effective_strain**2 * strain_variation * np.random.uniform(0.95, 1.05)
            
            # 根據物理模型計算疲勞壽命
            nf = delta_w_to_nf(delta_w)
            
            # 添加小隨機變異
            nf_variation = np.random.lognormal(0, 0.05)
            nf = nf * nf_variation
            
            # 更新目標值
            variation_df.at[idx, target_col] = nf
            
            # 生成時間序列數據
            up_series, down_series = generate_time_series(delta_w, noise_level=0.03)
            
            # 更新時間序列數據
            for j, (up_val, down_val) in enumerate(zip(up_series, down_series)):
                time_point = (j + 1) * 3600
                variation_df.at[idx, f'NLPLWK_up_{time_point}'] = up_val
                variation_df.at[idx, f'NLPLWK_down_{time_point}'] = down_val
        
        all_samples.append(variation_df)
    
    # 合併原始和變體樣本
    augmented_df = pd.concat(all_samples, ignore_index=True)
    
    return augmented_df

def create_mixed_sample(df, mix_factor=0.5):
    """
    創建混合樣本，對現有樣本的特徵進行混合
    
    參數:
        df (pd.DataFrame): 原始數據
        mix_factor (float): 混合因子，控制混合程度
        
    返回:
        pd.DataFrame: 包含混合樣本的數據框
    """
    n_samples = len(df)
    n_mixed = min(n_samples * 2, 100)  # 限制混合樣本數量
    
    # 提取結構參數列
    structure_cols = ['Die', 'stud', 'mold', 'PCB', 'Unit_warpage']
    
    # 提取時間序列列
    time_series_cols = [col for col in df.columns if 'NLPLWK' in col]
    
    # 目標列
    target_col = 'Nf_pred (cycles)'
    
    # 創建混合樣本
    mixed_samples = []
    
    for _ in range(n_mixed):
        # 隨機選擇兩個樣本
        idx1, idx2 = np.random.choice(n_samples, 2, replace=False)
        sample1 = df.iloc[idx1].to_dict()
        sample2 = df.iloc[idx2].to_dict()
        
        # 混合結構參數
        mixed_sample = {}
        for col in structure_cols:
            if col in df.columns:
                # 使用加權平均進行混合
                weight = np.random.uniform(mix_factor, 1.0 - mix_factor)
                mixed_sample[col] = weight * sample1[col] + (1.0 - weight) * sample2[col]
        
        # 計算有效應變
        effective_strain = calculate_effective_strain(mixed_sample)
        
        # 計算delta_w
        strain_variation = np.random.uniform(0.9, 1.1)
        delta_w = effective_strain**2 * strain_variation * np.random.uniform(0.95, 1.05)
        
        # 根據物理模型計算疲勞壽命
        nf = delta_w_to_nf(delta_w)
        
        # 添加小隨機變異
        nf_variation = np.random.lognormal(0, 0.05)
        mixed_sample[target_col] = nf * nf_variation
        
        # 生成時間序列數據
        up_series, down_series = generate_time_series(delta_w, noise_level=0.03)
        
        # 添加時間序列數據
        for j, (up_val, down_val) in enumerate(zip(up_series, down_series)):
            time_point = (j + 1) * 3600
            mixed_sample[f'NLPLWK_up_{time_point}'] = up_val
            mixed_sample[f'NLPLWK_down_{time_point}'] = down_val
        
        mixed_samples.append(mixed_sample)
    
    # 創建混合樣本的DataFrame
    mixed_df = pd.DataFrame(mixed_samples)
    
    # 確保列順序與原始數據相同
    if len(df.columns) == len(mixed_df.columns):
        mixed_df = mixed_df[df.columns]
    
    return mixed_df

def perform_comprehensive_augmentation(original_df, synthetic_samples=50, perturbation_variations=1, 
                                        mix_factor=0.3, noise_level=0.1, validate_samples=True):
    """
    執行綜合資料增強
    
    參數:
        original_df (pd.DataFrame): 原始數據
        synthetic_samples (int): 生成的合成樣本數量
        perturbation_variations (int): 每個原始樣本生成的變體數量
        mix_factor (float): 混合因子
        noise_level (float): 噪聲水平
        validate_samples (bool): 是否驗證生成的樣本
        
    返回:
        pd.DataFrame: 增強後的數據框
    """
    logger.info(f"開始綜合資料增強，原始數據大小: {len(original_df)}")
    
    # 1. 生成合成樣本
    if synthetic_samples > 0:
        logger.info(f"生成 {synthetic_samples} 個合成樣本...")
        synthetic_df = generate_synthetic_samples(
            original_df, 
            n_samples=synthetic_samples, 
            noise_level=noise_level, 
            validate_samples=validate_samples
        )
        logger.info(f"成功生成 {len(synthetic_df)} 個合成樣本")
    else:
        synthetic_df = pd.DataFrame(columns=original_df.columns)
    
    # 2. 應用參數微擾
    if perturbation_variations > 0:
        logger.info(f"對原始數據應用參數微擾，每個樣本生成 {perturbation_variations} 個變體...")
        perturbed_df = apply_parameter_perturbation(
            original_df, 
            perturbation_factor=noise_level/2, 
            n_variations=perturbation_variations
        )
        logger.info(f"成功生成 {len(perturbed_df) - len(original_df)} 個微擾變體")
    else:
        perturbed_df = original_df.copy()
    
    # 3. 創建混合樣本
    if mix_factor > 0:
        logger.info("創建混合樣本...")
        mixed_df = create_mixed_sample(
            original_df, 
            mix_factor=mix_factor
        )
        logger.info(f"成功生成 {len(mixed_df)} 個混合樣本")
    else:
        mixed_df = pd.DataFrame(columns=original_df.columns)
    
    # 4. 合併所有增強數據
    all_df = pd.concat([perturbed_df, synthetic_df, mixed_df], ignore_index=True)
    
    # 5. 驗證和清理
    if validate_samples:
        logger.info("驗證和清理增強後的數據...")
        # 移除重複行
        all_df = all_df.drop_duplicates(subset=original_df.columns)
        
        # 移除不合理的樣本
        target_col = 'Nf_pred (cycles)'
        target_min = original_df[target_col].min() * 0.5
        target_max = original_df[target_col].max() * 2.0
        
        invalid_mask = (all_df[target_col] < target_min) | (all_df[target_col] > target_max)
        if invalid_mask.sum() > 0:
            logger.info(f"移除 {invalid_mask.sum()} 個不合理的樣本")
            all_df = all_df[~invalid_mask]
    
    logger.info(f"資料增強完成，最終數據大小: {len(all_df)}")
    
    return all_df

if __name__ == "__main__":
    # 簡單的測試代碼
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 創建示例數據
    np.random.seed(42)
    n_samples = 10
    
    data = {
        'Die': np.random.uniform(150, 250, n_samples),
        'stud': np.random.uniform(60, 80, n_samples),
        'mold': np.random.uniform(55, 75, n_samples),
        'PCB': np.random.uniform(0.6, 1.0, n_samples),
        'Unit_warpage': np.random.uniform(5, 15, n_samples),
        'Nf_pred (cycles)': np.random.uniform(500, 3000, n_samples)
    }
    
    # 添加時間序列數據
    for i in range(1, 5):
        time_point = i * 3600
        data[f'NLPLWK_up_{time_point}'] = np.random.uniform(0.001, 0.02, n_samples) * i / 4
        data[f'NLPLWK_down_{time_point}'] = np.random.uniform(0.001, 0.02, n_samples) * i / 4
    
    df = pd.DataFrame(data)
    
    # 測試資料增強
    logger.info(f"原始數據大小: {len(df)}")
    
    augmented_df = perform_comprehensive_augmentation(
        df, 
        synthetic_samples=20, 
        perturbation_variations=2, 
        mix_factor=0.3, 
        noise_level=0.1
    )