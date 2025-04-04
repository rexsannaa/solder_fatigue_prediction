#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
metrics.py - 評估指標模組
本模組提供了用於評估銲錫接點疲勞壽命預測模型性能的各種指標函數，
包括均方根誤差、決定係數、平均絕對誤差、相對誤差等指標。

主要功能:
1. 計算常見的迴歸模型評估指標
2. 提供綜合評估函數
3. 支援批次評估和結果對比
4. 處理對數尺度上的評估需求
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, 
    r2_score, 
    mean_absolute_error, 
    mean_absolute_percentage_error
)
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


def calculate_rmse(y_true, y_pred, sample_weight=None):
    """
    計算均方根誤差 (Root Mean Squared Error)
    
    參數:
        y_true (array-like): 真實值
        y_pred (array-like): 預測值
        sample_weight (array-like, optional): 樣本權重
        
    返回:
        float: RMSE值
    """
    return np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight))


def calculate_r2(y_true, y_pred, sample_weight=None):
    """
    計算決定係數 (R-squared)
    
    參數:
        y_true (array-like): 真實值
        y_pred (array-like): 預測值
        sample_weight (array-like, optional): 樣本權重
        
    返回:
        float: R²值
    """
    return r2_score(y_true, y_pred, sample_weight=sample_weight)


def calculate_mae(y_true, y_pred, sample_weight=None):
    """
    計算平均絕對誤差 (Mean Absolute Error)
    
    參數:
        y_true (array-like): 真實值
        y_pred (array-like): 預測值
        sample_weight (array-like, optional): 樣本權重
        
    返回:
        float: MAE值
    """
    return mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)


def calculate_mape(y_true, y_pred, sample_weight=None):
    """
    計算平均絕對百分比誤差 (Mean Absolute Percentage Error)
    
    參數:
        y_true (array-like): 真實值
        y_pred (array-like): 預測值
        sample_weight (array-like, optional): 樣本權重
        
    返回:
        float: MAPE值 (百分比)
    """
    # 避免除以零
    y_true_safe = np.maximum(np.abs(y_true), 1e-8)
    
    try:
        # 使用sklearn的MAPE實現（較新版本）
        return mean_absolute_percentage_error(y_true, y_pred) * 100
    except (AttributeError, ImportError):
        # 手動實現MAPE（較舊版本相容）
        mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
        return mape


def calculate_relative_error(y_true, y_pred, percentile=None):
    """
    計算相對誤差及其分佈特性
    
    參數:
        y_true (array-like): 真實值
        y_pred (array-like): 預測值
        percentile (float or list, optional): 計算相對誤差百分位數，默認為[25, 50, 75, 90, 95]
        
    返回:
        dict: 包含相對誤差統計信息的字典
    """
    # 確保輸入為numpy數組
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # 避免除以零
    y_true_safe = np.maximum(np.abs(y_true), 1e-8)
    
    # 計算相對誤差
    rel_error = np.abs((y_true - y_pred) / y_true_safe) * 100
    
    # 計算相對誤差統計量
    stats = {
        'mean': np.mean(rel_error),
        'std': np.std(rel_error),
        'min': np.min(rel_error),
        'max': np.max(rel_error),
    }
    
    # 計算百分位數
    if percentile is None:
        percentile = [25, 50, 75, 90, 95]
    
    if isinstance(percentile, (list, tuple, np.ndarray)):
        for p in percentile:
            stats[f'p{p}'] = np.percentile(rel_error, p)
    else:
        stats[f'p{percentile}'] = np.percentile(rel_error, percentile)
    
    return stats


def calculate_log_metrics(y_true, y_pred):
    """
    計算對數尺度上的評估指標
    適合處理範圍跨度大的疲勞壽命數據
    
    參數:
        y_true (array-like): 真實值
        y_pred (array-like): 預測值
        
    返回:
        dict: 包含對數尺度指標的字典
    """
    # 避免對小於等於零的值取對數
    y_true_safe = np.maximum(y_true, 1e-8)
    y_pred_safe = np.maximum(y_pred, 1e-8)
    
    # 取對數
    log_y_true = np.log10(y_true_safe)
    log_y_pred = np.log10(y_pred_safe)
    
    # 計算對數尺度指標
    log_metrics = {
        'log_rmse': calculate_rmse(log_y_true, log_y_pred),
        'log_r2': calculate_r2(log_y_true, log_y_pred),
        'log_mae': calculate_mae(log_y_true, log_y_pred)
    }
    
    return log_metrics


def evaluate_model(y_true, y_pred, model_name=None, verbose=True, return_dataframe=False):
    """
    綜合評估模型性能
    
    參數:
        y_true (array-like): 真實值
        y_pred (array-like): 預測值
        model_name (str, optional): 模型名稱，用於輸出
        verbose (bool): 是否輸出評估結果
        return_dataframe (bool): 是否以DataFrame格式返回結果
        
    返回:
        dict or pandas.DataFrame: 包含各項評估指標的字典或DataFrame
    """
    # 確保輸入為numpy數組
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # 計算常規指標
    metrics = {
        'rmse': calculate_rmse(y_true, y_pred),
        'r2': calculate_r2(y_true, y_pred),
        'mae': calculate_mae(y_true, y_pred),
        'mape': calculate_mape(y_true, y_pred)
    }
    
    # 計算相對誤差統計量
    rel_error_stats = calculate_relative_error(y_true, y_pred)
    metrics.update({
        'rel_error_mean': rel_error_stats['mean'],
        'rel_error_median': rel_error_stats['p50'],
        'rel_error_p90': rel_error_stats['p90']
    })
    
    # 計算對數尺度指標
    log_metrics = calculate_log_metrics(y_true, y_pred)
    metrics.update(log_metrics)
    
    # 將模型名稱添加到指標中
    if model_name:
        metrics['model'] = model_name
    
    # 輸出評估結果
    if verbose:
        output_str = []
        if model_name:
            output_str.append(f"模型: {model_name}")
        
        output_str.append(f"RMSE: {metrics['rmse']:.4f}")
        output_str.append(f"R²: {metrics['r2']:.4f}")
        output_str.append(f"MAE: {metrics['mae']:.4f}")
        output_str.append(f"MAPE: {metrics['mape']:.2f}%")
        output_str.append(f"相對誤差 (平均): {metrics['rel_error_mean']:.2f}%")
        output_str.append(f"相對誤差 (中位數): {metrics['rel_error_median']:.2f}%")
        output_str.append(f"相對誤差 (90百分位): {metrics['rel_error_p90']:.2f}%")
        output_str.append(f"對數RMSE: {metrics['log_rmse']:.4f}")
        
        logger.info('\n'.join(output_str))
    
    if return_dataframe:
        return pd.DataFrame([metrics])
    
    return metrics


def compare_models(models_results, sort_by='rmse', ascending=True):
    """
    比較多個模型的性能
    
    參數:
        models_results (list): 多個模型評估結果的列表，每個元素為evaluate_model的返回值
        sort_by (str): 用於排序的指標名稱
        ascending (bool): 排序方向，True為升序（適用於誤差類指標），False為降序（適用於R²）
        
    返回:
        pandas.DataFrame: 包含各模型評估指標的DataFrame
    """
    # 將結果轉換為DataFrame
    if all(isinstance(result, dict) for result in models_results):
        results_df = pd.DataFrame(models_results)
    else:
        results_df = pd.concat(models_results, ignore_index=True)
    
    # 排序結果
    if sort_by in results_df.columns:
        results_df = results_df.sort_values(by=sort_by, ascending=ascending)
    
    return results_df


def plot_metrics_comparison(models_results, metrics=None, figsize=(12, 8)):
    """
    繪製多個模型的指標比較圖
    
    參數:
        models_results (pandas.DataFrame): compare_models的返回值
        metrics (list, optional): 要比較的指標列表，默認為['rmse', 'r2', 'mae', 'mape']
        figsize (tuple): 圖像尺寸
        
    返回:
        matplotlib.figure.Figure: 圖像對象
    """
    if metrics is None:
        metrics = ['rmse', 'r2', 'mae', 'mape']
    
    # 確保models_results是DataFrame
    if not isinstance(models_results, pd.DataFrame):
        models_results = compare_models(models_results)
    
    # 獲取模型名稱
    model_names = models_results['model'] if 'model' in models_results.columns else models_results.index
    
    # 創建圖表
    fig, axes = plt.subplots(len(metrics), 1, figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        if metric in models_results.columns:
            # 為不同指標設定合適的排序
            ascending = metric != 'r2'  # R²值越大越好，其他指標越小越好
            
            # 獲取並排序數據
            data = models_results[[metric, 'model' if 'model' in models_results.columns else models_results.index.name]]
            data = data.sort_values(by=metric, ascending=ascending)
            
            # 繪製條形圖
            colors = plt.cm.viridis(np.linspace(0, 0.8, len(data)))
            bars = axes[i].bar(data['model'] if 'model' in data.columns else data.index, 
                     data[metric], 
                     color=colors)
            
            # 在條形上添加數值標籤
            for bar in bars:
                height = bar.get_height()
                axes[i].text(
                    bar.get_x() + bar.get_width()/2.,
                    height,
                    f'{height:.4f}',
                    ha='center', va='bottom', 
                    fontsize=8, rotation=0
                )
            
            # 設置標題和標籤
            axes[i].set_title(f'{metric.upper()} 比較')
            axes[i].set_ylabel(metric.upper())
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # 簡單的測試代碼
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 創建測試數據
    np.random.seed(42)
    y_true = np.exp(np.random.normal(10, 2, size=50))  # 模擬疲勞壽命真實值
    
    # 模擬三個不同模型的預測結果
    noise_levels = [0.1, 0.3, 0.5]
    model_names = ['PINN-LSTM', 'PINN', 'LSTM']
    
    all_results = []
    
    for noise, name in zip(noise_levels, model_names):
        # 加入不同程度的噪聲模擬預測
        y_pred = y_true * (1 + np.random.normal(0, noise, size=y_true.shape))
        
        # 評估性能
        result = evaluate_model(y_true, y_pred, model_name=name, verbose=True)
        all_results.append(result)
    
    # 比較模型性能
    comparison = compare_models(all_results)
    print("\n模型性能比較:")
    print(comparison)
    
    # 繪製比較圖
    try:
        fig = plot_metrics_comparison(comparison)
        plt.show()
    except Exception as e:
        logger.error(f"繪圖錯誤: {str(e)}")


import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging

logger = logging.getLogger(__name__)

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    logger.info(f"模型評估完成: RMSE={rmse}, MAE={mae}, R2={r2}")
    return {"RMSE": rmse, "MAE": mae, "R2": r2}

def compare_models(results_dict):
    for model_name, metrics in results_dict.items():
        logger.info(f"模型 {model_name} 評估結果: {metrics}")
    best_model = min(results_dict, key=lambda k: results_dict[k]["RMSE"])
    logger.info(f"最佳模型為: {best_model}")
    return best_model
