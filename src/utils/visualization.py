#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
visualization.py - 視覺化工具模組
本模組提供用於視覺化銲錫接點疲勞壽命預測模型的各種工具函數，
協助使用者瞭解模型性能、資料特性和預測結果。

主要功能:
1. 預測結果視覺化：比較預測值與真實值
2. 模型訓練歷史視覺化：損失曲線和指標變化
3. 特徵重要性視覺化：瞭解哪些結構參數更影響疲勞壽命
4. 注意力權重視覺化：分析LSTM分支對不同時間步的關注程度
5. 誤差分析和分佈視覺化：評估預測誤差的分佈特性
6. 物理約束可視化：驗證模型預測是否符合物理規律
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import logging
from pathlib import Path
import os
import torch

# 設定中文字體支援
try:
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei',
                                      'WenQuanYi Micro Hei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 正確顯示負號
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"設定中文字體支援時出錯: {str(e)}，圖表中的中文可能無法正確顯示")

logger = logging.getLogger(__name__)

def _save_figure(fig, save_path):
    """
    保存圖像的通用函數
    
    參數:
        fig (matplotlib.figure.Figure): 圖像對象
        save_path (str): 保存圖像的路徑
    """
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"圖像已保存至: {save_path}")
        except Exception as e:
            logger.error(f"保存圖像失敗: {str(e)}")

def plot_prediction_vs_true(y_true, y_pred, model_name=None, figsize=(10, 6), 
                           save_path=None, show_metrics=True, log_scale=True):
    """
    繪製預測值與真實值的對比圖
    
    參數:
        y_true (array-like): 真實值
        y_pred (array-like): 預測值
        model_name (str, optional): 模型名稱，用於標題
        figsize (tuple): 圖像尺寸
        save_path (str, optional): 保存圖像的路徑
        show_metrics (bool): 是否在圖上顯示評估指標
        log_scale (bool): 是否使用對數刻度
        
    返回:
        matplotlib.figure.Figure: 圖像對象
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=figsize)
    
    if log_scale and np.all(y_true > 0) and np.all(y_pred > 0):
        ax.set_xscale('log')
        ax.set_yscale('log')
        
    scatter = ax.scatter(y_true, y_pred, alpha=0.6, edgecolor='k', s=50)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    range_val = max_val - min_val
    min_val = max(0, min_val - range_val * 0.05)
    max_val = max_val + range_val * 0.05
    
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (True = Predicted)')
    
    if not log_scale:
        x_range = np.linspace(min_val, max_val, 100)
        ax.plot(x_range, x_range * 1.2, 'g--', alpha=0.5, label='+20%')
        ax.plot(x_range, x_range * 0.8, 'g--', alpha=0.5, label='-20%')
        ax.plot(x_range, x_range * 1.1, 'y--', alpha=0.5, label='+10%')
        ax.plot(x_range, x_range * 0.9, 'y--', alpha=0.5, label='-10%')
    
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    title = f'{model_name}: Prediction vs True Values' if model_name else 'Prediction vs True Values'
    ax.set_title(title)
    
    if show_metrics:
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rel_error = np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8)) * 100
        metrics_text = (f"RMSE: {rmse:.4f}\nR²: {r2:.4f}\nMAE: {mae:.4f}\n"
                       f"Mean Rel. Error: {np.mean(rel_error):.2f}%\n"
                       f"Median Rel. Error: {np.median(rel_error):.2f}%")
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
               verticalalignment='top', bbox=props, fontsize=9)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    plt.figtext(0.5, 0.01, f"Data Range: [{min_val:.2e}, {max_val:.2e}]", ha='center', fontsize=9)
    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig

def plot_training_history(history, figsize=(12, 5), save_path=None):
    """
    繪製訓練歷史曲線
    
    參數:
        history (dict): 包含訓練歷史的字典，應包含'train_loss'和'val_loss'
        figsize (tuple): 圖像尺寸
        save_path (str, optional): 保存圖像的路徑
        
    返回:
        matplotlib.figure.Figure: 圖像對象
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 損失曲線
    ax1 = axes[0]
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # 評估指標曲線
    ax2 = axes[1]
    has_metrics = False
    metrics_to_plot = ['rmse', 'r2', 'mae', 'delta_w_log_mse']
    for metric in metrics_to_plot:
        if metric in history.get('val_metrics', {}):
            metric_values = history['val_metrics'][metric]
            ax2.plot(epochs, metric_values, label=metric.upper())
            has_metrics = True
    
    if has_metrics:
        ax2.set_title('Validation Metrics')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Value')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
    else:
        # 如果沒有指標，顯示學習率（如果有）
        if 'learning_rate' in history:
            ax2.plot(epochs, history['learning_rate'], 'g-', label='Learning Rate')
            ax2.set_title('Learning Rate')
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Learning Rate')
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.legend()
        else:
            fig.delaxes(ax2)
            plt.tight_layout()
    
    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig

def plot_feature_importance(model, feature_names, figsize=(10, 6), save_path=None, method='permutation'):
    """
    繪製特徵重要性圖
    
    參數:
        model (torch.nn.Module): 訓練好的模型
        feature_names (list): 特徵名稱列表
        figsize (tuple): 圖像尺寸
        save_path (str, optional): 保存圖像的路徑
        method (str): 特徵重要性計算方法，'permutation'或'gradient'
        
    返回:
        matplotlib.figure.Figure: 圖像對象
    """
    # 這裡需要根據模型類型和方法具體實現
    # 示例實現：隨機生成特徵重要性值
    importances = np.random.rand(len(feature_names))
    indices = np.argsort(importances)[::-1]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bar_colors = plt.cm.viridis(np.linspace(0, 0.8, len(feature_names)))
    ax.bar(range(len(feature_names)), importances[indices], color=bar_colors)
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    ax.set_title('Feature Importance')
    ax.set_ylabel('Importance Score')
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig

def plot_attention_weights(attention_weights, time_points=None, figsize=(12, 6), save_path=None):
    """
    繪製注意力權重圖
    
    參數:
        attention_weights (array-like): 注意力權重，形狀為(batch_size, seq_len)
        time_points (list, optional): 時間點標籤
        figsize (tuple): 圖像尺寸
        save_path (str, optional): 保存圖像的路徑
        
    返回:
        matplotlib.figure.Figure: 圖像對象
    """
    attention_weights = np.asarray(attention_weights)
    
    # 設定時間點標籤
    if time_points is None:
        time_points = [f"t{i+1}" for i in range(attention_weights.shape[1])]
    
    # 平均每個樣本的注意力權重
    avg_weights = np.mean(attention_weights, axis=0)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 繪製折線圖
    axes[0].plot(range(len(time_points)), avg_weights, 'o-', linewidth=2, markersize=8)
    axes[0].set_xticks(range(len(time_points)))
    axes[0].set_xticklabels(time_points)
    axes[0].set_title('Average Attention Weights')
    axes[0].set_ylabel('Weight')
    axes[0].set_xlabel('Time Point')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # 繪製熱力圖
    sample_weights = attention_weights[:min(10, attention_weights.shape[0])]
    im = axes[1].imshow(sample_weights, aspect='auto', cmap='viridis')
    axes[1].set_xticks(range(len(time_points)))
    axes[1].set_xticklabels(time_points)
    axes[1].set_yticks(range(sample_weights.shape[0]))
    axes[1].set_yticklabels([f'Sample {i+1}' for i in range(sample_weights.shape[0])])
    axes[1].set_title('Attention Weights (First 10 Samples)')
    plt.colorbar(im, ax=axes[1], label='Weight')
    
    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig

def create_error_histogram(y_true, y_pred, bins=20, figsize=(10, 6), save_path=None, log_scale=False):
    """
    創建預測誤差分佈直方圖
    
    參數:
        y_true (array-like): 真實值
        y_pred (array-like): 預測值
        bins (int): 直方圖箱數
        figsize (tuple): 圖像尺寸
        save_path (str, optional): 保存圖像的路徑
        log_scale (bool): 是否在對數尺度下計算誤差
        
    返回:
        matplotlib.figure.Figure: 圖像對象
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if log_scale and np.all(y_true > 0) and np.all(y_pred > 0):
        y_true = np.log10(y_true)
        y_pred = np.log10(y_pred)
        errors = y_true - y_pred
        error_label = 'Log Scale Error'
    else:
        # 計算相對誤差
        errors = (y_true - y_pred) / np.maximum(y_true, 1e-8) * 100
        error_label = 'Relative Error (%)'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 繪製直方圖
    n, bins, patches = ax.hist(errors, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    
    # 添加核密度估計
    sns.kdeplot(errors, ax=ax, color='red', label='Density')
    
    # 標記統計信息
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    std_error = np.std(errors)
    
    # 添加統計信息文本
    stats_text = f"Mean: {mean_error:.4f}\nMedian: {median_error:.4f}\nStd: {std_error:.4f}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
           fontsize=9)
    
    ax.set_xlabel(error_label)
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig

def plot_parameter_impact(parameters, predictions, parameter_name, figsize=(10, 6), save_path=None):
    """
    繪製結構參數對疲勞壽命的影響圖
    
    參數:
        parameters (array-like): 結構參數值
        predictions (array-like): 對應的預測壽命值
        parameter_name (str): 參數名稱
        figsize (tuple): 圖像尺寸
        save_path (str, optional): 保存圖像的路徑
        
    返回:
        matplotlib.figure.Figure: 圖像對象
    """
    parameters = np.asarray(parameters)
    predictions = np.asarray(predictions)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    scatter = ax.scatter(parameters, predictions, alpha=0.7, edgecolor='k')
    
    try:
        from scipy.stats import linregress
        slope, intercept, r_value, _, _ = linregress(parameters, predictions)
        trend_x = np.linspace(np.min(parameters), np.max(parameters), 100)
        trend_y = slope * trend_x + intercept
        ax.plot(trend_x, trend_y, 'r--',
               label=f'Trend: y={slope:.4e}x+{intercept:.4e}, R²={r_value**2:.4f}')
    except:
        pass
    
    ax.set_xlabel(parameter_name)
    ax.set_ylabel('預測疲勞壽命')
    ax.set_title(f'{parameter_name}對疲勞壽命的影響')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig

def plot_delta_w_prediction_vs_theory(delta_w_pred, delta_w_theory, model_name=None, figsize=(10, 6), 
                                     save_path=None, show_metrics=True, log_scale=True):
    """
    繪製預測的delta_w與理論delta_w的對比圖
    
    參數:
        delta_w_pred (array-like): 預測的delta_w值
        delta_w_theory (array-like): 理論的delta_w值
        model_name (str, optional): 模型名稱，用於標題
        figsize (tuple): 圖像尺寸
        save_path (str, optional): 保存圖像的路徑
        show_metrics (bool): 是否顯示評估指標
        log_scale (bool): 是否使用對數刻度
        
    返回:
        matplotlib.figure.Figure: 圖像對象
    """
    # 確保輸入為numpy數組
    if isinstance(delta_w_pred, torch.Tensor):
        delta_w_pred = delta_w_pred.detach().cpu().numpy()
    if isinstance(delta_w_theory, torch.Tensor):
        delta_w_theory = delta_w_theory.detach().cpu().numpy()
        
    delta_w_pred = np.asarray(delta_w_pred)
    delta_w_theory = np.asarray(delta_w_theory)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if log_scale and np.all(delta_w_pred > 0) and np.all(delta_w_theory > 0):
        ax.set_xscale('log')
        ax.set_yscale('log')
        
    scatter = ax.scatter(delta_w_theory, delta_w_pred, alpha=0.6, edgecolor='k', s=50)
    
    min_val = min(np.min(delta_w_theory), np.min(delta_w_pred))
    max_val = max(np.max(delta_w_theory), np.max(delta_w_pred))
    range_val = max_val - min_val
    min_val = max(0, min_val - range_val * 0.05)
    max_val = max_val + range_val * 0.05
    
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (Theory = Predicted)')
    
    if not log_scale:
        x_range = np.linspace(min_val, max_val, 100)
        ax.plot(x_range, x_range * 1.2, 'g--', alpha=0.5, label='+20%')
        ax.plot(x_range, x_range * 0.8, 'g--', alpha=0.5, label='-20%')
        ax.plot(x_range, x_range * 1.1, 'y--', alpha=0.5, label='+10%')
        ax.plot(x_range, x_range * 0.9, 'y--', alpha=0.5, label='-10%')
    
    ax.set_xlabel('Theoretical ΔW Values')
    ax.set_ylabel('Predicted ΔW Values')
    title = f'{model_name}: ΔW Prediction vs Theory' if model_name else 'ΔW Prediction vs Theory'
    ax.set_title(title)
    
    if show_metrics:
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        # 計算對數空間的指標
        log_delta_w_pred = np.log10(np.maximum(delta_w_pred, 1e-8))
        log_delta_w_theory = np.log10(np.maximum(delta_w_theory, 1e-8))
        
        log_mse = mean_squared_error(log_delta_w_theory, log_delta_w_pred)
        log_r2 = r2_score(log_delta_w_theory, log_delta_w_pred)
        
        # 計算相對誤差
        rel_error = np.abs((delta_w_theory - delta_w_pred) / np.maximum(delta_w_theory, 1e-8)) * 100
        
        metrics_text = (f"Log MSE: {log_mse:.4f}\nLog R²: {log_r2:.4f}\n"
                       f"Mean Rel. Error: {np.mean(rel_error):.2f}%\n"
                       f"Median Rel. Error: {np.median(rel_error):.2f}%")
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
               verticalalignment='top', bbox=props, fontsize=9)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig

def plot_physical_constraint_validation(delta_w_values, nf_values, a=55.83, b=-2.259, 
                                       figsize=(10, 6), save_path=None):
    """
    繪製物理約束驗證圖
    驗證預測結果是否符合 Nf = a * (ΔW)^b 的物理關係
    
    參數:
        delta_w_values (array-like): 非線性塑性應變能密度變化量
        nf_values (array-like): 疲勞壽命值
        a (float): 物理模型係數a
        b (float): 物理模型係數b
        figsize (tuple): 圖像尺寸
        save_path (str, optional): 保存圖像的路徑
        
    返回:
        matplotlib.figure.Figure: 圖像對象
    """
    # 確保輸入為numpy數組
    if isinstance(delta_w_values, torch.Tensor):
        delta_w_values = delta_w_values.detach().cpu().numpy()
    if isinstance(nf_values, torch.Tensor):
        nf_values = nf_values.detach().cpu().numpy()
    
    delta_w_values = np.asarray(delta_w_values)
    nf_values = np.asarray(nf_values)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 使用對數坐標
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # 繪製預測點
    scatter = ax.scatter(delta_w_values, nf_values, alpha=0.7, edgecolor='k', label='Predicted Values')
    
    # 繪製理論曲線
    x_range = np.logspace(np.log10(np.min(delta_w_values)*0.5), np.log10(np.max(delta_w_values)*2), 100)
    y_theory = a * np.power(x_range, b)
    ax.plot(x_range, y_theory, 'r-', label=f'Physical Model: Nf={a}*(ΔW)^{b}')
    
    # 計算理論值 - 確保使用numpy運算
    y_theory_at_x = a * np.power(delta_w_values, b)
    
    # 計算相對誤差
    relative_error = np.abs((nf_values - y_theory_at_x) / y_theory_at_x) * 100
    
    # 顯示統計信息
    stats_text = (f"Model Deviation Statistics:\nMean Rel. Error: {np.mean(relative_error):.2f}%\n"
                 f"Median Rel. Error: {np.median(relative_error):.2f}%\n"
                 f"Max Rel. Error: {np.max(relative_error):.2f}%\n"
                 f"Min Rel. Error: {np.min(relative_error):.2f}%")
    
    ax.set_xlabel('Nonlinear Plastic Strain Energy Density Change (ΔW)')
    ax.set_ylabel('Fatigue Life (Nf)')
    ax.set_title('Physical Constraint Validation: Nf vs ΔW')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # 添加文本框顯示統計信息
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
           fontsize=9)
    
    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig

def visualize_model_results(results, output_dir=None):
    """
    對模型結果進行綜合視覺化
    
    參數:
        results (dict): 包含各種預測結果的字典
        output_dir (str, optional): 輸出目錄
        
    返回:
        dict: 各視覺化圖表的路徑
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    visualization_paths = {}
    
    # 預測vs真實值
    if 'predictions' in results and 'targets' in results:
        # 確保數據是 numpy 數組
        predictions = results['predictions']
        targets = results['targets']
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
            
        plot_path = os.path.join(output_dir, "prediction_vs_true.png") if output_dir else None
        fig = plot_prediction_vs_true(
            targets, 
            predictions,
            save_path=plot_path
        )
        plt.close(fig)
        visualization_paths['prediction_vs_true'] = plot_path
    
    # Delta_W預測vs理論
    if 'delta_w' in results and 'targets' in results:
        try:
            plot_path = os.path.join(output_dir, "delta_w_prediction.png") if output_dir else None
            # 從目標值計算理論delta_w
            # 確保是numpy數組
            targets = results['targets']
            delta_w = results['delta_w']
            
            if isinstance(targets, torch.Tensor):
                targets = targets.detach().cpu().numpy()
            if isinstance(delta_w, torch.Tensor):
                delta_w = delta_w.detach().cpu().numpy()
                
            targets = np.asarray(targets)
            
            delta_w_theory = np.power(targets / 55.83, 1 / -2.259)
            fig = plot_delta_w_prediction_vs_theory(
                delta_w,
                delta_w_theory,
                save_path=plot_path
            )
            plt.close(fig)
            visualization_paths['delta_w_prediction'] = plot_path
        except Exception as e:
            logger.warning(f"生成 delta_w 預測圖時出錯: {str(e)}")
    
    # 物理約束驗證
    if 'delta_w' in results and 'predictions' in results:
        try:
            plot_path = os.path.join(output_dir, "physical_constraint.png") if output_dir else None
            
            # 確保數據是 numpy 數組
            delta_w = results['delta_w']
            predictions = results['predictions']
            
            if isinstance(delta_w, torch.Tensor):
                delta_w = delta_w.detach().cpu().numpy()
            if isinstance(predictions, torch.Tensor):
                predictions = predictions.detach().cpu().numpy()
                
            fig = plot_physical_constraint_validation(
                delta_w,
                predictions,
                save_path=plot_path
            )
            plt.close(fig)
            visualization_paths['physical_constraint'] = plot_path
        except Exception as e:
            logger.warning(f"生成物理約束驗證圖時出錯: {str(e)}")
    
    # 預測誤差分布
    if 'predictions' in results and 'targets' in results:
        try:
            plot_path = os.path.join(output_dir, "error_distribution.png") if output_dir else None
            
            # 確保數據是 numpy 數組
            predictions = results['predictions']
            targets = results['targets']
            
            if isinstance(predictions, torch.Tensor):
                predictions = predictions.detach().cpu().numpy()
            if isinstance(targets, torch.Tensor):
                targets = targets.detach().cpu().numpy()
                
            fig = create_error_histogram(
                targets,
                predictions,
                save_path=plot_path
            )
            plt.close(fig)
            visualization_paths['error_distribution'] = plot_path
        except Exception as e:
            logger.warning(f"生成誤差分布圖時出錯: {str(e)}")
    
    # 注意力權重
    if 'attention_weights' in results:
        try:
            plot_path = os.path.join(output_dir, "attention_weights.png") if output_dir else None
            
            # 確保數據是 numpy 數組
            attention_weights = results['attention_weights']
            if isinstance(attention_weights, torch.Tensor):
                attention_weights = attention_weights.detach().cpu().numpy()
                
            fig = plot_attention_weights(
                attention_weights,
                save_path=plot_path
            )
            plt.close(fig)
            visualization_paths['attention_weights'] = plot_path
        except Exception as e:
            logger.warning(f"生成注意力權重圖時出錯: {str(e)}")
    
    return visualization_paths