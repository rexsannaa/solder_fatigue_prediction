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

# 設定中文字體支援（保留原註解）
try:
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei',
                                        'WenQuanYi Micro Hei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 正確顯示負號
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"設定中文字體支援時出錯: {str(e)}，圖表中的中文可能無法正確顯示")

logger = logging.getLogger(__name__)

# 輔助函數：計算並返回所有參數的 L2 範數（用於正則化）
def _l2_penalty(parameters):
    return sum(p.norm(2) for p in parameters)
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

def plot_delta_w_prediction_vs_theory(delta_w_pred, delta_w_theory, model_name=None, figsize=(10, 6), 
                                      save_path=None, show_metrics=True, log_scale=True):
    """
    繪製預測的 delta_w 與理論 delta_w 的對比圖
    
    參數:
        delta_w_pred (array-like): 預測的 delta_w 值
        delta_w_theory (array-like): 理論的 delta_w 值
        model_name (str, optional): 模型名稱，用於標題
        figsize (tuple): 圖像尺寸
        save_path (str, optional): 保存圖像的路徑
        show_metrics (bool): 是否顯示評估指標
        log_scale (bool): 是否使用對數刻度
        
    返回:
        matplotlib.figure.Figure: 圖像對象
    """
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
    
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='理想線 (Theory = Predicted)')
    
    if not log_scale:
        x_range = np.linspace(min_val, max_val, 100)
        ax.plot(x_range, x_range * 1.2, 'g--', alpha=0.5, label='+20%')
        ax.plot(x_range, x_range * 0.8, 'g--', alpha=0.5, label='-20%')
        ax.plot(x_range, x_range * 1.1, 'y--', alpha=0.5, label='+10%')
        ax.plot(x_range, x_range * 0.9, 'y--', alpha=0.5, label='-10%')
    
    ax.set_xlabel('理論 ΔW 值')
    ax.set_ylabel('預測 ΔW 值')
    title = f'{model_name}: ΔW Prediction vs Theory' if model_name else 'ΔW Prediction vs Theory'
    ax.set_title(title)
    
    if show_metrics:
        # 計算評估指標
        log_delta_w_pred = np.log10(delta_w_pred)
        log_delta_w_theory = np.log10(delta_w_theory)
        
        mse = np.mean((delta_w_theory - delta_w_pred) ** 2)
        log_mse = np.mean((log_delta_w_theory - log_delta_w_pred) ** 2)
        
        rel_error = np.abs((delta_w_theory - delta_w_pred) / delta_w_theory) * 100
        metrics_text = (f"MSE: {mse:.6f}\nLog-MSE: {log_mse:.6f}\n"
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


def plot_parameter_impact(parameters, predictions, parameter_name="結構參數", 
                          figsize=(10, 6), save_path=None):
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
        if len(parameters) > 5:
            from numpy.polynomial.polynomial import Polynomial
            p = Polynomial.fit(parameters, predictions, 2)
            poly_x = np.linspace(np.min(parameters), np.max(parameters), 100)
            ax.plot(poly_x, p(poly_x), 'g-', label='Polynomial Fit (degree=2)')
    except Exception:
        pass
    
    ax.set_xlabel(parameter_name)
    ax.set_ylabel('預測疲勞壽命')
    ax.set_title(f'{parameter_name}對疲勞壽命的影響')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    plt.figtext(0.5, 0.01, f"{parameter_name}範圍: [{np.min(parameters)}, {np.max(parameters)}], 壽命範圍: [{np.min(predictions):.2e}, {np.max(predictions):.2e}]",
                ha='center', fontsize=9)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
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
    # 將輸入轉換為標準 numpy 數組
    if 'torch' in globals() or 'torch' in locals():
        import torch
        if isinstance(delta_w_values, torch.Tensor):
            delta_w_values = delta_w_values.detach().cpu().numpy()
        if isinstance(nf_values, torch.Tensor):
            nf_values = nf_values.detach().cpu().numpy()
    
    delta_w_values = np.asarray(delta_w_values)
    nf_values = np.asarray(nf_values)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    scatter = ax.scatter(delta_w_values, nf_values, alpha=0.7, edgecolor='k', label='預測值')
    
    x_range = np.logspace(np.log10(np.min(delta_w_values)*0.5), np.log10(np.max(delta_w_values)*2), 100)
    y_theory = a * np.power(x_range, b)
    ax.plot(x_range, y_theory, 'r-', label=f'物理模型: Nf={a}*(ΔW)^{b}')
    
    # 計算理論值並確保它是 numpy 數組
    y_theory_at_x = a * np.power(delta_w_values, b)
    
    # 防止任何類型不匹配
    if 'torch' in globals() or 'torch' in locals():
        import torch
        if isinstance(y_theory_at_x, torch.Tensor):
            y_theory_at_x = y_theory_at_x.detach().cpu().numpy()
        if isinstance(nf_values, torch.Tensor):
            nf_values = nf_values.detach().cpu().numpy()
    
    # 確保都是 numpy 數組
    y_theory_at_x = np.asarray(y_theory_at_x)
    nf_values = np.asarray(nf_values)
    
    relative_error = np.abs((nf_values - y_theory_at_x) / y_theory_at_x) * 100
    stats_text = (f"與物理模型偏差統計:\n平均相對誤差: {np.mean(relative_error):.2f}%\n"
                  f"中位數相對誤差: {np.median(relative_error):.2f}%\n"
                  f"最大相對誤差: {np.max(relative_error):.2f}%\n"
                  f"最小相對誤差: {np.min(relative_error):.2f}%")
                  
    ax.set_xlabel('非線性塑性應變能密度變化量 (ΔW)')
    ax.set_ylabel('疲勞壽命 (Nf)')
    ax.set_title('物理約束驗證: Nf vs ΔW')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=9)
            
    plt.tight_layout()
    
    _save_figure(fig, save_path)
    
    return fig

def _process_fusion_weights(fusion_weights):
    """
    處理不同形式的融合權重數據，返回一個長度至少為2的平均權重陣列
    """
    if isinstance(fusion_weights, np.ndarray):
        if fusion_weights.ndim > 1:
            avg_weights = np.mean(fusion_weights, axis=0)
        elif fusion_weights.ndim == 1 and fusion_weights.size >= 2:
            avg_weights = fusion_weights[:2]
        else:
            scalar_value = float(fusion_weights.item() if hasattr(fusion_weights, 'item') else fusion_weights)
            avg_weights = np.array([scalar_value, 1.0 - scalar_value])
    else:
        try:
            scalar_value = float(fusion_weights)
            avg_weights = np.array([scalar_value, 1.0 - scalar_value])
        except (TypeError, ValueError):
            avg_weights = np.array([0.5, 0.5])
    if not hasattr(avg_weights, '__len__') or len(avg_weights) < 2:
        scalar_value = float(avg_weights) if np.isscalar(avg_weights) else 0.5
        avg_weights = np.array([scalar_value, 1.0 - scalar_value])
    return avg_weights
def plot_delta_w_prediction_vs_theory(delta_w_pred, delta_w_theory, model_name=None, figsize=(10, 6), 
                                      save_path=None, show_metrics=True, log_scale=True):
    """
    繪製預測的 delta_w 與理論 delta_w 的對比圖
    
    參數:
        delta_w_pred (array-like): 預測的 delta_w 值
        delta_w_theory (array-like): 理論的 delta_w 值
        model_name (str, optional): 模型名稱，用於標題
        figsize (tuple): 圖像尺寸
        save_path (str, optional): 保存圖像的路徑
        show_metrics (bool): 是否顯示評估指標
        log_scale (bool): 是否使用對數刻度
        
    返回:
        matplotlib.figure.Figure: 圖像對象
    """
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
    
    ax.set_xlabel('理論 ΔW 值')
    ax.set_ylabel('預測 ΔW 值')
    title = f'{model_name}: ΔW Prediction vs Theory' if model_name else 'ΔW Prediction vs Theory'
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig
def plot_delta_w_prediction_vs_theory(delta_w_pred, delta_w_theory, model_name=None, figsize=(10, 6), 
                                      save_path=None, show_metrics=True, log_scale=True):
    """
    繪製預測的 delta_w 與理論 delta_w 的對比圖
    
    參數:
        delta_w_pred (array-like): 預測的 delta_w 值
        delta_w_theory (array-like): 理論的 delta_w 值
        model_name (str, optional): 模型名稱，用於標題
        figsize (tuple): 圖像尺寸
        save_path (str, optional): 保存圖像的路徑
        show_metrics (bool): 是否顯示評估指標
        log_scale (bool): 是否使用對數刻度
        
    返回:
        matplotlib.figure.Figure: 圖像對象
    """
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
    
    ax.set_xlabel('理論 ΔW 值')
    ax.set_ylabel('預測 ΔW 值')
    title = f'{model_name}: ΔW Prediction vs Theory' if model_name else 'ΔW Prediction vs Theory'
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig
def visualize_model_results(results, output_dir="visualization", prefix="", 
                           a_coefficient=55.83, b_coefficient=-2.259):
    """
    生成模型預測結果的全面視覺化，包括delta_w預測評估
    
    參數:
        results (dict): 模型預測結果，包含targets, predictions, delta_w等
        output_dir (str): 輸出目錄
        prefix (str): 文件名前綴
        a_coefficient (float): 物理模型係數a
        b_coefficient (float): 物理模型係數b
        
    返回:
        dict: 包含各圖像路徑的字典
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = {}
    
    # 1. 預測vs真實值圖
    if 'targets' in results and 'predictions' in results:
        path = os.path.join(output_dir, f"{prefix}prediction_vs_true.png")
        from src.utils.visualization import plot_prediction_vs_true
        fig = plot_prediction_vs_true(
            results['targets'], 
            results['predictions'],
            save_path=path
        )
        plt.close(fig)
        paths['prediction_vs_true'] = path
    
    # 2. delta_w預測vs理論值圖
    if 'delta_w' in results and 'targets' in results:
        # 從目標計算理論delta_w
        delta_w_theory = np.power(results['targets'] / a_coefficient, 1.0 / b_coefficient)
        path = os.path.join(output_dir, f"{prefix}delta_w_vs_theory.png")
        fig = plot_delta_w_prediction_vs_theory(
            results['delta_w'],
            delta_w_theory,
            save_path=path
        )
        plt.close(fig)
        paths['delta_w_vs_theory'] = path
    
    # 3. 物理約束驗證圖
    if 'delta_w' in results and 'predictions' in results:
        path = os.path.join(output_dir, f"{prefix}physical_constraint.png")
        fig = plot_physical_constraint_validation(
            results['delta_w'],
            results['predictions'],
            a=a_coefficient,
            b=b_coefficient,
            save_path=path
        )
        plt.close(fig)
        paths['physical_constraint'] = path
    
    # 4. PINN和LSTM分支delta_w對比圖（如果有）
    if 'pinn_delta_w' in results and 'lstm_delta_w' in results and 'delta_w' in results:
        path = os.path.join(output_dir, f"{prefix}branch_delta_w_comparison.png")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        if 'targets' in results:
            delta_w_theory = np.power(results['targets'] / a_coefficient, 1.0 / b_coefficient)
            ax.scatter(delta_w_theory, results['delta_w'], alpha=0.7, label='融合預測', s=50)
            ax.scatter(delta_w_theory, results['pinn_delta_w'], alpha=0.5, label='PINN分支', s=30)
            ax.scatter(delta_w_theory, results['lstm_delta_w'], alpha=0.5, label='LSTM分支', s=30)
            
            min_val = min(np.min(delta_w_theory), np.min(results['delta_w']))
            max_val = max(np.max(delta_w_theory), np.max(results['delta_w']))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='理想線')
            
            ax.set_xlabel('理論 ΔW 值')
        else:
            ax.scatter(range(len(results['delta_w'])), results['delta_w'], alpha=0.7, label='融合預測')
            ax.scatter(range(len(results['pinn_delta_w'])), results['pinn_delta_w'], alpha=0.5, label='PINN分支')
            ax.scatter(range(len(results['lstm_delta_w'])), results['lstm_delta_w'], alpha=0.5, label='LSTM分支')
            
            ax.set_xlabel('樣本索引')
        
        ax.set_ylabel('預測 ΔW 值')
        ax.set_title('分支預測對比: PINN vs LSTM vs 融合')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        plt.tight_layout()
        _save_figure(fig, path)
        plt.close(fig)
        
        paths['branch_comparison'] = path
    
    # 5. 注意力權重視覺化（如果有）
    if 'attention_weights' in results:
        path = os.path.join(output_dir, f"{prefix}attention_weights.png")
        
        attention_weights = results['attention_weights']
        if isinstance(attention_weights, np.ndarray) and attention_weights.ndim == 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(attention_weights, aspect='auto', cmap='viridis')
            ax.set_xlabel('時間步')
            ax.set_ylabel('樣本')
            ax.set_title('注意力權重分布')
            plt.colorbar(im, ax=ax, label='權重值')
            
            _save_figure(fig, path)
            plt.close(fig)
            
            paths['attention_weights'] = path
    
    return paths