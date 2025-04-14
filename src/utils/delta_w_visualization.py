#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
delta_w_visualization.py - 專門用於非線性塑性應變能密度變化量(delta_w)預測的視覺化工具
本模組提供特化的視覺化工具，用於評估和分析delta_w的預測精度，
對物理導向預測模型的效能提供深入的理解。

主要功能:
1. delta_w預測與理論值的對比視覺化
2. delta_w預測誤差分析
3. delta_w與疲勞壽命的物理關係視覺化
4. 分支模型delta_w預測一致性分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
import os
import torch

logger = logging.getLogger(__name__)

# 物理模型常數
A_COEFFICIENT = 55.83  # 係數 a
B_COEFFICIENT = -2.259  # 係數 b

def _save_figure(fig, save_path):
    """保存圖像的通用函數"""
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"圖像已保存至: {save_path}")
        except Exception as e:
            logger.error(f"保存圖像失敗: {str(e)}")

def calculate_delta_w_theory(nf_true, a=A_COEFFICIENT, b=B_COEFFICIENT):
    """
    根據真實疲勞壽命計算理論delta_w
    
    參數:
        nf_true (array-like): 真實的疲勞壽命
        a (float): 物理模型係數a
        b (float): 物理模型係數b
    
    返回:
        array-like: 理論delta_w值
    """
    nf_true = np.asarray(nf_true)
    nf_true = np.maximum(nf_true, 1e-10)  # 避免除零錯誤
    delta_w_theory = np.power(nf_true / a, 1.0 / b)
    return delta_w_theory

def plot_delta_w_prediction_vs_theory(delta_w_pred, delta_w_theory=None, nf_true=None, 
                                     model_name=None, figsize=(10, 6), 
                                     save_path=None, show_metrics=True, log_scale=True,
                                     a=A_COEFFICIENT, b=B_COEFFICIENT):
    """
    繪製預測的delta_w與理論delta_w的對比圖
    
    參數:
        delta_w_pred (array-like): 預測的delta_w值
        delta_w_theory (array-like, optional): 理論delta_w值
        nf_true (array-like, optional): 真實的疲勞壽命，如果提供則計算delta_w_theory
        model_name (str, optional): 模型名稱，用於標題
        figsize (tuple): 圖像尺寸
        save_path (str, optional): 保存圖像的路徑
        show_metrics (bool): 是否顯示評估指標
        log_scale (bool): 是否使用對數刻度
        a (float): 物理模型係數a
        b (float): 物理模型係數b
    
    返回:
        matplotlib.figure.Figure: 圖像對象
    """
    # 確保輸入為NumPy數組
    delta_w_pred = np.asarray(delta_w_pred)
    
    # 如果沒有提供delta_w_theory但提供了nf_true，則計算delta_w_theory
    if delta_w_theory is None and nf_true is not None:
        delta_w_theory = calculate_delta_w_theory(nf_true, a, b)
    
    if delta_w_theory is None:
        raise ValueError("必須提供delta_w_theory或nf_true")
    
    delta_w_theory = np.asarray(delta_w_theory)
    
    # 創建圖像
    fig, ax = plt.subplots(figsize=figsize)
    
    # 設置對數刻度
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    # 繪製散點圖
    scatter = ax.scatter(delta_w_theory, delta_w_pred, alpha=0.7, edgecolor='k', s=60)
    
    # 設置坐標軸範圍
    min_val = min(np.min(delta_w_theory), np.min(delta_w_pred))
    max_val = max(np.max(delta_w_theory), np.max(delta_w_pred))
    range_val = max_val - min_val
    min_val = max(0, min_val - range_val * 0.05)
    max_val = max_val + range_val * 0.05
    
    # 繪製理想線
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='理想預測 (預測=理論)')
    
    # 繪製誤差範圍線
    if not log_scale:
        x_range = np.linspace(min_val, max_val, 100)
        ax.plot(x_range, x_range * 1.2, 'g--', alpha=0.5, label='+20%')
        ax.plot(x_range, x_range * 0.8, 'g--', alpha=0.5, label='-20%')
        ax.plot(x_range, x_range * 1.1, 'y--', alpha=0.5, label='+10%')
        ax.plot(x_range, x_range * 0.9, 'y--', alpha=0.5, label='-10%')
    
    # 設置坐標軸標籤和標題
    ax.set_xlabel('理論 ΔW 值')
    ax.set_ylabel('預測 ΔW 值')
    title = f'{model_name}: ΔW 預測評估' if model_name else 'ΔW 預測評估'
    ax.set_title(title)
    
    # 顯示評估指標
    if show_metrics:
        # 計算評估指標
        mse = np.mean((delta_w_theory - delta_w_pred) ** 2)
        rmse = np.sqrt(mse)
        
        # 對數空間指標
        log_delta_w_theory = np.log10(delta_w_theory)
        log_delta_w_pred = np.log10(delta_w_pred)
        log_mse = np.mean((log_delta_w_theory - log_delta_w_pred) ** 2)
        log_rmse = np.sqrt(log_mse)
        
        # 相對誤差
        rel_error = np.abs((delta_w_theory - delta_w_pred) / delta_w_theory) * 100
        mean_rel_error = np.mean(rel_error)
        median_rel_error = np.median(rel_error)
        
        # 顯示指標
        metrics_text = (f"RMSE: {rmse:.6f}\n"
                        f"Log10 RMSE: {log_rmse:.4f}\n"
                        f"平均相對誤差: {mean_rel_error:.2f}%\n"
                        f"中位數相對誤差: {median_rel_error:.2f}%")
        
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 添加網格和圖例
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # 添加數據範圍信息
    plt.figtext(0.5, 0.01, f"ΔW 範圍: [{min_val:.2e}, {max_val:.2e}]", ha='center', fontsize=9)
    
    # 調整布局
    plt.tight_layout()
    
    # 保存圖像
    _save_figure(fig, save_path)
    
    return fig

def plot_delta_w_error_distribution(delta_w_pred, delta_w_theory=None, nf_true=None,
                                   model_name=None, figsize=(10, 6), save_path=None,
                                   a=A_COEFFICIENT, b=B_COEFFICIENT):
    """
    繪製delta_w預測誤差分佈圖
    
    參數:
        delta_w_pred (array-like): 預測的delta_w值
        delta_w_theory (array-like, optional): 理論delta_w值
        nf_true (array-like, optional): 真實的疲勞壽命，如果提供則計算delta_w_theory
        model_name (str, optional): 模型名稱，用於標題
        figsize (tuple): 圖像尺寸
        save_path (str, optional): 保存圖像的路徑
        a (float): 物理模型係數a
        b (float): 物理模型係數b
    
    返回:
        matplotlib.figure.Figure: 圖像對象
    """
    # 確保輸入為NumPy數組
    delta_w_pred = np.asarray(delta_w_pred)
    
    # 如果沒有提供delta_w_theory但提供了nf_true，則計算delta_w_theory
    if delta_w_theory is None and nf_true is not None:
        delta_w_theory = calculate_delta_w_theory(nf_true, a, b)
    
    if delta_w_theory is None:
        raise ValueError("必須提供delta_w_theory或nf_true")
    
    delta_w_theory = np.asarray(delta_w_theory)
    
    # 計算不同類型的誤差
    # 絕對誤差
    abs_error = delta_w_pred - delta_w_theory
    
    # 相對誤差
    rel_error = (delta_w_pred - delta_w_theory) / delta_w_theory * 100
    
    # 對數空間誤差
    log_delta_w_pred = np.log10(delta_w_pred)
    log_delta_w_theory = np.log10(delta_w_theory)
    log_error = log_delta_w_pred - log_delta_w_theory
    
    # 創建圖像
    fig, axs = plt.subplots(3, 1, figsize=figsize)
    
    # 繪製絕對誤差分佈
    sns.histplot(abs_error, kde=True, ax=axs[0])
    axs[0].set_title('ΔW 絕對誤差分佈')
    axs[0].set_xlabel('絕對誤差 (預測 - 理論)')
    axs[0].axvline(x=0, color='r', linestyle='--')
    
    # 繪製相對誤差分佈
    sns.histplot(rel_error, kde=True, ax=axs[1])
    axs[1].set_title('ΔW 相對誤差分佈 (%)')
    axs[1].set_xlabel('相對誤差 %')
    axs[1].axvline(x=0, color='r', linestyle='--')
    
    # 繪製對數空間誤差分佈
    sns.histplot(log_error, kde=True, ax=axs[2])
    axs[2].set_title('ΔW 對數空間誤差分佈')
    axs[2].set_xlabel('對數誤差 (log10(預測) - log10(理論))')
    axs[2].axvline(x=0, color='r', linestyle='--')
    
    # 添加一些統計信息
    stats_text = (f"絕對誤差: 平均={np.mean(abs_error):.6f}, 標準差={np.std(abs_error):.6f}\n"
                 f"相對誤差: 平均={np.mean(rel_error):.2f}%, 標準差={np.std(rel_error):.2f}%\n"
                 f"對數誤差: 平均={np.mean(log_error):.4f}, 標準差={np.std(log_error):.4f}")
    
    plt.figtext(0.5, 0.01, stats_text, ha='center', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 設置主標題
    main_title = f'{model_name}: ΔW 預測誤差分析' if model_name else 'ΔW 預測誤差分析'
    fig.suptitle(main_title)
    
    # 調整布局
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存圖像
    _save_figure(fig, save_path)
    
    return fig

def plot_delta_w_nf_relationship(delta_w_pred, nf_pred=None, nf_true=None,
                               model_name=None, figsize=(10, 6), save_path=None,
                               a=A_COEFFICIENT, b=B_COEFFICIENT):
    """
    繪製delta_w與疲勞壽命的物理關係圖
    
    參數:
        delta_w_pred (array-like): 預測的delta_w值
        nf_pred (array-like, optional): 預測的疲勞壽命
        nf_true (array-like, optional): 真實的疲勞壽命
        model_name (str, optional): 模型名稱，用於標題
        figsize (tuple): 圖像尺寸
        save_path (str, optional): 保存圖像的路徑
        a (float): 物理模型係數a
        b (float): 物理模型係數b
    
    返回:
        matplotlib.figure.Figure: 圖像對象
    """
    # 確保輸入為NumPy數組
    delta_w_pred = np.asarray(delta_w_pred)
    
    # 如果沒有提供nf_pred，則使用物理模型計算
    if nf_pred is None:
        nf_pred = a * np.power(delta_w_pred, b)
    else:
        nf_pred = np.asarray(nf_pred)
    
    # 創建圖像
    fig, ax = plt.subplots(figsize=figsize)
    
    # 設置對數刻度
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # 繪製預測的delta_w與nf的關係
    ax.scatter(delta_w_pred, nf_pred, alpha=0.7, edgecolor='k', label='模型預測')
    
    # 如果有真實的nf，也繪製出來
    if nf_true is not None:
        nf_true = np.asarray(nf_true)
        delta_w_theory = calculate_delta_w_theory(nf_true, a, b)
        ax.scatter(delta_w_theory, nf_true, alpha=0.5, edgecolor='r', 
                  facecolor='none', label='真實數據')
    
    # 繪製物理模型曲線
    delta_w_range = np.logspace(
        np.log10(max(1e-6, delta_w_pred.min() * 0.5)), 
        np.log10(delta_w_pred.max() * 2), 
        100
    )
    nf_physics = a * np.power(delta_w_range, b)
    ax.plot(delta_w_range, nf_physics, 'r-', linewidth=2, 
           label=f'物理模型: Nf = {a} × (ΔW)^{b}')
    
    # 設置坐標軸標籤和標題
    ax.set_xlabel('非線性塑性應變能密度變化量 (ΔW)')
    ax.set_ylabel('疲勞壽命 (Nf)')
    title = f'{model_name}: ΔW 與 Nf 關係分析' if model_name else 'ΔW 與 Nf 關係分析'
    ax.set_title(title)
    
    # 計算一些統計信息
    if nf_true is not None:
        # 計算物理一致性
        nf_from_delta_w = a * np.power(delta_w_pred, b)
        rel_diff = np.abs((nf_pred - nf_from_delta_w) / nf_from_delta_w) * 100
        
        stats_text = (f"物理一致性:\n"
                     f"平均相對差異: {np.mean(rel_diff):.2f}%\n"
                     f"最大相對差異: {np.max(rel_diff):.2f}%\n"
                     f"模型是否完全遵循物理公式: {'是' if np.max(rel_diff) < 1e-5 else '否'}")
        
        ax.text(0.05, 0.05, stats_text, transform=ax.transAxes,
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 添加網格和圖例
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # 調整布局
    plt.tight_layout()
    
    # 保存圖像
    _save_figure(fig, save_path)
    
    return fig

def plot_branch_delta_w_consistency(pinn_delta_w, lstm_delta_w, delta_w_theory=None, nf_true=None,
                                  model_name=None, figsize=(10, 6), save_path=None,
                                  a=A_COEFFICIENT, b=B_COEFFICIENT):
    """
    繪製PINN和LSTM分支delta_w預測一致性對比圖
    
    參數:
        pinn_delta_w (array-like): PINN分支預測的delta_w值
        lstm_delta_w (array-like): LSTM分支預測的delta_w值
        delta_w_theory (array-like, optional): 理論delta_w值
        nf_true (array-like, optional): 真實的疲勞壽命，如果提供則計算delta_w_theory
        model_name (str, optional): 模型名稱，用於標題
        figsize (tuple): 圖像尺寸
        save_path (str, optional): 保存圖像的路徑
        a (float): 物理模型係數a
        b (float): 物理模型係數b
    
    返回:
        matplotlib.figure.Figure: 圖像對象
    """
    # 確保輸入為NumPy數組
    pinn_delta_w = np.asarray(pinn_delta_w)
    lstm_delta_w = np.asarray(lstm_delta_w)
    
    # 如果沒有提供delta_w_theory但提供了nf_true，則計算delta_w_theory
    if delta_w_theory is None and nf_true is not None:
        delta_w_theory = calculate_delta_w_theory(nf_true, a, b)
    
    # 創建圖像
    if delta_w_theory is not None:
        fig, axs = plt.subplots(1, 3, figsize=figsize)
    else:
        fig, axs = plt.subplots(1, 2, figsize=figsize)
    
    # 設置對數刻度
    for ax in axs:
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    # 1. PINN vs LSTM的delta_w預測對比
    axs[0].scatter(pinn_delta_w, lstm_delta_w, alpha=0.7, edgecolor='k')
    min_val = min(pinn_delta_w.min(), lstm_delta_w.min())
    max_val = max(pinn_delta_w.max(), lstm_delta_w.max())
    axs[0].plot([min_val, max_val], [min_val, max_val], 'r--')
    axs[0].set_xlabel('PINN 預測的 ΔW')
    axs[0].set_ylabel('LSTM 預測的 ΔW')
    axs[0].set_title('PINN vs LSTM: ΔW 預測對比')
    
    # 計算一致性指標
    rel_diff = np.abs((pinn_delta_w - lstm_delta_w) / pinn_delta_w) * 100
    log_pinn = np.log10(pinn_delta_w)
    log_lstm = np.log10(lstm_delta_w)
    log_mse = np.mean((log_pinn - log_lstm) ** 2)
    
    stats_text = (f"一致性指標:\n"
                 f"平均相對差異: {np.mean(rel_diff):.2f}%\n"
                 f"最大相對差異: {np.max(rel_diff):.2f}%\n"
                 f"對數MSE: {log_mse:.4f}")
    
    axs[0].text(0.05, 0.95, stats_text, transform=axs[0].transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. 預測分佈對比
    kde_x = np.logspace(
        np.log10(min(pinn_delta_w.min(), lstm_delta_w.min()) * 0.9),
        np.log10(max(pinn_delta_w.max(), lstm_delta_w.max()) * 1.1),
        1000
    )
    
    # 計算核密度估計
    from scipy.stats import gaussian_kde
    try:
        pinn_kde = gaussian_kde(np.log10(pinn_delta_w))
        lstm_kde = gaussian_kde(np.log10(lstm_delta_w))
        
        # 轉換回原始空間
        pinn_pdf = pinn_kde(np.log10(kde_x)) / kde_x / np.log(10)
        lstm_pdf = lstm_kde(np.log10(kde_x)) / kde_x / np.log(10)
        
        # 繪製密度曲線
        axs[1].plot(kde_x, pinn_pdf, 'b-', label='PINN')
        axs[1].plot(kde_x, lstm_pdf, 'g-', label='LSTM')
        
        if delta_w_theory is not None:
            theory_kde = gaussian_kde(np.log10(delta_w_theory))
            theory_pdf = theory_kde(np.log10(kde_x)) / kde_x / np.log(10)
            axs[1].plot(kde_x, theory_pdf, 'r--', label='理論值')
    except Exception as e:
        # 如果核密度估計失敗，則使用直方圖
        axs[1].hist(np.log10(pinn_delta_w), bins=15, alpha=0.5, label='PINN')
        axs[1].hist(np.log10(lstm_delta_w), bins=15, alpha=0.5, label='LSTM')
        if delta_w_theory is not None:
            axs[1].hist(np.log10(delta_w_theory), bins=15, alpha=0.5, label='理論值')
        axs[1].set_xscale('linear')  # 對數轉換過的數據用線性尺度表示
        axs[1].set_xlabel('log10(ΔW)')
    
    axs[1].set_title('ΔW 預測分佈對比')
    if axs[1].get_xscale() == 'log':
        axs[1].set_xlabel('ΔW')
    axs[1].set_ylabel('概率密度')
    axs[1].legend()
    
    # 3. 與理論值的對比
    if delta_w_theory is not None:
        delta_w_theory = np.asarray(delta_w_theory)
        
        # 繪製PINN和LSTM與理論值的相對誤差箱線圖
        pinn_rel_error = (pinn_delta_w - delta_w_theory) / delta_w_theory * 100
        lstm_rel_error = (lstm_delta_w - delta_w_theory) / delta_w_theory * 100
        
        data = [pinn_rel_error, lstm_rel_error]
        axs[2].boxplot(data, labels=['PINN', 'LSTM'])
        axs[2].set_title('相對於理論值的誤差對比')
        axs[2].set_ylabel('相對誤差 (%)')
        axs[2].grid(True, linestyle='--', alpha=0.7)
        axs[2].axhline(y=0, color='r', linestyle='--')
        
        # 添加一些統計信息
        pinn_stats = f"PINN: 平均={np.mean(pinn_rel_error):.2f}%, 中位數={np.median(pinn_rel_error):.2f}%"
        lstm_stats = f"LSTM: 平均={np.mean(lstm_rel_error):.2f}%, 中位數={np.median(lstm_rel_error):.2f}%"
        
        axs[2].text(0.5, 0.05, f"{pinn_stats}\n{lstm_stats}", 
                   transform=axs[2].transAxes, ha='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 設置主標題
    main_title = f'{model_name}: 分支 ΔW 預測一致性分析' if model_name else '分支 ΔW 預測一致性分析'
    fig.suptitle(main_title)
    
    # 調整布局
    plt.tight_layout()
    
    # 保存圖像
    _save_figure(fig, save_path)
    
    return fig

def evaluate_delta_w_prediction(delta_w_pred, delta_w_theory=None, nf_true=None,
                              a=A_COEFFICIENT, b=B_COEFFICIENT):
    """
    評估delta_w預測的準確性
    
    參數:
        delta_w_pred (array-like): 預測的delta_w值
        delta_w_theory (array-like, optional): 理論delta_w值
        nf_true (array-like, optional): 真實的疲勞壽命，如果提供則計算delta_w_theory
        a (float): 物理模型係數a
        b (float): 物理模型係數b
    
    返回:
        dict: 評估指標
    """
    # 確保輸入為NumPy數組
    delta_w_pred = np.asarray(delta_w_pred)
    
    # 如果沒有提供delta_w_theory但提供了nf_true，則計算delta_w_theory
    if delta_w_theory is None and nf_true is not None:
        delta_w_theory = calculate_delta_w_theory(nf_true, a, b)
    
    if delta_w_theory is None:
        raise ValueError("必須提供delta_w_theory或nf_true")
    
    delta_w_theory = np.asarray(delta_w_theory)
    
    # 計算各種評估指標
    # 1. 絕對誤差指標
    mse = np.mean((delta_w_theory - delta_w_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(delta_w_theory - delta_w_pred))
    
    # 2. 對數空間指標
    log_delta_w_theory = np.log10(delta_w_theory)
    log_delta_w_pred = np.log10(delta_w_pred)
    log_mse = np.mean((log_delta_w_theory - log_delta_w_pred) ** 2)
    log_rmse = np.sqrt(log_mse)
    log_mae = np.mean(np.abs(log_delta_w_theory - log_delta_w_pred))
    
    # 3. 相對誤差指標
    rel_error = np.abs((delta_w_theory - delta_w_pred) / delta_w_theory) * 100
    mean_rel_error = np.mean(rel_error)
    median_rel_error = np.median(rel_error)
    min_rel_error = np.min(rel_error)
    max_rel_error = np.max(rel_error)
    
    # 4. 10%和20%閾值內的樣本比例
    within_10_percent = np.mean(rel_error <= 10.0) * 100
    within_20_percent = np.mean(rel_error <= 20.0) * 100
    
    # 使用對數誤差計算相關係數
    from scipy.stats import pearsonr
    log_correlation, _ = pearsonr(log_delta_w_theory, log_delta_w_pred)
    
    # 收集指標
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'log_rmse': log_rmse,
        'log_mae': log_mae,
        'mean_rel_error': mean_rel_error,
        'median_rel_error': median_rel_error,
        'min_rel_error': min_rel_error,
        'max_rel_error': max_rel_error,
        'within_10_percent': within_10_percent,
        'within_20_percent': within_20_percent,
        'log_correlation': log_correlation
    }
    
    return metrics

def generate_delta_w_report(delta_w_pred, delta_w_theory=None, nf_true=None, nf_pred=None,
                          pinn_delta_w=None, lstm_delta_w=None, model_name=None, 
                          output_dir=None, a=A_COEFFICIENT, b=B_COEFFICIENT):
    """
    生成完整的delta_w預測分析報告
    
    參數:
        delta_w_pred (array-like): 預測的delta_w值
        delta_w_theory (array-like, optional): 理論delta_w值
        nf_true (array-like, optional): 真實的疲勞壽命，如果提供則計算delta_w_theory
        nf_pred (array-like, optional): 預測的疲勞壽命
        pinn_delta_w (array-like, optional): PINN分支預測的delta_w值
        lstm_delta_w (array-like, optional): LSTM分支預測的delta_w值
        model_name (str, optional): 模型名稱
        output_dir (str, optional): 輸出目錄
        a (float): 物理模型係數a
        b (float): 物理模型係數b
    
    返回:
        dict: 評估報告，包含指標和圖像路徑
    """
    # 如果沒有提供輸出目錄，使用默認目錄
    if output_dir is None:
        output_dir = './delta_w_analysis'
    
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 確保輸入為NumPy數組
    delta_w_pred = np.asarray(delta_w_pred)
    
    # 如果沒有提供delta_w_theory但提供了nf_true，則計算delta_w_theory
    if delta_w_theory is None and nf_true is not None:
        delta_w_theory = calculate_delta_w_theory(nf_true, a, b)
    
    # 創建報告字典
    report = {
        'metrics': None,
        'figures': {}
    }
    
    # 評估delta_w預測
    if delta_w_theory is not None or nf_true is not None:
        metrics = evaluate_delta_w_prediction(delta_w_pred, delta_w_theory, nf_true, a, b)
        report['metrics'] = metrics
        
        # 記錄評估結果
        logger.info(f"\nDelta_W預測評估結果: ")
        logger.info(f"RMSE: {metrics['rmse']:.6f}")
        logger.info(f"Log10 RMSE: {metrics['log_rmse']:.4f}")
        logger.info(f"平均相對誤差: {metrics['mean_rel_error']:.2f}%")
        logger.info(f"中位數相對誤差: {metrics['median_rel_error']:.2f}%")
        logger.info(f"10%閾值內樣本比例: {metrics['within_10_percent']:.2f}%")
        logger.info(f"20%閾值內樣本比例: {metrics['within_20_percent']:.2f}%")
        logger.info(f"對數空間相關係數: {metrics['log_correlation']:.4f}")
        
        # 生成delta_w預測與理論值的對比圖
        fig1_path = os.path.join(output_dir, 'delta_w_prediction_vs_theory.png')
        fig1 = plot_delta_w_prediction_vs_theory(
            delta_w_pred, delta_w_theory, nf_true, 
            model_name=model_name, save_path=fig1_path,
            a=a, b=b
        )
        report['figures']['prediction_vs_theory'] = fig1_path
        plt.close(fig1)
        
        # 生成delta_w預測誤差分佈圖
        fig2_path = os.path.join(output_dir, 'delta_w_error_distribution.png')
        fig2 = plot_delta_w_error_distribution(
            delta_w_pred, delta_w_theory, nf_true,
            model_name=model_name, save_path=fig2_path,
            a=a, b=b
        )
        report['figures']['error_distribution'] = fig2_path
        plt.close(fig2)
    
    # 生成delta_w與疲勞壽命關係圖
    fig3_path = os.path.join(output_dir, 'delta_w_nf_relationship.png')
    fig3 = plot_delta_w_nf_relationship(
        delta_w_pred, nf_pred, nf_true,
        model_name=model_name, save_path=fig3_path,
        a=a, b=b
    )
    report['figures']['nf_relationship'] = fig3_path
    plt.close(fig3)
    
    # 如果有PINN和LSTM分支的delta_w預測，生成一致性分析圖
    if pinn_delta_w is not None and lstm_delta_w is not None:
        fig4_path = os.path.join(output_dir, 'branch_delta_w_consistency.png')
        fig4 = plot_branch_delta_w_consistency(
            pinn_delta_w, lstm_delta_w, delta_w_theory, nf_true,
            model_name=model_name, save_path=fig4_path,
            a=a, b=b
        )
        report['figures']['branch_consistency'] = fig4_path
        plt.close(fig4)
    
    # 保存評估指標到CSV文件
    if report['metrics'] is not None:
        metrics_df = pd.DataFrame([report['metrics']])
        metrics_path = os.path.join(output_dir, 'delta_w_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Delta_W評估指標已保存至: {metrics_path}")
    
    return report


if __name__ == "__main__":
    # 設定日誌
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 創建模擬數據進行工具測試
    logger.info("測試Delta_W可視化工具")
    
    # 生成模擬真實值
    np.random.seed(42)
    n_samples = 50
    
    # 生成理論delta_w值（對數正態分佈）
    delta_w_theory = np.exp(np.random.normal(-4, 1, n_samples))
    
    # 計算理論疲勞壽命值
    nf_true = A_COEFFICIENT * np.power(delta_w_theory, B_COEFFICIENT)
    
    # 生成帶有一定誤差的預測delta_w值
    log_error = np.random.normal(0, 0.2, n_samples)
    delta_w_pred = delta_w_theory * np.exp(log_error)
    
    # 生成PINN和LSTM分支的預測值
    pinn_log_error = np.random.normal(0.1, 0.3, n_samples)
    lstm_log_error = np.random.normal(-0.1, 0.3, n_samples)
    pinn_delta_w = delta_w_theory * np.exp(pinn_log_error)
    lstm_delta_w = delta_w_theory * np.exp(lstm_log_error)
    
    # 計算預測的疲勞壽命
    nf_pred = A_COEFFICIENT * np.power(delta_w_pred, B_COEFFICIENT)
    
    # 測試生成完整報告
    logger.info("生成模擬Delta_W預測分析報告")
    test_dir = './test_delta_w_analysis'
    report = generate_delta_w_report(
        delta_w_pred=delta_w_pred,
        delta_w_theory=delta_w_theory,
        nf_true=nf_true,
        nf_pred=nf_pred,
        pinn_delta_w=pinn_delta_w,
        lstm_delta_w=lstm_delta_w,
        model_name="測試模型",
        output_dir=test_dir
    )
    
    logger.info(f"測試完成，報告已生成在: {test_dir}")
    logger.info(f"生成的圖像文件: {list(report['figures'].values())}")