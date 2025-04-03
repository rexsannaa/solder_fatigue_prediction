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

# 設定中文字體支援
try:
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 正確顯示負號
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"設定中文字體支援時出錯: {str(e)}，圖表中的中文可能無法正確顯示")

logger = logging.getLogger(__name__)


def plot_prediction_vs_true(y_true, y_pred, model_name=None, figsize=(10, 6), 
                           save_path=None, show_metrics=True, log_scale=False):
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
    # 確保輸入為numpy數組
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # 創建圖像
    fig, ax = plt.subplots(figsize=figsize)
    
    # 設置刻度
    if log_scale and np.all(y_true > 0) and np.all(y_pred > 0):
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    # 繪製散點圖
    scatter = ax.scatter(y_true, y_pred, alpha=0.6, 
                        edgecolor='k', s=50)
    
    # 繪製理想的對角線
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    
    # 擴展一點範圍，使圖像更美觀
    range_val = max_val - min_val
    min_val = max(0, min_val - range_val * 0.05)
    max_val = max_val + range_val * 0.05
    
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', 
           label='Ideal (True = Predicted)')
    
    # 繪製誤差範圍線（±10%和±20%）
    if not log_scale:
        x_range = np.linspace(min_val, max_val, 100)
        ax.plot(x_range, x_range * 1.2, 'g--', alpha=0.5, label='+20%')
        ax.plot(x_range, x_range * 0.8, 'g--', alpha=0.5, label='-20%')
        ax.plot(x_range, x_range * 1.1, 'y--', alpha=0.5, label='+10%')
        ax.plot(x_range, x_range * 0.9, 'y--', alpha=0.5, label='-10%')
    
    # 設置標籤和標題
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    
    title = 'Prediction vs True Values'
    if model_name:
        title = f'{model_name}: {title}'
    ax.set_title(title)
    
    # 如果顯示指標，計算並添加指標文本
    if show_metrics:
        from src.utils.metrics import calculate_rmse, calculate_r2, calculate_mae
        
        rmse = calculate_rmse(y_true, y_pred)
        r2 = calculate_r2(y_true, y_pred)
        mae = calculate_mae(y_true, y_pred)
        
        # 計算相對誤差
        rel_error = np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8)) * 100
        mean_rel_error = np.mean(rel_error)
        median_rel_error = np.median(rel_error)
        
        metrics_text = (
            f"RMSE: {rmse:.4f}\n"
            f"R²: {r2:.4f}\n"
            f"MAE: {mae:.4f}\n"
            f"Mean Rel. Error: {mean_rel_error:.2f}%\n"
            f"Median Rel. Error: {median_rel_error:.2f}%"
        )
        
        # 使用文本框顯示指標
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=props, fontsize=9)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # 添加一個標題行說明數據範圍
    plt.figtext(0.5, 0.01, 
               f"Data Range: [{min_val:.2e}, {max_val:.2e}]", 
               ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # 保存圖像
    if save_path:
        try:
            # 確保目錄存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"圖像已保存至: {save_path}")
        except Exception as e:
            logger.error(f"保存圖像失敗: {str(e)}")
    
    # 保存圖像
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"圖像已保存至: {save_path}")
        except Exception as e:
            logger.error(f"保存圖像失敗: {str(e)}")
    
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
    # 確保輸入為numpy數組
    parameters = np.asarray(parameters)
    predictions = np.asarray(predictions)
    
    # 創建圖像
    fig, ax = plt.subplots(figsize=figsize)
    
    # 繪製散點圖
    scatter = ax.scatter(parameters, predictions, 
                        alpha=0.7, edgecolor='k')
    
    # 嘗試繪製趨勢線
    try:
        from scipy.stats import linregress
        
        slope, intercept, r_value, p_value, std_err = linregress(parameters, predictions)
        trend_x = np.linspace(np.min(parameters), np.max(parameters), 100)
        trend_y = slope * trend_x + intercept
        
        ax.plot(trend_x, trend_y, 'r--', 
               label=f'Trend: y={slope:.4e}x+{intercept:.4e}, R²={r_value**2:.4f}')
        
        # 如果數據點足夠多，嘗試多項式擬合
        if len(parameters) > 5:
            try:
                from numpy.polynomial.polynomial import Polynomial
                
                # 2次多項式擬合
                p = Polynomial.fit(parameters, predictions, 2)
                poly_x = np.linspace(np.min(parameters), np.max(parameters), 100)
                poly_y = p(poly_x)
                
                ax.plot(poly_x, poly_y, 'g-', 
                       label=f'Polynomial Fit (degree=2)')
            except Exception:
                pass
    except Exception:
        pass
    
    # 設置標籤和標題
    ax.set_xlabel(parameter_name)
    ax.set_ylabel('預測疲勞壽命')
    ax.set_title(f'{parameter_name}對疲勞壽命的影響')
    
    # 添加網格和圖例
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # 添加數據範圍說明
    plt.figtext(0.5, 0.01, 
               f"{parameter_name}範圍: [{np.min(parameters)}, {np.max(parameters)}], "
               f"壽命範圍: [{np.min(predictions):.2e}, {np.max(predictions):.2e}]",
               ha='center', fontsize=9)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # 保存圖像
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"圖像已保存至: {save_path}")
        except Exception as e:
            logger.error(f"保存圖像失敗: {str(e)}")
    
    return fig


def plot_physical_constraint_validation(delta_w_values, nf_values, figsize=(10, 6), 
                                       a=55.83, b=-2.259, save_path=None):
    """
    繪製物理約束驗證圖
    驗證預測結果是否符合Nf=a*(ΔW)^b的物理關係
    
    參數:
        delta_w_values (array-like): 非線性塑性應變能密度變化量
        nf_values (array-like): 疲勞壽命值
        figsize (tuple): 圖像尺寸
        a (float): 物理模型係數a
        b (float): 物理模型係數b
        save_path (str, optional): 保存圖像的路徑
        
    返回:
        matplotlib.figure.Figure: 圖像對象
    """
    # 確保輸入為numpy數組
    delta_w_values = np.asarray(delta_w_values)
    nf_values = np.asarray(nf_values)
    
    # 創建圖像
    fig, ax = plt.subplots(figsize=figsize)
    
    # 使用對數刻度
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # 繪製散點圖
    scatter = ax.scatter(delta_w_values, nf_values, 
                        alpha=0.7, edgecolor='k', label='預測值')
    
    # 生成理論曲線
    x_range = np.logspace(np.log10(np.min(delta_w_values) * 0.5), 
                         np.log10(np.max(delta_w_values) * 2), 100)
    y_theory = a * np.power(x_range, b)
    
    # 繪製理論曲線
    ax.plot(x_range, y_theory, 'r-', label=f'物理模型: Nf={a}*(ΔW)^{b}')
    
    # 計算理論值與預測值之間的偏差
    y_theory_at_x = a * np.power(delta_w_values, b)
    relative_error = np.abs((nf_values - y_theory_at_x) / y_theory_at_x) * 100
    mean_rel_error = np.mean(relative_error)
    median_rel_error = np.median(relative_error)
    
    # 設置標籤和標題
    ax.set_xlabel('非線性塑性應變能密度變化量 (ΔW)')
    ax.set_ylabel('疲勞壽命 (Nf)')
    ax.set_title('物理約束驗證: Nf vs ΔW')
    
    # 添加網格和圖例
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # 添加偏差統計信息
    stats_text = (
        f"與物理模型偏差統計:\n"
        f"平均相對誤差: {mean_rel_error:.2f}%\n"
        f"中位數相對誤差: {median_rel_error:.2f}%\n"
        f"最大相對誤差: {np.max(relative_error):.2f}%\n"
        f"最小相對誤差: {np.min(relative_error):.2f}%"
    )
    
    # 放置文本框
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
           fontsize=9)
    
    plt.tight_layout()
    
    # 保存圖像
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"圖像已保存至: {save_path}")
        except Exception as e:
            logger.error(f"保存圖像失敗: {str(e)}")
    
    return fig


def visualize_model_results(results, output_dir="./visualizations", prefix=""):
    """
    將模型預測結果進行全面視覺化，生成多個圖表
    
    參數:
        results (dict): 模型預測結果，包含predictions, targets, delta_w等
        output_dir (str): 輸出目錄
        prefix (str): 檔案名前綴
        
    返回:
        list: 保存的圖像路徑列表
    """
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    saved_paths = []
    
    # 1. 預測值與真實值對比圖
    if 'predictions' in results and 'targets' in results and results['targets'] is not None:
        pred_vs_true_path = os.path.join(output_dir, f"{prefix}pred_vs_true.png")
        fig = plot_prediction_vs_true(
            results['targets'], results['predictions'], 
            save_path=pred_vs_true_path
        )
        plt.close(fig)
        saved_paths.append(pred_vs_true_path)
        
        # 2. 誤差分析圖
        error_analysis_path = os.path.join(output_dir, f"{prefix}error_analysis.png")
        fig = plot_prediction_error_analysis(
            results['targets'], results['predictions'],
            save_path=error_analysis_path
        )
        plt.close(fig)
        saved_paths.append(error_analysis_path)
        
        # 3. 相對誤差直方圖
        rel_error_hist_path = os.path.join(output_dir, f"{prefix}rel_error_hist.png")
        fig = create_error_histogram(
            results['targets'], results['predictions'],
            save_path=rel_error_hist_path,
            error_type='relative'
        )
        plt.close(fig)
        saved_paths.append(rel_error_hist_path)
    
    # 4. 物理約束驗證圖
    if 'delta_w' in results and 'predictions' in results:
        physics_path = os.path.join(output_dir, f"{prefix}physics_validation.png")
        fig = plot_physical_constraint_validation(
            results['delta_w'], results['predictions'],
            save_path=physics_path
        )
        plt.close(fig)
        saved_paths.append(physics_path)
    
    # 5. 注意力權重視覺化
    if 'attention_weights' in results:
        attention_path = os.path.join(output_dir, f"{prefix}attention_weights.png")
        fig = plot_attention_weights(
            results['attention_weights'],
            save_path=attention_path
        )
        plt.close(fig)
        saved_paths.append(attention_path)
    
    # 6. 融合權重視覺化
    if 'fusion_weights' in results:
        fusion_path = os.path.join(output_dir, f"{prefix}fusion_weights.png")
        fusion_weights = results['fusion_weights']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        labels = ['PINN', 'LSTM']
        avg_weights = np.mean(fusion_weights, axis=0)
        ax.bar(labels, avg_weights, color=['blue', 'orange'])
        ax.set_ylabel('平均權重')
        ax.set_title('PINN與LSTM分支融合權重')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 在條形上添加數值
        for i, w in enumerate(avg_weights):
            ax.text(i, w + 0.01, f'{w:.3f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(fusion_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        saved_paths.append(fusion_path)
    
    logger.info(f"已生成 {len(saved_paths)} 個視覺化圖表於 {output_dir}")
    return saved_paths


if __name__ == "__main__":
    # 簡單的測試代碼
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 創建測試數據
    np.random.seed(42)
    y_true = np.exp(np.random.normal(10, 2, size=50))  # 模擬疲勞壽命真實值
    y_pred = y_true * (1 + np.random.normal(0, 0.3, size=y_true.shape))  # 模擬預測值
    
    # 測試預測值與真實值對比圖
    logger.info("測試預測值與真實值對比圖")
    fig = plot_prediction_vs_true(y_true, y_pred, model_name="PINN-LSTM")
    plt.show()
    
    # 測試誤差分析圖
    logger.info("測試誤差分析圖")
    fig = plot_prediction_error_analysis(y_true, y_pred)
    plt.show()
    
    # 測試物理約束驗證圖
    logger.info("測試物理約束驗證圖")
    delta_w = np.power(y_true / 55.83, 1/-2.259) * (1 + np.random.normal(0, 0.1, size=y_true.shape))
    fig = plot_physical_constraint_validation(delta_w, y_pred)
    plt.show()
    
    # 測試注意力權重視覺化
    logger.info("測試注意力權重視覺化")
    attention_weights = np.random.dirichlet(np.ones(4), size=1)[0]  # 模擬注意力權重
    fig = plot_attention_weights(attention_weights)
    plt.show()
    
    logger.info("所有測試完成")


def plot_prediction_error_analysis(y_true, y_pred, x_feature=None, feature_name=None,
                                 figsize=(15, 8), save_path=None):
    """
    繪製預測誤差分析圖
    
    參數:
        y_true (array-like): 真實值
        y_pred (array-like): 預測值
        x_feature (array-like, optional): 用於分析誤差與特徵關係的特徵值
        feature_name (str, optional): 特徵名稱
        figsize (tuple): 圖像尺寸
        save_path (str, optional): 保存圖像的路徑
        
    返回:
        matplotlib.figure.Figure: 圖像對象
    """
    # 確保輸入為numpy數組
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # 計算絕對誤差和相對誤差
    abs_error = np.abs(y_true - y_pred)
    rel_error = np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8)) * 100
    
    # 創建圖像
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 繪製絕對誤差直方圖
    axes[0, 0].hist(abs_error, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_xlabel('Absolute Error')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Absolute Error Distribution')
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    # 繪製相對誤差直方圖
    axes[0, 1].hist(rel_error, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_xlabel('Relative Error (%)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Relative Error Distribution')
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)
    
    # 繪製真實值與絕對誤差的關係
    axes[1, 0].scatter(y_true, abs_error, alpha=0.6, edgecolor='k')
    axes[1, 0].set_xlabel('True Values')
    axes[1, 0].set_ylabel('Absolute Error')
    axes[1, 0].set_title('True Values vs Absolute Error')
    axes[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    # 如果提供了特徵，繪製特徵與誤差的關係
    if x_feature is not None:
        x_feature = np.asarray(x_feature)
        feature_title = feature_name if feature_name else 'Feature'
        
        axes[1, 1].scatter(x_feature, rel_error, alpha=0.6, edgecolor='k')
        axes[1, 1].set_xlabel(feature_title)
        axes[1, 1].set_ylabel('Relative Error (%)')
        axes[1, 1].set_title(f'{feature_title} vs Relative Error')
        
        # 如果點數較多，嘗試繪製趨勢線
        if len(x_feature) > 10:
            try:
                from scipy.stats import linregress
                
                slope, intercept, r_value, p_value, std_err = linregress(x_feature, rel_error)
                trend_x = np.linspace(np.min(x_feature), np.max(x_feature), 100)
                trend_y = slope * trend_x + intercept
                
                axes[1, 1].plot(trend_x, trend_y, 'r--', 
                               label=f'Trend: y={slope:.4f}x+{intercept:.4f}, R²={r_value**2:.4f}')
                axes[1, 1].legend()
            except Exception:
                pass
    else:
        # 如果沒有提供特徵，繪製預測值與相對誤差的關係
        axes[1, 1].scatter(y_pred, rel_error, alpha=0.6, edgecolor='k')
        axes[1, 1].set_xlabel('Predicted Values')
        axes[1, 1].set_ylabel('Relative Error (%)')
        axes[1, 1].set_title('Predicted Values vs Relative Error')
    
    axes[1, 1].grid(True, linestyle='--', alpha=0.7)
    
    # 添加摘要統計信息
    summary_text = (
        f"Mean Abs. Error: {np.mean(abs_error):.4f}\n"
        f"Median Abs. Error: {np.median(abs_error):.4f}\n"
        f"Mean Rel. Error: {np.mean(rel_error):.2f}%\n"
        f"Median Rel. Error: {np.median(rel_error):.2f}%\n"
        f"90th Percentile Rel. Error: {np.percentile(rel_error, 90):.2f}%"
    )
    
    plt.figtext(0.5, 0.01, summary_text, ha='center', 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # 為底部文本留出空間
    
    # 保存圖像
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"圖像已保存至: {save_path}")
        except Exception as e:
            logger.error(f"保存圖像失敗: {str(e)}")
    
    return fig


def plot_training_history(history, metrics=None, figsize=(12, 6), 
                         save_path=None, start_epoch=0):
    """
    繪製模型訓練歷史曲線
    
    參數:
        history (dict): 訓練歷史記錄，包含訓練損失和驗證損失等
        metrics (list, optional): 要繪製的指標列表，默認為訓練損失和驗證損失
        figsize (tuple): 圖像尺寸
        save_path (str, optional): 保存圖像的路徑
        start_epoch (int): 起始輪次，用於截取歷史記錄
        
    返回:
        matplotlib.figure.Figure: 圖像對象
    """
    if not history:
        logger.warning("訓練歷史記錄為空")
        return None
    
    # 默認繪製訓練損失和驗證損失
    if metrics is None:
        metrics = ['loss']
        if 'val_loss' in history:
            metrics.append('val_loss')
    
    # 創建圖像
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # 獲取指標數據
        if metric == 'loss':
            values = history.get('train_loss', [])
            label = 'Training Loss'
            color = 'blue'
        elif metric == 'val_loss':
            values = history.get('val_loss', [])
            label = 'Validation Loss'
            color = 'red'
        else:
            # 嘗試從metrics_history中獲取
            if 'metrics_history' in history and metric in history['metrics_history']:
                values = history['metrics_history'][metric]
                label = f'{metric.replace("_", " ").title()}'
                color = 'green'
            else:
                values = history.get(metric, [])
                label = f'{metric.replace("_", " ").title()}'
                color = 'green'
        
        # 截取歷史記錄
        if start_epoch > 0 and len(values) > start_epoch:
            values = values[start_epoch:]
            epochs = np.arange(start_epoch, start_epoch + len(values))
        else:
            epochs = np.arange(len(values))
        
        # 繪製曲線
        ax.plot(epochs, values, marker='o', linestyle='-', 
               color=color, label=label, markersize=3)
        
        # 標記最佳值
        if len(values) > 0:
            if metric in ['val_loss', 'loss', 'rmse', 'mae', 'mape']:
                # 對於這些指標，值越小越好
                best_epoch = np.argmin(values)
                best_value = values[best_epoch]
                best_text = 'Minimum'
            else:
                # 對於其他指標（如R²），值越大越好
                best_epoch = np.argmax(values)
                best_value = values[best_epoch]
                best_text = 'Maximum'
            
            ax.scatter(epochs[best_epoch], best_value, 
                      color='red', s=100, zorder=10, alpha=0.8)
            ax.annotate(f'{best_text}: {best_value:.4f} (Epoch {epochs[best_epoch]})', 
                       xy=(epochs[best_epoch], best_value),
                       xytext=(epochs[best_epoch], best_value * 1.1 if best_value > 0 else best_value * 0.9),
                       arrowprops=dict(arrowstyle='->'), fontsize=8)
        
        # 設置標籤和標題
        ax.set_xlabel('Epoch')
        ax.set_ylabel(label)
        ax.set_title(f'{label} vs. Epoch')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 如果指標是損失，添加標記
        if 'loss' in metric.lower():
            if len(values) > 10:
                # 計算移動平均線以顯示趨勢
                window_size = min(5, len(values) // 5)
                if window_size > 1:
                    moving_avg = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
                    moving_avg_epochs = epochs[window_size-1:]
                    ax.plot(moving_avg_epochs, moving_avg, 'r--', alpha=0.5, 
                           label=f'{window_size}-Epoch Moving Avg')
            
            # 如果有其他損失成分，也可以一併繪製
            if 'loss_components' in history:
                components = history['loss_components']
                for comp_name, comp_values in components.items():
                    if len(comp_values) > 0:
                        comp_epochs = np.arange(len(comp_values))
                        ax.plot(comp_epochs, comp_values, '--', alpha=0.5, 
                               label=f'{comp_name.replace("_", " ").title()}')
        
        ax.legend()
    
    plt.tight_layout()
    
    # 保存圖像
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"圖像已保存至: {save_path}")
        except Exception as e:
            logger.error(f"保存圖像失敗: {str(e)}")
    
    return fig


def plot_feature_importance(feature_names, importance_values, model_name=None, 
                          figsize=(10, 6), save_path=None, top_n=None):
    """
    繪製特徵重要性條形圖
    
    參數:
        feature_names (list): 特徵名稱列表
        importance_values (array-like): 特徵重要性值
        model_name (str, optional): 模型名稱，用於標題
        figsize (tuple): 圖像尺寸
        save_path (str, optional): 保存圖像的路徑
        top_n (int, optional): 顯示前N個最重要的特徵
        
    返回:
        matplotlib.figure.Figure: 圖像對象
    """
    # 確保輸入為列表或數組
    feature_names = list(feature_names)
    importance_values = np.asarray(importance_values)
    
    # 如果設置了top_n，只顯示前N個特徵
    if top_n is not None and top_n < len(feature_names):
        # 獲取重要性排序的索引
        idx = np.argsort(importance_values)[-top_n:]
        feature_names = [feature_names[i] for i in idx]
        importance_values = importance_values[idx]
    
    # 創建圖像
    fig, ax = plt.subplots(figsize=figsize)
    
    # 繪製水平條形圖，按重要性降序排列
    sorted_idx = np.argsort(importance_values)
    ax.barh([feature_names[i] for i in sorted_idx], 
           [importance_values[i] for i in sorted_idx],
           color='skyblue', edgecolor='navy')
    
    # 設置標籤和標題
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    
    title = 'Feature Importance'
    if model_name:
        title = f'{model_name}: {title}'
    ax.set_title(title)
    
    # 在條形上添加數值
    for i, v in enumerate(sorted_idx):
        ax.text(importance_values[v] + 0.01, i, f'{importance_values[v]:.3f}', 
               va='center', fontsize=9)
    
    plt.tight_layout()
    
    # 保存圖像
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"圖像已保存至: {save_path}")
        except Exception as e:
            logger.error(f"保存圖像失敗: {str(e)}")
    
    return fig


def plot_attention_weights(attention_weights, sequence_labels=None, figsize=(10, 6), 
                         save_path=None, cmap='viridis', title=None):
    """
    繪製注意力權重熱力圖
    
    參數:
        attention_weights (array-like): 注意力權重矩陣，形狀為(batch_size, seq_len)或(seq_len,)
        sequence_labels (list, optional): 序列標籤列表
        figsize (tuple): 圖像尺寸
        save_path (str, optional): 保存圖像的路徑
        cmap (str): 色彩映射名稱
        title (str, optional): 圖像標題
        
    返回:
        matplotlib.figure.Figure: 圖像對象
    """
    # 確保輸入為numpy數組
    attention_weights = np.asarray(attention_weights)
    
    # 檢查維度，根據需要調整
    if attention_weights.ndim == 1:
        # 單樣本的權重
        attention_weights = attention_weights.reshape(1, -1)
    
    batch_size, seq_len = attention_weights.shape
    
    # 創建序列標籤（如果未提供）
    if sequence_labels is None:
        if seq_len == 4:
            # 假設是時間步: 3600, 7200, 10800, 14400
            sequence_labels = ['3600s', '7200s', '10800s', '14400s']
        else:
            sequence_labels = [f'Step {i+1}' for i in range(seq_len)]
    
    # 創建圖像
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    
    # 如果批次大小為1，直接繪製單個條形圖
    if batch_size == 1:
        ax = axes
        weights = attention_weights[0]
        
        # 繪製條形圖
        bars = ax.bar(sequence_labels, weights, 
                     color='skyblue', edgecolor='navy')
        
        # 在條形上添加數值
        for i, (bar, weight) in enumerate(zip(bars, weights)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{weight:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 設置標籤和標題
        ax.set_ylabel('Attention Weight')
        ax.set_ylim(0, max(weights) * 1.2)
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Attention Weights')
    else:
        # 批次大小大於1，繪製熱力圖
        max_samples = min(5, batch_size)  # 最多顯示5個樣本
        
        # 截取前max_samples個樣本
        weights = attention_weights[:max_samples]
        
        # 為熱力圖創建適當的Y軸標籤
        y_labels = [f'Sample {i+1}' for i in range(max_samples)]
        
        # 繪製熱力圖
        im = axes.imshow(weights, cmap=cmap, aspect='auto')
        
        # 添加顏色條
        cbar = fig.colorbar(im, ax=axes)
        cbar.set_label('Attention Weight')
        
        # 設置標籤
        axes.set_xticks(np.arange(len(sequence_labels)))
        axes.set_xticklabels(sequence_labels)
        axes.set_yticks(np.arange(len(y_labels)))
        axes.set_yticklabels(y_labels)
        
        # 在每個單元格中添加文本
        for i in range(len(y_labels)):
            for j in range(len(sequence_labels)):
                text = axes.text(j, i, f'{weights[i, j]:.2f}',
                                ha="center", va="center", color="w" if weights[i, j] > 0.5 else "k")
        
        if title:
            axes.set_title(title)
        else:
            axes.set_title('Attention Weights Heatmap')
    
    plt.tight_layout()
    
    # 保存圖像
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"圖像已保存至: {save_path}")
        except Exception as e:
            logger.error(f"保存圖像失敗: {str(e)}")
    
    return fig


def create_error_histogram(y_true, y_pred, bins=20, figsize=(10, 6), 
                         save_path=None, title=None, error_type='relative'):
    """
    創建誤差直方圖
    
    參數:
        y_true (array-like): 真實值
        y_pred (array-like): 預測值
        bins (int): 直方圖的箱數
        figsize (tuple): 圖像尺寸
        save_path (str, optional): 保存圖像的路徑
        title (str, optional): 圖像標題
        error_type (str): 誤差類型，'relative'或'absolute'
        
    返回:
        matplotlib.figure.Figure: 圖像對象
    """
    # 確保輸入為numpy數組
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # 計算誤差
    if error_type.lower() == 'relative':
        # 相對誤差（百分比）
        errors = np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8)) * 100
        x_label = 'Relative Error (%)'
        color = 'green'
    else:
        # 絕對誤差
        errors = np.abs(y_true - y_pred)
        x_label = 'Absolute Error'
        color = 'blue'
    
    # 創建圖像
    fig, ax = plt.subplots(figsize=figsize)
    
    # 繪製直方圖
    n, bins, patches = ax.hist(errors, bins=bins, 
                              color=color, alpha=0.7, 
                              edgecolor='black')
    
    # 計算統計量
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    p90_error = np.percentile(errors, 90)
    
    # 在圖上標記統計量
    ax.axvline(mean_error, color='red', linestyle='--', 
              label=f'Mean: {mean_error:.2f}')
    ax.axvline(median_error, color='orange', linestyle='-', 
              label=f'Median: {median_error:.2f}')
    ax.axvline(p90_error, color='purple', linestyle='-.', 
              label=f'90th Percentile: {p90_error:.2f}')
    
    # 設置標籤和標題
    ax.set_xlabel(x_label)
    ax.set_ylabel('Frequency')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'{error_type.title()} Error Distribution')
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # 添加統計摘要文本
    stats_text = (
        f"Mean: {mean_error:.2f}\n"
        f"Median: {median_error:.2f}\n"
        f"Std. Dev.: {np.std(errors):.2f}\n"
        f"Min: {np.min(errors):.2f}\n"
        f"Max: {np.max(errors):.2f}\n"
        f"90th Percentile: {p90_error:.2f}"
    )
    
    # 放置文本框在右上角
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
           fontsize=9)