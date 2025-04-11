#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
model_diagnostics.py - 模型診斷工具
用於診斷和分析銲錫接點疲勞壽命預測模型的問題，幫助找出預測偏差的根源。

此工具可以：
1. 詳細檢查模型的各個分支預測輸出
2. 分析物理模型的精確性
3. 比較不同分支貢獻度
4. 進行對數空間線性關係分析
5. 生成相關診斷圖表
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

def diagnose_model_outputs(outputs, targets, a_coefficient=55.83, b_coefficient=-2.259, 
                         save_dir=None, prefix=""):
    """
    詳細診斷模型輸出，查找問題根源
    
    參數:
        outputs (dict): 模型輸出字典
        targets (numpy.ndarray): 目標值
        a_coefficient (float): 物理模型係數a
        b_coefficient (float): 物理模型係數b
        save_dir (str): 保存結果的目錄
        prefix (str): 文件名前綴
        
    返回:
        dict: 診斷結果
    """
    # 確保目錄存在
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # 1. 檢查預測值與真實值
    predictions = outputs.get('nf_pred')
    if predictions is None:
        logger.warning("預測值不存在於輸出中")
        return {}
    
    # 2. 計算各類指標
    log_targets = np.log10(targets)
    log_predictions = np.log10(predictions)
    
    # 計算相對誤差
    rel_error = np.abs((targets - predictions) / targets) * 100
    log_rel_error = np.abs((log_targets - log_predictions)) * 100
    
    # 計算統計指標
    stats = {
        'mean_error': np.mean(targets - predictions),
        'median_error': np.median(targets - predictions),
        'mean_rel_error': np.mean(rel_error),
        'median_rel_error': np.median(rel_error),
        'mean_log_rel_error': np.mean(log_rel_error),
        'median_log_rel_error': np.median(log_rel_error),
        'min_target': np.min(targets),
        'max_target': np.max(targets),
        'mean_target': np.mean(targets),
        'min_prediction': np.min(predictions),
        'max_prediction': np.max(predictions),
        'mean_prediction': np.mean(predictions)
    }
    
    # 3. 檢查物理模型預測 vs 實際預測
    if 'delta_w' in outputs:
        delta_w = outputs['delta_w']
        
        # 用物理公式計算理論疲勞壽命
        physics_nf = a_coefficient * np.power(delta_w, b_coefficient)
        
        # 計算物理模型與真實值的誤差
        physics_rel_error = np.abs((targets - physics_nf) / targets) * 100
        
        # 計算物理模型與最終預測的誤差
        model_physics_rel_error = np.abs((predictions - physics_nf) / physics_nf) * 100
        
        # 添加到統計指標
        stats.update({
            'mean_physics_rel_error': np.mean(physics_rel_error),
            'median_physics_rel_error': np.median(physics_rel_error),
            'mean_model_physics_rel_error': np.mean(model_physics_rel_error),
            'median_model_physics_rel_error': np.median(model_physics_rel_error),
            'min_physics_nf': np.min(physics_nf),
            'max_physics_nf': np.max(physics_nf),
            'mean_physics_nf': np.mean(physics_nf)
        })
        
        # 計算相關系數
        target_physics_corr = np.corrcoef(targets, physics_nf)[0, 1]
        pred_physics_corr = np.corrcoef(predictions, physics_nf)[0, 1]
        target_pred_corr = np.corrcoef(targets, predictions)[0, 1]
        
        stats.update({
            'target_physics_corr': target_physics_corr,
            'pred_physics_corr': pred_physics_corr,
            'target_pred_corr': target_pred_corr
        })
        
        # 4. 對數空間線性關係分析
        log_delta_w = np.log10(delta_w)
        log_physics_nf = np.log10(physics_nf)
        
        # 理論物理關係
        theory_intercept = np.log10(a_coefficient)
        theory_slope = b_coefficient
        
        # 從數據擬合關係
        data_slope, data_intercept, data_r, data_p, data_std = linregress(log_delta_w, log_targets)
        pred_slope, pred_intercept, pred_r, pred_p, pred_std = linregress(log_delta_w, log_predictions)
        physics_slope, physics_intercept, physics_r, physics_p, physics_std = linregress(log_delta_w, log_physics_nf)
        
        # 添加到統計指標
        stats.update({
            'theory_intercept': theory_intercept,
            'theory_slope': theory_slope,
            'data_intercept': data_intercept,
            'data_slope': data_slope,
            'data_r2': data_r ** 2,
            'pred_intercept': pred_intercept,
            'pred_slope': pred_slope,
            'pred_r2': pred_r ** 2,
            'physics_intercept': physics_intercept,
            'physics_slope': physics_slope,
            'physics_r2': physics_r ** 2
        })
        
        # 5. 生成診斷圖表
        if save_dir:
            # 圖1：預測值 vs 真實值
            plt.figure(figsize=(10, 8))
            plt.scatter(targets, predictions, alpha=0.7, label='預測值')
            plt.scatter(targets, physics_nf, alpha=0.7, label='物理模型計算值')
            
            # 添加1:1對角線
            min_val = min(np.min(targets), np.min(predictions), np.min(physics_nf))
            max_val = max(np.max(targets), np.max(predictions), np.max(physics_nf))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='理想1:1線')
            
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('真實壽命')
            plt.ylabel('預測壽命')
            plt.title('預測值 vs 真實值 (對數尺度)')
            plt.legend()
            plt.grid(True, which='both', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{prefix}pred_vs_true_log.png"), dpi=300)
            plt.close()
            
            # 圖2：delta_w vs Nf關係
            plt.figure(figsize=(10, 8))
            plt.scatter(delta_w, targets, alpha=0.7, label='真實數據')
            plt.scatter(delta_w, predictions, alpha=0.7, label='模型預測')
            plt.scatter(delta_w, physics_nf, alpha=0.7, label='物理模型計算')
            
            # 添加理論曲線
            dw_range = np.logspace(np.log10(np.min(delta_w)), np.log10(np.max(delta_w)), 100)
            theory_nf = a_coefficient * np.power(dw_range, b_coefficient)
            plt.plot(dw_range, theory_nf, 'r-', label=f'理論曲線 Nf={a_coefficient}*(ΔW)^{b_coefficient}')
            
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('非線性塑性應變能密度變化量 (ΔW)')
            plt.ylabel('疲勞壽命 (Nf)')
            plt.title('ΔW vs Nf關係分析')
            plt.legend()
            plt.grid(True, which='both', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{prefix}delta_w_vs_nf.png"), dpi=300)
            plt.close()
            
            # 圖3：對數空間線性關係分析
            plt.figure(figsize=(10, 8))
            plt.scatter(log_delta_w, log_targets, alpha=0.7, label='真實數據')
            plt.scatter(log_delta_w, log_predictions, alpha=0.7, label='模型預測')
            
            # 添加理論線
            x_range = np.linspace(np.min(log_delta_w), np.max(log_delta_w), 100)
            theory_line = theory_intercept + theory_slope * x_range
            data_line = data_intercept + data_slope * x_range
            pred_line = pred_intercept + pred_slope * x_range
            
            plt.plot(x_range, theory_line, 'r-', 
                   label=f'理論關係: log(Nf)={theory_intercept:.4f}{theory_slope:.4f}*log(ΔW)')
            plt.plot(x_range, data_line, 'g--', 
                   label=f'真實數據擬合: log(Nf)={data_intercept:.4f}{data_slope:.4f}*log(ΔW), R²={data_r**2:.4f}')
            plt.plot(x_range, pred_line, 'b--', 
                   label=f'預測數據擬合: log(Nf)={pred_intercept:.4f}{pred_slope:.4f}*log(ΔW), R²={pred_r**2:.4f}')
            
            plt.xlabel('log(ΔW)')
            plt.ylabel('log(Nf)')
            plt.title('對數空間線性關係分析')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{prefix}log_linear_analysis.png"), dpi=300)
            plt.close()
            
            # 圖4：相對誤差分析
            plt.figure(figsize=(10, 8))
            plt.hist(rel_error, bins=20, alpha=0.7, label='模型預測相對誤差')
            plt.hist(physics_rel_error, bins=20, alpha=0.7, label='物理模型相對誤差')
            plt.xlabel('相對誤差 (%)')
            plt.ylabel('頻率')
            plt.title('相對誤差分佈')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{prefix}rel_error_hist.png"), dpi=300)
            plt.close()
            
            # 保存診斷結果到JSON文件
            with open(os.path.join(save_dir, f"{prefix}diagnostics.json"), 'w', encoding='utf-8') as f:
                # 將numpy數組轉換為列表以便JSON序列化
                json_stats = {}
                for k, v in stats.items():
                    if isinstance(v, np.ndarray):
                        json_stats[k] = v.tolist()
                    elif isinstance(v, np.float64) or isinstance(v, np.float32):
                        json_stats[k] = float(v)
                    else:
                        json_stats[k] = v
                json.dump(json_stats, f, indent=2, ensure_ascii=False)
            
            # 生成診斷報告文本文件
            with open(os.path.join(save_dir, f"{prefix}diagnostics_report.txt"), 'w', encoding='utf-8') as f:
                f.write("===== 銲錫接點疲勞壽命預測模型診斷報告 =====\n\n")
                
                f.write("1. 基本統計資訊\n")
                f.write(f"   真實值範圍: {stats['min_target']:.2f} 到 {stats['max_target']:.2f}, 平均: {stats['mean_target']:.2f}\n")
                f.write(f"   預測值範圍: {stats['min_prediction']:.2f} 到 {stats['max_prediction']:.2f}, 平均: {stats['mean_prediction']:.2f}\n")
                if 'mean_physics_nf' in stats:
                    f.write(f"   物理模型計算值範圍: {stats['min_physics_nf']:.2f} 到 {stats['max_physics_nf']:.2f}, 平均: {stats['mean_physics_nf']:.2f}\n")
                
                f.write("\n2. 誤差分析\n")
                f.write(f"   平均絕對誤差: {stats['mean_error']:.4f}\n")
                f.write(f"   中位絕對誤差: {stats['median_error']:.4f}\n")
                f.write(f"   平均相對誤差: {stats['mean_rel_error']:.2f}%\n")
                f.write(f"   中位相對誤差: {stats['median_rel_error']:.2f}%\n")
                f.write(f"   平均對數相對誤差: {stats['mean_log_rel_error']:.2f}%\n")
                f.write(f"   中位對數相對誤差: {stats['median_log_rel_error']:.2f}%\n")
                
                if 'mean_physics_rel_error' in stats:
                    f.write("\n3. 物理模型比較\n")
                    f.write(f"   物理模型平均相對誤差: {stats['mean_physics_rel_error']:.2f}%\n")
                    f.write(f"   物理模型中位相對誤差: {stats['median_physics_rel_error']:.2f}%\n")
                    f.write(f"   模型與物理模型差異平均相對誤差: {stats['mean_model_physics_rel_error']:.2f}%\n")
                    f.write(f"   模型與物理模型差異中位相對誤差: {stats['median_model_physics_rel_error']:.2f}%\n")
                
                if 'target_physics_corr' in stats:
                    f.write("\n4. 相關係數分析\n")
                    f.write(f"   真實值與物理模型相關係數: {stats['target_physics_corr']:.4f}\n")
                    f.write(f"   預測值與物理模型相關係數: {stats['pred_physics_corr']:.4f}\n")
                    f.write(f"   真實值與預測值相關係數: {stats['target_pred_corr']:.4f}\n")
                
                if 'theory_slope' in stats:
                    f.write("\n5. 對數空間線性關係分析\n")
                    f.write(f"   理論關係: log(Nf) = {stats['theory_intercept']:.4f} + {stats['theory_slope']:.4f}*log(ΔW)\n")
                    f.write(f"   真實數據擬合: log(Nf) = {stats['data_intercept']:.4f} + {stats['data_slope']:.4f}*log(ΔW), R² = {stats['data_r2']:.4f}\n")
                    f.write(f"   預測數據擬合: log(Nf) = {stats['pred_intercept']:.4f} + {stats['pred_slope']:.4f}*log(ΔW), R² = {stats['pred_r2']:.4f}\n")
                    f.write(f"   物理模型擬合: log(Nf) = {stats['physics_intercept']:.4f} + {stats['physics_slope']:.4f}*log(ΔW), R² = {stats['physics_r2']:.4f}\n")
                
                f.write("\n6. 問題診斷與建議\n")
                
                # 根據診斷結果自動生成問題診斷和建議
                model_vs_physics_diff = abs(stats['pred_r2'] - stats['physics_r2'])
                model_vs_theory_slope_diff = abs(stats['pred_slope'] - stats['theory_slope']) / abs(stats['theory_slope'])
                
                # 問題1: 檢查模型與物理模型的偏差
                if stats['mean_model_physics_rel_error'] > 20.0:
                    f.write("   問題: 模型預測與物理模型計算存在較大偏差，平均相對誤差為 {:.2f}%\n".format(
                        stats['mean_model_physics_rel_error']))
                    if stats['pred_r2'] < 0.9 * stats['physics_r2']:
                        f.write("   建議: 模型預測的確定係數(R²={:.4f})明顯低於物理模型(R²={:.4f})，".format(
                            stats['pred_r2'], stats['physics_r2']))
                        f.write("應調整模型架構，直接使用物理模型計算最終預測\n")
                
                # 問題2: 檢查斜率偏差
                if model_vs_theory_slope_diff > 0.2:
                    f.write("   問題: 模型預測的對數線性關係斜率({:.4f})與理論值({:.4f})存在較大偏差\n".format(
                        stats['pred_slope'], stats['theory_slope']))
                    f.write("   建議: 加強物理約束，確保預測結果符合理論關係 Nf = a * (ΔW)^b\n")
                
                # 問題3: 檢查模型預測與真實值的相關性
                if stats['target_pred_corr'] < 0.8:
                    f.write("   問題: 模型預測與真實值相關性較低，相關係數僅為 {:.4f}\n".format(
                        stats['target_pred_corr']))
                    if stats['target_physics_corr'] > stats['target_pred_corr'] + 0.1:
                        f.write("   建議: 物理模型與真實值相關性({:.4f})優於模型預測，".format(
                            stats['target_physics_corr']))
                        f.write("建議使用物理模型直接計算預測結果，或重新訓練模型\n")
                
                # 問題4: 檢查整體預測準確度
                if stats['mean_rel_error'] > 50.0:
                    f.write("   問題: 模型預測整體準確度較低，平均相對誤差為 {:.2f}%\n".format(
                        stats['mean_rel_error']))
                    if stats['mean_physics_rel_error'] < stats['mean_rel_error'] * 0.8:
                        f.write("   建議: 物理模型預測更準確(誤差為{:.2f}%)，".format(
                            stats['mean_physics_rel_error']))
                        f.write("應直接使用物理模型計算，避免神經網絡干擾物理關係\n")
                    else:
                        f.write("   建議: 檢查數據預處理流程，確保使用對數尺度處理疲勞壽命數據\n")
                
                # 問題5: 檢查大值預測區域
                large_value_mask = log_targets > np.log10(1000)
                if np.any(large_value_mask):
                    large_rel_error = np.mean(rel_error[large_value_mask])
                    f.write("   問題: 大值區域(Nf>1000)的預測相對誤差為 {:.2f}%\n".format(
                        large_rel_error))
                    if large_rel_error > stats['mean_rel_error'] * 1.5:
                        f.write("   建議: 大值區域預測誤差顯著高於平均，應加強對這些區域的訓練權重\n")
    
    return stats