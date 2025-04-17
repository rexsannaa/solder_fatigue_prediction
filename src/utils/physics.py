#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
physics.py - 物理約束函數模組
本模組提供銲錫接點疲勞壽命預測所需的物理約束方程和物理知識轉換函數，
基於能量密度法疲勞壽命模型的物理關係和能量守恆原理。

主要功能:
1. 實現疲勞壽命與非線性塑性應變能密度變化量(ΔW)的轉換關係
2. 提供CAE結構參數與物理特性之間的轉換函數
3. 實現物理約束方程驗證功能
4. 支援基於物理知識的數據增強與驗證
"""

import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

# 常數定義
# 根據文獻資料，非線性塑性應變能與疲勞壽命的關係: Nf = a * (ΔW)^b
# 參考文獻: SACQ Solder Board Level Reliability Evaluation and Life Prediction Model
A_COEFFICIENT = 55.83  # 係數 a
B_COEFFICIENT = -2.259  # 係數 b (負值表示反比關係)
def calculate_delta_w(up_interface, down_interface, weight_factor=0.5):
    """
    計算非線性塑性應變能密度變化量(ΔW)
    
    參數:
        up_interface (array-like): 上界面的非線性塑性應變功數據
        down_interface (array-like): 下界面的非線性塑性應變功數據
        weight_factor (float): 上下界面權重因子，預設為0.5（等權重）
        
    返回:
        float or array: 計算的非線性塑性應變能密度變化量
    """
    # 確保輸入為numpy數組
    up_interface = np.asarray(up_interface)
    down_interface = np.asarray(down_interface)
    
    # 檢查數據形狀是否一致
    if up_interface.shape != down_interface.shape:
        logger.warning(f"上下界面數據形狀不一致: {up_interface.shape} vs {down_interface.shape}")
    
    # 如果為時間序列數據，計算首尾差值；否則直接取數值
    if up_interface.ndim == 1 and len(up_interface) > 1:
        delta_w_up = up_interface[-1] - up_interface[0]
        delta_w_down = down_interface[-1] - down_interface[0]
    else:
        delta_w_up = up_interface
        delta_w_down = down_interface

    # 加權平均計算ΔW，並保證為正值
    delta_w = weight_factor * delta_w_up + (1 - weight_factor) * delta_w_down 
    delta_w = np.maximum(delta_w, 1e-10)
    
    return delta_w


def nf_from_delta_w(delta_w, a=A_COEFFICIENT, b=B_COEFFICIENT):
    """
    根據非線性塑性應變能密度變化量(ΔW)計算疲勞壽命(Nf)
    公式: Nf = a * (ΔW)^b
    
    參數:
        delta_w (float or array): 非線性塑性應變能密度變化量
        a (float): 係數a
        b (float): 係數b
        
    返回:
        float or array: 計算的疲勞壽命
    """

    delta_w = np.maximum(np.asarray(delta_w), 1e-10)
     # 添加調試輸出
    if isinstance(delta_w, np.ndarray) and len(delta_w) < 10:
        print(f"[DEBUG] delta_w 範圍: {np.min(delta_w):.6e} - {np.max(delta_w):.6e}")
        print(f"[DEBUG] delta_w 樣本: {delta_w}")
    nf = a * np.power(delta_w, b)
     # 添加調試輸出
    if isinstance(nf, np.ndarray) and len(nf) < 10:
        print(f"[DEBUG] 計算的 Nf 範圍: {np.min(nf):.2f} - {np.max(nf):.2f}")
        print(f"[DEBUG] 計算的 Nf 樣本: {nf}")

    return nf


def delta_w_from_nf(nf, a=A_COEFFICIENT, b=B_COEFFICIENT):
    """
    根據疲勞壽命(Nf)計算非線性塑性應變能密度變化量(ΔW)
    公式: ΔW = (Nf/a)^(1/b)
    
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
def strain_energy_equation(nf, delta_w, a=A_COEFFICIENT, b=B_COEFFICIENT):
    """
    計算應變能方程的殘差
    殘差 = Nf - a * (ΔW)^b
    
    參數:
        nf (float or array): 疲勞壽命
        delta_w (float or array): 非線性塑性應變能密度變化量
        a (float): 係數a
        b (float): 係數b
        
    返回:
        float or array: 殘差值，以及相對殘差（百分比）
    """
    nf = np.asarray(nf)
    delta_w = np.asarray(delta_w)
    nf_theory = a * np.power(delta_w, b)
    residual = nf - nf_theory
    relative_residual = np.abs(residual / np.maximum(nf, 1e-10)) * 100
    return residual, relative_residual


def validate_physical_constraints(delta_w, nf, a=A_COEFFICIENT, b=B_COEFFICIENT, 
                                 threshold=20.0, verbose=True):
    """
    驗證預測結果是否符合物理約束條件
    
    參數:
        delta_w (array-like): 非線性塑性應變能密度變化量
        nf (array-like): 疲勞壽命
        a (float): 係數a
        b (float): 係數b
        threshold (float): 相對誤差閾值（百分比）
        verbose (bool): 是否輸出詳細資訊
        
    返回:
        tuple: (是否通過驗證, 相對殘差, 違反約束的索引)
    """
    delta_w = np.asarray(delta_w)
    nf = np.asarray(nf)
    
    # 計算理論nf值（考慮放大因子）
    nf_theory = a * np.power(delta_w, b) * nf_amp_factor
    
    # 計算相對誤差
    residual = nf - nf_theory
    relative_residual = np.abs(residual / np.maximum(nf_theory, 1e-10)) * 100
    violated_idx = np.where(relative_residual > threshold)[0]
    passed = (len(violated_idx) == 0)
    
    # 添加診斷信息
    if verbose:
        mean_delta_w = np.mean(delta_w)
        mean_nf = np.mean(nf)
        mean_nf_theory = np.mean(nf_theory)
        
        logger.info(f"物理診斷 - 平均delta_w: {mean_delta_w:.6e}, 對應理論Nf: {mean_nf_theory:.2f}")
        logger.info(f"物理診斷 - 平均預測Nf: {mean_nf:.2f}, 理論與預測比例: {mean_nf_theory/mean_nf:.4f}")
        
        if passed:
            logger.info(f"物理約束驗證通過! 最大相對殘差: {np.max(relative_residual):.2f}%")
        else:
            logger.warning(f"物理約束驗證失敗! 有 {len(violated_idx)} 個樣本違反約束")
            logger.warning(f"違反約束的樣本索引: {violated_idx}")
            logger.warning(f"違反約束的樣本相對殘差: {relative_residual[violated_idx]}")
    
    return passed, relative_residual, violated_idx

def estimate_fatigue_life_from_cae(nlplwk_up, nlplwk_down, warpage=None, 
                                  a=A_COEFFICIENT, b=B_COEFFICIENT):
    """
    從CAE模擬數據估計疲勞壽命
    
    參數:
        nlplwk_up (array-like): 上界面非線性塑性應變功時間序列
        nlplwk_down (array-like): 下界面非線性塑性應變功時間序列
        warpage (float, optional): 翹曲變形量，可用於調整估計
        a (float): 係數a
        b (float): 係數b
        
    返回:
        float: 估計的疲勞壽命, 以及計算的ΔW
    """
    delta_w = calculate_delta_w(nlplwk_up, nlplwk_down)
    if warpage is not None:
        warpage_factor = 1.0 + 0.01 * warpage  # 假設每增加1單位翹曲增加1%
        delta_w *= warpage_factor
    nf = nf_from_delta_w(delta_w, a, b)
    return nf, delta_w


def calculate_effective_strain(structure_params, temperature_range=180.0):
    """
    計算有效應變
    根據結構參數和溫度範圍估算有效應變
    
    參數:
        structure_params (dict): 結構參數，包含die, stud, mold, pcb等
        temperature_range (float): 溫度循環範圍，預設為180°C
        
    返回:
        float: 估計的有效應變
    """
    die = structure_params.get('die', 200)  # (μm)
    stud = structure_params.get('stud', 70)  # (μm)
    mold = structure_params.get('mold', 65)  # (μm)
    pcb = structure_params.get('pcb', 0.8)   # (mm)
    cte_die = 2.8    # ppm/°C
    cte_stud = 17.0
    cte_mold = 12.0
    cte_pcb = 16.0
    pcb_um = pcb * 1000
    strain_die_pcb = temperature_range * abs(cte_die - cte_pcb) * 1e-6
    strain_mold_pcb = temperature_range * abs(cte_mold - cte_pcb) * 1e-6
    solder_height = stud * 0.5
    die_factor = die / (die + mold + pcb_um)
    mold_factor = mold / (die + mold + pcb_um)
    effective_strain = (strain_die_pcb * die_factor + strain_mold_pcb * mold_factor) * (100 / solder_height)**0.3
    return effective_strain


def plot_nf_delta_w_relationship(delta_w_range=None, a=A_COEFFICIENT, b=B_COEFFICIENT, 
                                 figsize=(10, 6), save_path=None):
    """
    繪製非線性塑性應變能密度變化量(ΔW)與疲勞壽命(Nf)的關係圖
    
    參數:
        delta_w_range (tuple): ΔW 的範圍，預設自動計算
        a (float): 係數a
        b (float): 係數b
        figsize (tuple): 圖像尺寸
        save_path (str): 保存圖像的路徑
        
    返回:
        matplotlib.figure.Figure: 圖像對象
    """
    if delta_w_range is None:
        nf_min, nf_max = 100, 10000
        delta_w_max = delta_w_from_nf(nf_min, a, b)
        delta_w_min = delta_w_from_nf(nf_max, a, b)
        delta_w_range = (delta_w_min, delta_w_max)
        
    delta_w_values = np.logspace(np.log10(delta_w_range[0]), np.log10(delta_w_range[1]), 100)
    nf_values = nf_from_delta_w(delta_w_values, a, b)
    fig, ax = plt.subplots(figsize=figsize)
    ax.loglog(delta_w_values, nf_values, 'b-', linewidth=2)
    key_deltas = [delta_w_range[0], np.sqrt(delta_w_range[0]*delta_w_range[1]), delta_w_range[1]]
    key_nfs = nf_from_delta_w(np.array(key_deltas), a, b)
    ax.scatter(key_deltas, key_nfs, color='red', s=100, zorder=5)
    for dw, nf in zip(key_deltas, key_nfs):
        ax.annotate(f'ΔW={dw:.2e}, Nf={nf:.1f}', xy=(dw, nf), xytext=(dw*1.1, nf*1.1),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5), fontsize=9)
    ax.set_title(f'ΔW與Nf的關係: Nf = {a} × (ΔW)^{b}')
    ax.set_xlabel('非線性塑性應變能密度變化量 (ΔW)')
    ax.set_ylabel('疲勞壽命 (Nf)')
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    equation_text = f'Nf = {a} × (ΔW)^{b}'
    ax.text(0.05, 0.95, equation_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"圖像已保存至: {save_path}")
        except Exception as e:
            logger.error(f"保存圖像失敗: {str(e)}")
    return fig
def generate_physics_guided_data(structure_params_list, n_samples=100, noise_level=0.1, 
                                 a=A_COEFFICIENT, b=B_COEFFICIENT):
    """
    生成基於物理知識的合成數據
    用於數據增強或模型預訓練
    
    參數:
        structure_params_list (list): 結構參數列表
        n_samples (int): 生成的樣本數量
        noise_level (float): 噪聲水平
        a (float): 係數a
        b (float): 係數b
        
    返回:
        dict: 生成的數據，包含結構參數、應變能和壽命等
    """
    if not structure_params_list:
        logger.warning("結構參數列表為空，使用默認參數")
        structure_params_list = [
            {'die': 250, 'stud': 80, 'mold': 75, 'pcb': 1.0},
            {'die': 200, 'stud': 70, 'mold': 65, 'pcb': 0.8},
            {'die': 150, 'stud': 60, 'mold': 55, 'pcb': 0.6}
        ]
    
    synthetic_data = {
        'structure_params': [],
        'delta_w': [],
        'nf': [],
        'nlplwk_up': [],
        'nlplwk_down': []
    }
    
    n_params = len(structure_params_list)
    samples_per_param = n_samples // n_params
    
    for params in structure_params_list:
        base_strain = calculate_effective_strain(params)
        for _ in range(samples_per_param):
            strain_variation = np.random.normal(1.0, noise_level)
            effective_strain = base_strain * strain_variation
            delta_w = effective_strain**2 * np.random.uniform(1.0, 2.0)
            nf = nf_from_delta_w(delta_w, a, b)
            time_points = np.array([3600, 7200, 10800, 14400])
            nlplwk_up_base = delta_w * np.cumsum(np.array([0.2, 0.3, 0.3, 0.2]))
            nlplwk_down_base = delta_w * np.cumsum(np.array([0.15, 0.25, 0.35, 0.25]))
            noise_up = np.random.normal(0, noise_level * 0.1, 4) * delta_w
            noise_down = np.random.normal(0, noise_level * 0.1, 4) * delta_w
            nlplwk_up = np.maximum.accumulate(nlplwk_up_base + noise_up)
            nlplwk_down = np.maximum.accumulate(nlplwk_down_base + noise_down)
            synthetic_data['structure_params'].append(params.copy())
            synthetic_data['delta_w'].append(delta_w)
            synthetic_data['nf'].append(nf)
            synthetic_data['nlplwk_up'].append(nlplwk_up)
            synthetic_data['nlplwk_down'].append(nlplwk_down)
    
    return synthetic_data


def map_structure_to_warpage(structure_params):
    """
    根據結構參數估算翹曲變形量
    
    參數:
        structure_params (dict): 結構參數，包含die, stud, mold, pcb等
        
    返回:
        float: 估算的翹曲變形量
    """
    die = structure_params.get('die', 200)
    stud = structure_params.get('stud', 70)
    mold = structure_params.get('mold', 65)
    pcb = structure_params.get('pcb', 0.8)
    pcb_um = pcb * 1000
    thickness_factor = 1000 / (die + stud + mold + pcb_um)
    balance_factor = abs((die + stud) - (mold + pcb_um/2)) / (die + stud + mold + pcb_um)
    base_warpage = 5.0
    warpage = base_warpage * thickness_factor * (1 + 2 * balance_factor)
    return warpage
if __name__ == "__main__":
    # 設定 logging 基本設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 測試ΔW與疲勞壽命關係
    logger.info("測試ΔW與Nf的關係")
    delta_w_test = np.array([0.001, 0.01, 0.1])
    nf_test = nf_from_delta_w(delta_w_test)
    for dw, nf in zip(delta_w_test, nf_test):
        logger.info(f"ΔW = {dw:.4f} => Nf = {nf:.2f}")
    
    # 測試物理約束驗證
    logger.info("\n測試物理約束驗證")
    delta_w_samples = np.array([0.005, 0.02, 0.05])
    nf_samples = nf_from_delta_w(delta_w_samples)
    nf_samples_disturbed = nf_samples * (1 + np.random.normal(0, 0.1, size=len(nf_samples)))
    passed, residuals, violated = validate_physical_constraints(delta_w_samples, nf_samples_disturbed, threshold=15.0)
    
    # 測試物理關係圖
    logger.info("\n繪製物理關係圖")
    fig = plot_nf_delta_w_relationship()
    plt.show()
    
    # 測試從CAE數據估計疲勞壽命
    logger.info("\n測試從CAE數據估計疲勞壽命")
    nlplwk_up = np.array([0, 0.005, 0.012, 0.020])
    nlplwk_down = np.array([0, 0.004, 0.010, 0.018])
    warpage = 10.0
    nf_est, delta_w_est = estimate_fatigue_life_from_cae(nlplwk_up, nlplwk_down, warpage)
    logger.info(f"估計的ΔW = {delta_w_est:.6f}, 估計的Nf = {nf_est:.2f}")
    
    # 測試生成合成數據
    logger.info("\n測試生成物理引導的合成數據")
    structure_params = [
        {'die': 250, 'stud': 80, 'mold': 75, 'pcb': 1.0},
        {'die': 200, 'stud': 70, 'mold': 65, 'pcb': 0.8}
    ]
    synthetic_data = generate_physics_guided_data(structure_params, n_samples=10)
    logger.info(f"生成了 {len(synthetic_data['nf'])} 個合成樣本")
    for i in range(min(3, len(synthetic_data['nf']))):
        logger.info(f"樣本 {i+1}: ΔW = {synthetic_data['delta_w'][i]:.6f}, Nf = {synthetic_data['nf'][i]:.2f}")
    
    logger.info("所有測試完成")
