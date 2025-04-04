#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
improved_dataset.py - 改進的資料集封裝模組
本模組提供自定義資料集的封裝，用於銲錫接點疲勞壽命預測模型訓練和評估。
實現了適合混合PINN-LSTM模型所需的資料載入與批次處理功能，並針對小樣本進行優化。

主要改進:
1. 針對小樣本數據集增強的資料封裝和批次處理策略
2. 提供更靈活的時間序列資料處理機制
3. 支援物理知識驅動的資料增強和特徵轉換
4. 加強資料集的可視化和統計分析功能
5. 優化分層抽樣的訓練-驗證-測試集分割策略
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, WeightedRandomSampler
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from pathlib import Path
import os
import math

logger = logging.getLogger(__name__)

class SolderFatigueDataset(Dataset):
    """
    銲錫接點疲勞壽命資料集
    處理靜態特徵（結構參數）與預測目標（疲勞壽命）
    
    增強功能：
    1. 支援權重賦值，平衡稀有樣本
    2. 提供基本資料擴增
    3. 支援特徵轉換與標準化
    """
    def __init__(self, features, targets=None, transform=None, normalization=None, 
                 enable_augmentation=False, augmentation_factor=0.1, sample_weights=None,
                 feature_names=None):
        """
        初始化資料集
        
        參數:
            features (numpy.ndarray): 特徵資料，形狀為 (樣本數, 特徵數)
            targets (numpy.ndarray, optional): 目標資料，形狀為 (樣本數,)
            transform (callable, optional): 應用於特徵的轉換函數
            normalization (dict, optional): 標準化參數，包含'mean'和'std'
            enable_augmentation (bool): 是否啟用資料擴增
            augmentation_factor (float): 擴增強度係數
            sample_weights (numpy.ndarray, optional): 樣本權重
            feature_names (list, optional): 特徵名稱列表
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
        self.transform = transform
        self.normalization = normalization
        self.enable_augmentation = enable_augmentation
        self.augmentation_factor = augmentation_factor
        self.sample_weights = sample_weights
        self.feature_names = feature_names
        
        # 應用標準化（如果有）
        if self.normalization is not None:
            self._apply_normalization()
        
        self.n_samples = len(self.features)
        self.n_features = self.features.shape[1]
        
        logger.debug(f"初始化SolderFatigueDataset，特徵形狀: {self.features.shape}, "
                    f"目標形狀: {None if self.targets is None else self.targets.shape}, "
                    f"啟用擴增: {enable_augmentation}")

    def _apply_normalization(self):
        """應用標準化"""
        if 'mean' in self.normalization and 'std' in self.normalization:
            mean = torch.FloatTensor(self.normalization['mean'])
            std = torch.FloatTensor(self.normalization['std'])
            
            # 確保維度匹配
            if mean.shape[0] == self.features.shape[1]:
                self.features = (self.features - mean) / (std + 1e-8)
    
    def _augment_sample(self, features, target=None):
        """
        對單個樣本進行擴增
        
        參數:
            features (torch.Tensor): 特徵張量
            target (torch.Tensor, optional): 目標張量
            
        返回:
            tuple: (擴增特徵, 擴增目標)
        """
        # 簡單的高斯噪聲擴增
        noise = torch.randn_like(features) * self.augmentation_factor
        augmented_features = features + noise
        
        augmented_target = target
        if target is not None:
            # 對目標值進行輕微擾動
            # 使用對數正態分佈避免負值
            noise_factor = torch.exp(torch.randn(1) * 0.05)  # 小擾動
            augmented_target = target * noise_factor
        
        return augmented_features, augmented_target

    def __len__(self):
        """返回資料集中的樣本數量"""
        return self.n_samples

    def __getitem__(self, idx):
        """獲取指定索引的樣本"""
        features = self.features[idx]
        
        # 應用轉換（如果有）
        if self.transform:
            features = self.transform(features)
        
        # 如果啟用擴增且不是在評估模式（有目標值）
        if self.enable_augmentation and self.targets is not None and torch.rand(1).item() < 0.5:
            # 50%機率進行擴增
            features, target = self._augment_sample(features, self.targets[idx] if self.targets is not None else None)
        else:
            target = self.targets[idx] if self.targets is not None else None
        
        # 如果有目標，返回特徵和目標；否則只返回特徵
        if target is not None:
            return features, target
        else:
            return features
    
    def get_sample_weights(self):
        """獲取樣本權重用於加權抽樣"""
        if self.sample_weights is not None:
            return self.sample_weights
        
        # 如果未提供權重且有目標值，根據目標分佈計算權重
        if self.targets is not None:
            # 將目標分成bins以計算權重
            bins = min(10, len(self.targets) // 5)  # 確保每個bin至少有5個樣本
            target_np = self.targets.numpy()
            
            # 使用分位數確保每個bin有均衡數量的樣本
            quantiles = np.percentile(target_np, np.linspace(0, 100, bins+1))
            
            # 計算每個bin的樣本數
            bin_counts = np.zeros(bins)
            for i in range(bins):
                if i == bins - 1:
                    bin_counts[i] = np.sum((log_targets >= quantiles[i]) & (log_targets <= quantiles[i+1]))
                else:
                    bin_counts[i] = np.sum((log_targets >= quantiles[i]) & (log_targets < quantiles[i+1]))
            
            # 為每個樣本分配權重
            weights = np.ones(len(log_targets))
            for i in range(bins):
                if i == bins - 1:
                    mask = (log_targets >= quantiles[i]) & (log_targets <= quantiles[i+1])
                else:
                    mask = (log_targets >= quantiles[i]) & (log_targets < quantiles[i+1])
                
                if bin_counts[i] > 0:
                    weights[mask] = 1.0 / (bin_counts[i] / len(log_targets))
            
            # 標準化權重使總和為len(log_targets)
            weights = weights * len(log_targets) / np.sum(weights)
            
            return weights
        
        # 默認使用均勻權重
        return np.ones(self.n_samples)
    
    def visualize_distributions(self, figsize=(12, 8), save_path=None):
        """
        視覺化資料分佈
        
        參數:
            figsize (tuple): 圖像尺寸
            save_path (str, optional): 保存路徑
            
        返回:
            matplotlib.figure.Figure: 圖像對象
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        
        # 創建圖像
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 2)
        
        # 1. 目標分佈（如果有）
        if self.targets is not None:
            target_np = self.targets.numpy()
            
            # 線性空間
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.hist(target_np, bins=30, alpha=0.7, color='blue')
            ax1.set_title('目標分佈（線性空間）')
            ax1.set_xlabel('壽命值')
            ax1.set_ylabel('頻率')
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # 對數空間
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.hist(np.log10(target_np + 1e-8), bins=30, alpha=0.7, color='green')
            ax2.set_title('目標分佈（對數空間）')
            ax2.set_xlabel('log10(壽命值)')
            ax2.set_ylabel('頻率')
            ax2.grid(True, linestyle='--', alpha=0.7)
        
        # 2. 靜態特徵分佈
        ax3 = fig.add_subplot(gs[1, 0])
        feature_np = self.features.numpy()
        
        if self.feature_names is not None and len(self.feature_names) == feature_np.shape[1]:
            labels = self.feature_names
        else:
            labels = [f'特徵{i}' for i in range(feature_np.shape[1])]
        
        # 使用箱形圖顯示每個特徵的分佈
        ax3.boxplot(feature_np, labels=labels)
        ax3.set_title('靜態特徵分佈')
        ax3.set_xlabel('特徵')
        ax3.set_ylabel('值')
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        # 3. 時間序列特徵分佈
        ax4 = fig.add_subplot(gs[1, 1])
        
        # 計算每個時間步的平均值和標準差
        ts_mean = self.time_series.mean(dim=0).numpy()
        ts_std = self.time_series.std(dim=0).numpy()
        
        # 繪製時間序列平均值
        time_steps = np.arange(self.n_time_steps)
        for i in range(self.n_ts_features):
            ax4.plot(time_steps, ts_mean[:, i], label=f'特徵{i+1}')
            ax4.fill_between(
                time_steps, 
                ts_mean[:, i] - ts_std[:, i], 
                ts_mean[:, i] + ts_std[:, i], 
                alpha=0.3
            )
        
        ax4.set_title('時間序列特徵分佈')
        ax4.set_xlabel('時間步')
        ax4.set_ylabel('平均值±標準差')
        ax4.legend()
        ax4.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # 保存圖像
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            except Exception as e:
                print(f"保存圖像失敗: {str(e)}")
        
        return fig bins - 1:
                    bin_counts[i] = np.sum((target_np >= quantiles[i]) & (target_np <= quantiles[i+1]))
                else:
                    bin_counts[i] = np.sum((target_np >= quantiles[i]) & (target_np < quantiles[i+1]))
            
            # 為每個樣本分配權重
            weights = np.ones(len(target_np))
            for i in range(bins):
                if i == bins - 1:
                    mask = (target_np >= quantiles[i]) & (target_np <= quantiles[i+1])
                else:
                    mask = (target_np >= quantiles[i]) & (target_np < quantiles[i+1])
                
                if bin_counts[i] > 0:
                    weights[mask] = 1.0 / (bin_counts[i] / len(target_np))
            
            # 標準化權重使總和為len(target_np)
            weights = weights * len(target_np) / np.sum(weights)
            
            return weights
        
        # 默認使用均勻權重
        return np.ones(self.n_samples)


class TimeSeriesDataset(Dataset):
    """
    時間序列資料集
    處理時間序列特徵（非線性塑性應變功時間序列）
    
    增強功能：
    1. 支援多種時間序列標準化策略
    2. 提供時間序列擴增
    3. 允許可變長度時間序列處理
    4. 支援時間步重要性加權
    """
    def __init__(self, time_series, targets=None, transform=None, normalization=None,
                 normalize_per_feature=True, normalize_per_sample=False,
                 enable_augmentation=False, augmentation_factor=0.05,
                 time_step_weights=None, mask_value=None):
        """
        初始化時間序列資料集
        
        參數:
            time_series (numpy.ndarray): 時間序列資料，形狀為 (樣本數, 時間步數, 特徵數)
            targets (numpy.ndarray, optional): 目標資料，形狀為 (樣本數,)
            transform (callable, optional): 應用於時間序列的轉換函數
            normalization (dict, optional): 標準化參數
            normalize_per_feature (bool): 是否對每個特徵單獨標準化
            normalize_per_sample (bool): 是否對每個樣本單獨標準化
            enable_augmentation (bool): 是否啟用資料擴增
            augmentation_factor (float): 擴增強度係數
            time_step_weights (numpy.ndarray, optional): 時間步權重
            mask_value (float, optional): 用於填充缺失值的遮罩值
        """
        self.time_series = torch.FloatTensor(time_series)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
        self.transform = transform
        self.normalization = normalization
        self.normalize_per_feature = normalize_per_feature
        self.normalize_per_sample = normalize_per_sample
        self.enable_augmentation = enable_augmentation
        self.augmentation_factor = augmentation_factor
        self.time_step_weights = time_step_weights
        self.mask_value = mask_value
        
        # 應用標準化（如果有）
        if self.normalization is not None:
            self._apply_normalization()
        elif self.normalize_per_feature or self.normalize_per_sample:
            self._normalize_time_series()
        
        self.n_samples = len(self.time_series)
        self.n_time_steps = self.time_series.shape[1]
        self.n_features = self.time_series.shape[2]
        
        logger.debug(f"初始化TimeSeriesDataset，時間序列形狀: {self.time_series.shape}, "
                    f"目標形狀: {None if self.targets is None else self.targets.shape}, "
                    f"啟用擴增: {enable_augmentation}")

    def _apply_normalization(self):
        """應用提供的標準化參數"""
        if 'mean' in self.normalization and 'std' in self.normalization:
            mean = torch.FloatTensor(self.normalization['mean'])
            std = torch.FloatTensor(self.normalization['std'])
            
            # 根據維度擴展均值和標準差
            if mean.dim() == 1 and mean.shape[0] == self.time_series.shape[2]:
                # 對每個特徵標準化
                mean = mean.view(1, 1, -1)
                std = std.view(1, 1, -1)
                self.time_series = (self.time_series - mean) / (std + 1e-8)
    
    def _normalize_time_series(self):
        """根據策略標準化時間序列"""
        # 創建標準化參數儲存
        self._normalization_params = {}
        
        if self.normalize_per_feature:
            # 對每個特徵進行標準化
            mean = torch.mean(self.time_series.reshape(-1, self.n_features), dim=0)
            std = torch.std(self.time_series.reshape(-1, self.n_features), dim=0)
            
            # 儲存標準化參數
            self._normalization_params['feature_mean'] = mean
            self._normalization_params['feature_std'] = std
            
            # 應用標準化
            mean = mean.view(1, 1, self.n_features)
            std = std.view(1, 1, self.n_features)
            self.time_series = (self.time_series - mean) / (std + 1e-8)
        
        if self.normalize_per_sample:
            # 對每個樣本分別標準化
            # 這個操作會覆蓋之前的標準化（如果有）
            for i in range(self.n_samples):
                sample = self.time_series[i]
                mean = torch.mean(sample, dim=0, keepdim=True)
                std = torch.std(sample, dim=0, keepdim=True)
                self.time_series[i] = (sample - mean) / (std + 1e-8)
    
    def _augment_time_series(self, time_series, target=None):
        """
        對時間序列樣本進行擴增
        
        參數:
            time_series (torch.Tensor): 時間序列張量
            target (torch.Tensor, optional): 目標張量
            
        返回:
            tuple: (擴增時間序列, 擴增目標)
        """
        # 時間序列擴增策略
        aug_type = torch.randint(0, 4, (1,)).item()  # 隨機選擇擴增類型
        
        if aug_type == 0:
            # 添加高斯噪聲
            noise = torch.randn_like(time_series) * self.augmentation_factor
            augmented_ts = time_series + noise
        elif aug_type == 1:
            # 時間尺度擴展：輕微縮放時間軸
            scale = 1.0 + (torch.rand(1).item() * 2 - 1) * 0.1  # 0.9到1.1之間
            indices = torch.linspace(0, time_series.shape[0]-1, time_series.shape[0])
            new_indices = torch.clamp(indices * scale, 0, time_series.shape[0]-1).long()
            augmented_ts = time_series[new_indices]
        elif aug_type == 2:
            # 值尺度擴展：輕微縮放值
            scale = 1.0 + (torch.rand(1).item() * 2 - 1) * 0.1  # 0.9到1.1之間
            augmented_ts = time_series * scale
        else:
            # 微小時間偏移
            shift = int((torch.rand(1).item() * 2 - 1) * 2)  # -2到2之間的整數
            if shift > 0:
                augmented_ts = torch.cat([time_series[shift:], time_series[-1].repeat(shift, 1)])
            elif shift < 0:
                augmented_ts = torch.cat([time_series[0].repeat(-shift, 1), time_series[:shift]])
            else:
                augmented_ts = time_series.clone()
        
        augmented_target = target
        if target is not None:
            # 時間序列擴增可能影響目標值
            # 使用小擾動模擬這種影響
            noise_factor = torch.exp(torch.randn(1) * 0.03)  # 較小擾動
            augmented_target = target * noise_factor
        
        return augmented_ts, augmented_target

    def __len__(self):
        """返回資料集中的樣本數量"""
        return self.n_samples

    def __getitem__(self, idx):
        """獲取指定索引的樣本"""
        time_series = self.time_series[idx]
        
        # 應用轉換（如果有）
        if self.transform:
            time_series = self.transform(time_series)
        
        # 如果啟用擴增且不是在評估模式（有目標值）
        if self.enable_augmentation and self.targets is not None and torch.rand(1).item() < 0.3:
            # 30%機率進行擴增
            time_series, target = self._augment_time_series(
                time_series, 
                self.targets[idx] if self.targets is not None else None
            )
        else:
            target = self.targets[idx] if self.targets is not None else None
        
        # 如果有目標，返回時間序列和目標；否則只返回時間序列
        if target is not None:
            return time_series, target
        else:
            return time_series


class HybridDataset(Dataset):
    """
    混合資料集
    同時處理靜態特徵和時間序列特徵
    
    增強功能：
    1. 支援兩種特徵類型的統一處理
    2. 提供更靈活的擴增策略
    3. 支援物理知識驅動的樣本生成
    4. 允許樣本重要性加權
    """
    def __init__(self, features, time_series, targets=None, 
                 feature_transform=None, time_series_transform=None,
                 feature_normalization=None, time_series_normalization=None,
                 enable_augmentation=False, augmentation_policy=None,
                 sample_weights=None, use_physics_constraints=False,
                 physics_params=None, feature_names=None):
        """
        初始化混合資料集
        
        參數:
            features (numpy.ndarray): 靜態特徵資料，形狀為 (樣本數, 特徵數)
            time_series (numpy.ndarray): 時間序列資料，形狀為 (樣本數, 時間步數, 特徵數)
            targets (numpy.ndarray, optional): 目標資料，形狀為 (樣本數,)
            feature_transform (callable, optional): 應用於靜態特徵的轉換函數
            time_series_transform (callable, optional): 應用於時間序列的轉換函數
            feature_normalization (dict, optional): 靜態特徵標準化參數
            time_series_normalization (dict, optional): 時間序列標準化參數
            enable_augmentation (bool): 是否啟用資料擴增
            augmentation_policy (dict, optional): 擴增策略
            sample_weights (numpy.ndarray, optional): 樣本權重
            use_physics_constraints (bool): 是否使用物理約束
            physics_params (dict, optional): 物理模型參數
            feature_names (list, optional): 特徵名稱列表
        """
        if len(features) != len(time_series):
            raise ValueError("特徵資料和時間序列資料的樣本數量必須相同")
        
        self.features = torch.FloatTensor(features)
        self.time_series = torch.FloatTensor(time_series)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
        self.feature_transform = feature_transform
        self.time_series_transform = time_series_transform
        self.feature_normalization = feature_normalization
        self.time_series_normalization = time_series_normalization
        self.enable_augmentation = enable_augmentation
        self.augmentation_policy = augmentation_policy or {'prob': 0.3, 'strength': 0.1}
        self.sample_weights = sample_weights
        self.use_physics_constraints = use_physics_constraints
        self.physics_params = physics_params or {'a': 55.83, 'b': -2.259}
        self.feature_names = feature_names
        
        # 應用標準化（如果有）
        if self.feature_normalization is not None:
            self._apply_feature_normalization()
        
        if self.time_series_normalization is not None:
            self._apply_time_series_normalization()
        
        self.n_samples = len(self.features)
        self.n_features = self.features.shape[1]
        self.n_time_steps = self.time_series.shape[1]
        self.n_ts_features = self.time_series.shape[2]
        
        logger.debug(f"初始化HybridDataset，特徵形狀: {self.features.shape}, "
                    f"時間序列形狀: {self.time_series.shape}, "
                    f"目標形狀: {None if self.targets is None else self.targets.shape}, "
                    f"啟用擴增: {enable_augmentation}")

    def _apply_feature_normalization(self):
        """應用靜態特徵標準化"""
        if 'mean' in self.feature_normalization and 'std' in self.feature_normalization:
            mean = torch.FloatTensor(self.feature_normalization['mean'])
            std = torch.FloatTensor(self.feature_normalization['std'])
            
            # 確保維度匹配
            if mean.shape[0] == self.features.shape[1]:
                self.features = (self.features - mean) / (std + 1e-8)
    
    def _apply_time_series_normalization(self):
        """應用時間序列標準化"""
        if 'mean' in self.time_series_normalization and 'std' in self.time_series_normalization:
            mean = torch.FloatTensor(self.time_series_normalization['mean'])
            std = torch.FloatTensor(self.time_series_normalization['std'])
            
            # 根據維度擴展均值和標準差
            if mean.dim() == 1 and mean.shape[0] == self.time_series.shape[2]:
                # 對每個特徵標準化
                mean = mean.view(1, 1, -1)
                std = std.view(1, 1, -1)
                self.time_series = (self.time_series - mean) / (std + 1e-8)
    
    def _augment_sample(self, features, time_series, target=None):
        """
        對樣本進行綜合擴增
        
        參數:
            features (torch.Tensor): 靜態特徵張量
            time_series (torch.Tensor): 時間序列張量
            target (torch.Tensor, optional): 目標張量
            
        返回:
            tuple: (擴增特徵, 擴增時間序列, 擴增目標)
        """
        # 獲取擴增參數
        strength = self.augmentation_policy.get('strength', 0.1)
        
        # 結構參數擴增
        # 使用小幅高斯噪聲
        feature_noise = torch.randn_like(features) * strength
        augmented_features = features + feature_noise
        
        # 時間序列擴增
        aug_type = torch.randint(0, 3, (1,)).item()  # 隨機選擇擴增類型
        
        if aug_type == 0:
            # 添加高斯噪聲
            ts_noise = torch.randn_like(time_series) * strength
            augmented_ts = time_series + ts_noise
        elif aug_type == 1:
            # 輕微縮放
            scale = 1.0 + (torch.rand(1).item() * 2 - 1) * 0.1  # 0.9到1.1之間
            augmented_ts = time_series * scale
        else:
            # 時間軸抖動
            # 對每個時間步添加小擾動
            jitter = torch.randn_like(time_series) * strength * 0.5
            augmented_ts = time_series + jitter
            # 確保時間序列單調遞增（如果適用）
            if self.use_physics_constraints:
                # 對每個特徵分別確保單調性
                for i in range(augmented_ts.shape[1]):
                    augmented_ts[:, i] = torch.cummax(augmented_ts[:, i], dim=0)[0]
        
        augmented_target = target
        if target is not None and self.use_physics_constraints:
            # 使用物理約束計算調整後的目標
            # 估算應變能密度變化量
            try:
                # 假設時間序列最後一步包含最終的應變能密度
                delta_w_up = augmented_ts[-1, 0].item()
                delta_w_down = augmented_ts[-1, 1].item() if augmented_ts.shape[1] > 1 else delta_w_up
                
                # 取加權平均
                delta_w = 0.5 * delta_w_up + 0.5 * delta_w_down
                
                # 使用物理模型計算壽命
                a = self.physics_params['a']
                b = self.physics_params['b']
                physics_nf = a * (delta_w ** b)
                
                # 添加小擾動
                noise_factor = torch.exp(torch.randn(1) * 0.05)  # 小擾動
                augmented_target = torch.tensor(physics_nf) * noise_factor
            except:
                # 如果物理計算失敗，使用原始目標加擾動
                noise_factor = torch.exp(torch.randn(1) * 0.05)  # 小擾動
                augmented_target = target * noise_factor
        elif target is not None:
            # 簡單擾動
            noise_factor = torch.exp(torch.randn(1) * 0.03)  # 更小擾動
            augmented_target = target * noise_factor
        
        return augmented_features, augmented_ts, augmented_target

    def __len__(self):
        """返回資料集中的樣本數量"""
        return self.n_samples

    def __getitem__(self, idx):
        """獲取指定索引的樣本"""
        features = self.features[idx]
        time_series = self.time_series[idx]
        
        # 應用轉換（如果有）
        if self.feature_transform:
            features = self.feature_transform(features)
        
        if self.time_series_transform:
            time_series = self.time_series_transform(time_series)
        
        # 如果啟用擴增且不是在評估模式（有目標值）
        if self.enable_augmentation and self.targets is not None and torch.rand(1).item() < self.augmentation_policy.get('prob', 0.3):
            # 對樣本進行擴增
            features, time_series, target = self._augment_sample(
                features, 
                time_series, 
                self.targets[idx] if self.targets is not None else None
            )
        else:
            target = self.targets[idx] if self.targets is not None else None
        
        # 如果有目標，返回特徵、時間序列和目標；否則只返回特徵和時間序列
        if target is not None:
            return features, time_series, target
        else:
            return features, time_series
    
    def get_sample_weights(self):
        """獲取樣本權重用於加權抽樣"""
        if self.sample_weights is not None:
            return self.sample_weights
        
        # 如果未提供權重且有目標值，根據目標分佈計算權重
        if self.targets is not None:
            # 使用對數空間計算權重，適合疲勞壽命這種跨度大的數據
            target_np = self.targets.numpy()
            log_targets = np.log(target_np + 1e-8)
            
            # 分成bins
            bins = min(10, len(log_targets) // 5)  # 確保每個bin至少有5個樣本
            
            # 使用分位數確保每個bin有均衡數量的樣本
            quantiles = np.percentile(log_targets, np.linspace(0, 100, bins+1))
            
            # 計算每個bin的樣本數
            bin_counts = np.zeros(bins)
            for i in range(bins):
                if i ==