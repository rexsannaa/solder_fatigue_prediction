#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
dataset.py - 資料集封裝模組
本模組提供自定義資料集的封裝，用於銲錫接點疲勞壽命預測模型訓練和評估。
實現了適合混合PINN-LSTM模型所需的資料載入與批次處理功能，並針對小樣本進行優化。

主要功能:
1. 針對小樣本數據集優化的資料封裝和批次處理策略
2. 提供靈活的時間序列資料處理機制
3. 支援物理知識驅動的資料增強和特徵轉換
4. 處理結構參數與時間序列的混合輸入
5. 針對不平衡數據提供加權採樣機制
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, WeightedRandomSampler
import logging
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
import math
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class SolderFatigueDataset(Dataset):
    """
    銲錫接點疲勞壽命資料集
    處理靜態特徵（結構參數）與預測目標（疲勞壽命）
    """
    def __init__(self, static_features, time_series_features, targets=None, 
                 static_transform=None, time_transform=None,
                 static_normalizer=None, time_normalizer=None,
                 augmentation=False, augmentation_factor=0.1):
        """
        初始化資料集
        
        參數:
            static_features (numpy.ndarray): 靜態特徵，形狀為 (樣本數, 特徵數)
            time_series_features (numpy.ndarray): 時間序列特徵，形狀為 (樣本數, 時間步數, 特徵數)
            targets (numpy.ndarray, optional): 目標值，形狀為 (樣本數,)
            static_transform (callable, optional): 靜態特徵轉換函數
            time_transform (callable, optional): 時間序列轉換函數
            static_normalizer (object, optional): 靜態特徵標準化器
            time_normalizer (object, optional): 時間序列標準化器
            augmentation (bool): 是否啟用資料增強
            augmentation_factor (float): 增強因子
        """
        self.static_features = torch.FloatTensor(static_features)
        self.time_series_features = torch.FloatTensor(time_series_features)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
        
        self.static_transform = static_transform
        self.time_transform = time_transform
        self.static_normalizer = static_normalizer
        self.time_normalizer = time_normalizer
        
        self.augmentation = augmentation
        self.augmentation_factor = augmentation_factor
        
        # 應用標準化
        if self.static_normalizer is not None:
            self._normalize_static_features()
        
        if self.time_normalizer is not None:
            self._normalize_time_series()
        
        self.n_samples = len(self.static_features)
        logger.debug(f"初始化SolderFatigueDataset，樣本數: {self.n_samples}")
    
    def _normalize_static_features(self):
        """標準化靜態特徵"""
        if hasattr(self.static_normalizer, 'transform'):
            # scikit-learn風格標準化器
            self.static_features = torch.FloatTensor(
                self.static_normalizer.transform(self.static_features.numpy())
            )
        elif isinstance(self.static_normalizer, dict) and 'mean' in self.static_normalizer and 'std' in self.static_normalizer:
            # 使用提供的均值和標準差
            mean = torch.FloatTensor(self.static_normalizer['mean'])
            std = torch.FloatTensor(self.static_normalizer['std'])
            self.static_features = (self.static_features - mean) / (std + 1e-8)
    
    def _normalize_time_series(self):
        """標準化時間序列特徵"""
        if hasattr(self.time_normalizer, 'transform'):
            # 將時間序列展平進行標準化，然後恢復形狀
            orig_shape = self.time_series_features.shape
            flattened = self.time_series_features.view(-1, orig_shape[-1])
            normalized = torch.FloatTensor(
                self.time_normalizer.transform(flattened.numpy())
            )
            self.time_series_features = normalized.view(orig_shape)
        elif isinstance(self.time_normalizer, dict) and 'mean' in self.time_normalizer and 'std' in self.time_normalizer:
            # 使用提供的均值和標準差
            mean = torch.FloatTensor(self.time_normalizer['mean'])
            std = torch.FloatTensor(self.time_normalizer['std'])
            # 擴展維度以匹配時間序列形狀
            if mean.dim() == 1:
                mean = mean.view(1, 1, -1)
                std = std.view(1, 1, -1)
            self.time_series_features = (self.time_series_features - mean) / (std + 1e-8)
    
    def _augment_sample(self, static, time_series, target=None):
        """
        對樣本進行資料增強
    
        參數:
            static (torch.Tensor): 靜態特徵
            time_series (torch.Tensor): 時間序列特徵
            target (torch.Tensor, optional): 目標值
        
        返回:
            tuple: (增強的靜態特徵, 增強的時間序列特徵, 增強的目標值)
        """
        # 對靜態特徵增加小噪聲
        static_noise = torch.randn_like(static) * self.augmentation_factor
        augmented_static = static + static_noise
    
        # 對時間序列增加小噪聲，並保持時間順序
        time_noise = torch.randn_like(time_series) * self.augmentation_factor
        augmented_time = time_series + time_noise
    
        # 確保時間序列保持單調性（如果適用）
        # 假設時間序列是單調遞增的（如累積應變能量）
        if augmented_time.dim() == 2:  # (time_steps, features)
            for i in range(augmented_time.shape[1]):
                augmented_time[:, i] = torch.cummax(augmented_time[:, i], dim=0)[0]
    
        # 如果有目標值，也對其增加小噪聲
        augmented_target = target
        if target is not None:
            # 使用對數正態噪聲確保目標值保持正數
            target_noise = torch.exp(torch.randn(1) * 0.05)  # 小噪聲
            augmented_target = target * target_noise
        
            # 確保目標值的維度一致
            if isinstance(augmented_target, torch.Tensor) and augmented_target.dim() == 0:
                augmented_target = augmented_target.unsqueeze(0)
    
        return augmented_static, augmented_time, augmented_target
    
    def __len__(self):
        """返回樣本數量"""
        return self.n_samples
    

    def __getitem__(self, idx):
        """
        獲取指定索引的樣本
    
        參數:
            idx (int): 樣本索引
    
        返回:
            tuple: (時間序列, 目標值) 或 時間序列
        """
        time_series = self.time_series[idx]
    
        # 應用轉換（如果有）
        if self.transform is not None:
            time_series = self.transform(time_series)
    
        # 檢查是否有目標值
        if self.targets is not None:
            target = self.targets[idx]
        
            # 確保目標值是一維張量
            if isinstance(target, torch.Tensor):
                if target.dim() == 0:
                    # 如果是標量張量，轉為一維
                    target = target.unsqueeze(0)
                elif target.dim() > 1:
                    # 如果是多維張量，展平為一維
                    target = target.reshape(-1)
        
            return time_series, target
        else:
            return time_series

def augment_training_data(X_train, time_series_train, y_train, synthetic_samples=20, 
                       noise_level=0.05, physics_guided=True, a_coefficient=55.83, 
                       b_coefficient=-2.259):
    """
    進行資料增強，適用於小樣本數據集
    
    參數:
        X_train (numpy.ndarray): 靜態特徵訓練集
        time_series_train (numpy.ndarray): 時間序列特徵訓練集
        y_train (numpy.ndarray): 目標訓練集
        synthetic_samples (int): 生成的合成樣本數量
        noise_level (float): 噪聲水平
        physics_guided (bool): 是否使用物理知識引導增強
        a_coefficient (float): 物理模型係數a
        b_coefficient (float): 物理模型係數b
        
    返回:
        dict: 包含增強後資料的字典
    """
    logger.info(f"進行資料增強，原始訓練集大小: {len(X_train)} 樣本")
    
    try:
        # 導入資料增強相關模組
        from scipy.stats import norm
        import copy
        
        # 基本混合/擾動增強
        augmented_X = []
        augmented_time_series = []
        augmented_y = []
        
        # 1. 添加原始數據
        augmented_X.append(X_train)
        augmented_time_series.append(time_series_train)
        augmented_y.append(y_train)
        
        # 2. 添加小噪聲擾動樣本
        n_samples = len(X_train)
        for _ in range(min(10, synthetic_samples // 2)):
            # 複製原始樣本
            X_noise = X_train.copy()
            ts_noise = time_series_train.copy()
            y_noise = y_train.copy()
            
            # 添加小噪聲
            X_noise += np.random.normal(0, noise_level * np.std(X_train, axis=0), X_train.shape)
            
            # 對時間序列添加噪聲，但保持單調性
            for i in range(ts_noise.shape[0]):
                for j in range(ts_noise.shape[2]):  # 對每個特徵
                    # 添加噪聲
                    noise = np.random.normal(0, noise_level * np.std(ts_noise[:, :, j]), 
                                           (ts_noise.shape[0], ts_noise.shape[1]))
                    ts_noise[:, :, j] += noise
                    
                    # 確保時間序列單調增加
                    for sample_idx in range(ts_noise.shape[0]):
                        ts_noise[sample_idx, :, j] = np.maximum.accumulate(ts_noise[sample_idx, :, j])
            
            # 根據物理知識調整目標值
            if physics_guided:
                # 計算物理模型中的delta_w（從時間序列）
                # 假設最後時間步與初始時間步的差值代表delta_w
                delta_w = np.mean(ts_noise[:, -1, :] - ts_noise[:, 0, :], axis=1)
                
                # 使用物理模型計算新的目標值: Nf = a * (delta_w)^b
                y_noise = a_coefficient * np.power(np.maximum(delta_w, 1e-10), b_coefficient)
            else:
                # 添加一些對數正態噪聲以保持正值
                y_noise = y_noise * np.exp(np.random.normal(0, noise_level, y_noise.shape))
            
            augmented_X.append(X_noise)
            augmented_time_series.append(ts_noise)
            augmented_y.append(y_noise)
        
        # 3. 插值/混合樣本（僅在樣本數足夠時使用）
        if n_samples >= 5:
            for _ in range(min(10, synthetic_samples // 2)):
                # 隨機選擇兩個樣本進行混合
                idx1, idx2 = np.random.choice(n_samples, 2, replace=False)
                alpha = np.random.uniform(0.3, 0.7)  # 混合比例
                
                # 線性插值
                X_mix = alpha * X_train[idx1] + (1 - alpha) * X_train[idx2]
                ts_mix = alpha * time_series_train[idx1] + (1 - alpha) * time_series_train[idx2]
                
                # 使用物理模型計算目標值
                if physics_guided:
                    # 對混合後的時間序列計算delta_w
                    delta_w_mix = np.mean(ts_mix[-1, :] - ts_mix[0, :])
                    
                    # 使用物理模型計算新的目標值
                    y_mix = a_coefficient * np.power(max(delta_w_mix, 1e-10), b_coefficient)
                else:
                    # 線性插值目標值
                    y_mix = alpha * y_train[idx1] + (1 - alpha) * y_train[idx2]
                
                # 添加單個樣本
                augmented_X.append(np.expand_dims(X_mix, axis=0))
                augmented_time_series.append(np.expand_dims(ts_mix, axis=0))
                augmented_y.append(np.array([y_mix]))
        
        # 合併所有增強樣本
        X_augmented = np.vstack(augmented_X)
        time_series_augmented = np.vstack(augmented_time_series)
        y_augmented = np.concatenate(augmented_y)
        
        logger.info(f"資料增強完成，增強後的訓練集大小: {len(X_augmented)} 樣本")
        
        return {
            "X_train": X_augmented,
            "time_series_train": time_series_augmented,
            "y_train": y_augmented
        }
    except Exception as e:
        logger.error(f"資料增強時出錯: {str(e)}")
        logger.warning("返回原始資料繼續訓練")
        
        return {
            "X_train": X_train,
            "time_series_train": time_series_train,
            "y_train": y_train
        }


class TimeSeriesDataset(Dataset):
    """
    時間序列資料集
    專門處理銲錫接點的時間序列資料（如非線性塑性應變功時間序列）
    """
    def __init__(self, time_series, targets=None, transform=None, normalizer=None,
                normalize_per_feature=True, normalize_per_sample=False):
        """
        初始化時間序列資料集
        
        參數:
            time_series (numpy.ndarray): 時間序列資料，形狀為 (樣本數, 時間步數, 特徵數)
            targets (numpy.ndarray, optional): 目標資料，形狀為 (樣本數,)
            transform (callable, optional): 轉換函數
            normalizer (object, optional): 標準化器
            normalize_per_feature (bool): 是否對每個特徵分別標準化
            normalize_per_sample (bool): 是否對每個樣本分別標準化
        """
        self.time_series = torch.FloatTensor(time_series)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
        self.transform = transform
        self.normalizer = normalizer
        self.normalize_per_feature = normalize_per_feature
        self.normalize_per_sample = normalize_per_sample
        
        # 應用標準化
        if self.normalizer is not None:
            self._normalize_time_series_with_normalizer()
        elif self.normalize_per_feature or self.normalize_per_sample:
            self._normalize_time_series()
        
        self.n_samples = len(self.time_series)
        self.n_time_steps = self.time_series.shape[1]
        self.n_features = self.time_series.shape[2]
        
        logger.debug(f"初始化TimeSeriesDataset，樣本數: {self.n_samples}, "
                    f"時間步數: {self.n_time_steps}, 特徵數: {self.n_features}")
    
    def _normalize_time_series_with_normalizer(self):
        """使用提供的標準化器標準化時間序列"""
        if hasattr(self.normalizer, 'transform'):
            # scikit-learn風格標準化器
            # 將時間序列展平進行標準化，然後恢復形狀
            orig_shape = self.time_series.shape
            flattened = self.time_series.reshape(-1, orig_shape[-1])
            normalized = self.normalizer.transform(flattened)
            self.time_series = torch.FloatTensor(normalized.reshape(orig_shape))
    
    def _normalize_time_series(self):
        """根據指定策略標準化時間序列"""
        if self.normalize_per_feature:
            # 對每個特徵分別計算均值和標準差
            reshaped = self.time_series.reshape(-1, self.n_features)
            mean = torch.mean(reshaped, dim=0)
            std = torch.std(reshaped, dim=0)
            
            # 儲存標準化參數
            self._norm_params = {'mean': mean, 'std': std}
            
            # 應用標準化
            for i in range(self.n_samples):
                for j in range(self.n_features):
                    self.time_series[i, :, j] = (self.time_series[i, :, j] - mean[j]) / (std[j] + 1e-8)
        
        if self.normalize_per_sample:
            # 對每個樣本分別標準化
            for i in range(self.n_samples):
                for j in range(self.n_features):
                    series = self.time_series[i, :, j]
                    mean = torch.mean(series)
                    std = torch.std(series)
                    self.time_series[i, :, j] = (series - mean) / (std + 1e-8)
    
    def __len__(self):
        """返回樣本數量"""
        return self.n_samples
    
    

    def __getitem__(self, idx):
        """
        獲取指定索引的樣本
    
        參數:
            idx (int): 樣本索引
    
        返回:
            tuple: (靜態特徵, 時間序列特徵, 目標值) 或 (靜態特徵, 時間序列特徵)
        """
        static = self.static_features[idx]
        time_series = self.time_series_features[idx]

        # 應用轉換（如果有）
        if self.static_transform is not None:
            static = self.static_transform(static)

        if self.time_transform is not None:
            time_series = self.time_transform(time_series)
    
        # 檢查是否應用資料增強
        target = None
        if self.targets is not None:
            target = self.targets[idx]
        
            # 機率性應用資料增強
            if self.augmentation and torch.rand(1).item() < 0.3:  # 30%機率增強
                static, time_series, target = self._augment_sample(
                    static, time_series, target
                )
    
        # 檢查是否有目標值
        if target is not None:
            # 確保目標值是一維張量
            if isinstance(target, torch.Tensor):
                if target.dim() == 0:
                    # 將標量轉為一維張量
                    target = target.unsqueeze(0)
                elif target.dim() > 1:
                    # 減少多餘維度
                    target = target.squeeze()
                    # 如果squeeze後還是0維度，則轉為一維
                    if target.dim() == 0:
                        target = target.unsqueeze(0)
            else:
                # 如果不是張量，轉換為張量
                target = torch.tensor([target], dtype=torch.float32)
            
            return static, time_series, target
        else:
            return static, time_series


def create_dataloaders(X_train, X_val, X_test, 
                      time_series_train, time_series_val, time_series_test,
                      y_train, y_val, y_test,
                      batch_size=8, num_workers=0, pin_memory=False,
                      use_weighted_sampler=False, augmentation=False):
    """
    創建訓練、驗證和測試資料載入器
    
    參數:
        X_train, X_val, X_test (numpy.ndarray): 靜態特徵
        time_series_train, time_series_val, time_series_test (numpy.ndarray): 時間序列特徵
        y_train, y_val, y_test (numpy.ndarray): 目標值
        batch_size (int): 批次大小
        num_workers (int): 資料載入工作執行緒數量
        pin_memory (bool): 是否使用固定記憶體
        use_weighted_sampler (bool): 是否使用加權採樣器
        augmentation (bool): 是否啟用資料增強
        
    返回:
        dict: 包含訓練、驗證和測試資料載入器的字典
    """
    # 創建訓練資料集
    train_dataset = SolderFatigueDataset(
        static_features=X_train,
        time_series_features=time_series_train,
        targets=y_train,
        augmentation=augmentation
    )
    
    # 創建驗證資料集
    val_dataset = SolderFatigueDataset(
        static_features=X_val,
        time_series_features=time_series_val,
        targets=y_val
    )
    
    # 創建測試資料集
    test_dataset = SolderFatigueDataset(
        static_features=X_test,
        time_series_features=time_series_test,
        targets=y_test
    )
    
    # 若樣本數量少於批次大小，則調整批次大小
    actual_batch_size = min(batch_size, len(train_dataset))
    if actual_batch_size < batch_size:
        logger.warning(f"樣本數量({len(train_dataset)})少於批次大小({batch_size})，調整批次大小為{actual_batch_size}")
        batch_size = actual_batch_size
    
    # 如果需要，創建加權採樣器
    train_sampler = None
    if use_weighted_sampler and y_train is not None and len(y_train) >= 5:  # 確保有足夠樣本進行加權
        # 計算類別權重（對於回歸問題，可以將目標值分組）
        y_train_np = y_train
        
        # 對疲勞壽命進行對數轉換以更均勻地分組
        log_targets = np.log(y_train_np + 1e-8)
        
        # 將目標值分成n_bins組
        n_bins = min(5, len(log_targets) // 2)  # 確保每組至少有2個樣本
        if n_bins >= 2:  # 確保至少有2個分組
            bins = np.linspace(log_targets.min(), log_targets.max(), n_bins + 1)
            bin_indices = np.digitize(log_targets, bins) - 1
            
            # 計算每組的權重（反比於樣本數量）
            bin_counts = np.bincount(bin_indices, minlength=n_bins)
            bin_weights = 1.0 / (bin_counts + 1e-8)
            
            # 標準化權重使總和為len(log_targets)
            bin_weights = bin_weights * len(log_targets) / np.sum(bin_weights)
            
            # 為每個樣本分配權重
            weights = np.array([bin_weights[i] for i in bin_indices])
            
            # 創建加權採樣器
            train_sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True
            )
    
    # 創建資料載入器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,  # 如果有採樣器就不需要shuffle
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(batch_size, len(val_dataset)),  # 確保批次大小不超過樣本數
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=min(batch_size, len(test_dataset)),  # 確保批次大小不超過樣本數
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    logger.info(f"已創建資料載入器 - 訓練: {len(train_dataset)} 樣本, "
                f"驗證: {len(val_dataset)} 樣本, 測試: {len(test_dataset)} 樣本, "
                f"批次大小: {batch_size}")
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader
    }

def create_stratified_split(X, time_series, y, test_size=0.15, val_size=0.15, random_state=42):
    """
    使用分層抽樣創建訓練、驗證和測試資料分割
    處理小樣本數據集的特殊情況
    
    參數:
        X (numpy.ndarray): 靜態特徵
        time_series (numpy.ndarray): 時間序列特徵
        y (numpy.ndarray): 目標值
        test_size (float): 測試集比例
        val_size (float): 驗證集比例
        random_state (int): 隨機種子
        
    返回:
        dict: 包含分割後資料的字典
    """
    # 檢查樣本數
    n_samples = len(X)
    if n_samples <= 10:  # 小樣本數據集特殊處理
        logger.warning(f"樣本數少於10，使用簡單隨機抽樣而非分層抽樣")
        
        # 計算測試集和驗證集樣本數
        n_test = max(1, int(test_size * n_samples))
        n_val = max(1, int(val_size * n_samples))
        n_train = n_samples - n_test - n_val
        
        # 確保至少有1個樣本分配給每個集合
        if n_train < 1 or n_val < 1 or n_test < 1:
            logger.warning("樣本數過少，無法按比例分割，使用最小分配: 訓練=60%, 驗證=20%, 測試=20%")
            n_test = max(1, int(0.2 * n_samples))
            n_val = max(1, int(0.2 * n_samples))
            n_train = n_samples - n_test - n_val
        
        # 創建索引
        indices = np.random.RandomState(random_state).permutation(n_samples)
        test_idx = indices[:n_test]
        val_idx = indices[n_test:n_test+n_val]
        train_idx = indices[n_test+n_val:]
        
        # 分割數據
        X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
        time_series_train = time_series[train_idx]
        time_series_val = time_series[val_idx]
        time_series_test = time_series[test_idx]
        y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    else:
        # 原有的分層抽樣處理
        try:
            # 為分層抽樣創建類別標籤
            # 對於回歸問題，可以將目標值分組
            y_log = np.log(y + 1e-8)
            
            # 將目標值分成n_bins組
            n_bins = min(10, len(y) // 8)  # 確保每組至少有8個樣本
            bins = np.linspace(y_log.min(), y_log.max(), n_bins + 1)
            y_binned = np.digitize(y_log, bins)
            
            # 首先分割出測試集
            X_temp, X_test, time_series_temp, time_series_test, y_temp, y_test, y_binned_temp = train_test_split(
                X, time_series, y, y_binned, test_size=test_size, random_state=random_state, stratify=y_binned
            )
            
            # 從剩餘資料中分割出驗證集
            val_ratio = val_size / (1 - test_size)
            X_train, X_val, time_series_train, time_series_val, y_train, y_val = train_test_split(
                X_temp, time_series_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_binned_temp
            )
        except ValueError as e:
            # 如果分層抽樣失敗（例如，每個類別樣本數不足）
            logger.warning(f"分層抽樣失敗: {str(e)}，使用簡單隨機抽樣")
            
            # 首先分割出測試集
            X_temp, X_test, time_series_temp, time_series_test, y_temp, y_test = train_test_split(
                X, time_series, y, test_size=test_size, random_state=random_state
            )
            
            # 從剩餘資料中分割出驗證集
            val_ratio = val_size / (1 - test_size)
            X_train, X_val, time_series_train, time_series_val, y_train, y_val = train_test_split(
                X_temp, time_series_temp, y_temp, test_size=val_ratio, random_state=random_state
            )
    
    logger.info(f"資料分割 - 訓練: {len(X_train)} 樣本, "
                f"驗證: {len(X_val)} 樣本, 測試: {len(X_test)} 樣本")
    
    return {
        'X_train': X_train, 'time_series_train': time_series_train, 'y_train': y_train,
        'X_val': X_val, 'time_series_val': time_series_val, 'y_val': y_val,
        'X_test': X_test, 'time_series_test': time_series_test, 'y_test': y_test
    }


def create_k_fold_cv(X, time_series, y, n_splits=5, shuffle=True, random_state=42):
    """
    創建k折交叉驗證分割
    
    參數:
        X (numpy.ndarray): 靜態特徵
        time_series (numpy.ndarray): 時間序列特徵
        y (numpy.ndarray): 目標值
        n_splits (int): 折數
        shuffle (bool): 是否打亂資料
        random_state (int): 隨機種子
        
    返回:
        list: 包含每折分割索引的列表
    """
    # 為分層交叉驗證創建類別標籤
    y_log = np.log(y + 1e-8)
    
    # 將目標值分成n_bins組
    n_bins = min(10, len(y) // (2 * n_splits))  # 確保每組至少有2*n_splits個樣本
    bins = np.linspace(y_log.min(), y_log.max(), n_bins + 1)
    y_binned = np.digitize(y_log, bins)
    
    # 創建分層k折交叉驗證
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    # 創建分割索引
    fold_indices = []
    for train_idx, val_idx in skf.split(X, y_binned):
        fold_indices.append((train_idx, val_idx))
    
    logger.info(f"創建{n_splits}折交叉驗證分割")
    
    return fold_indices


def create_leave_one_out_cv(X, time_series, y):
    """
    創建留一法交叉驗證分割
    適用於小樣本數據集
    
    參數:
        X (numpy.ndarray): 靜態特徵
        time_series (numpy.ndarray): 時間序列特徵
        y (numpy.ndarray): 目標值
        
    返回:
        list: 包含每折分割索引的列表
    """
    n_samples = len(X)
    fold_indices = []
    
    for i in range(n_samples):
        val_idx = [i]
        train_idx = [j for j in range(n_samples) if j != i]
        fold_indices.append((train_idx, val_idx))
    
    logger.info(f"創建留一法交叉驗證分割，共{n_samples}折")
    
    return fold_indices


def create_dataset_from_fold(X, time_series, y, fold_indices, fold_idx,
                           normalizers=None, augmentation=False):
    """
    從交叉驗證分割創建資料集
    
    參數:
        X (numpy.ndarray): 靜態特徵
        time_series (numpy.ndarray): 時間序列特徵
        y (numpy.ndarray): 目標值
        fold_indices (list): 分割索引列表
        fold_idx (int): 當前折索引
        normalizers (dict, optional): 標準化器字典
        augmentation (bool): 是否啟用資料增強
        
    返回:
        dict: 包含訓練和驗證資料集的字典
    """
    train_idx, val_idx = fold_indices[fold_idx]
    
    # 分割資料
    X_train, X_val = X[train_idx], X[val_idx]
    time_series_train, time_series_val = time_series[train_idx], time_series[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # 創建標準化器（如果沒有提供）
    if normalizers is None:
        normalizers = {}
        
        # 靜態特徵標準化
        from sklearn.preprocessing import StandardScaler
        static_scaler = StandardScaler()
        static_scaler.fit(X_train)
        normalizers['static_scaler'] = static_scaler
        
        # 時間序列標準化（對每個特徵單獨標準化）
        time_series_reshaped = time_series_train.reshape(-1, time_series_train.shape[-1])
        time_scaler = StandardScaler()
        time_scaler.fit(time_series_reshaped)
        normalizers['time_scaler'] = time_scaler
    
    # 創建資料集
    train_dataset = SolderFatigueDataset(
        static_features=X_train,
        time_series_features=time_series_train,
        targets=y_train,
        static_normalizer=normalizers.get('static_scaler'),
        time_normalizer=normalizers.get('time_scaler'),
        augmentation=augmentation
    )
    
    val_dataset = SolderFatigueDataset(
        static_features=X_val,
        time_series_features=time_series_val,
        targets=y_val,
        static_normalizer=normalizers.get('static_scaler'),
        time_normalizer=normalizers.get('time_scaler')
    )
    
    logger.info(f"從第{fold_idx+1}折創建資料集 - 訓練: {len(train_dataset)} 樣本, "
                f"驗證: {len(val_dataset)} 樣本")
    
    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'normalizers': normalizers
    }


if __name__ == "__main__":
    # 設定日誌
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 測試代碼
    logger.info("創建測試資料")
    
    # 生成隨機測試資料
    n_samples = 100
    n_static_features = 5
    n_time_steps = 4
    n_time_features = 2
    
    # 靜態特徵
    X = np.random.rand(n_samples, n_static_features)
    
    # 時間序列特徵（確保遞增）
    time_series = np.zeros((n_samples, n_time_steps, n_time_features))
    for i in range(n_samples):
        for j in range(n_time_features):
            time_series[i, :, j] = np.sort(np.random.rand(n_time_steps))
    
    # 目標值（疲勞壽命）- 使用指數分佈模擬跨度大的壽命數據
    y = np.exp(2 + 3 * np.random.rand(n_samples))
    
    # 測試資料集
    logger.info("測試SolderFatigueDataset")
    dataset = SolderFatigueDataset(X, time_series, y, augmentation=True)
    logger.info(f"資料集大小: {len(dataset)}")
    
    static, ts, target = dataset[0]
    logger.info(f"靜態特徵形狀: {static.shape}")
    logger.info(f"時間序列形狀: {ts.shape}")
    logger.info(f"目標形狀: {target.shape if hasattr(target, 'shape') else 'scalar'}")
    
    # 測試加權採樣
    logger.info("測試加權採樣資料載入器")
    dataloaders = create_dataloaders(
        X[:70], X[70:85], X[85:],
        time_series[:70], time_series[70:85], time_series[85:],
        y[:70], y[70:85], y[85:],
        batch_size=16,
        use_weighted_sampler=True,
        augmentation=True
    )
    
    train_loader = dataloaders['train_loader']
    logger.info(f"訓練資料載入器批次數: {len(train_loader)}")
    
    # 取出一個批次查看形狀
    for static_batch, ts_batch, target_batch in train_loader:
        logger.info(f"批次靜態特徵形狀: {static_batch.shape}")
        logger.info(f"批次時間序列形狀: {ts_batch.shape}")
        logger.info(f"批次目標形狀: {target_batch.shape}")
        break
    
    # 測試分層資料分割
    logger.info("測試分層資料分割")
    split_data = create_stratified_split(X, time_series, y, 
                                        test_size=0.15, val_size=0.15)
    
    # 檢查分割後的資料大小
    logger.info(f"訓練集: {len(split_data['X_train'])}")
    logger.info(f"驗證集: {len(split_data['X_val'])}")
    logger.info(f"測試集: {len(split_data['X_test'])}")
    
    # 測試k折交叉驗證
    logger.info("測試k折交叉驗證")
    fold_indices = create_k_fold_cv(X, time_series, y, n_splits=5)
    logger.info(f"交叉驗證折數: {len(fold_indices)}")
    
    # 測試從交叉驗證折創建資料集
    logger.info("測試從交叉驗證折創建資料集")
    fold_data = create_dataset_from_fold(X, time_series, y, fold_indices, fold_idx=0,
                                       augmentation=True)
    
    train_dataset = fold_data['train_dataset']
    val_dataset = fold_data['val_dataset']
    
    logger.info(f"訓練集大小: {len(train_dataset)}")
    logger.info(f"驗證集大小: {len(val_dataset)}")
    
    # 如果數據集很小，測試留一法交叉驗證
    if n_samples <= 100:
        logger.info("測試留一法交叉驗證")
        loo_indices = create_leave_one_out_cv(X, time_series, y)
        logger.info(f"留一法折數: {len(loo_indices)}")
        
        # 使用第一折測試
        loo_data = create_dataset_from_fold(X, time_series, y, loo_indices, fold_idx=0)
        logger.info(f"留一法訓練集大小: {len(loo_data['train_dataset'])}")
        logger.info(f"留一法驗證集大小: {len(loo_data['val_dataset'])}")
    
    logger.info("測試完成")



def load_dataset(
    csv_path,
    static_features,
    time_series_features,
    target_feature,
    sequence_length=10,
    batch_size=32,
    val_ratio=0.2,
    random_seed=42,
    num_workers=0
):
    """從 CSV 載入資料並建立 DataLoader"""
    df = pd.read_csv(csv_path)

    # 靜態特徵
    static_data = df[static_features].values

    # 時間序列特徵需要 reshape 成 (樣本數, 時間步數, 特徵數)
    time_data = df[time_series_features].values
    num_samples = len(df)
    time_data = time_data.reshape((num_samples, sequence_length, -1))

    # 預測目標
    targets = df[target_feature].values

    # 切分訓練與驗證
    X_static_train, X_static_val, X_time_train, X_time_val, y_train, y_val = train_test_split(
        static_data, time_data, targets, test_size=val_ratio, random_state=random_seed
    )

    # 建立 Dataset
    train_dataset = SolderFatigueDataset(X_static_train, X_time_train, y_train)
    val_dataset = SolderFatigueDataset(X_static_val, X_time_val, y_val)

    # 建立 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset
    }
