#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
dataset.py - 資料集封裝模組
本模組提供自定義資料集的封裝，用於銲錫接點疲勞壽命預測模型訓練和評估。
實現了適合混合PINN-LSTM模型所需的資料載入與批次處理功能。

主要組件:
1. SolderFatigueDataset - 基本資料集類別，處理靜態特徵與目標
2. TimeSeriesDataset - 時間序列資料集類別，處理時間序列特徵
3. 資料載入器創建功能
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import logging

logger = logging.getLogger(__name__)

class SolderFatigueDataset(Dataset):
    """
    銲錫接點疲勞壽命資料集
    處理靜態特徵（結構參數）與預測目標（疲勞壽命）
    """
    def __init__(self, features, targets=None, transform=None):
        """
        初始化資料集
        
        參數:
            features (numpy.ndarray): 特徵資料，形狀為 (樣本數, 特徵數)
            targets (numpy.ndarray, optional): 目標資料，形狀為 (樣本數,)
            transform (callable, optional): 應用於特徵的轉換函數
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
        self.transform = transform
        
        logger.debug(f"初始化SolderFatigueDataset，特徵形狀: {self.features.shape}, "
                    f"目標形狀: {None if self.targets is None else self.targets.shape}")

    def __len__(self):
        """返回資料集中的樣本數量"""
        return len(self.features)

    def __getitem__(self, idx):
        """獲取指定索引的樣本"""
        features = self.features[idx]
        
        # 應用轉換（如果有）
        if self.transform:
            features = self.transform(features)
        
        # 如果有目標，返回特徵和目標；否則只返回特徵
        if self.targets is not None:
            return features, self.targets[idx]
        else:
            return features


class TimeSeriesDataset(Dataset):
    """
    時間序列資料集
    處理時間序列特徵（非線性塑性應變功時間序列）
    """
    def __init__(self, time_series, targets=None, transform=None):
        """
        初始化時間序列資料集
        
        參數:
            time_series (numpy.ndarray): 時間序列資料，形狀為 (樣本數, 時間步數, 特徵數)
            targets (numpy.ndarray, optional): 目標資料，形狀為 (樣本數,)
            transform (callable, optional): 應用於時間序列的轉換函數
        """
        self.time_series = torch.FloatTensor(time_series)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
        self.transform = transform
        
        logger.debug(f"初始化TimeSeriesDataset，時間序列形狀: {self.time_series.shape}, "
                    f"目標形狀: {None if self.targets is None else self.targets.shape}")

    def __len__(self):
        """返回資料集中的樣本數量"""
        return len(self.time_series)

    def __getitem__(self, idx):
        """獲取指定索引的樣本"""
        time_series = self.time_series[idx]
        
        # 應用轉換（如果有）
        if self.transform:
            time_series = self.transform(time_series)
        
        # 如果有目標，返回時間序列和目標；否則只返回時間序列
        if self.targets is not None:
            return time_series, self.targets[idx]
        else:
            return time_series


class HybridDataset(Dataset):
    """
    混合資料集
    同時處理靜態特徵和時間序列特徵
    """
    def __init__(self, features, time_series, targets=None, 
                 feature_transform=None, time_series_transform=None):
        """
        初始化混合資料集
        
        參數:
            features (numpy.ndarray): 靜態特徵資料，形狀為 (樣本數, 特徵數)
            time_series (numpy.ndarray): 時間序列資料，形狀為 (樣本數, 時間步數, 特徵數)
            targets (numpy.ndarray, optional): 目標資料，形狀為 (樣本數,)
            feature_transform (callable, optional): 應用於靜態特徵的轉換函數
            time_series_transform (callable, optional): 應用於時間序列的轉換函數
        """
        if len(features) != len(time_series):
            raise ValueError("特徵資料和時間序列資料的樣本數量必須相同")
        
        self.features = torch.FloatTensor(features)
        self.time_series = torch.FloatTensor(time_series)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
        self.feature_transform = feature_transform
        self.time_series_transform = time_series_transform
        
        logger.debug(f"初始化HybridDataset，特徵形狀: {self.features.shape}, "
                    f"時間序列形狀: {self.time_series.shape}, "
                    f"目標形狀: {None if self.targets is None else self.targets.shape}")

    def __len__(self):
        """返回資料集中的樣本數量"""
        return len(self.features)

    def __getitem__(self, idx):
        """獲取指定索引的樣本"""
        features = self.features[idx]
        time_series = self.time_series[idx]
        
        # 應用轉換（如果有）
        if self.feature_transform:
            features = self.feature_transform(features)
        
        if self.time_series_transform:
            time_series = self.time_series_transform(time_series)
        
        # 如果有目標，返回特徵、時間序列和目標；否則只返回特徵和時間序列
        if self.targets is not None:
            return features, time_series, self.targets[idx]
        else:
            return features, time_series


def create_dataloaders(X_train, X_val, X_test, 
                      time_series_train, time_series_val, time_series_test,
                      y_train, y_val, y_test,
                      batch_size=16, num_workers=0, pin_memory=False):
    """
    創建資料載入器
    
    參數:
        X_train, X_val, X_test (numpy.ndarray): 訓練、驗證和測試集的靜態特徵
        time_series_train, time_series_val, time_series_test (numpy.ndarray): 
            訓練、驗證和測試集的時間序列特徵
        y_train, y_val, y_test (numpy.ndarray): 訓練、驗證和測試集的目標變數
        batch_size (int): 批次大小
        num_workers (int): 資料載入器使用的工作執行緒數量
        pin_memory (bool): 是否使用固定內存
        
    返回:
        dict: 包含訓練、驗證和測試集資料載入器的字典
    """
    # 考慮小樣本資料集（81筆），調整批次大小
    if len(y_train) < batch_size:
        original_batch_size = batch_size
        batch_size = max(1, len(y_train) // 2)  # 確保批次大小至少為1
        logger.warning(f"訓練集樣本數 ({len(y_train)}) 小於設定的批次大小 ({original_batch_size})，"
                     f"已調整批次大小為 {batch_size}")
    
    # 創建混合資料集
    train_dataset = HybridDataset(X_train, time_series_train, y_train)
    val_dataset = HybridDataset(X_val, time_series_val, y_val)
    test_dataset = HybridDataset(X_test, time_series_test, y_test)
    
    # 創建資料載入器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,  # 小樣本資料集不丟棄最後的批次
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    
    logger.info(f"創建資料載入器完成，批次大小={batch_size}，"
               f"訓練集批次數={len(train_loader)}，"
               f"驗證集批次數={len(val_loader)}，"
               f"測試集批次數={len(test_loader)}")
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'batch_size': batch_size
    }


if __name__ == "__main__":
    # 簡單的測試代碼
    import os
    from src.data.preprocess import process_pipeline
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 獲取相對於專案根目錄的資料路徑
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    data_path = os.path.join(project_root, "data/raw/Training_data_warpage_final_20250321_v1.2.csv")
    
    if os.path.exists(data_path):
        # 預處理資料
        data_dict = process_pipeline(data_path)
        
        # 測試資料載入器
        loaders = create_dataloaders(
            data_dict['X_train'], data_dict['X_val'], data_dict['X_test'],
            data_dict['time_series_train'], data_dict['time_series_val'], data_dict['time_series_test'],
            data_dict['y_train'], data_dict['y_val'], data_dict['y_test'],
            batch_size=8
        )
        
        # 檢驗載入的批次
        for i, (features, time_series, targets) in enumerate(loaders['train_loader']):
            logger.info(f"批次 {i+1}:")
            logger.info(f"  特徵形狀: {features.shape}")
            logger.info(f"  時間序列形狀: {time_series.shape}")
            logger.info(f"  目標形狀: {targets.shape}")
            if i >= 2:  # 只檢查前幾個批次
                break
    else:
        logger.error(f"找不到資料檔案: {data_path}")