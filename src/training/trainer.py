#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
trainer.py - 訓練器模組
本模組提供銲錫接點疲勞壽命預測模型的訓練和評估功能，
包括基礎訓練器、早停機制和學習率調度等功能。

主要組件:
1. Trainer - 基礎訓練器，提供訓練和評估的核心功能
2. EarlyStopping - 早停機制，避免過擬合
3. LearningRateScheduler - 學習率調度器，動態調整學習率
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

from src.utils.metrics import calculate_rmse, calculate_r2, calculate_mae

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    早停機制
    監控指標不再改善時提前結束訓練，避免過擬合
    """
    def __init__(self, patience=10, min_delta=0.0001, verbose=True, mode='min'):
        """
        初始化早停機制
        
        參數:
            patience (int): 容忍的輪數，指標不再改善後等待多少輪停止
            min_delta (float): 最小改善閾值，小於此值視為沒有改善
            verbose (bool): 是否輸出詳細日誌
            mode (str): 監控模式，'min'表示指標越小越好，'max'表示指標越大越好
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf') if mode == 'min' else float('-inf')
    
    def __call__(self, score, model=None, save_path=None):
        """
        執行早停檢查
        
        參數:
            score (float): 當前監控指標值
            model (torch.nn.Module, optional): 如果提供模型，則在指標改善時保存模型
            save_path (str, optional): 模型保存路徑
            
        返回:
            bool: 是否達到早停條件
        """
        # 檢查模式
        if self.mode == 'min':
            # 對於需要最小化的指標（如損失）
            score_improved = self.best_score is None or score < self.best_score - self.min_delta
        else:
            # 對於需要最大化的指標（如準確率）
            score_improved = self.best_score is None or score > self.best_score + self.min_delta
        
        if score_improved:
            # 指標改善，重置計數器
            if self.verbose:
                improvement = "" if self.best_score is None else f", 從 {self.best_score:.6f} 改善到 {score:.6f}"
                logger.info(f"{self.mode}模式下監控指標改善{improvement}")
            
            self.best_score = score
            self.counter = 0
            
            # 如果提供了模型和保存路徑，保存模型
            if model is not None and save_path is not None:
                self._save_checkpoint(model, score, save_path)
        else:
            # 指標未改善，增加計數器
            self.counter += 1
            if self.verbose:
                logger.info(f"監控指標未改善，已連續 {self.counter} 輪。最佳值: {self.best_score:.6f}, 當前值: {score:.6f}")
            
            if self.counter >= self.patience:
                # 達到早停條件
                self.early_stop = True
                if self.verbose:
                    logger.info(f"早停觸發！{self.counter} 輪未改善")
        
        return self.early_stop
    
    def _save_checkpoint(self, model, score, save_path):
        """
        保存模型檢查點
        
        參數:
            model (torch.nn.Module): 模型
            score (float): 監控指標值
            save_path (str): 保存路徑
        """
        # 確保目錄存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存模型
        torch.save(model.state_dict(), save_path)
        
        if self.verbose:
            logger.info(f"監控指標改善，模型已保存至 {save_path}")
    
    def reset(self):
        """重置早停狀態"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf') if self.mode == 'min' else float('-inf')


class LearningRateScheduler:
    """
    學習率調度器
    動態調整學習率，加速收斂過程
    """
    def __init__(self, optimizer, mode='step', step_size=10, gamma=0.1, 
                 min_lr=0.00001, factor=0.5, patience=5, T_max=100, 
                 monitor='val_loss', monitor_mode='min', verbose=True):
        """
        初始化學習率調度器
        
        參數:
            optimizer (torch.optim.Optimizer): 優化器
            mode (str): 調度模式，'step', 'exp', 'plateau', 'cosine'
            step_size (int): step模式下的步長
            gamma (float): 學習率衰減因子
            min_lr (float): 最小學習率
            factor (float): plateau模式下的衰減因子
            patience (int): plateau模式下的耐心值
            T_max (int): cosine模式下的週期長度
            monitor (str): plateau模式下監控的指標
            monitor_mode (str): 監控模式，'min'或'max'
            verbose (bool): 是否輸出詳細日誌
        """
        self.optimizer = optimizer
        self.mode = mode
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.verbose = verbose
        self.min_lr = min_lr
        
        # 根據模式創建調度器
        if mode == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=step_size, 
                gamma=gamma
            )
            logger.info(f"創建步進式學習率調度器，步長={step_size}，衰減因子={gamma}")
        elif mode == 'exp':
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer, 
                gamma=gamma
            )
            logger.info(f"創建指數式學習率調度器，衰減因子={gamma}")
        elif mode == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode=monitor_mode, 
                factor=factor, 
                patience=patience, 
                min_lr=min_lr,
                verbose=verbose
            )
            logger.info(f"創建平原式學習率調度器，模式={monitor_mode}，"
                       f"衰減因子={factor}，耐心值={patience}，最小學習率={min_lr}")
        elif mode == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=T_max, 
                eta_min=min_lr
            )
            logger.info(f"創建餘弦退火學習率調度器，週期={T_max}，最小學習率={min_lr}")
        else:
            raise ValueError(f"不支援的調度模式: {mode}")
        
        # 當前學習率
        self.current_lr = [param_group['lr'] for param_group in optimizer.param_groups]
    
    def step(self, val=None):
        """
        執行學習率調度
        
        參數:
            val (float, optional): 監控指標的值，用於plateau模式
        """
        if self.mode == 'plateau' and val is not None:
            # 平原式調度需要監控指標值
            self.scheduler.step(val)
        else:
            # 其他調度模式直接步進
            self.scheduler.step()
        
        # 更新當前學習率
        self.current_lr = [param_group['lr'] for param_group in self.optimizer.param_groups]
        
        if self.verbose:
            logger.debug(f"學習率更新為: {self.current_lr}")
    
    def step_with_metrics(self, val):
        """
        使用指標值執行學習率調度
        
        參數:
            val (float): 監控指標的值
        """
        self.step(val)
    
    def get_lr(self):
        """
        獲取當前學習率
        
        返回:
            list: 當前學習率列表（可能有多個參數組）
        """
        return self.current_lr
    
    def state_dict(self):
        """
        獲取調度器狀態字典
        
        返回:
            dict: 調度器狀態
        """
        return self.scheduler.state_dict()
    
    def load_state_dict(self, state_dict):
        """
        載入調度器狀態字典
        
        參數:
            state_dict (dict): 調度器狀態
        """
        self.scheduler.load_state_dict(state_dict)
        self.current_lr = [param_group['lr'] for param_group in self.optimizer.param_groups]


class Trainer:
    """
    訓練器
    提供模型訓練和評估的核心功能
    """
    def __init__(self, model, criterion, optimizer, device, 
                 metrics=None, scheduler=None, clip_grad_norm=0.0,
                 log_interval=10, use_amp=False):
        """
        初始化訓練器
        
        參數:
            model (torch.nn.Module): 模型
            criterion (callable): 損失函數
            optimizer (torch.optim.Optimizer): 優化器
            device (torch.device): 計算設備
            metrics (dict, optional): 評估指標函數字典
            scheduler (LearningRateScheduler, optional): 學習率調度器
            clip_grad_norm (float): 梯度裁剪範數，0表示不裁剪
            log_interval (int): 日誌輸出間隔（每隔多少批次輸出一次）
            use_amp (bool): 是否使用自動混合精度
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.metrics = metrics or {
            'rmse': calculate_rmse,
            'r2': calculate_r2,
            'mae': calculate_mae
        }
        self.scheduler = scheduler
        self.clip_grad_norm = clip_grad_norm
        self.log_interval = log_interval
        self.use_amp = use_amp
        
        # 初始化訓練記錄
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        self.learning_rates = []
        
        # 初始化混合精度訓練
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("啟用自動混合精度訓練")
        
        logger.info(f"初始化訓練器完成，模型: {type(model).__name__}, "
                   f"優化器: {type(optimizer).__name__}, 設備: {device}")
    
    def train_epoch(self, train_loader):
        """
        訓練一個輪次
        
        參數:
            train_loader (DataLoader): 訓練數據載入器
            
        返回:
            dict: 包含訓練損失和指標的字典
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        all_targets = []
        all_predictions = []
        
        # 使用tqdm顯示進度條（如果可用）
        try:
            from tqdm import tqdm
            train_iter = tqdm(train_loader, desc="Training", leave=False)
        except ImportError:
            train_iter = train_loader
        
        for batch_idx, data in enumerate(train_iter):
            # 處理不同形式的數據批次
            if isinstance(data, (list, tuple)) and len(data) >= 2:
                if len(data) == 2:
                    # 基本形式: (inputs, targets)
                    inputs, targets = data
                    if isinstance(inputs, (list, tuple)):
                        # 多輸入形式: ([input1, input2, ...], targets)
                        inputs = [x.to(self.device) for x in inputs]
                    else:
                        # 單輸入形式: (input, targets)
                        inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                elif len(data) == 3:
                    # 混合PINN-LSTM形式: (static_features, time_series, targets)
                    static_features, time_series, targets = data
                    static_features = static_features.to(self.device)
                    time_series = time_series.to(self.device)
                    targets = targets.to(self.device)
                    # 保存為元組用於前向傳播
                    inputs = (static_features, time_series)
            else:
                # 自定義批次格式
                raise ValueError("不支援的批次數據格式")
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 根據是否使用自動混合精度選擇不同的前向和反向傳播方式
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    # 前向傳播
                    if isinstance(inputs, (list, tuple)) and len(inputs) == 2 and hasattr(self.model, 'forward') and 'static_features' in self.model.forward.__code__.co_varnames:
                        # 專門用於混合PINN-LSTM模型
                        outputs = self.model(inputs[0], inputs[1])
                        
                        # 檢查outputs是否為字典（用於混合模型）
                        if isinstance(outputs, dict) and 'nf_pred' in outputs:
                            predictions = outputs['nf_pred']
                            losses = self.criterion(outputs, targets, self.model)
                            loss = losses['total_loss']
                        else:
                            predictions = outputs
                            loss = self.criterion(predictions, targets)
                    else:
                        # 一般模型
                        outputs = self.model(inputs)
                        
                        # 檢查outputs是否為字典
                        if isinstance(outputs, dict) and 'output' in outputs:
                            predictions = outputs['output']
                        else:
                            predictions = outputs
                        
                        # 計算損失
                        loss = self.criterion(predictions, targets)
                
                # 使用scaler處理反向傳播和優化器步驟
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                if self.clip_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 常規前向傳播
                if isinstance(inputs, (list, tuple)) and len(inputs) == 2 and hasattr(self.model, 'forward') and 'static_features' in self.model.forward.__code__.co_varnames:
                    # 專門用於混合PINN-LSTM模型
                    outputs = self.model(inputs[0], inputs[1])
                    
                    # 檢查outputs是否為字典（用於混合模型）
                    if isinstance(outputs, dict) and 'nf_pred' in outputs:
                        predictions = outputs['nf_pred']
                        losses = self.criterion(outputs, targets, self.model)
                        loss = losses['total_loss']
                    else:
                        predictions = outputs
                        loss = self.criterion(predictions, targets)
                else:
                    # 一般模型
                    outputs = self.model(inputs)
                    
                    # 檢查outputs是否為字典
                    if isinstance(outputs, dict) and 'output' in outputs:
                        predictions = outputs['output']
                    else:
                        predictions = outputs
                    
                    # 計算損失
                    loss = self.criterion(predictions, targets)
                
                # 反向傳播
                loss.backward()
                
                # 梯度裁剪
                if self.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                
                # 參數更新
                self.optimizer.step()
            
            # 收集損失和預測結果
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # 收集預測和目標，用於計算指標
            if isinstance(predictions, torch.Tensor):
                all_predictions.append(predictions.detach().cpu().numpy())
                all_targets.append(targets.detach().cpu().numpy())
            
            # 更新進度條
            if hasattr(train_iter, 'set_postfix'):
                train_iter.set_postfix({'loss': loss.item()})
        
        # 計算平均損失
        avg_loss = total_loss / total_samples
        
        # 計算指標
        metrics_values = {}
        if all_predictions and all_targets:
            all_predictions = np.concatenate([p.reshape(-1) for p in all_predictions])
            all_targets = np.concatenate([t.reshape(-1) for t in all_targets])
            
            for name, metric_fn in self.metrics.items():
                try:
                    metrics_values[name] = metric_fn(all_targets, all_predictions)
                except Exception as e:
                    logger.warning(f"計算指標 {name} 時出錯: {str(e)}")
                    metrics_values[name] = float('nan')
        
        return {
            'loss': avg_loss,
            'metrics': metrics_values,
            'predictions': all_predictions if len(all_predictions) > 0 else None,
            'targets': all_targets if all_targets else None
        }
    
    def validate(self, val_loader):
        """
        驗證模型
        
        參數:
            val_loader (DataLoader): 驗證數據載入器
            
        返回:
            dict: 包含驗證損失和指標的字典
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_targets = []
        all_predictions = []
        all_outputs = {}
        
        with torch.no_grad():
            for data in val_loader:
                # 處理不同形式的數據批次
                if isinstance(data, (list, tuple)) and len(data) >= 2:
                    if len(data) == 2:
                        # 基本形式: (inputs, targets)
                        inputs, targets = data
                        if isinstance(inputs, (list, tuple)):
                            # 多輸入形式: ([input1, input2, ...], targets)
                            inputs = [x.to(self.device) for x in inputs]
                        else:
                            # 單輸入形式: (input, targets)
                            inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                    elif len(data) == 3:
                        # 混合PINN-LSTM形式: (static_features, time_series, targets)
                        static_features, time_series, targets = data
                        static_features = static_features.to(self.device)
                        time_series = time_series.to(self.device)
                        targets = targets.to(self.device)
                        # 保存為元組用於前向傳播
                        inputs = (static_features, time_series)
                else:
                    # 自定義批次格式
                    raise ValueError("不支援的批次數據格式")
                
                # 前向傳播
                if isinstance(inputs, (list, tuple)) and len(inputs) == 2 and hasattr(self.model, 'forward') and 'static_features' in self.model.forward.__code__.co_varnames:
                    # 專門用於混合PINN-LSTM模型
                    outputs = self.model(inputs[0], inputs[1], return_features=True)
                    
                    # 檢查outputs是否為字典（用於混合模型）
                    if isinstance(outputs, dict) and 'nf_pred' in outputs:
                        predictions = outputs['nf_pred']
                        losses = self.criterion(outputs, targets, self.model)
                        loss = losses['total_loss']
                        
                        # 收集輸出
                        for key, value in outputs.items():
                            if key not in all_outputs:
                                all_outputs[key] = []
                            
                            if isinstance(value, torch.Tensor):
                                all_outputs[key].append(value.cpu().numpy())
                            else:
                                all_outputs[key].append(value)
                    else:
                        predictions = outputs
                        loss = self.criterion(predictions, targets)
                else:
                    # 一般模型
                    outputs = self.model(inputs)
                    
                    # 檢查outputs是否為字典
                    if isinstance(outputs, dict) and 'output' in outputs:
                        predictions = outputs['output']
                    else:
                        predictions = outputs
                    
                    # 計算損失
                    loss = self.criterion(predictions, targets)
                
                # 收集損失和預測結果
                batch_size = targets.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # 收集預測和目標，用於計算指標
                if isinstance(predictions, torch.Tensor):
                    all_predictions.append(predictions.detach().cpu().numpy())
                    all_targets.append(targets.detach().cpu().numpy())
        
        # 計算平均損失
        avg_loss = total_loss / total_samples
        
        # 計算指標
        metrics_values = {}
        if all_predictions and all_targets:
            try:
                all_predictions = np.concatenate([p.reshape(-1) for p in all_predictions])
                all_targets = np.concatenate([t.reshape(-1) for t in all_targets])
                
                for name, metric_fn in self.metrics.items():
                    try:
                        metrics_values[name] = metric_fn(all_targets, all_predictions)
                    except Exception as e:
                        logger.warning(f"計算指標 {name} 時出錯: {str(e)}")
                        metrics_values[name] = float('nan')
            except Exception as e:
                logger.error(f"合併預測結果時出錯: {str(e)}")
        
        # 合併所有輸出
        outputs_merged = {}
        for key, values in all_outputs.items():
            if values and isinstance(values[0], np.ndarray):
                outputs_merged[key] = np.concatenate(values)
            else:
                outputs_merged[key] = values
        
        return {
            'loss': avg_loss,
            'metrics': metrics_values,
            'predictions': all_predictions if len(all_predictions) > 0 else None,
            'targets': all_targets if len(all_targets) > 0 else None,
            'outputs': outputs_merged
        }
    
    def train(self, train_loader, val_loader, epochs, early_stopping=None,
             callbacks=None, save_path=None):
        """
        訓練模型
        
        參數:
            train_loader (DataLoader): 訓練數據載入器
            val_loader (DataLoader): 驗證數據載入器
            epochs (int): 訓練輪數
            early_stopping (EarlyStopping, optional): 早停機制
            callbacks (list, optional): 回調函數列表
            save_path (str, optional): 模型保存路徑
            
        返回:
            dict: 訓練歷史記錄
        """
        callbacks = callbacks or []
        start_time = time.time()
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': defaultdict(list),
            'val_metrics': defaultdict(list),
            'learning_rate': []
        }
        
        logger.info(f"開始訓練，總輪數: {epochs}")
        
        # 遍歷每個訓練輪次
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # 訓練一個輪次
            train_results = self.train_epoch(train_loader)
            train_loss = train_results['loss']
            train_metrics = train_results['metrics']
            
            # 驗證
            val_results = self.validate(val_loader)
            val_loss = val_results['loss']
            val_metrics = val_results['metrics']
            
            # 記錄學習率
            current_lr = self.optimizer.param_groups[0]['lr']
            history['learning_rate'].append(current_lr)
            
            # 更新學習率
            if self.scheduler is not None:
                # 使用驗證損失更新學習率（對於ReduceLROnPlateau）
                if hasattr(self.scheduler, 'step_with_metrics'):
                    self.scheduler.step_with_metrics(val_loss)
                else:
                    self.scheduler.step()
            
            # 記錄歷史
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            for name, value in train_metrics.items():
                history['train_metrics'][name].append(value)
            
            for name, value in val_metrics.items():
                history['val_metrics'][name].append(value)
            
            # 輸出日誌
            if epoch % self.log_interval == 0 or epoch == epochs - 1:
                # 計算輪次訓練時間
                epoch_time = time.time() - epoch_start_time
                
                # 準備日誌信息
                log_info = [
                    f"輪次 {epoch+1}/{epochs}",
                    f"訓練損失: {train_loss:.6f}",
                    f"驗證損失: {val_loss:.6f}",
                ]
                
                # 添加主要指標
                for name in ['rmse', 'r2', 'mae']:
                    if name in val_metrics:
                        log_info.append(f"{name}: {val_metrics[name]:.6f}")
                
                log_info.append(f"學習率: {current_lr:.6f}")
                log_info.append(f"時間: {epoch_time:.2f}秒")
                
                logger.info(" - ".join(log_info))
            
            # 執行回調函數
            for callback in callbacks:
                callback(epoch, {
                    'model': self.model,
                    'optimizer': self.optimizer,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'metrics': val_metrics,
                    'epoch': epoch
                })
            
            # 檢查早停條件
            if early_stopping is not None:
                if early_stopping(val_loss, self.model, save_path):
                    logger.info(f"早停觸發，在輪次 {epoch+1} 停止訓練")
                    break
            elif save_path is not None and epoch == epochs - 1:
                # 如果沒有早停但提供了保存路徑，在最後一輪保存模型
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.model.state_dict(), save_path)
                logger.info(f"模型已保存至 {save_path}")
        
        # 計算總訓練時間
        total_time = time.time() - start_time
        logger.info(f"訓練完成，總時間: {total_time:.2f}秒")
        
        return history
    
    def predict(self, data_loader):
        """
        使用模型進行預測
        
        參數:
            data_loader (DataLoader): 數據載入器
            
        返回:
            dict: 包含預測結果的字典
        """
        self.model.eval()
        all_predictions = []
        all_outputs = {}
        
        with torch.no_grad():
            for data in data_loader:
                # 處理不同形式的數據批次
                if isinstance(data, (list, tuple)):
                    if len(data) == 2:
                        # 基本形式: (inputs, targets)
                        inputs, targets = data
                        if isinstance(inputs, (list, tuple)):
                            # 多輸入形式: ([input1, input2, ...], targets)
                            inputs = [x.to(self.device) for x in inputs]
                        else:
                            # 單輸入形式: (input, targets)
                            inputs = inputs.to(self.device)
                    elif len(data) == 3:
                        # 混合PINN-LSTM形式: (static_features, time_series, targets)
                        static_features, time_series, _ = data
                        static_features = static_features.to(self.device)
                        time_series = time_series.to(self.device)
                        # 保存為元組用於前向傳播
                        inputs = (static_features, time_series)
                    else:
                        inputs = data[0].to(self.device)
                else:
                    # 自定義批次格式
                    inputs = data.to(self.device)
                
                # 前向傳播
                if isinstance(inputs, (list, tuple)) and len(inputs) == 2 and hasattr(self.model, 'forward') and 'static_features' in self.model.forward.__code__.co_varnames:
                    # 專門用於混合PINN-LSTM模型
                    outputs = self.model(inputs[0], inputs[1], return_features=True)
                    
                    # 檢查outputs是否為字典（用於混合模型）
                    if isinstance(outputs, dict) and 'nf_pred' in outputs:
                        predictions = outputs['nf_pred']
                        
                        # 收集輸出
                        for key, value in outputs.items():
                            if key not in all_outputs:
                                all_outputs[key] = []
                            
                            if isinstance(value, torch.Tensor):
                                all_outputs[key].append(value.cpu().numpy())
                            else:
                                all_outputs[key].append(value)
                    else:
                        predictions = outputs
                else:
                    # 一般模型
                    outputs = self.model(inputs)
                    
                    # 檢查outputs是否為字典
                    if isinstance(outputs, dict) and 'output' in outputs:
                        predictions = outputs['output']
                    else:
                        predictions = outputs
                
                # 收集預測結果
                if isinstance(predictions, torch.Tensor):
                    all_predictions.append(predictions.detach().cpu().numpy())
        
        # 合併預測結果
        merged_predictions = None
        if all_predictions:
            try:
                merged_predictions = np.concatenate([p.reshape(-1) for p in all_predictions])
            except Exception as e:
                logger.error(f"合併預測結果時出錯: {str(e)}")
                merged_predictions = all_predictions
        
        # 合併所有輸出
        outputs_merged = {}
        for key, values in all_outputs.items():
            if values and isinstance(values[0], np.ndarray):
                try:
                    outputs_merged[key] = np.concatenate(values)
                except Exception as e:
                    logger.warning(f"合併輸出 {key} 時出錯: {str(e)}")
                    outputs_merged[key] = values
            else:
                outputs_merged[key] = values
        
        # 添加預測結果
        if merged_predictions is not None:
            outputs_merged['predictions'] = merged_predictions
        
        return outputs_merged
    
    def save_model(self, path, epoch=None, train_loss=None, val_loss=None, metrics=None):
        """
        保存模型
        
        參數:
            path (str): 保存路徑
            epoch (int, optional): 當前輪次
            train_loss (float, optional): 訓練損失
            val_loss (float, optional): 驗證損失
            metrics (dict, optional): 指標字典
        """
        # 確保目錄存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 構建保存內容
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        
        # 添加額外信息
        if epoch is not None:
            save_dict['epoch'] = epoch
        
        if train_loss is not None:
            save_dict['train_loss'] = train_loss
        
        if val_loss is not None:
            save_dict['val_loss'] = val_loss
        
        if metrics is not None:
            save_dict['metrics'] = metrics
        
        if self.scheduler is not None and hasattr(self.scheduler, 'state_dict'):
            save_dict['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 保存模型
        torch.save(save_dict, path)
        logger.info(f"模型已保存至 {path}")
    
    def load_model(self, path, load_optimizer=True, load_scheduler=True):
        """
        載入模型
        
        參數:
            path (str): 模型路徑
            load_optimizer (bool): 是否載入優化器狀態
            load_scheduler (bool): 是否載入調度器狀態
            
        返回:
            dict: 載入的額外數據
        """
        # 檢查文件是否存在
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型檔案不存在: {path}")
        
        # 載入檢查點
        checkpoint = torch.load(path, map_location=self.device)
        
        # 載入模型權重
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            logger.error(f"載入模型權重時出錯: {str(e)}")
            raise
        
        # 載入優化器狀態（如果需要）
        if load_optimizer and 'optimizer_state_dict' in checkpoint and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                logger.warning(f"載入優化器狀態時出錯: {str(e)}")
        
        # 載入調度器狀態（如果需要）
        if load_scheduler and 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                logger.warning(f"載入調度器狀態時出錯: {str(e)}")
        
        # 提取其他數據
        extra_data = {}
        for key in ['epoch', 'train_loss', 'val_loss', 'metrics']:
            if key in checkpoint:
                extra_data[key] = checkpoint[key]
        
        logger.info(f"模型已從 {path} 載入")
        
        return extra_data
    
    def plot_training_history(self, save_path=None):
        """
        繪製訓練歷史曲線
        
        參數:
            save_path (str, optional): 圖像保存路徑
            
        返回:
            matplotlib.figure.Figure: 圖像對象
        """
        if not self.train_losses or not self.val_losses:
            logger.warning("訓練歷史記錄為空，無法繪製圖表")
            return None
        
        # 設置圖像
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 繪製損失曲線
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 繪製指標曲線（如果有）
        if self.val_metrics:
            for metric_name in ['rmse', 'r2', 'mae']:
                if metric_name in self.val_metrics and len(self.val_metrics[metric_name]) > 0:
                    ax2.plot(epochs, self.val_metrics[metric_name], label=metric_name.upper())
            
            ax2.set_title('Validation Metrics')
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Value')
            ax2.legend()
            ax2.grid(True)
        
        plt.tight_layout()
        
        # 保存圖像
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"訓練歷史圖像已保存至 {save_path}")
            except Exception as e:
                logger.error(f"保存圖像失敗: {str(e)}")
        
        return fig


def prepare_training_config(config, train_config, stage="all", start_epoch=0):
    """
    根據訓練階段準備訓練配置
    
    參數:
        config (dict): 模型配置
        train_config (dict): 訓練配置
        stage (str): 訓練階段: all, warmup, main, finetune
        start_epoch (int): 起始輪次
        
    返回:
        dict: 訓練配置
    """
    # 獲取分階段訓練配置
    training_strategy = train_config.get("training_strategy", {})
    stages = training_strategy.get("stages", [])
    
    # 如果沒有分階段配置或選擇了'all'，使用默認配置
    if not stages or stage == "all":
        return {
            "epochs": config["training"]["epochs"],
            "learning_rate": config["training"]["optimizer"]["learning_rate"],
            "lambda_physics": config["training"]["loss"].get("initial_lambda_physics", 0.1),
            "lambda_consistency": config["training"]["loss"].get("initial_lambda_consistency", 0.1),
            "batch_size": config["training"]["batch_size"],
            "clip_grad_norm": config["training"].get("clip_grad_norm", 1.0),
            "description": "完整訓練"
        }
    
    # 根據指定階段獲取配置
    for stage_config in stages:
        if stage_config["name"] == stage:
            # 計算學習率
            base_lr = config["training"]["optimizer"]["learning_rate"]
            lr_factor = stage_config.get("learning_rate_factor", 1.0)
            
            # 獲取物理約束和一致性權重
            if stage == "warmup":
                lambda_physics = stage_config.get("lambda_physics", 0.01)
                lambda_consistency = stage_config.get("lambda_consistency", 0.01)
            elif stage == "main":
                lambda_physics = stage_config.get("lambda_physics_start", 0.05)
                lambda_consistency = stage_config.get("lambda_consistency_start", 0.05)
            elif stage == "finetune":
                lambda_physics = stage_config.get("lambda_physics", 0.5)
                lambda_consistency = stage_config.get("lambda_consistency", 0.3)
            
            return {
                "epochs": stage_config["epochs"],
                "learning_rate": base_lr * lr_factor,
                "lambda_physics": lambda_physics,
                "lambda_consistency": lambda_consistency,
                "batch_size": config["training"]["batch_size"],
                "clip_grad_norm": config["training"].get("clip_grad_norm", 1.0),
                "description": stage_config.get("description", f"{stage}階段訓練")
            }
    
    # 如果找不到指定階段，使用默認配置
    logger.warning(f"找不到指定訓練階段: {stage}，使用默認配置")
    return {
        "epochs": config["training"]["epochs"],
        "learning_rate": config["training"]["optimizer"]["learning_rate"],
        "lambda_physics": config["training"]["loss"].get("initial_lambda_physics", 0.1),
        "lambda_consistency": config["training"]["loss"].get("initial_lambda_consistency", 0.1),
        "batch_size": config["training"]["batch_size"],
        "clip_grad_norm": config["training"].get("clip_grad_norm", 1.0),
        "description": "默認訓練"
    }


def create_optimizer(model, config, stage_config=None):
    """
    創建優化器
    
    參數:
        model (torch.nn.Module): 模型
        config (dict): 配置
        stage_config (dict, optional): 階段配置
        
    返回:
        torch.optim.Optimizer: 優化器
    """
    optimizer_config = config["training"]["optimizer"]
    
    # 使用階段配置中的學習率（如果有）
    if stage_config and "learning_rate" in stage_config:
        lr = stage_config["learning_rate"]
    else:
        lr = optimizer_config["learning_rate"]
    
    weight_decay = optimizer_config["weight_decay"]
    
    if optimizer_config["name"].lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_config["name"].lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_config["name"].lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        logger.warning(f"不支援的優化器類型: {optimizer_config['name']}，使用Adam")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    logger.info(f"創建{optimizer_config['name']}優化器，學習率: {lr}，權重衰減: {weight_decay}")
    return optimizer


def create_lr_scheduler(optimizer, config, stage_config=None, total_epochs=None):
    """
    創建學習率調度器
    
    參數:
        optimizer (torch.optim.Optimizer): 優化器
        config (dict): 配置
        stage_config (dict, optional): 階段配置
        total_epochs (int, optional): 總訓練輪數
        
    返回:
        LearningRateScheduler: 學習率調度器
    """
    scheduler_config = config["training"]["lr_scheduler"]
    
    if not scheduler_config:
        return None
    
    # 使用階段特定的輪數（如果有）
    if stage_config and "epochs" in stage_config:
        epochs = stage_config["epochs"]
    elif total_epochs:
        epochs = total_epochs
    else:
        epochs = config["training"]["epochs"]
    
    scheduler_type = scheduler_config["name"]
    
    if scheduler_type == "step":
        step_size = scheduler_config.get("step_size", 10)
        gamma = scheduler_config.get("gamma", 0.1)
        scheduler = LearningRateScheduler(optimizer, mode="step", step_size=step_size, gamma=gamma)
        logger.info(f"創建步進式學習率調度器，步長: {step_size}，衰減因子: {gamma}")
    elif scheduler_type == "exp":
        gamma = scheduler_config.get("gamma", 0.95)
        scheduler = LearningRateScheduler(optimizer, mode="exp", gamma=gamma)
        logger.info(f"創建指數式學習率調度器，衰減因子: {gamma}")
    elif scheduler_type == "cosine":
        T_max = scheduler_config.get("T_max", epochs)
        min_lr = scheduler_config.get("min_lr", 0)
        scheduler = LearningRateScheduler(optimizer, mode="cosine", T_max=T_max, min_lr=min_lr)
        logger.info(f"創建餘弦退火學習率調度器，週期: {T_max}，最小學習率: {min_lr}")
    elif scheduler_type == "plateau":
        factor = scheduler_config.get("factor", 0.1)
        patience = scheduler_config.get("patience", 10)
        min_lr = scheduler_config.get("min_lr", 0)
        scheduler = LearningRateScheduler(optimizer, mode="plateau", factor=factor, 
                                       patience=patience, min_lr=min_lr)
        logger.info(f"創建平原式學習率調度器，衰減因子: {factor}，耐心值: {patience}，最小學習率: {min_lr}")
    else:
        logger.warning(f"不支援的學習率調度器類型: {scheduler_type}，不使用學習率調度")
        scheduler = None
    
    return scheduler


def get_pinn_lstm_trainer(model, optimizer, config, device, 
                        lambda_physics=0.1, lambda_consistency=0.1, 
                        clip_grad_norm=1.0, scheduler=None):
    """
    創建PINN-LSTM訓練器
    
    參數:
        model (HybridPINNLSTMModel): 混合模型
        optimizer (torch.optim.Optimizer): 優化器
        config (dict): 配置
        device (torch.device): 計算設備
        lambda_physics (float): 物理約束損失權重
        lambda_consistency (float): 一致性損失權重
        clip_grad_norm (float): 梯度裁剪範數
        scheduler (LearningRateScheduler): 學習率調度器
        
    返回:
        PINNLSTMTrainer: PINN-LSTM訓練器
    """
    # 獲取損失配置
    loss_config = config["training"]["loss"]
    
    # 導入專用訓練器
    from src.models.hybrid_model import PINNLSTMTrainer
    
    # 創建專用訓練器
    trainer = PINNLSTMTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        lambda_physics_init=lambda_physics,
        lambda_physics_max=loss_config.get("max_lambda_physics", lambda_physics * 5),
        lambda_consistency_init=lambda_consistency,
        lambda_consistency_max=loss_config.get("max_lambda_consistency", lambda_consistency * 3),
        lambda_ramp_epochs=loss_config.get("epochs_to_max", 50),
        clip_grad_norm=clip_grad_norm,
        scheduler=scheduler,
        log_interval=config.get("debug", {}).get("verbose", 1)
    )
    
    logger.info(f"創建PINN-LSTM專用訓練器，"
               f"初始物理約束權重: {lambda_physics}，"
               f"初始一致性約束權重: {lambda_consistency}，"
               f"梯度裁剪範數: {clip_grad_norm}")
    
    return trainer


def train_hybrid_model_with_stages(model, dataloaders, config, train_config, device, output_dir, 
                                  use_physics=True, stages=None):
    """
    使用分階段策略訓練混合模型
    
    參數:
        model (HybridPINNLSTMModel): 混合模型
        dataloaders (dict): 資料載入器
        config (dict): 模型配置
        train_config (dict): 訓練配置
        device (torch.device): 計算設備
        output_dir (str): 輸出目錄
        use_physics (bool): 是否使用物理約束
        stages (list): 要執行的訓練階段列表
        
    返回:
        dict: 訓練歷史記錄
    """
    # 設定預設階段，如果沒有提供
    if stages is None:
        stages = ["warmup", "main", "finetune"]
    
    # 從hybrid_model模組導入PINNLSTMTrainer類
    from src.models.hybrid_model import PINNLSTMTrainer
    
    # 導入回調函數
    from src.training.callbacks import AdaptiveCallbacks
    
    all_history = {}
    
    # 定義階段訓練記錄檔案
    stage_log_file = os.path.join(output_dir, "stage_training_log.txt")
    with open(stage_log_file, "w") as f:
        f.write(f"分階段訓練日誌 - 開始時間: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型: {config['model']['name']}, 使用物理約束: {use_physics}\n")
        f.write("="*50 + "\n")
    
    current_epoch = 0
    
    # 遍歷每個訓練階段
    for stage in stages:
        logger.info("="*50)
        logger.info(f"開始 {stage} 階段訓練")
        
        # 獲取階段配置
        stage_config = prepare_training_config(config, train_config, stage=stage, start_epoch=current_epoch)
        logger.info(f"階段描述: {stage_config['description']}")
        logger.info(f"訓練輪數: {stage_config['epochs']}")
        logger.info(f"學習率: {stage_config['learning_rate']}")
        logger.info(f"物理約束權重: {stage_config['lambda_physics']}")
        logger.info(f"一致性約束權重: {stage_config['lambda_consistency']}")
        
        # 創建階段特定的輸出目錄
        stage_dir = os.path.join(output_dir, f"stage_{stage}")
        os.makedirs(stage_dir, exist_ok=True)
        
        # 創建階段優化器
        optimizer = create_optimizer(model, config, stage_config)
        
        # 創建階段學習率調度器
        scheduler = create_lr_scheduler(optimizer, config, stage_config)
        
        # 創建專用訓練器
        trainer = PINNLSTMTrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            lambda_physics_init=stage_config["lambda_physics"] if use_physics else 0.0,
            lambda_physics_max=stage_config.get("lambda_physics_max", stage_config["lambda_physics"] * 2),
            lambda_consistency_init=stage_config["lambda_consistency"],
            lambda_consistency_max=stage_config.get("lambda_consistency_max", stage_config["lambda_consistency"] * 2),
            lambda_ramp_epochs=stage_config.get("lambda_ramp_epochs", stage_config["epochs"] // 2),
            clip_grad_norm=stage_config["clip_grad_norm"],
            scheduler=scheduler
        )
        
        # 創建階段回調函數
        callbacks = AdaptiveCallbacks.create_callbacks(
            model=model,
            dataset_size=len(dataloaders["train_loader"].dataset),
            epochs=stage_config["epochs"],
            output_dir=stage_dir,
            use_tensorboard=True,
            use_progress_bar=True,
            use_early_stopping=True,
            patience=config["training"]["early_stopping"]["patience"],
            monitor="val_loss",
            mode="min",
            save_freq=5
        )
        
        # 訓練階段
        start_time = time.time()
        
        stage_history = trainer.train(
            train_loader=dataloaders["train_loader"],
            val_loader=dataloaders["val_loader"],
            epochs=stage_config["epochs"],
            early_stopping_patience=config["training"]["early_stopping"]["patience"],
            save_path=os.path.join(stage_dir, "models", "best_model.pt"),
            callbacks=callbacks
        )
        
        train_time = time.time() - start_time
        
        # 記錄階段訓練結果
        with open(stage_log_file, "a") as f:
            f.write(f"\n階段: {stage}\n")
            f.write(f"描述: {stage_config['description']}\n")
            f.write(f"訓練輪數: {stage_config['epochs']}\n")
            f.write(f"訓練時間: {train_time:.2f}秒\n")
            f.write(f"最佳驗證損失: {stage_history['best_val_loss']:.6f}\n")
            if 'val_metrics' in stage_history and 'rmse' in stage_history['val_metrics']:
                f.write(f"RMSE: {stage_history['val_metrics']['rmse'][-1]:.4f}\n")
            if 'val_metrics' in stage_history and 'r2' in stage_history['val_metrics']:
                f.write(f"R²: {stage_history['val_metrics']['r2'][-1]:.4f}\n")
            f.write("-"*40 + "\n")
        
        # 將階段歷史添加到總歷史
        all_history[stage] = stage_history
        
        # 更新當前輪次
        current_epoch += stage_config["epochs"]
        
        logger.info(f"{stage} 階段訓練完成 - 耗時: {train_time:.2f}秒, 最佳驗證損失: {stage_history['best_val_loss']:.6f}")
    
    # 最終模型保存
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    final_model_path = os.path.join(output_dir, "models", "final_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'training_history': all_history,
        'final_epoch': current_epoch
    }, final_model_path)
    
    logger.info(f"分階段訓練完成，最終模型已保存至: {final_model_path}")
    
    return all_history


if __name__ == "__main__":
    # 設定日誌
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 測試代碼
    logger.info("測試Trainer模組")
    
    # 創建簡單模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = nn.Linear(10, 1)
        
        def forward(self, x):
            return self.fc(x)
    
    # 初始化模型和訓練器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 創建訓練器
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device
    )
    
    logger.info("測試EarlyStopping")
    early_stopping = EarlyStopping(patience=5, verbose=True)
    
    logger.info("測試LearningRateScheduler")
    scheduler = LearningRateScheduler(
        optimizer=optimizer,
        mode="step",
        step_size=10,
        gamma=0.1
    )
    
    logger.info("Trainer模組測試完成")
