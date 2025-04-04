#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
trainer.py - 優化版模型訓練器模組
本模組實現了用於訓練銲錫接點疲勞壽命預測混合模型的訓練器，
提供完整的訓練循環、模型評估和早停機制等功能，並特別針對小樣本數據集優化。

主要改進:
1. 加強物理知識驅動的訓練流程
2. 改進早停機制以避免小樣本過擬合
3. 實現更靈活的學習率調度策略
4. 提供更詳細的訓練狀態監控
5. 優化模型評估流程以支援混合PINN-LSTM模型
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import math
from collections import defaultdict

# 導入自定義模組
from src.training.losses import HybridLoss, AdaptiveHybridLoss

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    增強版早停機制
    監控驗證集上的指標，當指標不再改善時提前停止訓練，
    特別針對小樣本數據集優化，避免過擬合。
    """
    def __init__(self, patience=20, min_delta=0, mode='min', verbose=True, 
                 save_path=None, metric_name='val_loss', restore_best_weights=True,
                 cooldown=0, min_epochs=0, smoothing_factor=0):
        """
        初始化早停機制
        
        參數:
            patience (int): 容忍指標不改善的輪數
            min_delta (float): 認為有改善的最小變化量
            mode (str): 監控模式，'min'表示指標越小越好，'max'表示指標越大越好
            verbose (bool): 是否輸出日誌
            save_path (str): 最佳模型保存路徑
            metric_name (str): 監控的指標名稱
            restore_best_weights (bool): 訓練結束時是否恢復最佳權重
            cooldown (int): 觸發早停後的冷卻輪數
            min_epochs (int): 訓練的最小輪數，小於此輪數不觸發早停
            smoothing_factor (float): 指標平滑因子，減少小樣本波動的影響
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.save_path = save_path
        self.metric_name = metric_name
        self.restore_best_weights = restore_best_weights
        self.cooldown = cooldown
        self.min_epochs = min_epochs
        self.smoothing_factor = max(0, min(smoothing_factor, 0.9))
        
        self.best_score = None
        self.best_weights = None
        self.counter = 0
        self.cooldown_counter = 0
        self.early_stop = False
        self.current_epoch = 0
        self.metric_history = []
        
        # 初始化最佳分數比較函數
        if mode == 'min':
            self._is_better = lambda score, best: score <= best - min_delta
        elif mode == 'max':
            self._is_better = lambda score, best: score >= best + min_delta
        else:
            raise ValueError(f"不支援的模式: {mode}，應為 'min' 或 'max'")
        
        logger.info(f"初始化EarlyStopping: "
                   f"patience={patience}, mode={mode}, metric={metric_name}, "
                   f"min_epochs={min_epochs}, smoothing_factor={smoothing_factor}")
    
    def __call__(self, score, model=None, epoch=None):
        """
        檢查是否應該早停
        
        參數:
            score (float): 當前指標值
            model (nn.Module, optional): 當前模型
            epoch (int, optional): 當前輪次
            
        返回:
            bool: 是否應該早停
        """
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        # 記錄指標歷史
        self.metric_history.append(score)
        
        # 如果使用平滑，則計算平滑後的分數
        if self.smoothing_factor > 0 and len(self.metric_history) > 1:
            smoothed_score = (1 - self.smoothing_factor) * score + self.smoothing_factor * self.metric_history[-2]
        else:
            smoothed_score = score
        
        # 檢查是否處於最小輪數或冷卻期
        if self.current_epoch < self.min_epochs:
            if self.verbose:
                logger.debug(f"EarlyStopping: 當前輪次 {self.current_epoch} < 最小輪次 {self.min_epochs}，忽略早停檢查")
            self._save_checkpoint(score, model)
            return False
        
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            if self.verbose:
                logger.debug(f"EarlyStopping: 處於冷卻期，剩餘 {self.cooldown_counter} 輪")
            return False
        
        if self.best_score is None:
            # 首次迭代，初始化最佳分數
            self.best_score = smoothed_score
            self._save_checkpoint(score, model)
        elif self._is_better(smoothed_score, self.best_score):
            # 指標有改善
            self.best_score = smoothed_score
            self.counter = 0
            self._save_checkpoint(score, model)
        else:
            # 指標沒有改善
            self.counter += 1
            if self.verbose:
                logger.debug(f"EarlyStopping: 指標未改善，計數器 {self.counter}/{self.patience}，"
                            f"當前: {score:.6f}, 最佳: {self.best_score:.6f}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info(f"EarlyStopping: 指標連續 {self.patience} 輪未改善，觸發早停")
                
                # 重置冷卻計數器
                self.cooldown_counter = self.cooldown
                
                # 如果需要，恢復最佳權重
                if self.restore_best_weights and self.best_weights is not None and model is not None:
                    if self.verbose:
                        logger.info("EarlyStopping: 恢復最佳權重")
                    model.load_state_dict(self.best_weights)
        
        return self.early_stop
    
    def _save_checkpoint(self, score, model):
        """保存模型檢查點"""
        if model is not None:
            # 保存模型權重
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            # 如果提供了保存路徑，則保存模型
            if self.save_path is not None:
                # 確保保存目錄存在
                Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
                
                if self.verbose:
                    if self.mode == 'min':
                        logger.info(f"指標改善: {self.best_score:.6f} -> {score:.6f}，保存模型至 {self.save_path}")
                    else:
                        logger.info(f"指標改善: {self.best_score:.6f} -> {score:.6f}，保存模型至 {self.save_path}")
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'score': score,
                    'epoch': self.current_epoch
                }, self.save_path)
    
    def reset(self):
        """重置早停機制"""
        self.best_score = None
        self.best_weights = None
        self.counter = 0
        self.cooldown_counter = 0
        self.early_stop = False
        self.current_epoch = 0
        self.metric_history = []


class LearningRateScheduler:
    """
    增強版學習率調度器
    提供多種學習率調度策略，支援動態調整、斷點續訓練，
    並專門為小樣本數據集優化學習率變化曲線。
    """
    def __init__(self, optimizer, mode='step', warm_up_epochs=0, **kwargs):
        """
        初始化學習率調度器
        
        參數:
            optimizer (torch.optim.Optimizer): 優化器
            mode (str): 調度模式
                - 'step': 階梯式下降，每隔step_size輪降低gamma倍
                - 'exp': 指數下降，每輪降低gamma倍
                - 'cosine': 餘弦退火，在T_max輪內從初始學習率降至min_lr
                - 'plateau': 當指標不再改善時降低學習率
                - 'one_cycle': One Cycle學習率策略
                - 'cyclic': 循環學習率
            warm_up_epochs (int): 預熱輪數，在這些輪次中逐漸增加學習率
            **kwargs: 根據不同模式傳遞的參數
        """
        self.optimizer = optimizer
        self.mode = mode
        self.warm_up_epochs = warm_up_epochs
        self.kwargs = kwargs
        
        # 獲取初始學習率
        self.initial_lr = []
        for param_group in optimizer.param_groups:
            self.initial_lr.append(param_group['lr'])
        
        # 當前輪次
        self.current_epoch = 0
        
        # 根據模式創建調度器
        if mode == 'step':
            step_size = kwargs.get('step_size', 10)
            gamma = kwargs.get('gamma', 0.1)
            self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            logger.info(f"創建階梯式學習率調度器: step_size={step_size}, gamma={gamma}")
            
        elif mode == 'exp':
            gamma = kwargs.get('gamma', 0.95)
            self.scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
            logger.info(f"創建指數式學習率調度器: gamma={gamma}")
            
        elif mode == 'cosine':
            T_max = kwargs.get('T_max', 100)
            eta_min = kwargs.get('min_lr', 0)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
            logger.info(f"創建餘弦退火學習率調度器: T_max={T_max}, min_lr={eta_min}")
            
        elif mode == 'plateau':
            factor = kwargs.get('factor', 0.1)
            patience = kwargs.get('patience', 10)
            min_lr = kwargs.get('min_lr', 0)
            cooldown = kwargs.get('cooldown', 0)
            threshold = kwargs.get('threshold', 1e-4)
            
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=factor, patience=patience,
                min_lr=min_lr, cooldown=cooldown, threshold=threshold
            )
            
            logger.info(f"創建平原式學習率調度器: factor={factor}, patience={patience}, "
                       f"min_lr={min_lr}, cooldown={cooldown}, threshold={threshold}")
            
        elif mode == 'one_cycle':
            max_lr = kwargs.get('max_lr', [lr * 10 for lr in self.initial_lr])
            total_steps = kwargs.get('total_steps', 100)
            pct_start = kwargs.get('pct_start', 0.3)
            
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=max_lr, total_steps=total_steps,
                pct_start=pct_start, div_factor=25.0, final_div_factor=10000.0
            )
            
            logger.info(f"創建One Cycle學習率調度器: max_lr={max_lr}, "
                       f"total_steps={total_steps}, pct_start={pct_start}")
            
        elif mode == 'cyclic':
            base_lr = kwargs.get('base_lr', [lr / 10 for lr in self.initial_lr])
            max_lr = kwargs.get('max_lr', [lr * 10 for lr in self.initial_lr])
            step_size_up = kwargs.get('step_size_up', 20)
            
            self.scheduler = optim.lr_scheduler.CyclicLR(
                optimizer, base_lr=base_lr, max_lr=max_lr,
                step_size_up=step_size_up, mode='triangular2'
            )
            
            logger.info(f"創建循環學習率調度器: base_lr={base_lr}, max_lr={max_lr}, "
                       f"step_size_up={step_size_up}")
            
        else:
            raise ValueError(f"不支援的學習率調度模式: {mode}")
        
        # 預熱階段的學習率增長率
        if warm_up_epochs > 0:
            self.warm_up_factor = []
            for i, lr in enumerate(self.initial_lr):
                self.warm_up_factor.append(lr / warm_up_epochs)
            logger.info(f"啟用學習率預熱: warm_up_epochs={warm_up_epochs}")
    
    def step(self, val_loss=None, epoch=None):
        """
        更新學習率
        
        參數:
            val_loss (float, optional): 驗證損失，用於'plateau'模式
            epoch (int, optional): 當前輪次，用於記錄
        """
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        # 處理預熱階段
        if self.warm_up_epochs > 0 and self.current_epoch < self.warm_up_epochs:
            # 在預熱階段逐漸增加學習率
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.warm_up_factor[i] * (self.current_epoch + 1)
            
            if self.current_epoch + 1 == self.warm_up_epochs:
                # 預熱結束，恢復初始學習率
                for i, param_group in enumerate(self.optimizer.param_groups):
                    param_group['lr'] = self.initial_lr[i]
                
                logger.info(f"學習率預熱完成，恢復初始學習率: {self.initial_lr}")
            
            return
        
        # 預熱後的常規調度
        if self.mode == 'plateau' and val_loss is not None:
            self.scheduler.step(val_loss)
        elif self.mode == 'one_cycle' or self.mode == 'cyclic':
            self.scheduler.step()
        else:
            self.scheduler.step()
    
    def get_last_lr(self):
        """獲取當前學習率"""
        if self.warm_up_epochs > 0 and self.current_epoch < self.warm_up_epochs:
            return [self.warm_up_factor[i] * (self.current_epoch + 1) for i in range(len(self.initial_lr))]
        elif self.mode == 'plateau':
            return [param_group['lr'] for param_group in self.optimizer.param_groups]
        else:
            return self.scheduler.get_last_lr()
    
    def state_dict(self):
        """獲取調度器狀態"""
        state = {
            'mode': self.mode,
            'warm_up_epochs': self.warm_up_epochs,
            'current_epoch': self.current_epoch,
            'initial_lr': self.initial_lr,
            'kwargs': self.kwargs
        }
        
        # 添加具體調度器的狀態
        if hasattr(self.scheduler, 'state_dict'):
            state['scheduler_state'] = self.scheduler.state_dict()
        
        return state
    
    def load_state_dict(self, state_dict):
        """載入調度器狀態"""
        self.mode = state_dict['mode']
        self.warm_up_epochs = state_dict['warm_up_epochs']
        self.current_epoch = state_dict['current_epoch']
        self.initial_lr = state_dict['initial_lr']
        self.kwargs = state_dict['kwargs']
        
        # 載入具體調度器的狀態
        if 'scheduler_state' in state_dict and hasattr(self.scheduler, 'load_state_dict'):
            self.scheduler.load_state_dict(state_dict['scheduler_state'])


class PhysicsGuidedTrainer:
    """
    物理知識驅動的訓練器
    專為混合PINN-LSTM模型設計，整合物理約束和時序特徵，
    提供高級訓練策略以處理小樣本數據集。
    """
    def __init__(self, model, optimizer, loss_fn, device=None, 
                 lr_scheduler=None, clip_grad_norm=1.0,
                 lambda_physics=0.1, lambda_consistency=0.1,
                 grad_accumulation_steps=1, use_amp=False):
        """
        初始化物理知識驅動訓練器
        
        參數:
            model (nn.Module): 待訓練的模型
            optimizer (torch.optim.Optimizer): 優化器
            loss_fn (nn.Module): 損失函數
            device (torch.device): 計算設備
            lr_scheduler (LearningRateScheduler): 學習率調度器
            clip_grad_norm (float): 梯度裁剪範數
            lambda_physics (float): 物理約束損失權重
            lambda_consistency (float): 一致性損失權重
            grad_accumulation_steps (int): 梯度累積步數，用於模擬更大批次
            use_amp (bool): 是否使用自動混合精度(AMP)訓練
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr_scheduler = lr_scheduler
        self.clip_grad_norm = clip_grad_norm
        self.lambda_physics = lambda_physics
        self.lambda_consistency = lambda_consistency
        self.grad_accumulation_steps = max(1, grad_accumulation_steps)
        self.use_amp = use_amp and torch.cuda.is_available()  # 只在CUDA可用時啟用AMP
        
        # 將模型移至指定設備
        self.model.to(self.device)
        
        # 訓練記錄
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        self.lr_history = []
        self.batch_loss_history = []
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_metric = float('-inf')
        self.best_model_state = None
        
        # 初始化混合精度訓練
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("已啟用自動混合精度(AMP)訓練")
        
        logger.info(f"初始化PhysicsGuidedTrainer: "
                   f"model={model.__class__.__name__}, device={self.device}, "
                   f"lambda_physics={lambda_physics}, lambda_consistency={lambda_consistency}, "
                   f"grad_accumulation_steps={grad_accumulation_steps}, use_amp={self.use_amp}")
    
    def train(self, train_loader, val_loader, epochs=100, early_stopping=None, 
              verbose=True, eval_interval=1, save_path=None, callbacks=None,
              val_metric='val_loss', val_mode='min', track_grad_norm=False,
              track_weights_norm=False, track_gradients=False):
        """
        訓練模型
        
        參數:
            train_loader (DataLoader): 訓練數據加載器
            val_loader (DataLoader): 驗證數據加載器
            epochs (int): 訓練輪數
            early_stopping (EarlyStopping): 早停機制
            verbose (bool): 是否輸出詳細日誌
            eval_interval (int): 評估間隔輪數
            save_path (str): 最佳模型保存路徑
            callbacks (list): 回調函數列表
            val_metric (str): 驗證指標名稱
            val_mode (str): 驗證模式，'min'或'max'
            track_grad_norm (bool): 是否追蹤梯度範數
            track_weights_norm (bool): 是否追蹤權重範數
            track_gradients (bool): 是否追蹤每層梯度
            
        返回:
            dict: 訓練歷史記錄
        """
        if callbacks is None:
            callbacks = []
        
        start_time = time.time()
        logger.info(f"開始訓練: epochs={epochs}, 訓練批次數={len(train_loader)}, "
                  f"驗證批次數={len(val_loader)}")
        
        # 初始化追蹤記錄
        if track_grad_norm:
            self.grad_norm_history = []
        if track_weights_norm:
            self.weights_norm_history = []
        if track_gradients:
            self.layer_gradients = defaultdict(list)
        
        # 主要訓練循環
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # 執行一輪訓練
            train_results = self._train_epoch(
                train_loader, 
                track_grad_norm=track_grad_norm,
                track_gradients=track_gradients
            )
            
            train_loss = train_results['loss']
            self.train_losses.append(train_loss)
            
            # 記錄訓練批次損失
            self.batch_loss_history.extend(train_results.get('batch_losses', []))
            
            # 記錄學習率
            if self.lr_scheduler:
                self.lr_history.append(self.lr_scheduler.get_last_lr())
            
            # 記錄權重範數
            if track_weights_norm:
                weights_norm = self._calculate_weights_norm()
                self.weights_norm_history.append(weights_norm)
            
            # 定期評估
            if epoch % eval_interval == 0 or epoch == epochs - 1:
                val_results = self.evaluate(val_loader)
                val_loss = val_results['loss']
                self.val_losses.append(val_loss)
                
                # 更新指標記錄
                for metric_name, metric_value in val_results['metrics'].items():
                    self.val_metrics[metric_name].append(metric_value)
                
                # 監控的驗證指標
                if val_metric == 'val_loss':
                    val_score = val_loss
                    is_better = val_score < self.best_val_loss if val_mode == 'min' else val_score > self.best_val_loss
                else:
                    val_score = val_results['metrics'].get(val_metric, val_loss)
                    is_better = val_score < self.best_val_metric if val_mode == 'min' else val_score > self.best_val_metric
                
                # 輸出訓練狀態
                if verbose:
                    # 基本日誌
                    log_msg = f"輪次 {epoch+1}/{epochs} - "
                    log_msg += f"訓練損失: {train_loss:.6f}, 驗證損失: {val_loss:.6f}"
                    
                    # 添加關鍵指標
                    if 'rmse' in val_results['metrics']:
                        log_msg += f", RMSE: {val_results['metrics']['rmse']:.4f}"
                    if 'r2' in val_results['metrics']:
                        log_msg += f", R²: {val_results['metrics']['r2']:.4f}"
                    
                    # 添加學習率
                    if self.lr_scheduler:
                        lrs = self.lr_scheduler.get_last_lr()
                        if len(lrs) == 1:
                            log_msg += f", 學習率: {lrs[0]:.2e}"
                        else:
                            log_msg += f", 學習率: {lrs[0]:.2e}/{lrs[-1]:.2e}"
                    
                    logger.info(log_msg)
                
                # 檢查是否是最佳模型
                if is_better:
                    if val_metric == 'val_loss':
                        self.best_val_loss = val_score
                    else:
                        self.best_val_metric = val_score
                    
                    self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    
                    if save_path:
                        self._save_model(save_path, val_results, epoch)
                
                # 早停檢查
                if early_stopping and early_stopping(val_score, self.model, epoch):
                    logger.info(f"早停觸發，在輪次 {epoch+1} 停止訓練")
                    break
                
                # 更新學習率調度器
                if self.lr_scheduler:
                    if hasattr(self.lr_scheduler, 'step_with_metrics'):
                        self.lr_scheduler.step(val_score, epoch)
                    else:
                        self.lr_scheduler.step(val_loss, epoch)
            else:
                # 如果不評估，仍然需要更新學習率調度器
                if self.lr_scheduler:
                    self.lr_scheduler.step(epoch=epoch)
            
            # 執行回調函數
            for callback in callbacks:
                callback(epoch, {
                    'model': self.model,
                    'optimizer': self.optimizer,
                    'train_loss': train_loss,
                    'val_loss': val_loss if epoch % eval_interval == 0 else None,
                    'metrics': val_results['metrics'] if epoch % eval_interval == 0 else None,
                    'lr_scheduler': self.lr_scheduler
                })
        
        total_time = time.time() - start_time
        logger.info(f"訓練完成: 總時間: {total_time:.2f}秒, 最佳驗證損失: {self.best_val_loss:.6f}")
        
        # 恢復最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': dict(self.train_metrics),
            'val_metrics': dict(self.val_metrics),
            'lr_history': self.lr_history,
            'best_val_loss': self.best_val_loss,
            'training_time': total_time
        }
    
    def _train_epoch(self, train_loader, track_grad_norm=False, track_gradients=False):
        """
        執行一輪訓練
        
        參數:
            train_loader (DataLoader): 訓練數據加載器
            track_grad_norm (bool): 是否追蹤梯度範數
            track_gradients (bool): 是否追蹤每層梯度
            
        返回:
            dict: 包含訓練損失和指標的字典
        """
        self.model.train()
        epoch_loss = 0.0
        batch_losses = []
        num_batches = len(train_loader)
        
        # 梯度範數記錄
        if track_grad_norm:
            epoch_grad_norm = 0.0
        
        # 啟用梯度累積
        self.optimizer.zero_grad()
        
        for batch_idx, batch_data in enumerate(train_loader):
            # 解析批次數據
            if len(batch_data) == 3:
                static_features, time_series, targets = batch_data
                static_features = static_features.to(self.device)
                time_series = time_series.to(self.device)
                targets = targets.to(self.device)
                
                # 使用自動混合精度
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        # 前向傳播
                        outputs = self.model(static_features, time_series)
                        
                        # 計算損失
                        if isinstance(self.loss_fn, (HybridLoss, AdaptiveHybridLoss)):
                            losses = self.loss_fn(outputs, targets, self.model)
                            loss = losses['total_loss'] / self.grad_accumulation_steps
                        else:
                            # 簡單損失函數
                            if isinstance(outputs, dict) and 'nf_pred' in outputs:
                                loss = self.loss_fn(outputs['nf_pred'], targets) / self.grad_accumulation_steps
                            else:
                                loss = self.loss_fn(outputs, targets) / self.grad_accumulation_steps
                    
                    # 反向傳播
                    self.scaler.scale(loss).backward()
                else:
                    # 前向傳播
                    outputs = self.model(static_features, time_series)
                    
                    # 計算損失
                    if isinstance(self.loss_fn, (HybridLoss, AdaptiveHybridLoss)):
                        losses = self.loss_fn(outputs, targets, self.model)
                        loss = losses['total_loss'] / self.grad_accumulation_steps
                    else:
                        # 簡單損失函數
                        if isinstance(outputs, dict) and 'nf_pred' in outputs:
                            loss = self.loss_fn(outputs['nf_pred'], targets) / self.grad_accumulation_steps
                        else:
                            loss = self.loss_fn(outputs, targets) / self.grad_accumulation_steps
                    
                    # 反向傳播
                    loss.backward()
            else:
                # 單分支模型或其他類型模型
                inputs, targets = batch_data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        # 前向傳播
                        outputs = self.model(inputs)
                        
                        # 計算損失
                        if isinstance(self.loss_fn, (HybridLoss, AdaptiveHybridLoss)):
                            losses = self.loss_fn(outputs, targets, self.model)
                            loss = losses['total_loss'] / self.grad_accumulation_steps
                        else:
                            loss = self.loss_fn(outputs, targets) / self.grad_accumulation_steps
                    
                    # 反向傳播
                    self.scaler.scale(loss).backward()
                else:
                    # 前向傳播
                    outputs = self.model(inputs)
                    
                    # 計算損失
                    if isinstance(self.loss_fn, (HybridLoss, AdaptiveHybridLoss)):
                        losses = self.loss_fn(outputs, targets, self.model)
                        loss = losses['total_loss'] / self.grad_accumulation_steps
                    else:
                        loss = self.loss_fn(outputs, targets) / self.grad_accumulation_steps
                    
                    # 反向傳播
                    loss.backward()
            
            # 追蹤每層梯度
            if track_gradients and (batch_idx + 1) % self.grad_accumulation_steps == 0:
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        if name not in self.layer_gradients:
                            self.layer_gradients[name] = []
                        self.layer_gradients[name].append(param.grad.abs().mean().item())
            
            # 累計損失
            epoch_loss += loss.item() * self.grad_accumulation_steps
            batch_losses.append(loss.item() * self.grad_accumulation_steps)
            
            # 梯度累積 - 當達到累積步數或最後一個批次時更新
            if (batch_idx + 1) % self.grad_accumulation_steps == 0 or batch_idx == len(train_loader) - 1:
                # 梯度裁剪
                if self.clip_grad_norm > 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        
                    # 計算梯度範數（如果需要追蹤）
                    if track_grad_norm:
                        # 完整梯度範數計算
                        total_norm = 0.0
                        for param in self.model.parameters():
                            if param.grad is not None:
                                param_norm = param.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** 0.5
                        epoch_grad_norm += total_norm
                    
                    # 執行梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                
                # 更新參數
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                # 重置梯度
                self.optimizer.zero_grad()
        
        # 計算平均損失
        avg_epoch_loss = epoch_loss / num_batches
        
        # 構建返回結果
        results = {
            'loss': avg_epoch_loss,
            'batch_losses': batch_losses
        }
        
        # 添加梯度範數
        if track_grad_norm:
            avg_grad_norm = epoch_grad_norm / (num_batches // self.grad_accumulation_steps + 1)
            results['grad_norm'] = avg_grad_norm
            self.grad_norm_history.append(avg_grad_norm)
        
        return results
    
    def evaluate(self, data_loader, return_predictions=False, calculate_physics=True):
        """
        評估模型
        
        參數:
            data_loader (DataLoader): 數據加載器
            return_predictions (bool): 是否返回預測結果
            calculate_physics (bool): 是否計算物理指標
            
        返回:
            dict: 包含評估損失、指標和預測結果的字典
        """
        self.model.eval()
        total_loss = 0.0
        all_outputs = {}
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_data in data_loader:
                # 解析批次數據
                if len(batch_data) == 3:
                    static_features, time_series, targets = batch_data
                    static_features = static_features.to(self.device)
                    time_series = time_series.to(self.device)
                    targets = targets.to(self.device)
                    
                    # 前向傳播
                    outputs = self.model(static_features, time_series)
                else:
                    # 單分支模型或其他類型模型
                    inputs, targets = batch_data
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    # 前向傳播
                    outputs = self.model(inputs)
                
                # 計算損失
                if isinstance(self.loss_fn, (HybridLoss, AdaptiveHybridLoss)):
                    losses = self.loss_fn(outputs, targets, self.model)
                    loss = losses['total_loss']
                else:
                    # 簡單損失函數
                    if isinstance(outputs, dict) and 'nf_pred' in outputs:
                        loss = self.loss_fn(outputs['nf_pred'], targets)
                        predictions = outputs['nf_pred']
                    else:
                        loss = self.loss_fn(outputs, targets)
                        predictions = outputs
                
                # 獲取預測值
                if isinstance(outputs, dict) and 'nf_pred' in outputs:
                    predictions = outputs['nf_pred']
                else:
                    predictions = outputs
                
                # 累計損失
                total_loss += loss.item()
                
                # 收集預測結果
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                
                # 收集其他輸出
                if isinstance(outputs, dict):
                    for key, value in outputs.items():
                        if key not in all_outputs:
                            all_outputs[key] = []
                        
                        if isinstance(value, torch.Tensor):
                            all_outputs[key].append(value.cpu().numpy())
                        else:
                            all_outputs[key].append(value)
        
        # 計算平均損失
        avg_loss = total_loss / len(data_loader)
        
        # 合併批次結果
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        
        # 合併模型輸出
        for key in all_outputs:
            if isinstance(all_outputs[key][0], np.ndarray):
                all_outputs[key] = np.concatenate(all_outputs[key])
        
        # 計算評估指標
        metrics = self._calculate_metrics(all_predictions, all_targets, all_outputs, calculate_physics)
        
        # 構建返回結果
        result = {
            'loss': avg_loss,
            'metrics': metrics
        }
        
        # 如果需要，添加預測結果
        if return_predictions:
            result['predictions'] = all_predictions
            result['targets'] = all_targets
            result['outputs'] = all_outputs
        
        return result
    
    def predict(self, data_loader, return_features=True, enforce_physics=False):
        """
        使用模型進行預測
        
        參數:
            data_loader (DataLoader): 數據加載器
            return_features (bool): 是否返回特徵
            enforce_physics (bool): 是否強制物理約束
            
        返回:
            dict: 包含預測結果的字典
        """
        self.model.eval()
        all_outputs = {}
        all_static_features = []
        all_time_series = []
        
        with torch.no_grad():
            for batch_data in data_loader:
                # 解析批次數據
                if len(batch_data) >= 2:
                    if len(batch_data) == 3:
                        static_features, time_series, targets = batch_data
                        static_features = static_features.to(self.device)
                        time_series = time_series.to(self.device)
                        
                        # 保存特徵
                        if return_features:
                            all_static_features.append(static_features.cpu().numpy())
                            all_time_series.append(time_series.cpu().numpy())
                        
                        # 前向傳播
                        outputs = self.model(static_features, time_series)
                    else:
                        inputs, targets = batch_data
                        inputs = inputs.to(self.device)
                        
                        # 保存特徵
                        if return_features:
                            if isinstance(inputs, list) and len(inputs) == 2:
                                all_static_features.append(inputs[0].cpu().numpy())
                                all_time_series.append(inputs[1].cpu().numpy())
                            else:
                                all_static_features.append(inputs.cpu().numpy())
                        
                        # 前向傳播
                        outputs = self.model(inputs)
                else:
                    # 單一輸入
                    inputs = batch_data[0]
                    inputs = inputs.to(self.device)
                    
                    # 保存特徵
                    if return_features:
                        all_static_features.append(inputs.cpu().numpy())
                    
                    # 前向傳播
                    outputs = self.model(inputs)
                
                # 強制物理約束（如果需要）
                if enforce_physics and 'delta_w' in outputs and hasattr(self.model, 'a_coefficient') and hasattr(self.model, 'b_coefficient'):
                    a = self.model.a_coefficient
                    b = self.model.b_coefficient
                    delta_w = outputs['delta_w']
                    physics_nf = a * torch.pow(delta_w, b)
                    outputs['nf_pred'] = physics_nf
                
                # 收集輸出
                if isinstance(outputs, dict):
                    for key, value in outputs.items():
                        if key not in all_outputs:
                            all_outputs[key] = []
                        
                        if isinstance(value, torch.Tensor):
                            all_outputs[key].append(value.cpu().numpy())
                        else:
                            all_outputs[key].append(value)
                else:
                    if 'output' not in all_outputs:
                        all_outputs['output'] = []
                    all_outputs['output'].append(outputs.cpu().numpy())
        
        # 合併批次結果
        for key in all_outputs:
            if isinstance(all_outputs[key][0], np.ndarray):
                all_outputs[key] = np.concatenate(all_outputs[key])
        
        # 添加特徵
        if return_features and all_static_features:
            all_outputs['static_features'] = np.concatenate(all_static_features)
        if return_features and all_time_series:
            all_outputs['time_series'] = np.concatenate(all_time_series)
        
        return all_outputs
    
    def _calculate_metrics(self, predictions, targets, outputs=None, calculate_physics=True):
        """
        計算評估指標
        
        參數:
            predictions (np.ndarray): 預測值
            targets (np.ndarray): 真實值
            outputs (dict, optional): 模型輸出
            calculate_physics (bool): 是否計算物理指標
            
        返回:
            dict: 包含評估指標的字典
        """
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        metrics = {}
        
        # 基本指標
        # 均方根誤差
        metrics['rmse'] = np.sqrt(mean_squared_error(targets, predictions))
        
        # 決定係數
        metrics['r2'] = r2_score(targets, predictions)
        
        # 平均絕對誤差
        metrics['mae'] = mean_absolute_error(targets, predictions)
        
        # 平均絕對百分比誤差
        metrics['mape'] = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
        
        # 中位絕對百分比誤差
        metrics['mdape'] = np.median(np.abs((targets - predictions) / (targets + 1e-8))) * 100
        
        # 標準化均方根誤差
        metrics['nrmse'] = metrics['rmse'] / (np.max(targets) - np.min(targets))
        
        # 對數空間指標
        log_targets = np.log(np.maximum(targets, 1e-8))
        log_predictions = np.log(np.maximum(predictions, 1e-8))
        metrics['log_rmse'] = np.sqrt(mean_squared_error(log_targets, log_predictions))
        metrics['log_r2'] = r2_score(log_targets, log_predictions)
        
        # 物理約束指標
        if calculate_physics and outputs is not None and 'delta_w' in outputs:
            try:
                delta_w = outputs['delta_w']
                
                # 檢查模型是否包含物理係數
                if hasattr(self.model, 'a_coefficient') and hasattr(self.model, 'b_coefficient'):
                    a = self.model.a_coefficient
                    b = self.model.b_coefficient
                elif hasattr(self.loss_fn, 'a') and hasattr(self.loss_fn, 'b'):
                    a = self.loss_fn.a
                    b = self.loss_fn.b
                else:
                    a = 55.83  # 默認值
                    b = -2.259  # 默認值
                
                # 計算物理約束誤差
                theo_nf = a * np.power(delta_w, b)
                physics_error = np.abs((predictions - theo_nf) / (theo_nf + 1e-8)) * 100
                metrics['physics_error_mean'] = np.mean(physics_error)
                metrics['physics_error_median'] = np.median(physics_error)
                metrics['physics_consistency'] = r2_score(predictions, theo_nf)
            except Exception as e:
                logger.warning(f"計算物理約束指標時出錯: {str(e)}")
        
        return metrics
    
    def _calculate_weights_norm(self):
        """
        計算模型權重的範數
        
        返回:
            dict: 包含每層權重範數的字典
        """
        weights_norm = {}
        
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                layer_name = name.split('.weight')[0]
                weights_norm[layer_name] = param.data.norm().item()
        
        # 計算總範數
        total_norm = 0.0
        for norm in weights_norm.values():
            total_norm += norm ** 2
        weights_norm['total'] = total_norm ** 0.5
        
        return weights_norm
    
    def _save_model(self, path, val_results, epoch):
        """
        保存模型
        
        參數:
            path (str): 保存路徑
            val_results (dict): 驗證結果
            epoch (int): 當前輪次
        """
        # 確保目錄存在
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # 構建保存數據
        save_data = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'val_loss': val_results['loss'],
            'val_metrics': val_results['metrics'],
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': dict(self.train_metrics),
            'val_metrics_history': dict(self.val_metrics)
        }
        
        # 添加學習率調度器（如果有）
        if self.lr_scheduler and hasattr(self.lr_scheduler, 'state_dict'):
            save_data['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()
        
        # 保存模型
        torch.save(save_data, path)
        logger.info(f"模型已保存至 {path}")
    
    def load_model(self, path, load_optimizer=True, load_scheduler=True, map_location=None):
        """
        載入模型
        
        參數:
            path (str): 模型路徑
            load_optimizer (bool): 是否載入優化器狀態
            load_scheduler (bool): 是否載入學習率調度器狀態
            map_location (str or torch.device): 將張量映射到哪個設備
            
        返回:
            dict: 載入的數據
        """
        if map_location is None:
            map_location = self.device
        
        # 載入數據
        checkpoint = torch.load(path, map_location=map_location)
        
        # 載入模型權重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 載入優化器狀態
        if load_optimizer and 'optimizer_state_dict' in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 載入學習率調度器狀態
        if load_scheduler and 'lr_scheduler_state_dict' in checkpoint and self.lr_scheduler and hasattr(self.lr_scheduler, 'load_state_dict'):
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        # 恢復訓練記錄
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        if 'train_metrics' in checkpoint:
            for k, v in checkpoint['train_metrics'].items():
                self.train_metrics[k] = v
        if 'val_metrics_history' in checkpoint:
            for k, v in checkpoint['val_metrics_history'].items():
                self.val_metrics[k] = v
        if 'epoch' in checkpoint:
            self.current_epoch = checkpoint['epoch'] + 1
        if 'val_loss' in checkpoint:
            self.best_val_loss = checkpoint['val_loss']
        
        logger.info(f"模型已從 {path} 載入，當前輪次: {self.current_epoch}")
        
        return checkpoint
    
    def plot_training_history(self, figsize=(15, 10), save_path=None):
        """
        繪製訓練歷史
        
        參數:
            figsize (tuple): 圖像尺寸
            save_path (str): 保存路徑
            
        返回:
            matplotlib.figure.Figure: 圖像物件
        """
        import matplotlib.pyplot as plt
        
        num_plots = 1 + min(4, len(self.val_metrics))  # 損失圖 + 最多4個指標圖
        
        fig, axes = plt.subplots(num_plots, 1, figsize=figsize)
        if num_plots == 1:
            axes = [axes]
        
        # 繪製損失曲線
        axes[0].plot(self.train_losses, 'b-', label='Training Loss')
        
        if self.val_losses:
            # 調整驗證損失的x軸
            if len(self.val_losses) < len(self.train_losses):
                val_x = np.linspace(0, len(self.train_losses) - 1, len(self.val_losses))
                axes[0].plot(val_x, self.val_losses, 'r-', label='Validation Loss')
            else:
                axes[0].plot(self.val_losses, 'r-', label='Validation Loss')
        
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # 繪製其他指標
        for i, (metric_name, metric_values) in enumerate(self.val_metrics.items()):
            if i >= num_plots - 1:
                break
            
            ax_idx = i + 1
            axes[ax_idx].plot(metric_values, 'g-', label=metric_name)
            axes[ax_idx].set_title(f'{metric_name} vs. Epoch')
            axes[ax_idx].set_xlabel('Epoch')
            axes[ax_idx].set_ylabel(metric_name)
            axes[ax_idx].grid(True)
            
            # 標記最佳值
            if metric_name in ['rmse', 'mae', 'mape', 'log_rmse']:
                best_idx = np.argmin(metric_values)
                best_value = metric_values[best_idx]
                best_label = 'Minimum'
            else:  # 假設其他指標是越大越好
                best_idx = np.argmax(metric_values)
                best_value = metric_values[best_idx]
                best_label = 'Maximum'
            
            axes[ax_idx].scatter(best_idx, best_value, c='r', s=100, zorder=5)
            axes[ax_idx].annotate(
                f'{best_label}: {best_value:.4f} (Epoch {best_idx})',
                xy=(best_idx, best_value),
                xytext=(best_idx + 1, best_value * 1.1 if best_value > 0 else best_value * 0.9),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=9
            )
        
        plt.tight_layout()
        
        # 保存圖像
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"訓練歷史圖已保存至 {save_path}")
            except Exception as e:
                logger.warning(f"保存訓練歷史圖失敗: {str(e)}")
        
        return fig
    
    def plot_predictions(self, predictions, targets, figsize=(10, 8), save_path=None, log_scale=False):
        """
        繪製預測結果
        
        參數:
            predictions (np.ndarray): 預測值
            targets (np.ndarray): 真實值
            figsize (tuple): 圖像尺寸
            save_path (str): 保存路徑
            log_scale (bool): 是否使用對數刻度
            
        返回:
            matplotlib.figure.Figure: 圖像物件
        """
        import matplotlib.pyplot as plt
        from sklearn.metrics import r2_score
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 設置對數刻度（如果需要）
        if log_scale:
            ax.set_xscale('log')
            ax.set_yscale('log')
        
        # 繪製散點圖
        scatter = ax.scatter(targets, predictions, alpha=0.6, edgecolor='k')
        
        # 繪製理想線
        min_val = min(np.min(targets), np.min(predictions))
        max_val = max(np.max(targets), np.max(predictions))
        
        # 增加一些邊界
        range_val = max_val - min_val
        plot_min = max(0, min_val - 0.05 * range_val)
        plot_max = max_val + 0.05 * range_val
        
        ax.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', 
               label='Ideal (y=x)')
        
        # 添加±20%線
        if not log_scale:
            x_vals = np.linspace(plot_min, plot_max, 100)
            ax.plot(x_vals, x_vals * 1.2, 'g--', alpha=0.5, label='+20%')
            ax.plot(x_vals, x_vals * 0.8, 'g--', alpha=0.5, label='-20%')
        
        # 計算並添加指標
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        r2 = r2_score(targets, predictions)
        
        # 計算平均絕對百分比誤差
        mape = np.mean(np.abs((targets - predictions) / targets)) * 100
        
        stats_text = (
            f'RMSE: {rmse:.4f}\n'
            f'R²: {r2:.4f}\n'
            f'MAPE: {mape:.2f}%'
        )
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 設置標籤和標題
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Prediction vs True Values')
        
        # 添加網格和圖例
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        plt.tight_layout()
        
        # 保存圖像
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"預測圖已保存至 {save_path}")
            except Exception as e:
                logger.warning(f"保存預測圖失敗: {str(e)}")
        
        return fig


class Trainer:
    """
    通用模型訓練器
    用於訓練銲錫接點疲勞壽命預測模型，支援不同類型的模型。
    """
    def __init__(self, model, optimizer, loss_fn, device=None, lr_scheduler=None, clip_grad_norm=1.0):
        """
        初始化訓練器
        
        參數:
            model (nn.Module): 待訓練的模型
            optimizer (torch.optim.Optimizer): 優化器
            loss_fn (nn.Module): 損失函數
            device (torch.device): 計算設備
            lr_scheduler (LearningRateScheduler): 學習率調度器
            clip_grad_norm (float): 梯度裁剪範數
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr_scheduler = lr_scheduler
        self.clip_grad_norm = clip_grad_norm
        
        # 將模型移至指定設備
        self.model.to(self.device)
        
        # 訓練記錄
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = {}
        self.lr_history = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        logger.info(f"初始化Trainer: model={model.__class__.__name__}, device={self.device}")
    
    def train(self, train_loader, val_loader, epochs=100, early_stopping=None, 
              verbose=True, eval_interval=1, save_path=None, callbacks=None):
        """
        訓練模型
        
        參數:
            train_loader (DataLoader): 訓練數據加載器
            val_loader (DataLoader): 驗證數據加載器
            epochs (int): 訓練輪數
            early_stopping (EarlyStopping): 早停機制
            verbose (bool): 是否輸出詳細日誌
            eval_interval (int): 評估間隔輪數
            save_path (str): 最佳模型保存路徑
            callbacks (list): 回調函數列表
            
        返回:
            dict: 訓練歷史記錄
        """
        if callbacks is None:
            callbacks = []
        
        start_time = time.time()
        logger.info(f"開始訓練: epochs={epochs}, 訓練批次數={len(train_loader)}, "
                  f"驗證批次數={len(val_loader)}")
        
        for epoch in range(epochs):
            # 執行一輪訓練
            train_loss = self._train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # 記錄學習率
            if self.lr_scheduler:
                self.lr_history.append(self.lr_scheduler.get_last_lr())
            
            # 定期評估
            if epoch % eval_interval == 0 or epoch == epochs - 1:
                val_loss, metrics = self.evaluate(val_loader)
                self.val_losses.append(val_loss)
                
                # 更新指標歷史記錄
                for metric_name, metric_value in metrics.items():
                    if metric_name not in self.metrics_history:
                        self.metrics_history[metric_name] = []
                    self.metrics_history[metric_name].append(metric_value)
                
                epoch_time = time.time() - start_time
                start_time = time.time()  # 重置開始時間
                
                if verbose:
                    # 格式化輸出，確保數字對齊
                    log_msg = f"輪次 {epoch+1}/{epochs} - 時間: {epoch_time:.2f}s - "
                    log_msg += f"訓練損失: {train_loss:.6f} - 驗證損失: {val_loss:.6f}"
                    
                    # 添加重要指標
                    if 'rmse' in metrics:
                        log_msg += f" - RMSE: {metrics['rmse']:.4f}"
                    if 'r2' in metrics:
                        log_msg += f" - R²: {metrics['r2']:.4f}"
                    
                    # 添加學習率
                    if self.lr_scheduler:
                        lrs = self.lr_scheduler.get_last_lr()
                        log_msg += f" - LR: {lrs[0]:.2e}"
                    
                    logger.info(log_msg)
                
                # 檢查是否是最佳模型
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    
                    if save_path is not None:
                        self._save_model(save_path, val_loss, metrics)
                
                # 早停檢查
                if early_stopping is not None and early_stopping(val_loss, self.model, epoch):
                    logger.info(f"早停觸發，在輪次 {epoch+1} 停止訓練")
                    break
                
                # 學習率調度
                if self.lr_scheduler is not None:
                    if hasattr(self.lr_scheduler, 'step_with_metrics'):
                        self.lr_scheduler.step(val_loss)
                    else:
                        self.lr_scheduler.step()
            else:
                # 如果不評估，仍然需要調整學習率
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
            
            # 執行回調函數
            for callback in callbacks:
                callback(epoch, {
                    'model': self.model,
                    'optimizer': self.optimizer,
                    'train_loss': train_loss,
                    'val_loss': val_loss if epoch % eval_interval == 0 else None,
                    'metrics': metrics if epoch % eval_interval == 0 else None
                })
        
        training_time = time.time() - start_time + epoch_time
        logger.info(f"訓練完成: 總時間: {training_time:.2f}秒, 最佳驗證損失: {self.best_val_loss:.6f}")
        
        # 恢復最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics_history': self.metrics_history,
            'best_val_loss': self.best_val_loss,
            'training_time': training_time
        }
    
    def _train_epoch(self, train_loader):
        """執行一輪訓練"""
        self.model.train()
        epoch_loss = 0.0
        
        for batch_idx, batch_data in enumerate(train_loader):
            # 解析批次數據
            if len(batch_data) == 3:
                static_features, time_series, targets = batch_data
                static_features = static_features.to(self.device)
                time_series = time_series.to(self.device)
                targets = targets.to(self.device)
                
                # 梯度清零
                self.optimizer.zero_grad()
                
                # 前向傳播
                outputs = self.model(static_features, time_series)
            else:
                # 單分支模型或其他類型模型
                inputs, targets = batch_data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # 梯度清零
                self.optimizer.zero_grad()
                
                # 前向傳播
                outputs = self.model(inputs)
            
            # 計算損失
            if isinstance(self.loss_fn, (HybridLoss, AdaptiveHybridLoss)):
                losses = self.loss_fn(outputs, targets)
                loss = losses['total_loss']
            else:
                # 簡單損失函數
                if isinstance(outputs, dict) and 'nf_pred' in outputs:
                    loss = self.loss_fn(outputs['nf_pred'], targets)
                else:
                    loss = self.loss_fn(outputs, targets)
            
            # 反向傳播和優化
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                
            self.optimizer.step()
            
            # 累計損失
            epoch_loss += loss.item()
        
        # 計算平均損失
        avg_epoch_loss = epoch_loss / len(train_loader)
        return avg_epoch_loss
    
    def evaluate(self, data_loader, return_predictions=False):
        """
        評估模型
        
        參數:
            data_loader (DataLoader): 數據加載器
            return_predictions (bool): 是否返回預測結果
            
        返回:
            tuple: (平均損失, 評估指標字典) 或 (平均損失, 評估指標字典, 預測值, 真實值)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_data in data_loader:
                # 解析批次數據
                if len(batch_data) == 3:
                    static_features, time_series, targets = batch_data
                    static_features = static_features.to(self.device)
                    time_series = time_series.to(self.device)
                    targets = targets.to(self.device)
                    
                    # 前向傳播
                    outputs = self.model(static_features, time_series)
                else:
                    # 單分支模型或其他類型模型
                    inputs, targets = batch_data
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    # 前向傳播
                    outputs = self.model(inputs)
                
                # 計算損失
                if isinstance(self.loss_fn, (HybridLoss, AdaptiveHybridLoss)):
                    losses = self.loss_fn(outputs, targets)
                    loss = losses['total_loss']
                else:
                    # 簡單損失函數
                    if isinstance(outputs, dict) and 'nf_pred' in outputs:
                        loss = self.loss_fn(outputs['nf_pred'], targets)
                        predictions = outputs['nf_pred']
                    else:
                        loss = self.loss_fn(outputs, targets)
                        predictions = outputs
                
                # 獲取預測值
                if isinstance(outputs, dict) and 'nf_pred' in outputs:
                    predictions = outputs['nf_pred']
                else:
                    predictions = outputs
                
                # 累計損失
                total_loss += loss.item()
                
                # 收集預測和目標
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # 計算平均損失
        avg_loss = total_loss / len(data_loader)
        
        # 合併批次結果
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        
        # 計算評估指標
        metrics = self._calculate_metrics(all_predictions, all_targets)
        
        if return_predictions:
            return avg_loss, metrics, all_predictions, all_targets
        else:
            return avg_loss, metrics
    
    def predict(self, data_loader):
        """
        使用模型進行預測
        
        參數:
            data_loader (DataLoader): 數據加載器
            
        返回:
            tuple: (預測值, 真實值)
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_outputs = {}
        
        with torch.no_grad():
            for batch_data in data_loader:
                # 解析批次數據
                if len(batch_data) == 3:
                    static_features, time_series, targets = batch_data
                    static_features = static_features.to(self.device)
                    time_series = time_series.to(self.device)
                    
                    if targets is not None:
                        targets = targets.to(self.device)
                    
                    # 前向傳播
                    outputs = self.model(static_features, time_series, return_features=True)
                else:
                    # 單分支模型或其他類型模型
                    if len(batch_data) == 2:
                        inputs, targets = batch_data
                        inputs = inputs.to(self.device)
                        
                        if targets is not None:
                            targets = targets.to(self.device)
                        
                        # 前向傳播
                        outputs = self.model(inputs)
                    else:
                        inputs = batch_data[0]
                        inputs = inputs.to(self.device)
                        targets = None
                        
                        # 前向傳播
                        outputs = self.model(inputs)
                
                # 獲取預測值
                if isinstance(outputs, dict) and 'nf_pred' in outputs:
                    predictions = outputs['nf_pred']
                    
                    # 收集所有輸出
                    for key, value in outputs.items():
                        if key not in all_outputs:
                            all_outputs[key] = []
                        all_outputs[key].append(value.cpu().numpy())
                else:
                    predictions = outputs
                
                # 收集預測和目標
                all_predictions.append(predictions.cpu().numpy())
                if targets is not None:
                    all_targets.append(targets.cpu().numpy())
        
        # 合併批次結果
        all_predictions = np.concatenate(all_predictions)
        
        if all_targets:
            all_targets = np.concatenate(all_targets)
        else:
            all_targets = None
        
        # 合併所有輸出
        for key in all_outputs:
            all_outputs[key] = np.concatenate(all_outputs[key])
        
        # 添加最終預測
        all_outputs['predictions'] = all_predictions
        all_outputs['targets'] = all_targets
        
        return all_outputs
    
    def _calculate_metrics(self, predictions, targets):
        """計算評估指標"""
        metrics = {}
        
        # 均方根誤差 (RMSE)
        metrics['rmse'] = np.sqrt(mean_squared_error(targets, predictions))
        
        # 決定係數 (R²)
        metrics['r2'] = r2_score(targets, predictions)
        
        # 平均絕對誤差 (MAE)
        metrics['mae'] = mean_absolute_error(targets, predictions)
        
        # 平均相對誤差 (MAPE)
        metrics['mape'] = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
        
        # 對數空間指標 (對於高動態範圍數據有用)
        try:
            log_targets = np.log(targets + 1e-8)
            log_predictions = np.log(predictions + 1e-8)
            metrics['log_rmse'] = np.sqrt(mean_squared_error(log_targets, log_predictions))
            metrics['log_r2'] = r2_score(log_targets, log_predictions)
        except Exception as e:
            logger.warning(f"計算對數空間指標時出錯: {str(e)}")
        
        return metrics
    
    def _save_model(self, path, val_loss, metrics):
        """保存模型"""
        # 確保保存目錄存在
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'metrics': metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics_history': self.metrics_history
        }, path)
        
        logger.info(f"模型已保存至 {path}")
    
    def load_model(self, path):
        """載入模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 恢復訓練歷史記錄
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        if 'metrics_history' in checkpoint:
            self.metrics_history = checkpoint['metrics_history']
        if 'val_loss' in checkpoint:
            self.best_val_loss = checkpoint['val_loss']
        
        logger.info(f"模型已從 {path} 載入")
        
        return checkpoint.get('metrics', {})
    
    def plot_losses(self, figsize=(10, 6), save_path=None):
        """繪製損失曲線"""
        plt.figure(figsize=figsize)
        plt.plot(self.train_losses, label='Training Loss')
        
        # 如果驗證損失不是每輪都記錄的，需要調整繪圖
        if len(self.val_losses) < len(self.train_losses):
            # 假設每 eval_interval 輪記錄一次驗證損失
            eval_interval = len(self.train_losses) // len(self.val_losses)
            val_epochs = np.arange(0, len(self.train_losses), eval_interval)
            if len(val_epochs) > len(self.val_losses):
                val_epochs = val_epochs[:len(self.val_losses)]
            plt.plot(val_epochs, self.val_losses, label='Validation Loss')
        else:
            plt.plot(self.val_losses, label='Validation Loss')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    def plot_metrics(self, metric_name='rmse', figsize=(10, 6), save_path=None):
        """繪製指標曲線"""
        if metric_name not in self.metrics_history:
            logger.warning(f"指標 {metric_name} 不在歷史記錄中")
            return
        
        plt.figure(figsize=figsize)
        plt.plot(self.metrics_history[metric_name])
        plt.xlabel('Evaluation')
        plt.ylabel(metric_name.upper())
        plt.title(f'{metric_name.upper()} During Training')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    def plot_predictions(self, predictions, targets, figsize=(10, 6), save_path=None):
        """繪製預測對比圖"""
        plt.figure(figsize=figsize)
        
        # 繪製預測值與真實值的散點圖
        plt.scatter(targets, predictions, alpha=0.6)
        
        # 繪製理想的對角線
        min_val = min(np.min(targets), np.min(predictions))
        max_val = max(np.max(targets), np.max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('Prediction vs True Values')
        plt.grid(True)
        
        # 計算並顯示指標
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        r2 = r2_score(targets, predictions)
        plt.text(0.05, 0.95, f'RMSE: {rmse:.4f}\nR²: {r2:.4f}', 
                 transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()


if __name__ == "__main__":
    # 簡單的測試代碼
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 創建一個簡單的模型和訓練器進行測試
    class SimpleModel(nn.Module):
        def __init__(self, input_dim=5, hidden_dim=10, output_dim=1):
            super(SimpleModel, self).__init__()
            self.layer1 = nn.Linear(input_dim, hidden_dim)
            self.layer2 = nn.Linear(hidden_dim, output_dim)
            
        def forward(self, x):
            x = torch.relu(self.layer1(x))
            x = self.layer2(x)
            return x
    
    # 創建模型和優化器
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    # 創建訓練器
    trainer = Trainer(model, optimizer, loss_fn)
    
    # 創建虛擬數據加載器
    class DummyDataLoader:
        def __init__(self, samples=100, input_dim=5, batch_size=16):
            self.samples = samples
            self.input_dim = input_dim
            self.batch_size = batch_size
            
        def __iter__(self):
            for i in range(0, self.samples, self.batch_size):
                batch_size = min(self.batch_size, self.samples - i)
                inputs = torch.randn(batch_size, self.input_dim)
                targets = torch.sum(inputs, dim=1, keepdim=True) / self.input_dim + 0.1 * torch.randn(batch_size, 1)
                yield inputs, targets
        
        def __len__(self):
            return (self.samples + self.batch_size - 1) // self.batch_size
    
    train_loader = DummyDataLoader(samples=100)
    val_loader = DummyDataLoader(samples=20)
    
    # 創建早停
    early_stopping = EarlyStopping(patience=5, mode='min')
    
    # 創建學習率調度器
    lr_scheduler = LearningRateScheduler(optimizer, mode='step', step_size=5, gamma=0.5)
    
    # 測試訓練
    logger.info("開始測試訓練...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=20,
        early_stopping=early_stopping,
        verbose=True
    )
    
    # 測試評估
    logger.info("測試評估...")
    val_loss, metrics = trainer.evaluate(val_loader)
    logger.info(f"評估結果: 損失={val_loss:.6f}, 指標={metrics}")
    
    # 測試預測
    logger.info("測試預測...")
    results = trainer.predict(val_loader)
    predictions = results['predictions']
    targets = results['targets']
    logger.info(f"預測結果: 形狀={predictions.shape}, 範圍=[{np.min(predictions):.4f}, {np.max(predictions):.4f}]")
    
    # 測試繪圖
    try:
        logger.info("測試繪圖...")
        trainer.plot_losses()
        
        if 'rmse' in trainer.metrics_history:
            trainer.plot_metrics('rmse')
        
        if predictions is not None and targets is not None:
            trainer.plot_predictions(predictions, targets)
            
        logger.info("測試繪圖完成")
    except Exception as e:
        logger.error(f"繪圖錯誤: {str(e)}")
    
    logger.info("所有測試完成")