#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
trainer.py - 模型訓練器模組
本模組實現了用於訓練銲錫接點疲勞壽命預測混合模型的訓練器，
包括完整的訓練循環、模型評估和早停機制等功能。

主要組件:
1. Trainer - 基本訓練器類別，提供通用的訓練和評估功能
2. EarlyStopping - 早停機制，防止模型過擬合
3. LearningRateScheduler - 學習率調度器，動態調整學習率
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

# 導入自定義模組
from src.training.losses import HybridLoss, AdaptiveHybridLoss

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    早停機制
    監控驗證集上的指標，當指標不再改善時提前停止訓練
    """
    def __init__(self, patience=20, min_delta=0, mode='min', verbose=True, save_path=None):
        """
        初始化早停機制
        
        參數:
            patience (int): 容忍指標不改善的輪數
            min_delta (float): 認為有改善的最小變化量
            mode (str): 監控模式，'min'表示指標越小越好，'max'表示指標越大越好
            verbose (bool): 是否輸出日誌
            save_path (str): 最佳模型保存路徑
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.save_path = save_path
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
        # 初始化最佳分數比較函數
        if mode == 'min':
            self._is_better = lambda score, best: score <= best - min_delta
        elif mode == 'max':
            self._is_better = lambda score, best: score >= best + min_delta
        else:
            raise ValueError(f"不支援的模式: {mode}，應為 'min' 或 'max'")
        
        logger.info(f"初始化EarlyStopping: patience={patience}, mode={mode}")
    
    def __call__(self, score, model=None):
        """
        檢查是否應該早停
        
        參數:
            score (float): 當前指標值
            model (nn.Module, optional): 當前模型
            
        返回:
            bool: 是否應該早停
        """
        if self.best_score is None:
            # 首次迭代，初始化最佳分數
            self.best_score = score
            self._save_checkpoint(score, model)
        elif self._is_better(score, self.best_score):
            # 指標有改善
            self.best_score = score
            self.counter = 0
            self._save_checkpoint(score, model)
        else:
            # 指標沒有改善
            self.counter += 1
            if self.verbose:
                logger.debug(f"EarlyStopping: 指標未改善，計數器 {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info(f"EarlyStopping: 指標連續 {self.patience} 輪未改善，觸發早停")
        
        return self.early_stop
    
    def _save_checkpoint(self, score, model):
        """保存模型檢查點"""
        if model is not None and self.save_path is not None:
            # 確保保存目錄存在
            Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
            
            if self.verbose:
                if self.mode == 'min':
                    logger.info(f"指標改善: {self.best_score:.6f} -> {score:.6f}，保存模型至 {self.save_path}")
                else:
                    logger.info(f"指標改善: {self.best_score:.6f} -> {score:.6f}，保存模型至 {self.save_path}")
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'score': score
            }, self.save_path)
    
    def reset(self):
        """重置早停機制"""
        self.best_score = None
        self.counter = 0
        self.early_stop = False


class LearningRateScheduler:
    """
    學習率調度器
    根據訓練進度動態調整學習率
    """
    def __init__(self, optimizer, mode='step', **kwargs):
        """
        初始化學習率調度器
        
        參數:
            optimizer (torch.optim.Optimizer): 優化器
            mode (str): 調度模式
                - 'step': 階梯式下降，每隔step_size輪降低gamma倍
                - 'exp': 指數下降，每輪降低gamma倍
                - 'cosine': 餘弦退火，在T_max輪內從初始學習率降至min_lr
                - 'plateau': 當指標不再改善時降低學習率
            **kwargs: 根據不同模式傳遞的參數
        """
        self.optimizer = optimizer
        self.mode = mode
        self.kwargs = kwargs
        
        # 獲取初始學習率
        self.initial_lr = []
        for param_group in optimizer.param_groups:
            self.initial_lr.append(param_group['lr'])
        
        # 根據模式創建調度器
        if mode == 'step':
            step_size = kwargs.get('step_size', 10)
            gamma = kwargs.get('gamma', 0.1)
            self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            
        elif mode == 'exp':
            gamma = kwargs.get('gamma', 0.95)
            self.scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
            
        elif mode == 'cosine':
            T_max = kwargs.get('T_max', 100)
            min_lr = kwargs.get('min_lr', 0)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=min_lr)
            
        elif mode == 'plateau':
            factor = kwargs.get('factor', 0.1)
            patience = kwargs.get('patience', 10)
            min_lr = kwargs.get('min_lr', 0)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=factor, patience=patience, min_lr=min_lr
            )
            
        else:
            raise ValueError(f"不支援的調度模式: {mode}")
        
        logger.info(f"初始化LearningRateScheduler: mode={mode}, 初始學習率={self.initial_lr}")
    
    def step(self, val_loss=None):
        """
        更新學習率
        
        參數:
            val_loss (float, optional): 驗證損失，僅適用於'plateau'模式
        """
        if self.mode == 'plateau' and val_loss is not None:
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()
    
    def get_last_lr(self):
        """獲取當前學習率"""
        if self.mode == 'plateau':
            return [param_group['lr'] for param_group in self.optimizer.param_groups]
        else:
            return self.scheduler.get_last_lr()


class Trainer:
    """
    模型訓練器
    提供完整的模型訓練、評估和預測功能
    """
    def __init__(self, model, optimizer, loss_fn, device=None, lr_scheduler=None):
        """
        初始化訓練器
        
        參數:
            model (nn.Module): 待訓練的模型
            optimizer (torch.optim.Optimizer): 優化器
            loss_fn (nn.Module): 損失函數
            device (torch.device): 計算設備
            lr_scheduler (LearningRateScheduler, optional): 學習率調度器
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr_scheduler = lr_scheduler
        
        # 將模型移至指定設備
        self.model.to(self.device)
        
        # 訓練記錄
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = {}
        
        # 訓練配置
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
            early_stopping (EarlyStopping, optional): 早停機制
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
            epoch_start_time = time.time()
            
            # 執行一輪訓練
            train_loss = self._train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # 定期評估
            if epoch % eval_interval == 0 or epoch == epochs - 1:
                val_loss, metrics = self.evaluate(val_loader)
                self.val_losses.append(val_loss)
                
                # 更新指標歷史記錄
                for metric_name, metric_value in metrics.items():
                    if metric_name not in self.metrics_history:
                        self.metrics_history[metric_name] = []
                    self.metrics_history[metric_name].append(metric_value)
                
                epoch_time = time.time() - epoch_start_time
                
                if verbose:
                    logger.info(f"輪次 {epoch+1}/{epochs} - 時間: {epoch_time:.2f}s - "
                              f"訓練損失: {train_loss:.6f} - 驗證損失: {val_loss:.6f} - "
                              f"R²: {metrics.get('r2_score', 0):.4f} - "
                              f"RMSE: {metrics.get('rmse', 0):.4f}")
                
                # 檢查是否是最佳模型
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_state = self.model.state_dict().copy()
                    
                    if save_path is not None:
                        self._save_model(save_path, val_loss, metrics)
                
                # 早停檢查
                if early_stopping is not None and early_stopping(val_loss, self.model):
                    logger.info(f"早停觸發，在輪次 {epoch+1} 停止訓練")
                    break
                
                # 學習率調度
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(val_loss)
            
            # 執行回調函數
            for callback in callbacks:
                callback(epoch, {
                    'model': self.model,
                    'optimizer': self.optimizer,
                    'train_loss': train_loss,
                    'val_loss': val_loss if epoch % eval_interval == 0 else None,
                    'metrics': metrics if epoch % eval_interval == 0 else None
                })
        
        total_time = time.time() - start_time
        logger.info(f"訓練完成: 總時間: {total_time:.2f}s, 最佳驗證損失: {self.best_val_loss:.6f}")
        
        # 恢復最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics_history': self.metrics_history,
            'best_val_loss': self.best_val_loss,
            'total_time': total_time
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
            if hasattr(self, 'clip_grad_norm'):
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
        metrics['r2_score'] = r2_score(targets, predictions)
        
        # 平均絕對誤差 (MAE)
        metrics['mae'] = mean_absolute_error(targets, predictions)
        
        # 平均相對誤差 (MAPE)
        metrics['mape'] = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
        
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