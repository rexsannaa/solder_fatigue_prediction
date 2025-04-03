#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
callbacks.py - 回調函數模組
本模組實現了用於訓練銲錫接點疲勞壽命預測混合模型的各種回調函數，
用於在訓練過程中執行額外操作，如模型檢查點保存、進度顯示和日誌記錄等。

主要組件:
1. ModelCheckpoint - 模型檢查點保存回調
2. TensorBoardLogger - TensorBoard日誌記錄回調
3. ProgressBar - 訓練進度條顯示回調
"""

import os
import time
import numpy as np
import torch
from pathlib import Path
import logging
import json
from datetime import datetime
try:
    from torch.utils.tensorboard import SummaryWriter
    tensorboard_available = True
except ImportError:
    tensorboard_available = False
    
try:
    from tqdm import tqdm
    tqdm_available = True
except ImportError:
    tqdm_available = False

logger = logging.getLogger(__name__)


class Callback:
    """
    回調函數基類
    定義回調函數的通用接口
    """
    def __call__(self, epoch, state):
        """
        回調函數調用接口
        
        參數:
            epoch (int): 當前訓練輪次
            state (dict): 當前訓練狀態，包含模型、優化器、損失等信息
        """
        pass


class ModelCheckpoint(Callback):
    """
    模型檢查點保存回調
    定期保存模型的檢查點，支持多種保存策略
    """
    def __init__(self, checkpoint_dir, filename_template='model_epoch_{epoch:03d}.pt',
                 save_freq=1, save_best_only=False, monitor='val_loss', mode='min',
                 max_save=5, verbose=True):
        """
        初始化模型檢查點回調
        
        參數:
            checkpoint_dir (str): 檢查點保存目錄
            filename_template (str): 檢查點文件名模板
            save_freq (int): 保存頻率（每隔多少輪保存一次）
            save_best_only (bool): 是否只保存最佳模型
            monitor (str): 監控指標名稱
            mode (str): 監控模式，'min'表示指標越小越好，'max'表示指標越大越好
            max_save (int): 最多保存的檢查點數量
            verbose (bool): 是否輸出詳細日誌
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.filename_template = filename_template
        self.save_freq = save_freq
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.max_save = max_save
        self.verbose = verbose
        
        # 確保保存目錄存在
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化監控指標比較函數和最佳值
        if mode == 'min':
            self.is_better = lambda current, best: current < best
            self.best_value = float('inf')
        elif mode == 'max':
            self.is_better = lambda current, best: current > best
            self.best_value = float('-inf')
        else:
            raise ValueError(f"不支援的模式: {mode}，應為 'min' 或 'max'")
        
        # 保存的檢查點列表
        self.saved_checkpoints = []
        
        logger.info(f"初始化ModelCheckpoint: 保存目錄={checkpoint_dir}, "
                  f"保存頻率={save_freq}, 只保存最佳={save_best_only}, "
                  f"監控指標={monitor}, 模式={mode}")
    
    def __call__(self, epoch, state):
        """執行回調"""
        if epoch % self.save_freq != 0 and epoch != -1:  # epoch=-1 表示訓練結束
            return
        
        # 獲取當前監控指標值
        current_value = None
        if self.monitor == 'val_loss' and state.get('val_loss') is not None:
            current_value = state['val_loss']
        elif state.get('metrics') is not None and self.monitor in state['metrics']:
            current_value = state['metrics'][self.monitor]
        
        # 判斷是否需要保存檢查點
        save_checkpoint = False
        if self.save_best_only:
            if current_value is not None and self.is_better(current_value, self.best_value):
                if self.verbose:
                    logger.info(f"輪次 {epoch+1}: {self.monitor} 改善 "
                              f"({self.best_value:.6f} -> {current_value:.6f})，保存模型")
                self.best_value = current_value
                save_checkpoint = True
        else:
            save_checkpoint = True
        
        if save_checkpoint:
            # 生成檢查點文件名
            if 'epoch' in self.filename_template:
                filename = self.filename_template.format(epoch=epoch+1)
            else:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{self.filename_template}_{timestamp}.pt"
            
            checkpoint_path = self.checkpoint_dir / filename
            
            # 保存檢查點
            self._save_checkpoint(checkpoint_path, state)
            
            # 管理檢查點數量
            self.saved_checkpoints.append(checkpoint_path)
            if self.max_save > 0 and len(self.saved_checkpoints) > self.max_save:
                oldest_checkpoint = self.saved_checkpoints.pop(0)
                if oldest_checkpoint.exists():
                    oldest_checkpoint.unlink()
                    if self.verbose:
                        logger.debug(f"移除舊檢查點: {oldest_checkpoint}")
    
    def _save_checkpoint(self, path, state):
        """保存檢查點"""
        # 獲取模型和優化器
        model = state.get('model')
        optimizer = state.get('optimizer')
        
        if model is None:
            logger.warning("無法保存檢查點：模型不存在")
            return
        
        # 構建檢查點內容
        checkpoint = {
            'model_state_dict': model.state_dict(),
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        # 保存訓練狀態
        for key in ['train_loss', 'val_loss', 'metrics']:
            if key in state and state[key] is not None:
                checkpoint[key] = state[key]
        
        # 保存其他額外信息
        checkpoint['epoch'] = state.get('epoch', 0)
        checkpoint['timestamp'] = datetime.now().isoformat()
        
        # 保存檢查點
        torch.save(checkpoint, path)
        
        if self.verbose:
            logger.info(f"模型檢查點已保存至 {path}")


class TensorBoardLogger(Callback):
    """
    TensorBoard日誌記錄回調
    將訓練過程中的損失和指標記錄到TensorBoard中
    """
    def __init__(self, log_dir, comment='', flush_secs=120):
        """
        初始化TensorBoard日誌記錄回調
        
        參數:
            log_dir (str): 日誌保存目錄
            comment (str): 附加註釋
            flush_secs (int): 刷新間隔（秒）
        """
        if not tensorboard_available:
            logger.warning("TensorBoard不可用，請安裝PyTorch版本的tensorboard")
            self.writer = None
            return
        
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 創建SummaryWriter
        self.writer = SummaryWriter(log_dir=str(log_dir), comment=comment, flush_secs=flush_secs)
        
        logger.info(f"初始化TensorBoardLogger: 日誌目錄={log_dir}")
    
    def __call__(self, epoch, state):
        """執行回調"""
        if self.writer is None:
            return
        
        # 記錄訓練損失
        if 'train_loss' in state and state['train_loss'] is not None:
            self.writer.add_scalar('Loss/train', state['train_loss'], epoch)
        
        # 記錄驗證損失
        if 'val_loss' in state and state['val_loss'] is not None:
            self.writer.add_scalar('Loss/validation', state['val_loss'], epoch)
        
        # 記錄指標
        if 'metrics' in state and state['metrics'] is not None:
            for metric_name, metric_value in state['metrics'].items():
                self.writer.add_scalar(f'Metrics/{metric_name}', metric_value, epoch)
        
        # 記錄學習率
        if 'optimizer' in state and state['optimizer'] is not None:
            for i, param_group in enumerate(state['optimizer'].param_groups):
                self.writer.add_scalar(f'LearningRate/group{i}', param_group['lr'], epoch)
    
    def close(self):
        """關閉TensorBoard寫入器"""
        if self.writer is not None:
            self.writer.close()


class ProgressBar(Callback):
    """
    進度條顯示回調
    顯示訓練進度條
    """
    def __init__(self, total_epochs, training_steps_per_epoch=None, update_freq=1, ascii=False):
        """
        初始化進度條回調
        
        參數:
            total_epochs (int): 總訓練輪數
            training_steps_per_epoch (int, optional): 每輪訓練步數
            update_freq (int): 更新頻率
            ascii (bool): 是否使用ASCII字符
        """
        if not tqdm_available:
            logger.warning("tqdm不可用，無法顯示進度條")
            self.progress_bar = None
            return
        
        self.total_epochs = total_epochs
        self.update_freq = update_freq
        self.ascii = ascii
        
        # 初始化進度條
        self.progress_bar = tqdm(
            total=total_epochs, 
            desc="Training", 
            ascii=ascii,
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        self.last_epoch = -1
    
    def __call__(self, epoch, state):
        """執行回調"""
        if self.progress_bar is None:
            return
        
        # 更新進度條
        if epoch > self.last_epoch:
            # 計算需要更新的步數
            steps = epoch - self.last_epoch
            
            # 更新進度條
            self.progress_bar.update(steps)
            
            # 更新描述信息
            desc_items = []
            if 'train_loss' in state and state['train_loss'] is not None:
                desc_items.append(f"loss: {state['train_loss']:.4f}")
            if 'val_loss' in state and state['val_loss'] is not None:
                desc_items.append(f"val_loss: {state['val_loss']:.4f}")
            if 'metrics' in state and state['metrics'] is not None:
                for metric_name, metric_value in state['metrics'].items():
                    if metric_name in ['rmse', 'r2_score', 'mae']:
                        desc_items.append(f"{metric_name}: {metric_value:.4f}")
            
            if desc_items:
                self.progress_bar.set_description(f"Epoch {epoch+1}/{self.total_epochs} - " + " - ".join(desc_items))
            
            self.last_epoch = epoch
    
    def close(self):
        """關閉進度條"""
        if self.progress_bar is not None:
            self.progress_bar.close()


class LossHistory(Callback):
    """
    損失歷史記錄回調
    記錄訓練過程中的損失和指標，並可以保存為JSON文件
    """
    def __init__(self, save_path=None, save_freq=10):
        """
        初始化損失歷史記錄回調
        
        參數:
            save_path (str, optional): 歷史記錄保存路徑
            save_freq (int): 保存頻率
        """
        self.save_path = save_path
        self.save_freq = save_freq
        
        # 初始化歷史記錄
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'metrics': {},
            'epoch': []
        }
        
        if save_path is not None:
            # 確保保存目錄存在
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"初始化LossHistory: 保存路徑={save_path}, 保存頻率={save_freq}")
    
    def __call__(self, epoch, state):
        """執行回調"""
        # 記錄訓練損失
        if 'train_loss' in state and state['train_loss'] is not None:
            self.history['train_loss'].append(float(state['train_loss']))
        else:
            self.history['train_loss'].append(None)
        
        # 記錄驗證損失
        if 'val_loss' in state and state['val_loss'] is not None:
            self.history['val_loss'].append(float(state['val_loss']))
        else:
            self.history['val_loss'].append(None)
        
        # 記錄指標
        if 'metrics' in state and state['metrics'] is not None:
            for metric_name, metric_value in state['metrics'].items():
                if metric_name not in self.history['metrics']:
                    self.history['metrics'][metric_name] = []
                self.history['metrics'][metric_name].append(float(metric_value))
        
        # 記錄輪次
        self.history['epoch'].append(epoch)
        
        # 定期保存歷史記錄
        if self.save_path is not None and (epoch % self.save_freq == 0 or epoch == -1):
            self.save()
    
    def save(self):
        """保存歷史記錄"""
        if self.save_path is None:
            return
        
        # 轉換為JSON可序列化格式
        json_history = {}
        for key, value in self.history.items():
            if isinstance(value, dict):
                json_history[key] = {}
                for sub_key, sub_value in value.items():
                    json_history[key][sub_key] = [float(v) if v is not None else None for v in sub_value]
            else:
                json_history[key] = [float(v) if v is not None else None for v in value]
        
        # 保存為JSON文件
        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(json_history, f, indent=2)
            
        logger.debug(f"訓練歷史記錄已保存至 {self.save_path}")


class AdaptiveCallbacks:
    """
    自適應回調工廠類
    根據數據集大小和模型複雜度自動創建適合的回調函數組合
    """
    @staticmethod
    def create_callbacks(
        model, 
        dataset_size,
        epochs,
        output_dir='./outputs',
        use_tensorboard=True,
        use_progress_bar=True,
        use_early_stopping=True,
        patience=20,
        monitor='val_loss',
        mode='min',
        save_freq=5,
        verbose=True
    ):
        """
        創建適合的回調函數組合
        
        參數:
            model (nn.Module): 模型
            dataset_size (int): 數據集大小
            epochs (int): 訓練輪數
            output_dir (str): 輸出目錄
            use_tensorboard (bool): 是否使用TensorBoard
            use_progress_bar (bool): 是否使用進度條
            use_early_stopping (bool): 是否使用早停
            patience (int): 早停耐心值
            monitor (str): 監控指標
            mode (str): 監控模式
            save_freq (int): 保存頻率
            verbose (bool): 是否輸出詳細日誌
            
        返回:
            list: 回調函數列表
        """
        callbacks = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 創建輸出目錄
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 模型名稱
        model_name = model.__class__.__name__
        
        # 1. 添加模型檢查點回調
        if dataset_size < 100:  # 小數據集
            # 小數據集容易過擬合，只保存最佳模型
            checkpoint_dir = output_dir / f"{model_name}_{timestamp}" / "checkpoints"
            checkpoint_callback = ModelCheckpoint(
                checkpoint_dir=checkpoint_dir,
                filename_template=f"{model_name}_best.pt",
                save_best_only=True,
                monitor=monitor,
                mode=mode,
                verbose=verbose
            )
        else:
            # 大數據集定期保存檢查點
            checkpoint_dir = output_dir / f"{model_name}_{timestamp}" / "checkpoints"
            checkpoint_callback = ModelCheckpoint(
                checkpoint_dir=checkpoint_dir,
                filename_template=f"{model_name}_epoch_{{epoch:03d}}.pt",
                save_freq=save_freq,
                save_best_only=False,
                max_save=min(5, epochs // save_freq + 1),
                verbose=verbose
            )
        
        callbacks.append(checkpoint_callback)
        
        # 2. 添加損失歷史記錄回調
        history_path = output_dir / f"{model_name}_{timestamp}" / "history.json"
        history_callback = LossHistory(
            save_path=history_path,
            save_freq=1
        )
        callbacks.append(history_callback)
        
        # 3. 添加TensorBoard回調
        if use_tensorboard and tensorboard_available:
            tensorboard_dir = output_dir / f"{model_name}_{timestamp}" / "tensorboard"
            tensorboard_callback = TensorBoardLogger(
                log_dir=tensorboard_dir,
                comment=f"_{model_name}_{dataset_size}_samples"
            )
            callbacks.append(tensorboard_callback)
        
        # 4. 添加進度條回調
        if use_progress_bar and tqdm_available:
            progress_bar = ProgressBar(
                total_epochs=epochs,
                update_freq=1,
                ascii=True  # 在某些環境中可能更兼容
            )
            callbacks.append(progress_bar)
        
        return callbacks


# 輔助函數
def create_default_callbacks(model_name, output_dir='./outputs', epochs=100):
    """
    創建默認回調函數集合
    
    參數:
        model_name (str): 模型名稱
        output_dir (str): 輸出目錄
        epochs (int): 訓練輪數
        
    返回:
        list: 回調函數列表
    """
    callbacks = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 創建輸出目錄
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_dir = output_dir / f"{model_name}_{timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 模型檢查點回調
    checkpoint_dir = model_dir / "checkpoints"
    checkpoint_callback = ModelCheckpoint(
        checkpoint_dir=checkpoint_dir,
        filename_template=f"{model_name}_epoch_{{epoch:03d}}.pt",
        save_freq=5,
        save_best_only=False,
        monitor='val_loss',
        mode='min',
        max_save=5,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # 2. 損失歷史記錄回調
    history_path = model_dir / "history.json"
    history_callback = LossHistory(
        save_path=history_path,
        save_freq=1
    )
    callbacks.append(history_callback)
    
    # 3. TensorBoard回調 (如果可用)
    if tensorboard_available:
        tensorboard_dir = model_dir / "tensorboard"
        tensorboard_callback = TensorBoardLogger(
            log_dir=tensorboard_dir,
            comment=f"_{model_name}"
        )
        callbacks.append(tensorboard_callback)
    
    # 4. 進度條回調 (如果可用)
    if tqdm_available:
        progress_bar = ProgressBar(
            total_epochs=epochs,
            update_freq=1,
            ascii=True
        )
        callbacks.append(progress_bar)
    
    return callbacks


if __name__ == "__main__":
    # 簡單的測試代碼
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 測試默認回調函數
    logger.info("測試默認回調函數集合:")
    callbacks = create_default_callbacks(
        model_name="HybridPINNLSTM",
        output_dir="./test_outputs",
        epochs=50
    )
    
    logger.info(f"創建了 {len(callbacks)} 個回調函數:")
    for i, callback in enumerate(callbacks):
        logger.info(f"  {i+1}. {callback.__class__.__name__}")
    
    # 測試自適應回調函數
    logger.info("\n測試自適應回調函數:")
    
    # 模擬模型
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.linear = torch.nn.Linear(10, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    model = DummyModel()
    adaptive_callbacks = AdaptiveCallbacks.create_callbacks(
        model=model,
        dataset_size=81,  # 小樣本數據集
        epochs=100,
        output_dir="./test_outputs",
        use_tensorboard=True,
        use_progress_bar=True,
        use_early_stopping=True,
        patience=15,
        monitor='val_loss',
        mode='min'
    )
    
    logger.info(f"為小樣本數據集創建了 {len(adaptive_callbacks)} 個回調函數:")
    for i, callback in enumerate(adaptive_callbacks):
        logger.info(f"  {i+1}. {callback.__class__.__name__}")
    
    # 清理測試目錄
    import shutil
    if os.path.exists("./test_outputs"):
        shutil.rmtree("./test_outputs")
        logger.info("已清理測試輸出目錄")