#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
hybrid_model.py - 混合PINN-LSTM模型
本模組實現了結合物理信息神經網絡(PINN)和長短期記憶網絡(LSTM)的混合模型，
用於準確預測銲錫接點的疲勞壽命。

主要特點:
1. 雙分支架構: PINN分支處理靜態特徵與物理約束，LSTM分支處理時間序列特徵
2. 特徵融合機制: 結合物理知識與時序特徵的注意力機制
3. 混合損失函數: 平衡物理約束與數據擬合
4. 針對小樣本數據集（81筆）優化的設計
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

from src.models.pinn import PINNModel
from src.models.lstm import LSTMModel

logger = logging.getLogger(__name__)

class FeatureFusionLayer(nn.Module):
    """
    特徵融合層
    融合PINN和LSTM分支提取的特徵，並透過注意力機制處理特徵的重要性
    """
    def __init__(self, pinn_feature_dim, lstm_feature_dim, fusion_dim=32):
        """
        初始化特徵融合層
        
        參數:
            pinn_feature_dim (int): PINN分支特徵維度
            lstm_feature_dim (int): LSTM分支特徵維度
            fusion_dim (int): 融合後的特徵維度
        """
        super(FeatureFusionLayer, self).__init__()
        
        self.pinn_feature_dim = pinn_feature_dim
        self.lstm_feature_dim = lstm_feature_dim
        self.fusion_dim = fusion_dim
        
        # 特徵投影層
        self.pinn_projection = nn.Linear(pinn_feature_dim, fusion_dim)
        self.lstm_projection = nn.Linear(lstm_feature_dim, fusion_dim)
        
        # 注意力門控機制
        self.attention_gate = nn.Sequential(
            nn.Linear(pinn_feature_dim + lstm_feature_dim, 2),
            nn.Softmax(dim=1)
        )
        
        # 融合後的特徵處理
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, pinn_features, lstm_features):
        """
        前向傳播
        
        參數:
            pinn_features (torch.Tensor): PINN分支特徵
            lstm_features (torch.Tensor): LSTM分支特徵
            
        返回:
            tuple: (融合特徵, 注意力權重)
        """
        # 計算特徵融合的注意力權重
        combined_features = torch.cat([pinn_features, lstm_features], dim=1)
        attention_weights = self.attention_gate(combined_features)
        
        # 投影特徵到相同的空間
        pinn_projected = self.pinn_projection(pinn_features)
        lstm_projected = self.lstm_projection(lstm_features)
        
        # 加權融合
        fused_features = (
            attention_weights[:, 0].unsqueeze(1) * pinn_projected + 
            attention_weights[:, 1].unsqueeze(1) * lstm_projected
        )
        
        # 進一步處理融合特徵
        output_features = self.fusion_layers(fused_features)
        
        return output_features, attention_weights


class HybridPINNLSTMModel(nn.Module):
    """
    混合PINN-LSTM模型
    結合物理信息神經網絡和長短期記憶網絡的優勢，實現高精度疲勞壽命預測
    """
    def __init__(self, 
                 static_input_dim=5,        # 靜態結構參數維度
                 time_input_dim=2,          # 時間序列特徵維度
                 time_steps=4,              # 時間步數
                 pinn_hidden_dims=[64, 32, 16],  # PINN隱藏層維度
                 lstm_hidden_size=64,       # LSTM隱藏層大小
                 lstm_num_layers=2,         # LSTM層數
                 fusion_dim=32,             # 特徵融合維度
                 dropout_rate=0.2,          # Dropout比率
                 bidirectional=True,        # 是否使用雙向LSTM
                 use_attention=True,        # 是否使用注意力機制
                 use_physics_layer=True):   # 是否使用物理約束層
        """
        初始化混合PINN-LSTM模型
        
        參數:
            static_input_dim (int): 靜態特徵輸入維度
            time_input_dim (int): 時間序列特徵輸入維度
            time_steps (int): 時間序列步數
            pinn_hidden_dims (list): PINN分支隱藏層維度列表
            lstm_hidden_size (int): LSTM隱藏層大小
            lstm_num_layers (int): LSTM層數
            fusion_dim (int): 特徵融合維度
            dropout_rate (float): Dropout比率
            bidirectional (bool): 是否使用雙向LSTM
            use_attention (bool): 是否使用LSTM的注意力機制
            use_physics_layer (bool): 是否使用物理約束層
        """
        super(HybridPINNLSTMModel, self).__init__()
        
        self.static_input_dim = static_input_dim
        self.time_input_dim = time_input_dim
        self.time_steps = time_steps
        self.use_physics_layer = use_physics_layer
        
        # 1. PINN分支，處理靜態結構參數
        self.pinn_branch = PINNModel(
            input_dim=static_input_dim,
            hidden_dims=pinn_hidden_dims,
            output_dim=1,
            dropout_rate=dropout_rate,
            use_physics_layer=use_physics_layer
        )
        
        # 2. LSTM分支，處理時間序列資料
        self.lstm_branch = LSTMModel(
            input_dim=time_input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            output_dim=1,
            bidirectional=bidirectional,
            dropout_rate=dropout_rate,
            use_attention=use_attention
        )
        
        # 3. 特徵融合層
        # 計算PINN分支最後隱藏層維度
        pinn_feature_dim = pinn_hidden_dims[-1]
        
        # 計算LSTM分支最後隱藏層維度
        lstm_feature_dim = lstm_hidden_size * 2 if bidirectional else lstm_hidden_size
        
        self.fusion_layer = FeatureFusionLayer(
            pinn_feature_dim=pinn_feature_dim,
            lstm_feature_dim=lstm_feature_dim,
            fusion_dim=fusion_dim
        )
        
        # 4. 輸出層
        self.output_layer = nn.Linear(fusion_dim, 1)
        
        logger.info(f"初始化HybridPINNLSTMModel: "
                  f"靜態特徵維度={static_input_dim}, 時間序列特徵維度={time_input_dim}, "
                  f"時間步數={time_steps}, 融合維度={fusion_dim}")
        
        # 初始化權重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化網絡權重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, static_features, time_series, return_features=False):
        """
        前向傳播
        
        參數:
            static_features (torch.Tensor): 靜態特徵，形狀為 (batch_size, static_input_dim)
            time_series (torch.Tensor): 時間序列特徵，形狀為 (batch_size, time_steps, time_input_dim)
            return_features (bool): 是否返回各分支特徵
            
        返回:
            dict: 包含預測結果和中間特徵的字典
        """
        # 1. PINN分支前向傳播
        pinn_output = self.pinn_branch(static_features)
        pinn_nf_pred = pinn_output['nf_pred']
        pinn_delta_w = pinn_output['delta_w']
        
        # 獲取PINN分支的特徵表示
        pinn_features = self.pinn_branch.feature_extractor(static_features)
        
        # 2. LSTM分支前向傳播
        lstm_output = self.lstm_branch(time_series, return_attention=return_features)
        lstm_nf_pred = lstm_output['output']
        
        # 獲取LSTM分支的特徵表示
        lstm_features = lstm_output['last_hidden']
        
        # 3. 特徵融合
        fused_features, attention_weights = self.fusion_layer(pinn_features, lstm_features)
        
        # 4. 最終輸出
        final_output = torch.exp(self.output_layer(fused_features)).squeeze(-1)
        
        # 組織返回結果
        result = {
            'nf_pred': final_output,
            'pinn_nf_pred': pinn_nf_pred,
            'lstm_nf_pred': lstm_nf_pred,
            'fusion_weights': attention_weights
        }
        
        if self.use_physics_layer:
            result['delta_w'] = pinn_delta_w
        
        if return_features:
            result['pinn_features'] = pinn_features
            result['lstm_features'] = lstm_features
            result['fused_features'] = fused_features
            if 'attention_weights' in lstm_output:
                result['lstm_attention_weights'] = lstm_output['attention_weights']
        
        return result
    
    def calculate_loss(self, outputs, targets, lambda_physics=0.1, lambda_consistency=0.1):
        """
        計算混合損失
        
        參數:
            outputs (dict): 模型輸出
            targets (torch.Tensor): 目標疲勞壽命
            lambda_physics (float): 物理約束損失權重
            lambda_consistency (float): 分支間一致性損失權重
            
        返回:
            dict: 包含各部分損失的字典
        """
        # 主要預測損失 (MSE)
        mse_loss = F.mse_loss(outputs['nf_pred'], targets)
        
        # 物理約束損失
        if self.use_physics_layer and 'delta_w' in outputs:
            physics_loss = self.pinn_branch.calculate_physics_loss(
                outputs['delta_w'], outputs['pinn_nf_pred'], targets
            )
        else:
            physics_loss = torch.tensor(0.0, device=targets.device)
        
        # 分支間一致性損失 (各分支預測應相近)
        consistency_loss = F.mse_loss(outputs['pinn_nf_pred'], outputs['lstm_nf_pred'])
        
        # 總損失
        total_loss = mse_loss + lambda_physics * physics_loss + lambda_consistency * consistency_loss
        
        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'physics_loss': physics_loss,
            'consistency_loss': consistency_loss
        }


class PINNLSTMTrainer:
    """
    PINN-LSTM混合模型訓練器
    提供完整的模型訓練、評估和預測功能
    """
    def __init__(self, model, optimizer, device='cuda' if torch.cuda.is_available() else 'cpu',
                lambda_physics=0.1, lambda_consistency=0.1, scheduler=None):
        """
        初始化訓練器
        
        參數:
            model (HybridPINNLSTMModel): 混合模型實例
            optimizer (torch.optim.Optimizer): 優化器
            device (str): 計算設備
            lambda_physics (float): 物理約束損失權重
            lambda_consistency (float): 分支間一致性損失權重
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): 學習率調度器
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.lambda_physics = lambda_physics
        self.lambda_consistency = lambda_consistency
        self.scheduler = scheduler
        
        # 將模型轉移到指定設備
        self.model.to(self.device)
        
        logger.info(f"初始化PINNLSTMTrainer: device={device}, "
                  f"lambda_physics={lambda_physics}, lambda_consistency={lambda_consistency}")
    
    def train_epoch(self, train_loader):
        """
        訓練一個epoch
        
        參數:
            train_loader (DataLoader): 訓練資料載入器
            
        返回:
            dict: 本epoch的訓練統計
        """
        self.model.train()
        epoch_losses = {
            'total_loss': 0.0,
            'mse_loss': 0.0,
            'physics_loss': 0.0,
            'consistency_loss': 0.0
        }
        
        num_batches = 0
        
        for static_features, time_series, targets in train_loader:
            # 將資料轉移到指定設備
            static_features = static_features.to(self.device)
            time_series = time_series.to(self.device)
            targets = targets.to(self.device)
            
            # 前向傳播
            outputs = self.model(static_features, time_series)
            
            # 計算損失
            losses = self.model.calculate_loss(
                outputs, targets, 
                lambda_physics=self.lambda_physics, 
                lambda_consistency=self.lambda_consistency
            )
            
            # 反向傳播和優化
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 累計損失
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            
            num_batches += 1
        
        # 計算平均損失
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        # 學習率調度
        if self.scheduler is not None:
            self.scheduler.step()
        
        return epoch_losses
    
    def evaluate(self, data_loader):
        """
        評估模型
        
        參數:
            data_loader (DataLoader): 評估資料載入器
            
        返回:
            tuple: (評估損失字典, 預測值, 真實值)
        """
        self.model.eval()
        eval_losses = {
            'total_loss': 0.0,
            'mse_loss': 0.0,
            'physics_loss': 0.0,
            'consistency_loss': 0.0
        }
        
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for static_features, time_series, targets in data_loader:
                # 將資料轉移到指定設備
                static_features = static_features.to(self.device)
                time_series = time_series.to(self.device)
                targets = targets.to(self.device)
                
                # 前向傳播
                outputs = self.model(static_features, time_series)
                
                # 計算損失
                losses = self.model.calculate_loss(
                    outputs, targets, 
                    lambda_physics=self.lambda_physics, 
                    lambda_consistency=self.lambda_consistency
                )
                
                # 累計損失
                for key in eval_losses:
                    eval_losses[key] += losses[key].item()
                
                # 收集預測和目標
                all_predictions.append(outputs['nf_pred'].cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                
                num_batches += 1
        
        # 計算平均損失
        for key in eval_losses:
            eval_losses[key] /= num_batches
        
        # 合併批次結果
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        
        return eval_losses, all_predictions, all_targets
    
    def predict(self, static_features, time_series):
        """
        使用模型進行預測
        
        參數:
            static_features (torch.Tensor or numpy.ndarray): 靜態特徵
            time_series (torch.Tensor or numpy.ndarray): 時間序列特徵
            
        返回:
            dict: 包含預測結果的字典
        """
        self.model.eval()
        
        # 轉換輸入為Tensor (如果是numpy數組)
        if isinstance(static_features, np.ndarray):
            static_features = torch.FloatTensor(static_features)
        if isinstance(time_series, np.ndarray):
            time_series = torch.FloatTensor(time_series)
        
        # 確保輸入有批次維度
        if static_features.dim() == 1:
            static_features = static_features.unsqueeze(0)
        if time_series.dim() == 2:
            time_series = time_series.unsqueeze(0)
        
        # 將資料轉移到指定設備
        static_features = static_features.to(self.device)
        time_series = time_series.to(self.device)
        
        # 前向傳播並獲取預測結果
        with torch.no_grad():
            outputs = self.model(static_features, time_series, return_features=True)
        
        # 將Tensor轉換為numpy數組
        for key in outputs:
            if isinstance(outputs[key], torch.Tensor):
                outputs[key] = outputs[key].cpu().numpy()
        
        return outputs
    
    def save_model(self, filepath):
        """
        保存模型
        
        參數:
            filepath (str): 模型保存路徑
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lambda_physics': self.lambda_physics,
            'lambda_consistency': self.lambda_consistency,
        }, filepath)
        
        logger.info(f"模型已保存至 {filepath}")
    
    def load_model(self, filepath):
        """
        載入模型
        
        參數:
            filepath (str): 模型路徑
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lambda_physics = checkpoint.get('lambda_physics', self.lambda_physics)
        self.lambda_consistency = checkpoint.get('lambda_consistency', self.lambda_consistency)
        
        logger.info(f"模型已從 {filepath} 載入")


if __name__ == "__main__":
    # 簡單的測試代碼
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 創建一個小型混合模型進行測試
    model = HybridPINNLSTMModel(
        static_input_dim=5,      # 靜態特徵維度
        time_input_dim=2,        # 時間序列特徵維度
        time_steps=4,            # 時間步數
        pinn_hidden_dims=[32, 16],  # PINN隱藏層
        lstm_hidden_size=32,     # LSTM隱藏層大小
        lstm_num_layers=1,       # LSTM層數
        fusion_dim=16,           # 融合特徵維度
        use_physics_layer=True   # 使用物理約束
    )
    
    # 創建隨機輸入資料
    batch_size = 8
    static_features = torch.randn(batch_size, 5)  # 靜態特徵
    time_series = torch.randn(batch_size, 4, 2)   # 時間序列
    targets = torch.abs(torch.randn(batch_size))  # 目標值(疲勞壽命，正值)
    
    # 前向傳播
    outputs = model(static_features, time_series, return_features=True)
    
    # 計算損失
    losses = model.calculate_loss(outputs, targets, lambda_physics=0.1, lambda_consistency=0.1)
    
    # 檢查輸出
    logger.info(f"模型輸出:")
    logger.info(f"  最終預測疲勞壽命形狀: {outputs['nf_pred'].shape}")
    logger.info(f"  PINN分支預測形狀: {outputs['pinn_nf_pred'].shape}")
    logger.info(f"  LSTM分支預測形狀: {outputs['lstm_nf_pred'].shape}")
    logger.info(f"  融合權重形狀: {outputs['fusion_weights'].shape}")
    
    if 'delta_w' in outputs:
        logger.info(f"  應變能密度變化量形狀: {outputs['delta_w'].shape}")
    
    # 檢查損失
    logger.info(f"損失計算:")
    logger.info(f"  總損失: {losses['total_loss'].item():.4f}")
    logger.info(f"  MSE損失: {losses['mse_loss'].item():.4f}")
    logger.info(f"  物理約束損失: {losses['physics_loss'].item():.4f}")
    logger.info(f"  一致性損失: {losses['consistency_loss'].item():.4f}")
    
    # 測試訓練器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = PINNLSTMTrainer(model, optimizer, device='cpu')
    
    # 模擬資料載入器
    class DummyDataLoader:
        def __init__(self, static_features, time_series, targets, batch_size=4):
            self.static_features = static_features
            self.time_series = time_series
            self.targets = targets
            self.batch_size = batch_size
            self.n_samples = len(static_features)
            
        def __iter__(self):
            for i in range(0, self.n_samples, self.batch_size):
                end = min(i + self.batch_size, self.n_samples)
                yield (self.static_features[i:end], 
                       self.time_series[i:end], 
                       self.targets[i:end])
        
        def __len__(self):
            return (self.n_samples + self.batch_size - 1) // self.batch_size
    
    dummy_loader = DummyDataLoader(static_features, time_series, targets)
    
    # 測試訓練一個epoch
    train_losses = trainer.train_epoch(dummy_loader)
    logger.info(f"訓練損失: {train_losses}")
    
    # 測試評估
    eval_losses, predictions, targets = trainer.evaluate(dummy_loader)
    logger.info(f"評估損失: {eval_losses}")
    logger.info(f"預測值形狀: {predictions.shape}, 目標值形狀: {targets.shape}")