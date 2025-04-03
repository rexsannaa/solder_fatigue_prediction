#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
losses.py - 損失函數模組
本模組實現了用於訓練銲錫接點疲勞壽命預測混合模型的各種損失函數，
包括標準損失函數、物理約束損失和混合損失等。

主要組件:
1. MSELoss - 均方誤差損失函數
2. PhysicsConstraintLoss - 物理約束損失函數
3. ConsistencyLoss - 模型分支間一致性損失函數
4. HybridLoss - 結合多種損失函數的混合損失函數
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MSELoss(nn.Module):
    """
    均方誤差損失函數
    計算預測值與真實值之間的均方誤差
    """
    def __init__(self, reduction='mean'):
        """
        初始化均方誤差損失
        
        參數:
            reduction (str): 誤差匯總方式，可選 'mean', 'sum', 'none'
        """
        super(MSELoss, self).__init__()
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction=reduction)
    
    def forward(self, pred, target):
        """
        計算均方誤差損失
        
        參數:
            pred (torch.Tensor): 預測值
            target (torch.Tensor): 目標值
            
        返回:
            torch.Tensor: 損失值
        """
        return self.mse(pred, target)


class PhysicsConstraintLoss(nn.Module):
    """
    物理約束損失函數
    基於能量密度法疲勞壽命模型的物理約束損失
    Nf=55.83⋅(ΔW)^(-2.259)
    """
    def __init__(self, a=55.83, b=-2.259, reduction='mean'):
        """
        初始化物理約束損失
        
        參數:
            a (float): 物理模型係數 a
            b (float): 物理模型係數 b
            reduction (str): 誤差匯總方式
        """
        super(PhysicsConstraintLoss, self).__init__()
        self.a = a
        self.b = b
        self.reduction = reduction
    
    def forward(self, delta_w, nf_pred, nf_true):
        """
        計算物理約束損失
        
        參數:
            delta_w (torch.Tensor): 預測的非線性塑性應變能密度變化量
            nf_pred (torch.Tensor): 預測的疲勞壽命
            nf_true (torch.Tensor): 真實的疲勞壽命
            
        返回:
            torch.Tensor: 物理約束損失
        """
        # 確保輸入為正值
        delta_w = torch.clamp(delta_w, min=1e-6)
        
        # 從物理模型計算疲勞壽命
        nf_physics = self.a * torch.pow(delta_w, self.b)
        
        # 從真實壽命反推理論上的delta_w
        delta_w_theory = torch.pow(nf_true / self.a, 1/self.b)
        delta_w_theory = torch.clamp(delta_w_theory, min=1e-6)
        
        # 計算三個物理約束損失
        # 1. 預測的delta_w應與理論值接近
        delta_w_loss = F.mse_loss(delta_w, delta_w_theory, reduction=self.reduction)
        
        # 2. 預測的nf應符合物理模型
        energy_loss = F.mse_loss(nf_pred, nf_physics, reduction=self.reduction)
        
        # 3. 確保物理守恆: 應變能量與壽命的反比關係
        conservation_loss = F.mse_loss(delta_w * torch.pow(nf_pred, 1/self.b), 
                                      torch.ones_like(nf_pred) * torch.pow(self.a, 1/self.b),
                                      reduction=self.reduction)
        
        # 綜合物理損失
        physics_loss = delta_w_loss + energy_loss + 0.1 * conservation_loss
        
        return physics_loss


class ConsistencyLoss(nn.Module):
    """
    一致性損失函數
    確保模型不同分支的預測結果保持一致
    """
    def __init__(self, reduction='mean'):
        """
        初始化一致性損失
        
        參數:
            reduction (str): 誤差匯總方式
        """
        super(ConsistencyLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, pinn_pred, lstm_pred):
        """
        計算一致性損失
        
        參數:
            pinn_pred (torch.Tensor): PINN分支預測值
            lstm_pred (torch.Tensor): LSTM分支預測值
            
        返回:
            torch.Tensor: 一致性損失
        """
        # 計算PINN和LSTM分支預測結果的一致性損失
        consistency_loss = F.mse_loss(pinn_pred, lstm_pred, reduction=self.reduction)
        
        # 對於小樣本數據集，還可以添加額外的約束
        # 例如確保兩個分支的預測趨勢一致（相關性約束）
        if pinn_pred.size(0) > 2:  # 至少需要3個樣本才能計算相關性
            # 標準化預測值
            pinn_norm = (pinn_pred - pinn_pred.mean()) / (pinn_pred.std() + 1e-8)
            lstm_norm = (lstm_pred - lstm_pred.mean()) / (lstm_pred.std() + 1e-8)
            
            # 計算相關性，確保兩個分支預測趨勢一致
            corr = torch.sum(pinn_norm * lstm_norm) / pinn_pred.size(0)
            correlation_loss = 1.0 - corr  # 相關性越高，損失越低
            
            # 結合MSE和相關性損失
            consistency_loss = consistency_loss + 0.2 * correlation_loss
        
        return consistency_loss


class HybridLoss(nn.Module):
    """
    混合損失函數
    結合MSE損失、物理約束損失和一致性損失
    """
    def __init__(self, lambda_physics=0.1, lambda_consistency=0.1, 
                 a=55.83, b=-2.259, reduction='mean'):
        """
        初始化混合損失
        
        參數:
            lambda_physics (float): 物理約束損失權重
            lambda_consistency (float): 一致性損失權重
            a (float): 物理模型係數 a
            b (float): 物理模型係數 b
            reduction (str): 誤差匯總方式
        """
        super(HybridLoss, self).__init__()
        self.lambda_physics = lambda_physics
        self.lambda_consistency = lambda_consistency
        
        self.mse_loss = MSELoss(reduction=reduction)
        self.physics_loss = PhysicsConstraintLoss(a=a, b=b, reduction=reduction)
        self.consistency_loss = ConsistencyLoss(reduction=reduction)
        
        logger.info(f"初始化HybridLoss: lambda_physics={lambda_physics}, "
                  f"lambda_consistency={lambda_consistency}, a={a}, b={b}")
    
    def forward(self, outputs, targets):
        """
        計算混合損失
        
        參數:
            outputs (dict): 模型輸出，包含多個預測結果
                - 'nf_pred': 最終預測的疲勞壽命
                - 'pinn_nf_pred': PINN分支預測的疲勞壽命
                - 'lstm_nf_pred': LSTM分支預測的疲勞壽命
                - 'delta_w': 預測的非線性塑性應變能密度變化量
            targets (torch.Tensor): 目標疲勞壽命
            
        返回:
            dict: 包含各部分損失和總損失的字典
        """
        # 計算主要預測損失 (MSE)
        mse_loss = self.mse_loss(outputs['nf_pred'], targets)
        
        # 計算物理約束損失
        if 'delta_w' in outputs:
            physics_loss = self.physics_loss(
                outputs['delta_w'], outputs['pinn_nf_pred'], targets
            )
        else:
            physics_loss = torch.tensor(0.0, device=targets.device)
        
        # 計算分支間一致性損失
        consistency_loss = self.consistency_loss(
            outputs['pinn_nf_pred'], outputs['lstm_nf_pred']
        )
        
        # 計算總損失
        total_loss = mse_loss + self.lambda_physics * physics_loss + self.lambda_consistency * consistency_loss
        
        # 返回損失字典
        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'physics_loss': physics_loss,
            'consistency_loss': consistency_loss
        }
    
    def update_lambda(self, lambda_physics=None, lambda_consistency=None):
        """
        更新損失權重
        
        參數:
            lambda_physics (float, optional): 新的物理約束損失權重
            lambda_consistency (float, optional): 新的一致性損失權重
        """
        if lambda_physics is not None:
            self.lambda_physics = lambda_physics
            logger.info(f"更新物理約束損失權重為: {lambda_physics}")
        
        if lambda_consistency is not None:
            self.lambda_consistency = lambda_consistency
            logger.info(f"更新一致性損失權重為: {lambda_consistency}")


class AdaptiveHybridLoss(HybridLoss):
    """
    自適應混合損失函數
    根據訓練進度自動調整損失權重
    """
    def __init__(self, initial_lambda_physics=0.01, max_lambda_physics=0.5,
                 initial_lambda_consistency=0.01, max_lambda_consistency=0.3,
                 epochs_to_max=50, a=55.83, b=-2.259, reduction='mean'):
        """
        初始化自適應混合損失
        
        參數:
            initial_lambda_physics (float): 初始物理約束損失權重
            max_lambda_physics (float): 最大物理約束損失權重
            initial_lambda_consistency (float): 初始一致性損失權重
            max_lambda_consistency (float): 最大一致性損失權重
            epochs_to_max (int): 達到最大權重的訓練輪數
            a (float): 物理模型係數 a
            b (float): 物理模型係數 b
            reduction (str): 誤差匯總方式
        """
        super(AdaptiveHybridLoss, self).__init__(
            lambda_physics=initial_lambda_physics,
            lambda_consistency=initial_lambda_consistency,
            a=a, b=b, reduction=reduction
        )
        
        self.initial_lambda_physics = initial_lambda_physics
        self.max_lambda_physics = max_lambda_physics
        self.initial_lambda_consistency = initial_lambda_consistency
        self.max_lambda_consistency = max_lambda_consistency
        self.epochs_to_max = epochs_to_max
        self.current_epoch = 0
        
        logger.info(f"初始化AdaptiveHybridLoss: "
                  f"physics權重從{initial_lambda_physics}增加到{max_lambda_physics}, "
                  f"consistency權重從{initial_lambda_consistency}增加到{max_lambda_consistency}, "
                  f"在{epochs_to_max}個輪次內達到最大值")
    
    def update_epoch(self, epoch):
        """
        更新當前訓練輪次並調整損失權重
        
        參數:
            epoch (int): 當前訓練輪次
        """
        self.current_epoch = epoch
        
        # 計算當前權重
        progress = min(epoch / self.epochs_to_max, 1.0)
        current_lambda_physics = self.initial_lambda_physics + (
            self.max_lambda_physics - self.initial_lambda_physics) * progress
        current_lambda_consistency = self.initial_lambda_consistency + (
            self.max_lambda_consistency - self.initial_lambda_consistency) * progress
        
        # 更新權重
        self.update_lambda(current_lambda_physics, current_lambda_consistency)
        
        logger.debug(f"輪次 {epoch}: 物理損失權重={self.lambda_physics:.4f}, "
                   f"一致性損失權重={self.lambda_consistency:.4f}")


# 實用工具函數
def get_loss_function(loss_type='hybrid', **kwargs):
    """
    獲取指定類型的損失函數
    
    參數:
        loss_type (str): 損失函數類型，可選 'mse', 'physics', 'consistency', 'hybrid', 'adaptive'
        **kwargs: 傳遞給損失函數的額外參數
    
    返回:
        nn.Module: 指定類型的損失函數實例
    """
    if loss_type.lower() == 'mse':
        return MSELoss(**kwargs)
    elif loss_type.lower() == 'physics':
        return PhysicsConstraintLoss(**kwargs)
    elif loss_type.lower() == 'consistency':
        return ConsistencyLoss(**kwargs)
    elif loss_type.lower() == 'hybrid':
        return HybridLoss(**kwargs)
    elif loss_type.lower() == 'adaptive':
        return AdaptiveHybridLoss(**kwargs)
    else:
        raise ValueError(f"不支援的損失函數類型: {loss_type}")


if __name__ == "__main__":
    # 簡單的測試代碼
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 創建隨機測試數據
    batch_size = 8
    device = torch.device('cpu')
    
    # 模擬模型輸出
    outputs = {
        'nf_pred': torch.abs(torch.randn(batch_size, device=device)),  # 最終預測
        'pinn_nf_pred': torch.abs(torch.randn(batch_size, device=device)),  # PINN分支預測
        'lstm_nf_pred': torch.abs(torch.randn(batch_size, device=device)),  # LSTM分支預測
        'delta_w': torch.abs(torch.randn(batch_size, device=device)),  # 應變能密度變化量
    }
    
    # 模擬目標值
    targets = torch.abs(torch.randn(batch_size, device=device))
    
    # 測試各種損失函數
    mse_loss = MSELoss()
    physics_loss = PhysicsConstraintLoss()
    consistency_loss = ConsistencyLoss()
    hybrid_loss = HybridLoss(lambda_physics=0.2, lambda_consistency=0.1)
    adaptive_loss = AdaptiveHybridLoss()
    
    # 計算損失
    mse_result = mse_loss(outputs['nf_pred'], targets)
    physics_result = physics_loss(outputs['delta_w'], outputs['pinn_nf_pred'], targets)
    consistency_result = consistency_loss(outputs['pinn_nf_pred'], outputs['lstm_nf_pred'])
    hybrid_result = hybrid_loss(outputs, targets)
    
    logger.info(f"MSE損失: {mse_result.item():.4f}")
    logger.info(f"物理約束損失: {physics_result.item():.4f}")
    logger.info(f"一致性損失: {consistency_result.item():.4f}")
    logger.info(f"混合損失: {hybrid_result['total_loss'].item():.4f}")
    logger.info(f"  - MSE部分: {hybrid_result['mse_loss'].item():.4f}")
    logger.info(f"  - 物理約束部分: {hybrid_result['physics_loss'].item():.4f}")
    logger.info(f"  - 一致性部分: {hybrid_result['consistency_loss'].item():.4f}")
    
    # 測試自適應損失
    logger.info("\n測試自適應損失:")
    for epoch in range(0, 61, 20):
        adaptive_loss.update_epoch(epoch)
        adaptive_result = adaptive_loss(outputs, targets)
        logger.info(f"輪次 {epoch}: 總損失={adaptive_result['total_loss'].item():.4f}, "
                  f"物理權重={adaptive_loss.lambda_physics:.4f}, "
                  f"一致性權重={adaptive_loss.lambda_consistency:.4f}")