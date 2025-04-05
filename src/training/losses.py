#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
losses.py - 損失函數模組
本模組實現了用於訓練銲錫接點疲勞壽命預測混合模型的各種損失函數，
包括物理約束損失、一致性損失和混合損失等。

主要功能:
1. 基礎MSE損失函數，支援對數空間和相對誤差
2. 物理約束損失，基於銲錫接點疲勞壽命的物理模型
3. 分支一致性損失，平衡PINN和LSTM分支的預測
4. 混合損失函數，整合上述所有損失，並支援權重調整
5. 自適應損失函數，根據訓練進度自動調整損失權重
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import math

logger = logging.getLogger(__name__)

# 物理模型常數 - 基於銲錫接點疲勞壽命模型: Nf = a * (ΔW)^b
A_COEFFICIENT = 55.83  # 係數 a
B_COEFFICIENT = -2.259  # 係數 b (負值表示反比關係)


class MSELoss(nn.Module):
    """
    均方誤差損失函數
    支援對數空間和相對誤差
    """
    def __init__(self, reduction='mean', log_space=False, relative_error_weight=0.0):
        """
        初始化均方誤差損失
        
        參數:
            reduction (str): 誤差匯總方式，可選 'mean', 'sum', 'none'
            log_space (bool): 是否在對數空間計算損失
            relative_error_weight (float): 相對誤差權重，0表示純MSE，1表示純相對誤差
        """
        super(MSELoss, self).__init__()
        self.reduction = reduction
        self.log_space = log_space
        self.relative_error_weight = relative_error_weight
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
        # 確保輸入為正值（對於對數變換）
        if self.log_space:
            pred_safe = torch.clamp(pred, min=1e-8)
            target_safe = torch.clamp(target, min=1e-8)
            
            # 對數空間MSE
            log_pred = torch.log(pred_safe)
            log_target = torch.log(target_safe)
            loss = self.mse(log_pred, log_target)
        else:
            # 常規MSE
            loss = self.mse(pred, target)
        
        # 如果需要，添加相對誤差成分
        if self.relative_error_weight > 0:
            # 計算相對誤差 |pred - target| / (|target| + epsilon)
            epsilon = 1e-8
            relative_error = torch.abs(pred - target) / (torch.abs(target) + epsilon)
            
            if self.reduction == 'mean':
                relative_loss = torch.mean(relative_error ** 2)
            elif self.reduction == 'sum':
                relative_loss = torch.sum(relative_error ** 2)
            else:  # 'none'
                relative_loss = relative_error ** 2
            
            # 加權組合
            loss = (1 - self.relative_error_weight) * loss + self.relative_error_weight * relative_loss
        
        return loss


class PhysicsConstraintLoss(nn.Module):
    """
    物理約束損失函數
    基於銲錫接點疲勞壽命的物理模型: Nf = a * (ΔW)^b
    """
    def __init__(self, a=A_COEFFICIENT, b=B_COEFFICIENT, reduction='mean', 
                 micro_weight=1.0, macro_weight=0.5):
        """
        初始化物理約束損失
        
        參數:
            a (float): 物理模型係數 a
            b (float): 物理模型係數 b
            reduction (str): 誤差匯總方式
            micro_weight (float): 微觀物理約束權重
            macro_weight (float): 宏觀物理約束權重
        """
        super(PhysicsConstraintLoss, self).__init__()
        self.a = a
        self.b = b
        self.reduction = reduction
        self.micro_weight = micro_weight
        self.macro_weight = macro_weight
    
    def forward(self, delta_w, nf_pred, nf_true):
        """
        計算物理約束損失
        
        參數:
            delta_w (torch.Tensor): 預測的非線性塑性應變能密度變化量
            nf_pred (torch.Tensor): 預測的疲勞壽命
            nf_true (torch.Tensor): 真實的疲勞壽命
            
        返回:
            dict: 包含各部分物理損失的字典
        """
        # 確保輸入為正值
        delta_w = torch.clamp(delta_w, min=1e-8)
        nf_pred = torch.clamp(nf_pred, min=1e-8)
        nf_true = torch.clamp(nf_true, min=1e-8)
        
        # 1. 微觀物理約束 - delta_w與理論值的一致性
        # 從真實壽命反推理論上的delta_w
        delta_w_theory = torch.pow(nf_true / self.a, 1/self.b)
        delta_w_theory = torch.clamp(delta_w_theory, min=1e-8)
        
        # 微觀損失: 預測的delta_w應與理論值一致
        if delta_w.dim() != delta_w_theory.dim():
            if delta_w.dim() > delta_w_theory.dim():
                delta_w_theory = delta_w_theory.view(delta_w.size())
            else:
                delta_w = delta_w.view(delta_w_theory.size())
        micro_loss = F.mse_loss(delta_w, delta_w_theory, reduction=self.reduction)

        
        
        # 2. 宏觀物理約束 - 預測值應符合物理模型
        # 從delta_w計算理論壽命
        nf_physics = self.a * torch.pow(delta_w, self.b)
        
        # 宏觀損失: 預測的nf應符合物理模型
        if nf_pred.dim() != nf_physics.dim():
            if nf_pred.dim() > nf_physics.dim():
                nf_physics = nf_physics.view(nf_pred.size())
        else:
            nf_pred = nf_pred.view(nf_physics.size())
        macro_loss = F.mse_loss(nf_pred, nf_physics, reduction=self.reduction)
        
        # 總物理約束損失
        physics_loss = self.micro_weight * micro_loss + self.macro_weight * macro_loss
        
        # 返回各部分損失
        return {
            'physics_loss': physics_loss,
            'micro_loss': micro_loss,
            'macro_loss': macro_loss
        }


class ConsistencyLoss(nn.Module):
    """
    一致性損失函數
    確保模型不同分支的預測結果保持一致
    """
    def __init__(self, reduction='mean', log_space=True, correlation_weight=0.3):
        """
        初始化一致性損失
        
        參數:
            reduction (str): 誤差匯總方式
            log_space (bool): 是否在對數空間計算一致性
            correlation_weight (float): 相關性約束權重
        """
        super(ConsistencyLoss, self).__init__()
        self.reduction = reduction
        self.log_space = log_space
        self.correlation_weight = correlation_weight
    
    def forward(self, pinn_pred, lstm_pred):
        """
        計算一致性損失
        
        參數:
            pinn_pred (torch.Tensor): PINN分支預測值
            lstm_pred (torch.Tensor): LSTM分支預測值
            
        返回:
            dict: 包含各部分一致性損失的字典
        """
        # 確保輸入為正值
        pinn_pred = torch.clamp(pinn_pred, min=1e-8)
        lstm_pred = torch.clamp(lstm_pred, min=1e-8)
        
        # 1. 基本一致性損失 - MSE或對數空間MSE
        if self.log_space:
            # 對數空間一致性
            log_pinn = torch.log(pinn_pred)
            log_lstm = torch.log(lstm_pred)
            basic_loss = F.mse_loss(log_pinn, log_lstm, reduction=self.reduction)
        else:
            # 常規一致性
            basic_loss = F.mse_loss(pinn_pred, lstm_pred, reduction=self.reduction)
        
        # 2. 相關性損失 - 確保預測趨勢一致
        correlation_loss = torch.tensor(0.0, device=pinn_pred.device)
        if self.correlation_weight > 0 and pinn_pred.size(0) > 2:
            # 標準化預測值
            pinn_norm = (pinn_pred - pinn_pred.mean()) / (pinn_pred.std() + 1e-8)
            lstm_norm = (lstm_pred - lstm_pred.mean()) / (lstm_pred.std() + 1e-8)
            
            # 計算相關性，確保兩個分支預測趨勢一致
            corr = torch.sum(pinn_norm * lstm_norm) / pinn_pred.size(0)
            correlation_loss = 1.0 - corr  # 相關性越高，損失越低
        
        # 總一致性損失
        consistency_loss = basic_loss + self.correlation_weight * correlation_loss
        
        # 返回各部分損失
        return {
            'consistency_loss': consistency_loss,
            'basic_loss': basic_loss,
            'correlation_loss': correlation_loss
        }


class HybridLoss(nn.Module):
    """
    混合損失函數
    結合MSE損失、物理約束損失和一致性損失
    """
    def __init__(self, lambda_physics=0.1, lambda_consistency=0.1, 
                 a=A_COEFFICIENT, b=B_COEFFICIENT, reduction='mean', log_space=True,
                 relative_error_weight=0.3, micro_weight=1.0, macro_weight=0.5,
                 correlation_weight=0.3, l1_reg=0.0, l2_reg=0.0):
        """
        初始化混合損失
        
        參數:
            lambda_physics (float): 物理約束損失權重
            lambda_consistency (float): 一致性損失權重
            a (float): 物理模型係數 a
            b (float): 物理模型係數 b
            reduction (str): 誤差匯總方式
            log_space (bool): 是否在對數空間計算損失
            relative_error_weight (float): 相對誤差權重
            micro_weight (float): 微觀物理約束權重
            macro_weight (float): 宏觀物理約束權重
            correlation_weight (float): 相關性約束權重
            l1_reg (float): L1正則化係數
            l2_reg (float): L2正則化係數
        """
        super(HybridLoss, self).__init__()
        self.lambda_physics = lambda_physics
        self.lambda_consistency = lambda_consistency
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        
        # 初始化子損失函數
        self.mse_loss = MSELoss(
            reduction=reduction, 
            log_space=log_space, 
            relative_error_weight=relative_error_weight
        )
        
        self.physics_loss = PhysicsConstraintLoss(
            a=a, 
            b=b, 
            reduction=reduction, 
            micro_weight=micro_weight, 
            macro_weight=macro_weight
        )
        
        self.consistency_loss = ConsistencyLoss(
            reduction=reduction, 
            log_space=log_space, 
            correlation_weight=correlation_weight
        )
        
        logger.info(f"初始化HybridLoss: lambda_physics={lambda_physics}, "
                  f"lambda_consistency={lambda_consistency}, a={a}, b={b}, "
                  f"log_space={log_space}, l1_reg={l1_reg}, l2_reg={l2_reg}")
    
    def forward(self, outputs, targets, model=None):
        """
        計算混合損失
        
        參數:
            outputs (dict): 模型輸出，包含多個預測結果
                - 'nf_pred': 最終預測的疲勞壽命
                - 'pinn_nf_pred': PINN分支預測的疲勞壽命
                - 'lstm_nf_pred': LSTM分支預測的疲勞壽命
                - 'delta_w': 預測的非線性塑性應變能密度變化量
            targets (torch.Tensor): 目標疲勞壽命
            model (torch.nn.Module, optional): 模型，用於計算正則化損失
            
        返回:
            dict: 包含各部分損失和總損失的字典
        """
        # 計算主要預測損失 (MSE)
        pred_loss = self.mse_loss(outputs['nf_pred'], targets)
        
        # 計算物理約束損失
        if 'delta_w' in outputs and 'pinn_nf_pred' in outputs:
            physics_results = self.physics_loss(
                outputs['delta_w'], outputs['pinn_nf_pred'], targets
            )
            physics_loss = physics_results['physics_loss']
        else:
            physics_loss = torch.tensor(0.0, device=targets.device)
            physics_results = {
                'micro_loss': torch.tensor(0.0, device=targets.device),
                'macro_loss': torch.tensor(0.0, device=targets.device)
            }
        
        # 計算分支間一致性損失
        if 'pinn_nf_pred' in outputs and 'lstm_nf_pred' in outputs:
            consistency_results = self.consistency_loss(
                outputs['pinn_nf_pred'], outputs['lstm_nf_pred']
            )
            consistency_loss = consistency_results['consistency_loss']
        else:
            consistency_loss = torch.tensor(0.0, device=targets.device)
            consistency_results = {
                'basic_loss': torch.tensor(0.0, device=targets.device),
                'correlation_loss': torch.tensor(0.0, device=targets.device)
            }
        
        # 計算正則化損失
        reg_loss = torch.tensor(0.0, device=targets.device)
        if (self.l1_reg > 0 or self.l2_reg > 0) and model is not None:
            l1_term = torch.tensor(0.0, device=targets.device)
            l2_term = torch.tensor(0.0, device=targets.device)
            
            for param in model.parameters():
                if self.l1_reg > 0:
                    l1_term += torch.sum(torch.abs(param))
                if self.l2_reg > 0:
                    l2_term += torch.sum(param ** 2)
            
            reg_loss = self.l1_reg * l1_term + self.l2_reg * l2_term
        
        # 計算總損失
        total_loss = (
            pred_loss + 
            self.lambda_physics * physics_loss + 
            self.lambda_consistency * consistency_loss +
            reg_loss
        )
        
        # 返回損失字典
        result = {
            'total_loss': total_loss,
            'pred_loss': pred_loss,
            'physics_loss': physics_loss,
            'consistency_loss': consistency_loss,
            'reg_loss': reg_loss
        }
        
        # 添加物理約束細節
        for key, value in physics_results.items():
            result[key] = value
        
        # 添加一致性細節
        for key, value in consistency_results.items():
            result[key] = value
        
        return result
    
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
                 epochs_to_max=50, warmup_epochs=5, 
                 a=A_COEFFICIENT, b=B_COEFFICIENT, reduction='mean', log_space=True,
                 relative_error_weight=0.3, micro_weight=1.0, macro_weight=0.5,
                 correlation_weight=0.3, l1_reg=0.0, l2_reg=0.0, adaptive_scheme='linear'):
        """
        初始化自適應混合損失
        
        參數:
            initial_lambda_physics (float): 初始物理約束損失權重
            max_lambda_physics (float): 最大物理約束損失權重
            initial_lambda_consistency (float): 初始一致性損失權重
            max_lambda_consistency (float): 最大一致性損失權重
            epochs_to_max (int): 達到最大權重的訓練輪數
            warmup_epochs (int): 預熱輪數，權重保持較低
            a (float): 物理模型係數 a
            b (float): 物理模型係數 b
            reduction (str): 誤差匯總方式
            log_space (bool): 是否在對數空間計算損失
            relative_error_weight (float): 相對誤差權重
            micro_weight (float): 微觀物理約束權重
            macro_weight (float): 宏觀物理約束權重
            correlation_weight (float): 相關性約束權重
            l1_reg (float): L1正則化係數
            l2_reg (float): L2正則化係數
            adaptive_scheme (str): 權重調整方案，'linear', 'exp', 'step', 'cosine'
        """
        super(AdaptiveHybridLoss, self).__init__(
            lambda_physics=initial_lambda_physics,
            lambda_consistency=initial_lambda_consistency,
            a=a, b=b, reduction=reduction, log_space=log_space,
            relative_error_weight=relative_error_weight,
            micro_weight=micro_weight, macro_weight=macro_weight,
            correlation_weight=correlation_weight, l1_reg=l1_reg, l2_reg=l2_reg
        )
        
        self.initial_lambda_physics = initial_lambda_physics
        self.max_lambda_physics = max_lambda_physics
        self.initial_lambda_consistency = initial_lambda_consistency
        self.max_lambda_consistency = max_lambda_consistency
        self.epochs_to_max = epochs_to_max
        self.warmup_epochs = warmup_epochs
        self.adaptive_scheme = adaptive_scheme
        self.current_epoch = 0
        
        logger.info(f"初始化AdaptiveHybridLoss: "
                  f"physics權重從{initial_lambda_physics}增加到{max_lambda_physics}, "
                  f"consistency權重從{initial_lambda_consistency}增加到{max_lambda_consistency}, "
                  f"在{epochs_to_max}個輪次內達到最大值, "
                  f"預熱輪次: {warmup_epochs}, 調整方案: {adaptive_scheme}")
    
    def update_epoch(self, epoch):
        """
        更新當前訓練輪次並調整損失權重
        
        參數:
            epoch (int): 當前訓練輪次
        """
        self.current_epoch = epoch
        
        # 計算調整因子
        if epoch < self.warmup_epochs:
            # 預熱期保持初始權重
            factor = 0.0
        else:
            # 按比例調整
            effective_epoch = epoch - self.warmup_epochs
            effective_max = self.epochs_to_max - self.warmup_epochs
            
            if effective_epoch >= effective_max:
                factor = 1.0
            else:
                if self.adaptive_scheme == 'linear':
                    # 線性增長
                    factor = effective_epoch / effective_max
                elif self.adaptive_scheme == 'exp':
                    # 指數增長
                    factor = 1.0 - math.exp(-5 * effective_epoch / effective_max)
                elif self.adaptive_scheme == 'step':
                    # 階梯式增長
                    steps = 4
                    factor = min(1.0, math.ceil(steps * effective_epoch / effective_max) / steps)
                elif self.adaptive_scheme == 'cosine':
                    # 餘弦退火增長
                    factor = 0.5 * (1 - math.cos(math.pi * effective_epoch / effective_max))
                else:
                    # 默認線性
                    factor = effective_epoch / effective_max
        
        # 計算當前權重
        current_lambda_physics = self.initial_lambda_physics + (
            self.max_lambda_physics - self.initial_lambda_physics) * factor
        current_lambda_consistency = self.initial_lambda_consistency + (
            self.max_lambda_consistency - self.initial_lambda_consistency) * factor
        
        # 更新權重
        self.update_lambda(current_lambda_physics, current_lambda_consistency)


def get_loss_function(loss_type='hybrid', **kwargs):
    """
    獲取指定類型的損失函數
    
    參數:
        loss_type (str): 損失函數類型，可選 'mse', 'physics', 'consistency', 
                        'hybrid', 'adaptive'
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
    
    # 測試混合損失
    hybrid_loss = HybridLoss(lambda_physics=0.2, lambda_consistency=0.1, log_space=True)
    hybrid_results = hybrid_loss(outputs, targets)
    
    print(f"混合損失: {hybrid_results['total_loss'].item():.4f}")
    print(f"  - 預測損失: {hybrid_results['pred_loss'].item():.4f}")
    print(f"  - 物理約束損失: {hybrid_results['physics_loss'].item():.4f}")
    print(f"  - 一致性損失: {hybrid_results['consistency_loss'].item():.4f}")
    
    # 測試自適應混合損失
    adaptive_loss = AdaptiveHybridLoss(
        initial_lambda_physics=0.01, 
        max_lambda_physics=0.5,
        initial_lambda_consistency=0.01,
        max_lambda_consistency=0.3,
        epochs_to_max=50,
        adaptive_scheme='cosine'
    )
    
    print("\n測試自適應混合損失:")
    for epoch in range(0, 61, 20):
        adaptive_loss.update_epoch(epoch)
        adaptive_result = adaptive_loss(outputs, targets)
        print(f"輪次 {epoch}: 總損失={adaptive_result['total_loss'].item():.4f}, "
             f"物理權重={adaptive_loss.lambda_physics:.4f}, "
             f"一致性權重={adaptive_loss.lambda_consistency:.4f}")

# 在檔案底部或者 PhysicsConstraintLoss 類別定義後添加
# 這樣可以保持原有功能，同時兼容__init__.py中的引用
EnhancedPhysicsConstraintLoss = PhysicsConstraintLoss
EnhancedConsistencyLoss = ConsistencyLoss  
EnhancedHybridLoss = HybridLoss