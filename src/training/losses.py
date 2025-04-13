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
        # 確保 pred 和 target 具有相同的形狀以避免廣播問題
        if pred.dim() != target.dim() or pred.shape != target.shape:
            if pred.dim() < target.dim():
                pred = pred.view(target.shape)
            elif target.dim() < pred.dim():
                target = target.view(pred.shape)
    
        # 對數變換處理
        if self.log_space:
            pred_safe = torch.clamp(pred, min=1e-8)
            target_safe = torch.clamp(target, min=1e-8)
            log_pred = torch.log(pred_safe)
            log_target = torch.log(target_safe)
            loss = self.mse(log_pred, log_target)
        else:
            loss = self.mse(pred, target)
    
        if self.relative_error_weight > 0:
            epsilon = 1e-8
            relative_error = torch.abs(pred - target) / (torch.abs(target) + epsilon)
        
            if self.reduction == 'mean':
                relative_loss = torch.mean(relative_error ** 2)
            elif self.reduction == 'sum':
                relative_loss = torch.sum(relative_error ** 2)
            else:
                relative_loss = relative_error ** 2
        
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
        delta_w = torch.clamp(delta_w, min=1e-8)
        nf_pred = torch.clamp(nf_pred, min=1e-8)
        nf_true = torch.clamp(nf_true, min=1e-8)
        
        if delta_w.dim() > 2:
            delta_w = delta_w.view(delta_w.size(0), -1)
        if nf_pred.dim() > 2:
            nf_pred = nf_pred.view(nf_pred.size(0), -1)
        if nf_true.dim() > 1:
            nf_true = nf_true.view(-1)
        
        if delta_w.dim() == 2 and delta_w.size(1) > 1:
            delta_w = delta_w[:, 0].unsqueeze(1)
        if nf_pred.dim() == 2 and nf_pred.size(1) > 1:
            nf_pred = nf_pred[:, 0].unsqueeze(1)

        delta_w_theory = torch.pow(nf_true / self.a, 1/self.b)
        delta_w_theory = torch.clamp(delta_w_theory, min=1e-8)
        
        if delta_w.dim() > delta_w_theory.dim():
            delta_w_theory = delta_w_theory.unsqueeze(-1)
        
        micro_loss = F.mse_loss(delta_w.squeeze(), delta_w_theory.squeeze(), reduction=self.reduction)
        nf_physics = self.a * torch.pow(delta_w, self.b)
        if nf_pred.dim() != nf_physics.dim():
            nf_physics = nf_physics.view_as(nf_pred)
        macro_loss = F.mse_loss(nf_pred.squeeze(), nf_physics.squeeze(), reduction=self.reduction)
        physics_loss = self.micro_weight * micro_loss + self.macro_weight * macro_loss
        
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
        pinn_pred = torch.clamp(pinn_pred, min=1e-8)
        lstm_pred = torch.clamp(lstm_pred, min=1e-8)
    
        if pinn_pred.dim() != lstm_pred.dim() or pinn_pred.shape != lstm_pred.shape:
            if pinn_pred.dim() < lstm_pred.dim():
                pinn_pred = pinn_pred.view(lstm_pred.shape)
            elif lstm_pred.dim() < pinn_pred.dim():
                lstm_pred = lstm_pred.view(pinn_pred.shape)
    
        if self.log_space:
            log_pinn = torch.log(pinn_pred)
            log_lstm = torch.log(lstm_pred)
            basic_loss = F.mse_loss(log_pinn, log_lstm, reduction=self.reduction)
        else:
            basic_loss = F.mse_loss(pinn_pred, lstm_pred, reduction=self.reduction)
    
        correlation_loss = torch.tensor(0.0, device=pinn_pred.device)
        if self.correlation_weight > 0 and pinn_pred.size(0) > 2:
            try:
                pinn_flat = pinn_pred.view(-1)
                lstm_flat = lstm_pred.view(-1)
                pinn_mean = pinn_flat.mean()
                pinn_std = pinn_flat.std() + 1e-8
                lstm_mean = lstm_flat.mean()
                lstm_std = lstm_flat.std() + 1e-8
                pinn_norm = (pinn_flat - pinn_mean) / pinn_std
                lstm_norm = (lstm_flat - lstm_mean) / lstm_std
                corr = torch.sum(pinn_norm * lstm_norm) / pinn_flat.size(0)
                correlation_loss = 1.0 - corr
            except Exception as e:
                logger.warning(f"計算相關性損失時出錯: {str(e)}")
        consistency_loss = basic_loss + self.correlation_weight * correlation_loss
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
                 correlation_weight=0.3, l1_reg=0.0, l2_reg=0.0,
                 delta_w_weight=1.5):
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
            delta_w_weight (float): delta_w預測損失權重 - 新增參數
        """
        super(HybridLoss, self).__init__()
        self.lambda_physics = lambda_physics
        self.lambda_consistency = lambda_consistency
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.delta_w_weight = delta_w_weight
        
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
                    f"log_space={log_space}, l1_reg={l1_reg}, l2_reg={l2_reg}, "
                    f"delta_w_weight={delta_w_weight}")
    
    def forward(self, outputs, targets, model=None):
        """
        計算混合損失 - 修改版：重點關注delta_w的預測精度
    
        參數:
            outputs (dict): 模型輸出，包含多個預測結果
                - 'nf_pred': 最終預測的疲勞壽命
                - 'pinn_nf_pred': PINN分支預測的疲勞壽命
                - 'lstm_nf_pred': LSTM分支預測的疲勞壽命
                - 'delta_w': 預測的非線性塑性應變能密度變化量
                - 'pinn_delta_w': PINN分支預測的delta_w  # 新增
                - 'lstm_delta_w': LSTM分支預測的delta_w  # 新增
            targets (torch.Tensor): 目標疲勞壽命
            model (torch.nn.Module, optional): 模型，用於計算正則化損失
        
        返回:
            dict: 包含各部分損失和總損失的字典
        """
        nf_pred = outputs['nf_pred']
        if nf_pred.dim() != targets.dim() or nf_pred.shape != targets.shape:
            if nf_pred.dim() < targets.dim():
                nf_pred = nf_pred.view(targets.shape)
            elif targets.dim() < nf_pred.dim():
                targets = targets.view(nf_pred.shape)
            outputs['nf_pred'] = nf_pred
            
        pred_loss = self.mse_loss(outputs['nf_pred'], targets)
    
        physics_results = {}
        physics_loss = torch.tensor(0.0, device=targets.device)
        
        delta_w_theory = torch.pow(targets / self.physics_loss.a, 1 / self.physics_loss.b)
        delta_w_theory = torch.clamp(delta_w_theory, min=1e-6)
        
        delta_w_loss = torch.tensor(0.0, device=targets.device)
        if 'delta_w' in outputs:
            delta_w_pred = torch.clamp(outputs['delta_w'], min=1e-6)
            log_delta_w_pred = torch.log10(delta_w_pred)
            log_delta_w_theory = torch.log10(delta_w_theory)
            delta_w_loss = F.mse_loss(log_delta_w_pred, log_delta_w_theory)
            physics_loss = physics_loss + self.delta_w_weight * delta_w_loss
        
        pinn_delta_w_loss = torch.tensor(0.0, device=targets.device)
        if 'pinn_delta_w' in outputs:
            pinn_delta_w = torch.clamp(outputs['pinn_delta_w'], min=1e-6)
            log_pinn_delta_w = torch.log10(pinn_delta_w)
            pinn_delta_w_loss = F.mse_loss(log_pinn_delta_w, log_delta_w_theory)
            physics_loss = physics_loss + 0.8 * pinn_delta_w_loss
        
        lstm_delta_w_loss = torch.tensor(0.0, device=targets.device)
        if 'lstm_delta_w' in outputs:
            lstm_delta_w = torch.clamp(outputs['lstm_delta_w'], min=1e-6)
            log_lstm_delta_w = torch.log10(lstm_delta_w)
            lstm_delta_w_loss = F.mse_loss(log_lstm_delta_w, log_delta_w_theory)
            physics_loss = physics_loss + 0.8 * lstm_delta_w_loss
        
        if 'delta_w' in outputs and 'nf_pred' in outputs:
            physics_temp = self.physics_loss(
                outputs['delta_w'], outputs['nf_pred'], targets
            )
            physics_loss = physics_loss + 0.5 * physics_temp['physics_loss']
            physics_results = physics_temp
        else:
            physics_results = {
                'micro_loss': torch.tensor(0.0, device=targets.device),
                'macro_loss': torch.tensor(0.0, device=targets.device)
            }
    
        consistency_results = {}
        consistency_loss = torch.tensor(0.0, device=targets.device)
        if 'pinn_nf_pred' in outputs and 'lstm_nf_pred' in outputs:
            consistency_temp = self.consistency_loss(
                outputs['pinn_nf_pred'], outputs['lstm_nf_pred']
            )
            consistency_loss = consistency_loss + 0.5 * consistency_temp['consistency_loss']
            consistency_results = consistency_temp
        else:
            consistency_results = {
                'basic_loss': torch.tensor(0.0, device=targets.device),
                'correlation_loss': torch.tensor(0.0, device=targets.device)
            }
        
        if 'pinn_delta_w' in outputs and 'lstm_delta_w' in outputs:
            log_pinn_delta_w = torch.log10(torch.clamp(outputs['pinn_delta_w'], min=1e-6))
            log_lstm_delta_w = torch.log10(torch.clamp(outputs['lstm_delta_w'], min=1e-6))
            delta_w_consistency = F.mse_loss(log_pinn_delta_w, log_lstm_delta_w)
            consistency_loss = consistency_loss + delta_w_consistency
            consistency_results['delta_w_consistency'] = delta_w_consistency
    
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
            if isinstance(reg_loss, torch.Tensor) and reg_loss.dim() > 0:
                reg_loss = reg_loss.mean()
        
        total_loss = (
            pred_loss + 
            self.lambda_physics * physics_loss + 
            self.lambda_consistency * consistency_loss +
            reg_loss
        )
    
        result = {
            'total_loss': total_loss,
            'pred_loss': pred_loss,
            'physics_loss': physics_loss,
            'consistency_loss': consistency_loss,
            'reg_loss': reg_loss,
            'delta_w_loss': delta_w_loss,
            'pinn_delta_w_loss': pinn_delta_w_loss,
            'lstm_delta_w_loss': lstm_delta_w_loss
        }
        for key, value in physics_results.items():
            result[key] = value
        for key, value in consistency_results.items():
            result[key] = value
        return result

    def update_lambda(self, lambda_physics=None, lambda_consistency=None, delta_w_weight=None):
        """
        更新損失權重
        
        參數:
            lambda_physics (float, optional): 新的物理約束損失權重
            lambda_consistency (float, optional): 新的一致性損失權重
            delta_w_weight (float, optional): 新的delta_w損失權重 - 新增參數
        """
        if lambda_physics is not None:
            self.lambda_physics = lambda_physics
            logger.info(f"更新物理約束損失權重為: {lambda_physics}")
        
        if lambda_consistency is not None:
            self.lambda_consistency = lambda_consistency
            logger.info(f"更新一致性損失權重為: {lambda_consistency}")
            
        if delta_w_weight is not None:
            self.delta_w_weight = delta_w_weight
            logger.info(f"更新delta_w損失權重為: {delta_w_weight}")
class AdaptiveHybridLoss(HybridLoss):
    """
    自適應混合損失函數
    根據訓練進度自動調整損失權重
    """
    def __init__(self, initial_lambda_physics=0.01, max_lambda_physics=0.5,
                 initial_lambda_consistency=0.01, max_lambda_consistency=0.3,
                 initial_delta_w_weight=1.5, max_delta_w_weight=3.0,
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
            initial_delta_w_weight (float): 初始delta_w損失權重  # 新增
            max_delta_w_weight (float): 最大delta_w損失權重      # 新增
            epochs_to_max (int): 達到最大權重的訓練輪數
            warmup_epochs (int): 預熱輪數，權重保持較低
            ...其他參數...
        """
        super(AdaptiveHybridLoss, self).__init__(
            lambda_physics=initial_lambda_physics,
            lambda_consistency=initial_lambda_consistency,
            delta_w_weight=initial_delta_w_weight,
            a=a, b=b, reduction=reduction, log_space=log_space,
            relative_error_weight=relative_error_weight,
            micro_weight=micro_weight, macro_weight=macro_weight,
            correlation_weight=correlation_weight, l1_reg=l1_reg, l2_reg=l2_reg
        )
        self.initial_lambda_physics = initial_lambda_physics
        self.max_lambda_physics = max_lambda_physics
        self.initial_lambda_consistency = initial_lambda_consistency
        self.max_lambda_consistency = max_lambda_consistency
        self.initial_delta_w_weight = initial_delta_w_weight
        self.max_delta_w_weight = max_delta_w_weight
        self.epochs_to_max = epochs_to_max
        self.warmup_epochs = warmup_epochs
        self.adaptive_scheme = adaptive_scheme
        self.current_epoch = 0
        
        logger.info(f"初始化AdaptiveHybridLoss: "
                    f"physics權重從{initial_lambda_physics}增加到{max_lambda_physics}, "
                    f"consistency權重從{initial_lambda_consistency}增加到{max_lambda_consistency}, "
                    f"delta_w權重從{initial_delta_w_weight}增加到{max_delta_w_weight}, "
                    f"在{epochs_to_max}個輪次內達到最大值, "
                    f"預熱輪次: {warmup_epochs}, 調整方案: {adaptive_scheme}")
    
    def update_epoch(self, epoch):
        """
        更新當前訓練輪次並調整損失權重
        
        參數:
            epoch (int): 當前訓練輪次
        """
        self.current_epoch = epoch
        
        if epoch < self.warmup_epochs:
            factor = 0.0
        else:
            effective_epoch = epoch - self.warmup_epochs
            effective_max = self.epochs_to_max - self.warmup_epochs
            if effective_epoch >= effective_max:
                factor = 1.0
            else:
                if self.adaptive_scheme == 'linear':
                    factor = effective_epoch / effective_max
                elif self.adaptive_scheme == 'exp':
                    factor = 1.0 - math.exp(-5 * effective_epoch / effective_max)
                elif self.adaptive_scheme == 'step':
                    steps = 4
                    factor = min(1.0, math.ceil(steps * effective_epoch / effective_max) / steps)
                elif self.adaptive_scheme == 'cosine':
                    factor = 0.5 * (1 - math.cos(math.pi * effective_epoch / effective_max))
                else:
                    factor = effective_epoch / effective_max
        
        current_lambda_physics = self.initial_lambda_physics + (
            self.max_lambda_physics - self.initial_lambda_physics) * factor
        current_lambda_consistency = self.initial_lambda_consistency + (
            self.max_lambda_consistency - self.initial_lambda_consistency) * factor
        current_delta_w_weight = self.initial_delta_w_weight + (
            self.max_delta_w_weight - self.initial_delta_w_weight) * factor
        
        self.update_lambda(
            current_lambda_physics, 
            current_lambda_consistency,
            current_delta_w_weight
        )


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
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 簡單的測試代碼
    logger.info("測試losses模組")
    
    # 測試MSELoss
    mse_loss_fn = MSELoss(reduction='mean', log_space=True, relative_error_weight=0.3)
    pred = torch.abs(torch.randn(8))
    target = torch.abs(torch.randn(8))
    loss_val = mse_loss_fn(pred, target)
    logger.info(f"MSELoss值: {loss_val.item():.4f}")
    
    # 測試PhysicsConstraintLoss
    phys_loss_fn = PhysicsConstraintLoss(a=A_COEFFICIENT, b=B_COEFFICIENT, reduction='mean', 
                                         micro_weight=1.0, macro_weight=0.5)
    delta_w = torch.abs(torch.randn(8))
    nf_pred = torch.abs(torch.randn(8))
    nf_true = torch.abs(torch.randn(8))
    phys_loss = phys_loss_fn(delta_w, nf_pred, nf_true)
    logger.info(f"PhysicsConstraintLoss - 總損失: {phys_loss['physics_loss'].item():.4f}")
    
    # 測試ConsistencyLoss
    cons_loss_fn = ConsistencyLoss(reduction='mean', log_space=True, correlation_weight=0.3)
    pinn_pred = torch.abs(torch.randn(8))
    lstm_pred = torch.abs(torch.randn(8))
    cons_loss = cons_loss_fn(pinn_pred, lstm_pred)
    logger.info(f"ConsistencyLoss - 一致性損失: {cons_loss['consistency_loss'].item():.4f}")
    
    # 測試HybridLoss
    hybrid_loss_fn = HybridLoss(lambda_physics=0.2, lambda_consistency=0.1, log_space=True)
    outputs = {
        'nf_pred': torch.abs(torch.randn(8)),
        'delta_w': torch.abs(torch.randn(8))
    }
    hybrid_results = hybrid_loss_fn(outputs, target)
    logger.info(f"HybridLoss總損失: {hybrid_results['total_loss'].item():.4f}")
    
    # 測試AdaptiveHybridLoss
    adaptive_loss_fn = AdaptiveHybridLoss(
        initial_lambda_physics=0.01, 
        max_lambda_physics=0.5,
        initial_lambda_consistency=0.01,
        max_lambda_consistency=0.3,
        epochs_to_max=50,
        adaptive_scheme='cosine'
    )
    for epoch in range(0, 61, 20):
        adaptive_loss_fn.update_epoch(epoch)
        adaptive_result = adaptive_loss_fn(outputs, target)
        logger.info(f"輪次 {epoch}: 總損失={adaptive_result['total_loss'].item():.4f}, "
                    f"物理權重={adaptive_loss_fn.lambda_physics:.4f}, "
                    f"一致性權重={adaptive_loss_fn.lambda_consistency:.4f}, "
                    f"delta_w權重={adaptive_loss_fn.delta_w_weight:.4f}")
    
    logger.info("losses模組測試完成")

