#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
losses.py - 優化的損失函數模組
本模組實現了用於訓練銲錫接點疲勞壽命預測混合模型的各種損失函數，
包括加強版物理約束損失、自適應多任務損失和對數空間損失等。

主要改進:
1. 增強物理約束損失函數，提供多層次的物理驅動
2. 實現自適應權重機制，平衡不同損失分量
3. 優化小樣本數據集的損失計算策略
4. 提供對數空間損失選項，處理跨度大的疲勞壽命值
5. 增加多樣化的正則化選項
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import math

logger = logging.getLogger(__name__)


class MSELoss(nn.Module):
    """
    增強版均方誤差損失函數
    支援相對誤差和對數空間損失計算
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


class EnhancedPhysicsConstraintLoss(nn.Module):
    """
    增強版物理約束損失函數
    提供多層次的物理驅動，包括能量、變形和結構約束
    """
    def __init__(self, a=55.83, b=-2.259, reduction='mean', micro_weight=1.0, 
                 macro_weight=0.5, conservation_weight=0.2, boundary_weight=0.1):
        """
        初始化增強版物理約束損失
        
        參數:
            a (float): 物理模型係數 a
            b (float): 物理模型係數 b
            reduction (str): 誤差匯總方式
            micro_weight (float): 微觀物理約束權重
            macro_weight (float): 宏觀物理約束權重
            conservation_weight (float): 守恆約束權重
            boundary_weight (float): 邊界條件約束權重
        """
        super(EnhancedPhysicsConstraintLoss, self).__init__()
        self.a = a
        self.b = b
        self.reduction = reduction
        self.micro_weight = micro_weight
        self.macro_weight = macro_weight
        self.conservation_weight = conservation_weight
        self.boundary_weight = boundary_weight
    
    def forward(self, delta_w, nf_pred, nf_true, static_features=None):
        """
        計算物理約束損失
        
        參數:
            delta_w (torch.Tensor): 預測的非線性塑性應變能密度變化量
            nf_pred (torch.Tensor): 預測的疲勞壽命
            nf_true (torch.Tensor): 真實的疲勞壽命
            static_features (torch.Tensor, optional): 靜態結構特徵
            
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
        micro_loss = F.mse_loss(delta_w, delta_w_theory, reduction=self.reduction)
        
        # 2. 宏觀物理約束 - 預測值應符合物理模型
        # 從delta_w計算理論壽命
        nf_physics = self.a * torch.pow(delta_w, self.b)
        
        # 宏觀損失: 預測的nf應符合物理模型
        macro_loss = F.mse_loss(nf_pred, nf_physics, reduction=self.reduction)
        
        # 3. 守恆約束 - 能量與壽命的反比關係
        # a * (delta_w)^b = nf 可轉換為 delta_w * nf^(1/b) = a^(1/b)
        conservation_term = delta_w * torch.pow(nf_pred, 1/self.b)
        conservation_target = torch.ones_like(delta_w) * math.pow(self.a, 1/self.b)
        conservation_loss = F.mse_loss(conservation_term, conservation_target, reduction=self.reduction)
        
        # 4. 邊界條件約束 (如果提供了靜態特徵)
        if static_features is not None:
            # 基於結構參數的邊界條件檢查
            # 例如，較大的翹曲變形(warpage)應導致較大的delta_w
            warpage_idx = 4  # 假設翹曲變形在索引4
            if static_features.shape[1] > warpage_idx:
                warpage = static_features[:, warpage_idx]
                # 計算warpage和delta_w的秩相關性
                # 使用softplus確保delta_w與warpage正相關
                boundary_loss = F.mse_loss(
                    F.softplus(delta_w), 
                    F.softplus(warpage * 1e-6), 
                    reduction=self.reduction
                )
            else:
                boundary_loss = torch.tensor(0.0, device=delta_w.device)
        else:
            boundary_loss = torch.tensor(0.0, device=delta_w.device)
        
        # 總物理約束損失
        physics_loss = (
            self.micro_weight * micro_loss + 
            self.macro_weight * macro_loss + 
            self.conservation_weight * conservation_loss +
            self.boundary_weight * boundary_loss
        )
        
        # 返回各部分損失
        return {
            'physics_loss': physics_loss,
            'micro_loss': micro_loss,
            'macro_loss': macro_loss,
            'conservation_loss': conservation_loss,
            'boundary_loss': boundary_loss
        }


class EnhancedConsistencyLoss(nn.Module):
    """
    增強版一致性損失函數
    確保模型不同分支的預測結果保持一致，同時考慮對數空間和預測趨勢
    """
    def __init__(self, reduction='mean', log_space=True, correlation_weight=0.3,
                gradient_weight=0.2, variance_weight=0.0):
        """
        初始化增強版一致性損失
        
        參數:
            reduction (str): 誤差匯總方式
            log_space (bool): 是否在對數空間計算一致性
            correlation_weight (float): 相關性約束權重
            gradient_weight (float): 梯度一致性權重
            variance_weight (float): 方差平衡權重
        """
        super(EnhancedConsistencyLoss, self).__init__()
        self.reduction = reduction
        self.log_space = log_space
        self.correlation_weight = correlation_weight
        self.gradient_weight = gradient_weight
        self.variance_weight = variance_weight
    
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
        
        # 初始化其他損失分量
        correlation_loss = torch.tensor(0.0, device=pinn_pred.device)
        gradient_loss = torch.tensor(0.0, device=pinn_pred.device)
        variance_loss = torch.tensor(0.0, device=pinn_pred.device)
        
        # 2. 相關性損失 - 確保預測趨勢一致
        if self.correlation_weight > 0 and pinn_pred.size(0) > 2:
            # 標準化預測值
            pinn_norm = (pinn_pred - pinn_pred.mean()) / (pinn_pred.std() + 1e-8)
            lstm_norm = (lstm_pred - lstm_pred.mean()) / (lstm_pred.std() + 1e-8)
            
            # 計算相關性，確保兩個分支預測趨勢一致
            corr = torch.sum(pinn_norm * lstm_norm) / pinn_pred.size(0)
            correlation_loss = 1.0 - corr  # 相關性越高，損失越低
        
        # 3. 梯度一致性損失 - 確保預測變化率一致
        if self.gradient_weight > 0 and pinn_pred.size(0) > 2:
            # 計算排序索引
            _, pinn_indices = torch.sort(pinn_pred)
            _, lstm_indices = torch.sort(lstm_pred)
            
            # 使用排序後的索引差異作為梯度一致性的度量
            if self.reduction == 'mean':
                gradient_loss = torch.mean(torch.abs(pinn_indices.float() - lstm_indices.float())) / pinn_pred.size(0)
            elif self.reduction == 'sum':
                gradient_loss = torch.sum(torch.abs(pinn_indices.float() - lstm_indices.float())) / pinn_pred.size(0)
            else:  # 'none'
                gradient_loss = torch.abs(pinn_indices.float() - lstm_indices.float()) / pinn_pred.size(0)
        
        # 4. 方差平衡損失 - 平衡兩個分支的預測方差
        if self.variance_weight > 0:
            pinn_var = torch.var(pinn_pred)
            lstm_var = torch.var(lstm_pred)
            variance_loss = torch.abs(pinn_var - lstm_var) / (torch.max(pinn_var, lstm_var) + 1e-8)
        
        # 總一致性損失
        consistency_loss = (
            basic_loss + 
            self.correlation_weight * correlation_loss + 
            self.gradient_weight * gradient_loss +
            self.variance_weight * variance_loss
        )
        
        # 返回各部分損失
        return {
            'consistency_loss': consistency_loss,
            'basic_loss': basic_loss,
            'correlation_loss': correlation_loss,
            'gradient_loss': gradient_loss,
            'variance_loss': variance_loss
        }


class EnhancedHybridLoss(nn.Module):
    """
    增強版混合損失函數
    結合MSE損失、物理約束損失和一致性損失，提供更精細的權重控制
    """
    def __init__(self, lambda_physics=0.1, lambda_consistency=0.1, 
                 a=55.83, b=-2.259, reduction='mean', log_space=True,
                 relative_error_weight=0.5, micro_weight=1.0, macro_weight=0.5,
                 conservation_weight=0.2, boundary_weight=0.1,
                 correlation_weight=0.3, gradient_weight=0.2,
                 l1_reg=0.0, l2_reg=0.0):
        """
        初始化增強版混合損失
        
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
            conservation_weight (float): 守恆約束權重
            boundary_weight (float): 邊界條件約束權重
            correlation_weight (float): 相關性約束權重
            gradient_weight (float): 梯度一致性權重
            l1_reg (float): L1正則化係數
            l2_reg (float): L2正則化係數
        """
        super(EnhancedHybridLoss, self).__init__()
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
        
        self.physics_loss = EnhancedPhysicsConstraintLoss(
            a=a, 
            b=b, 
            reduction=reduction, 
            micro_weight=micro_weight, 
            macro_weight=macro_weight,
            conservation_weight=conservation_weight,
            boundary_weight=boundary_weight
        )
        
        self.consistency_loss = EnhancedConsistencyLoss(
            reduction=reduction, 
            log_space=log_space, 
            correlation_weight=correlation_weight,
            gradient_weight=gradient_weight
        )
        
        logger.info(f"初始化EnhancedHybridLoss: lambda_physics={lambda_physics}, "
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
        if 'delta_w' in outputs:
            static_features = outputs.get('static_features', None)
            physics_results = self.physics_loss(
                outputs['delta_w'], outputs['pinn_nf_pred'], targets, static_features
            )
            physics_loss = physics_results['physics_loss']
        else:
            physics_loss = torch.tensor(0.0, device=targets.device)
            physics_results = {
                'micro_loss': torch.tensor(0.0, device=targets.device),
                'macro_loss': torch.tensor(0.0, device=targets.device),
                'conservation_loss': torch.tensor(0.0, device=targets.device),
                'boundary_loss': torch.tensor(0.0, device=targets.device)
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
                'correlation_loss': torch.tensor(0.0, device=targets.device),
                'gradient_loss': torch.tensor(0.0, device=targets.device),
                'variance_loss': torch.tensor(0.0, device=targets.device)
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
        elif 'l2_penalty' in outputs:
            # 如果模型已經計算了L2懲罰
            reg_loss = outputs['l2_penalty']
        
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


class AdaptiveHybridLoss(EnhancedHybridLoss):
    """
    自適應混合損失函數
    根據訓練進度自動調整損失權重
    """
    def __init__(self, initial_lambda_physics=0.01, max_lambda_physics=0.5,
                 initial_lambda_consistency=0.01, max_lambda_consistency=0.3,
                 epochs_to_max=50, warmup_epochs=5, 
                 a=55.83, b=-2.259, reduction='mean', log_space=True,
                 relative_error_weight=0.5, micro_weight=1.0, macro_weight=0.5,
                 conservation_weight=0.2, boundary_weight=0.1,
                 correlation_weight=0.3, gradient_weight=0.2,
                 l1_reg=0.0, l2_reg=0.0, adaptive_scheme='linear'):
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
            conservation_weight (float): 守恆約束權重
            boundary_weight (float): 邊界條件約束權重
            correlation_weight (float): 相關性約束權重
            gradient_weight (float): 梯度一致性權重
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
            conservation_weight=conservation_weight, boundary_weight=boundary_weight,
            correlation_weight=correlation_weight, gradient_weight=gradient_weight,
            l1_reg=l1_reg, l2_reg=l2_reg
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
        
        logger.debug(f"輪次 {epoch}: 物理損失權重={self.lambda_physics:.4f}, "
                   f"一致性損失權重={self.lambda_consistency:.4f}")


class AutobalancingLoss(nn.Module):
    """
    自動平衡損失函數
    根據各損失分量的梯度大小自動調整權重
    適合小樣本數據集，避免某一損失分量主導訓練過程
    """
    def __init__(self, losses_dict, initial_weights=None):
        """
        初始化自動平衡損失
        
        參數:
            losses_dict (dict): 損失函數字典，鍵為損失名稱，值為損失函數
            initial_weights (dict, optional): 初始權重字典，鍵為損失名稱，值為權重
        """
        super(AutobalancingLoss, self).__init__()
        self.losses_dict = losses_dict
        
        # 初始化權重
        if initial_weights is None:
            self.weights = {name: 1.0 for name in losses_dict.keys()}
        else:
            self.weights = initial_weights
        
        # 權重參數化
        self.log_weights = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(math.log(weight), dtype=torch.float32))
            for name, weight in self.weights.items()
        })
        
        # 梯度累積
        self.grad_norms = {name: 0.0 for name in losses_dict.keys()}
        self.update_steps = 0
        
        logger.info(f"初始化AutobalancingLoss，損失函數: {list(losses_dict.keys())}")
    
    def forward(self, outputs, targets, model=None):
        """
        計算自動平衡損失
        
        參數:
            outputs (dict): 模型輸出
            targets (torch.Tensor): 目標值
            model (torch.nn.Module, optional): 模型，用於某些損失函數
            
        返回:
            torch.Tensor: 總損失值
        """
        # 計算各損失分量
        losses = {}
        for name, loss_fn in self.losses_dict.items():
            losses[name] = loss_fn(outputs, targets, model)
            
            # 如果損失是字典（例如HybridLoss的返回），則使用總損失
            if isinstance(losses[name], dict):
                losses[name] = losses[name]['total_loss']
        
        # 應用權重
        weighted_losses = {}
        normalized_weights = {}
        
        # 計算權重總和以進行標準化
        weight_sum = sum(torch.exp(weight) for weight in self.log_weights.values())
        
        # 計算加權損失
        for name in losses:
            # 計算標準化權重
            normalized_weights[name] = torch.exp(self.log_weights[name]) / weight_sum
            
            # 應用權重
            weighted_losses[name] = normalized_weights[name] * losses[name]
        
        # 總損失
        total_loss = sum(weighted_losses.values())
        
        # 訓練時保存各分量的梯度信息
        if self.training:
            # 清零梯度累積
            for name in losses:
                losses[name].backward(retain_graph=True)
                
                # 計算梯度範數
                grad_norm = 0.0
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.norm().item()
                
                # 更新梯度累積
                self.grad_norms[name] = 0.9 * self.grad_norms[name] + 0.1 * grad_norm
                
                # 清零梯度
                model.zero_grad()
            
            # 更新次數增加
            self.update_steps += 1
            
            # 每隔一定步數更新權重
            if self.update_steps % 10 == 0:
                self._update_weights()
        
        # 返回結果
        return {
            'total_loss': total_loss,
            'losses': losses,
            'weighted_losses': weighted_losses,
            'weights': {name: normalized_weights[name].item() for name in normalized_weights}
        }
    
    def _update_weights(self):
        """更新權重以平衡各損失分量的貢獻"""
        # 計算平均梯度範數
        avg_grad_norm = sum(self.grad_norms.values()) / len(self.grad_norms)
        
        if avg_grad_norm > 0:
            # 計算調整因子
            adjustment_factors = {
                name: avg_grad_norm / (norm + 1e-8)
                for name, norm in self.grad_norms.items()
            }
            
            # 應用調整因子更新log權重
            with torch.no_grad():
                for name, factor in adjustment_factors.items():
                    # 限制調整幅度
                    factor = max(0.5, min(2.0, factor))
                    self.log_weights[name].mul_(factor)
        
        # 重置梯度累積
        self.grad_norms = {name: 0.0 for name in self.grad_norms.keys()}


# 實用工具函數
def get_loss_function(loss_type='hybrid', **kwargs):
    """
    獲取指定類型的損失函數
    
    參數:
        loss_type (str): 損失函數類型，可選 'mse', 'physics', 'consistency', 
                        'hybrid', 'adaptive', 'enhanced', 'autobalancing'
        **kwargs: 傳遞給損失函數的額外參數
    
    返回:
        nn.Module: 指定類型的損失函數實例
    """
    if loss_type.lower() == 'mse':
        return MSELoss(**kwargs)
    elif loss_type.lower() == 'physics':
        return EnhancedPhysicsConstraintLoss(**kwargs)
    elif loss_type.lower() == 'consistency':
        return EnhancedConsistencyLoss(**kwargs)
    elif loss_type.lower() == 'hybrid':
        return EnhancedHybridLoss(**kwargs)
    elif loss_type.lower() == 'adaptive':
        return AdaptiveHybridLoss(**kwargs)
    elif loss_type.lower() == 'enhanced':
        return EnhancedHybridLoss(**kwargs)
    elif loss_type.lower() == 'autobalancing':
        # 創建基本損失函數
        mse_loss = MSELoss(**kwargs)
        physics_loss = EnhancedPhysicsConstraintLoss(**kwargs)
        consistency_loss = EnhancedConsistencyLoss(**kwargs)
        
        # 初始權重
        initial_weights = {
            'mse': kwargs.get('mse_weight', 1.0),
            'physics': kwargs.get('lambda_physics', 0.1),
            'consistency': kwargs.get('lambda_consistency', 0.1)
        }
        
        # 創建自動平衡損失
        return AutobalancingLoss(
            losses_dict={
                'mse': mse_loss,
                'physics': physics_loss,
                'consistency': consistency_loss
            },
            initial_weights=initial_weights
        )
    else:
        raise ValueError(f"不支援的損失函數類型: {loss_type}")


def create_loss_function(config, use_physics=True, log_space=True):
    """
    根據配置創建損失函數
    
    參數:
        config (dict): 配置字典
        use_physics (bool): 是否使用物理約束
        log_space (bool): 是否在對數空間計算損失
        
    返回:
        nn.Module: 損失函數實例
    """
    # 獲取損失配置
    loss_config = config["training"]["loss"]
    loss_type = loss_config["type"]
    
    # 獲取物理模型參數
    a = config["model"]["physics"]["a_coefficient"]
    b = config["model"]["physics"]["b_coefficient"]
    
    # 基本參數
    params = {
        'a': a,
        'b': b,
        'log_space': log_space,
        'reduction': 'mean'
    }
    
    # 高級參數
    if 'relative_error_weight' in loss_config:
        params['relative_error_weight'] = loss_config['relative_error_weight']
    
    # 物理約束參數
    if use_physics:
        if 'lambda_physics' in loss_config:
            params['lambda_physics'] = loss_config['lambda_physics']
        else:
            params['lambda_physics'] = loss_config.get('initial_lambda_physics', 0.1)
            
        if 'micro_weight' in loss_config:
            params['micro_weight'] = loss_config['micro_weight']
        if 'macro_weight' in loss_config:
            params['macro_weight'] = loss_config['macro_weight']
        if 'conservation_weight' in loss_config:
            params['conservation_weight'] = loss_config['conservation_weight']
        if 'boundary_weight' in loss_config:
            params['boundary_weight'] = loss_config['boundary_weight']
    else:
        params['lambda_physics'] = 0.0
    
    # 一致性約束參數
    if 'lambda_consistency' in loss_config:
        params['lambda_consistency'] = loss_config['lambda_consistency']
    else:
        params['lambda_consistency'] = loss_config.get('initial_lambda_consistency', 0.1)
        
    if 'correlation_weight' in loss_config:
        params['correlation_weight'] = loss_config['correlation_weight']
    if 'gradient_weight' in loss_config:
        params['gradient_weight'] = loss_config['gradient_weight']
    
    # 正則化參數
    if 'l1_reg' in loss_config:
        params['l1_reg'] = loss_config['l1_reg']
    if 'l2_reg' in loss_config:
        params['l2_reg'] = loss_config['l2_reg']
    
    # 自適應損失參數
    if loss_type == 'adaptive':
        params['initial_lambda_physics'] = loss_config.get('initial_lambda_physics', 0.01) if use_physics else 0.0
        params['max_lambda_physics'] = loss_config.get('max_lambda_physics', 0.5) if use_physics else 0.0
        params['initial_lambda_consistency'] = loss_config.get('initial_lambda_consistency', 0.01)
        params['max_lambda_consistency'] = loss_config.get('max_lambda_consistency', 0.3)
        params['epochs_to_max'] = loss_config.get('epochs_to_max', 50)
        params['warmup_epochs'] = loss_config.get('warmup_epochs', 5)
        params['adaptive_scheme'] = loss_config.get('adaptive_scheme', 'linear')
    
    # 創建損失函數
    return get_loss_function(loss_type, **params)


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
    
    # 測試MSE損失
    mse_loss = MSELoss(log_space=True, relative_error_weight=0.3)
    mse_result = mse_loss(outputs['nf_pred'], targets)
    logger.info(f"MSE損失 (對數空間, 相對誤差權重=0.3): {mse_result.item():.4f}")
    
    # 測試增強版物理約束損失
    physics_loss = EnhancedPhysicsConstraintLoss()
    physics_results = physics_loss(outputs['delta_w'], outputs['pinn_nf_pred'], targets)
    logger.info(f"物理約束損失: {physics_results['physics_loss'].item():.4f}")
    logger.info(f"  - 微觀損失: {physics_results['micro_loss'].item():.4f}")
    logger.info(f"  - 宏觀損失: {physics_results['macro_loss'].item():.4f}")
    logger.info(f"  - 守恆損失: {physics_results['conservation_loss'].item():.4f}")
    
    # 測試增強版一致性損失
    consistency_loss = EnhancedConsistencyLoss(log_space=True, correlation_weight=0.3)
    consistency_results = consistency_loss(outputs['pinn_nf_pred'], outputs['lstm_nf_pred'])
    logger.info(f"一致性損失: {consistency_results['consistency_loss'].item():.4f}")
    logger.info(f"  - 基本損失: {consistency_results['basic_loss'].item():.4f}")
    logger.info(f"  - 相關性損失: {consistency_results['correlation_loss'].item():.4f}")
    
    # 測試增強版混合損失
    hybrid_loss = EnhancedHybridLoss(lambda_physics=0.2, lambda_consistency=0.1, log_space=True)
    hybrid_results = hybrid_loss(outputs, targets)
    logger.info(f"混合損失: {hybrid_results['total_loss'].item():.4f}")
    logger.info(f"  - 預測損失: {hybrid_results['pred_loss'].item():.4f}")
    logger.info(f"  - 物理約束損失: {hybrid_results['physics_loss'].item():.4f}")
    logger.info(f"  - 一致性損失: {hybrid_results['consistency_loss'].item():.4f}")
    
    # 測試自適應混合損失
    adaptive_loss = AdaptiveHybridLoss(
        initial_lambda_physics=0.01, 
        max_lambda_physics=0.5,
        initial_lambda_consistency=0.01,
        max_lambda_consistency=0.3,
        epochs_to_max=50,
        adaptive_scheme='cosine'
    )
    
    logger.info("\n測試自適應混合損失:")
    for epoch in range(0, 61, 20):
        adaptive_loss.update_epoch(epoch)
        adaptive_result = adaptive_loss(outputs, targets)
        logger.info(f"輪次 {epoch}: 總損失={adaptive_result['total_loss'].item():.4f}, "
                  f"物理權重={adaptive_loss.lambda_physics:.4f}, "
                  f"一致性權重={adaptive_loss.lambda_consistency:.4f}")


class HybridLoss(nn.Module):
    """
    基本版 Hybrid 損失函數：
    結合 MSE 與 Physics-based Loss，可透過權重調整貢獻度
    """
    def __init__(self, mse_weight=1.0, physics_weight=1.0, reduction='mean'):
        super(HybridLoss, self).__init__()
        self.mse_weight = mse_weight
        self.physics_weight = physics_weight
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, predictions, targets, physics_loss=None):
        """
        計算綜合損失
        :param predictions: 預測值
        :param targets: 真實值
        :param physics_loss: 額外傳入的物理損失 (tensor 或 None)
        """
        mse_loss = self.mse(predictions, targets)
        if physics_loss is not None:
            total_loss = self.mse_weight * mse_loss + self.physics_weight * physics_loss
        else:
            total_loss = self.mse_weight * mse_loss
        return total_loss
