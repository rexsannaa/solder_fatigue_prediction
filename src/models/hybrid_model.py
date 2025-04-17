#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
hybrid_model.py - 混合PINN-LSTM模型
本模組實現了結合物理信息神經網絡(PINN)和長短期記憶網絡(LSTM)的混合模型，
用於準確預測銲錫接點的疲勞壽命。

主要特點:
1. 明確的兩階段預測架構：先預測delta_w，再使用物理公式計算疲勞壽命
2. 簡化的網絡結構，降低層數和參數量
3. 直接物理計算為主導，神經網絡僅提供修正
4. 使用PINN分支從靜態特徵中提取物理關係
5. 使用LSTM分支從時間序列數據中提取動態特徵
6. 適用於小樣本數據集(81筆)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import math
import os
import time

logger = logging.getLogger(__name__)

class PhysicsLayer(nn.Module):
    """
    物理約束層
    實現銲錫接點疲勞壽命的物理模型約束
    基於疲勞壽命與非線性塑性應變能密度的關係: Nf=55.83⋅(ΔW)^(-2.259)
    """
    def __init__(self, a=55.83, b=-2.259, trainable=False):
        """
        初始化物理層
        
        參數:
            a (float): a係數
            b (float): b係數
            trainable (bool): 係數是否可訓練
        """
        super(PhysicsLayer, self).__init__()
        self.trainable = trainable
        if trainable:
            self.log_a = nn.Parameter(torch.tensor(np.log(a), dtype=torch.float32))
            self.log_neg_b = nn.Parameter(torch.tensor(np.log(-b), dtype=torch.float32))
        else:
            self.register_buffer('a', torch.tensor(a, dtype=torch.float32))
            self.register_buffer('b', torch.tensor(b, dtype=torch.float32))
    
    def forward(self, delta_w):
        """
        應用物理模型計算疲勞壽命
        
        參數:
            delta_w (torch.Tensor): 非線性塑性應變能密度變化量
        
        返回:
            torch.Tensor: 預測的疲勞壽命
        """
        # 確保輸入為正值 (物理上合理)
        delta_w = torch.clamp(delta_w, min=1e-8)
        
        if self.trainable:
            a = torch.exp(self.log_a)
            b = -torch.exp(self.log_neg_b)
            nf = a * torch.pow(delta_w, b)
        else:
            # 使用預設係數，確保它們是浮點數
            a_val = float(self.a) if hasattr(self.a, 'item') else float(self.a)
            b_val = float(self.b) if hasattr(self.b, 'item') else float(self.b)
            # 應用物理模型: Nf = a * (ΔW)^b
            nf = a_val * torch.pow(delta_w, b_val)
        
        nf = torch.clamp(nf, min=10.0)  # 疲勞壽命下限為10週期
        return nf

class SimpleAttentionLayer(nn.Module):
    """
    簡化的注意力機制層
    計算時間序列中不同時間步的重要性權重
    """
    def __init__(self, hidden_size):
        """
        初始化注意力層
        
        參數:
            hidden_size (int): 隱藏層大小
        """
        super(SimpleAttentionLayer, self).__init__()
        self.attention_weights = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, lstm_output):
        """
        前向傳播
        
        參數:
            lstm_output (torch.Tensor): LSTM輸出，形狀為 (batch_size, seq_len, hidden_size)
        
        返回:
            tuple: (加權後的特徵向量, 注意力權重)
        """
        # 計算注意力分數
        scores = self.attention_weights(lstm_output).squeeze(-1)  # (batch_size, seq_len)
        
        # 應用softmax獲取注意力權重
        weights = F.softmax(scores, dim=1)  # (batch_size, seq_len)
        
        # 將注意力權重應用於LSTM輸出
        context = torch.bmm(
            weights.unsqueeze(1),  # (batch_size, 1, seq_len)
            lstm_output  # (batch_size, seq_len, hidden_size)
        ).squeeze(1)  # (batch_size, hidden_size)
        
        return context, weights

class SimplePINNBranch(nn.Module):
    """
    簡化的物理信息神經網絡(PINN)分支
    處理靜態結構參數，專注於預測delta_w修正因子
    """
    def __init__(self, input_dim=5, hidden_dim=16, dropout_rate=0.1,
                 a_coefficient=55.83, b_coefficient=-2.259):
        """
        初始化簡化的PINN分支
        
        參數:
            input_dim (int): 輸入特徵維度
            hidden_dim (int): 隱藏層維度
            dropout_rate (float): Dropout率
            a_coefficient (float): 物理模型係數a
            b_coefficient (float): 物理模型係數b
        """
        super(SimplePINNBranch, self).__init__()
        
        # 註冊物理係數
        self.register_buffer('a_coefficient', torch.tensor(a_coefficient, dtype=torch.float32))
        self.register_buffer('b_coefficient', torch.tensor(b_coefficient, dtype=torch.float32))
        
        # 簡化的特徵提取層 - 僅一個隱藏層
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        )
        
        # 修正因子輸出層 - 預測一個接近1的修正因子
        self.correction_layer = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 輸出範圍 [0, 1]
        )
        
        # 初始化修正因子輸出層使其初始輸出接近0.5
        nn.init.constant_(self.correction_layer[0].bias, 0.0)
    
    def forward(self, x):
        """
        前向傳播 - 專注於預測delta_w修正因子
        
        參數:
            x (torch.Tensor): 靜態特徵輸入，形狀為 (batch_size, input_dim)
            
        返回:
            dict: 包含預測結果的字典
        """
        # 特徵提取
        features = self.feature_layer(x)
        
        # 預測修正因子 (範圍0-1，經過調整後)
        correction_factor = self.correction_layer(features).squeeze(-1)
        
        # 調整至更合適的範圍：0.5-1.5
        correction_factor = correction_factor + 0.5
        
        return {
            'correction_factor': correction_factor,
            'features': features
        }

class SimpleLSTMBranch(nn.Module):
    """
    簡化的長短期記憶網絡(LSTM)分支
    處理時間序列數據，專注於預測delta_w修正因子
    """
    def __init__(self, input_dim=2, hidden_size=32, bidirectional=True,
                 dropout_rate=0.1, a_coefficient=55.83, b_coefficient=-2.259):
        """
        初始化簡化的LSTM分支
        
        參數:
            input_dim (int): 輸入特徵維度
            hidden_size (int): LSTM隱藏層大小
            bidirectional (bool): 是否使用雙向LSTM
            dropout_rate (float): Dropout率
            a_coefficient (float): 物理模型係數a
            b_coefficient (float): 物理模型係數b
        """
        super(SimpleLSTMBranch, self).__init__()
        
        # 註冊物理係數
        self.register_buffer('a_coefficient', torch.tensor(a_coefficient, dtype=torch.float32))
        self.register_buffer('b_coefficient', torch.tensor(b_coefficient, dtype=torch.float32))
        
        self.bidirectional = bidirectional
        
        # 簡化的LSTM層 - 單層雙向
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0
        )
        
        # 計算LSTM輸出維度
        lstm_output_dim = hidden_size * 2 if bidirectional else hidden_size
        
        # 注意力層
        self.attention = SimpleAttentionLayer(lstm_output_dim)
        
        # 修正因子輸出層 - 預測一個接近1的修正因子
        self.correction_layer = nn.Sequential(
            nn.Linear(lstm_output_dim, 1),
            nn.Sigmoid()  # 輸出範圍 [0, 1]
        )
        
        # 初始化修正因子輸出層使其初始輸出接近0.5
        nn.init.constant_(self.correction_layer[0].bias, 0.0)
    
    def forward(self, x):
        """
        前向傳播 - 專注於預測delta_w修正因子
        
        參數:
            x (torch.Tensor): 時間序列輸入，形狀為 (batch_size, seq_len, input_dim)
            
        返回:
            dict: 包含預測結果的字典
        """
        # LSTM前向傳播
        lstm_output, _ = self.lstm(x)
        
        # 應用注意力機制
        context, attention_weights = self.attention(lstm_output)
        
        # 預測修正因子 (範圍0-1)
        correction_factor = self.correction_layer(context).squeeze(-1)
        
        # 調整至更合適的範圍：0.5-1.5
        correction_factor = correction_factor + 0.5
        
        return {
            'correction_factor': correction_factor,
            'features': context,
            'attention_weights': attention_weights
        }

class SimplifiedHybridModel(nn.Module):
    """
    簡化的混合PINN-LSTM模型
    強調直接物理計算，並使用神經網絡進行小規模修正
    明確專注於準確預測delta_w，再使用物理公式計算疲勞壽命
    """
    def __init__(self, 
                static_input_dim=5,
                time_input_dim=2,
                time_steps=4,
                hidden_dim=16,
                dropout_rate=0.1,
                pinn_weight=0.5,
                lstm_weight=0.5,
                a_coefficient=55.83,
                b_coefficient=-2.259):
        """
        初始化簡化的混合模型
        
        參數:
            static_input_dim (int): 靜態特徵維度
            time_input_dim (int): 時間序列特徵維度
            time_steps (int): 時間步數
            hidden_dim (int): 隱藏層維度
            dropout_rate (float): Dropout率
            pinn_weight (float): PINN分支權重
            lstm_weight (float): LSTM分支權重
            a_coefficient (float): 物理模型係數a
            b_coefficient (float): 物理模型係數b
        """
        super(SimplifiedHybridModel, self).__init__()
        
        # 註冊物理係數
        self.register_buffer('a_coefficient', torch.tensor(a_coefficient, dtype=torch.float32))
        self.register_buffer('b_coefficient', torch.tensor(b_coefficient, dtype=torch.float32))
        
        # 分支權重
        self.pinn_weight = pinn_weight
        self.lstm_weight = lstm_weight
        
        # 簡化的PINN分支
        self.pinn_branch = SimplePINNBranch(
            input_dim=static_input_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            a_coefficient=a_coefficient,
            b_coefficient=b_coefficient
        )
        
        # 簡化的LSTM分支
        self.lstm_branch = SimpleLSTMBranch(
            input_dim=time_input_dim,
            hidden_size=hidden_dim,
            bidirectional=True,
            dropout_rate=dropout_rate,
            a_coefficient=a_coefficient,
            b_coefficient=b_coefficient
        )
        
        # 物理層
        self.physics_layer = PhysicsLayer(
            a=a_coefficient, 
            b=b_coefficient, 
            trainable=False
        )
    
    def forward(self, static_input, time_series_input, return_features=False):
        """
        前向傳播 - 先直接計算delta_w，再使用神經網絡小幅修正
        
        參數:
            static_input (torch.Tensor): 靜態特徵輸入，形狀為 (batch_size, static_input_dim)
            time_series_input (torch.Tensor): 時間序列輸入，形狀為 (batch_size, time_steps, time_input_dim)
            return_features (bool): 是否返回內部特徵
            
        返回:
            dict: 包含預測結果的字典
        """
        # 1. 直接從時間序列計算物理量delta_w
        direct_delta_w = self._calculate_direct_delta_w(time_series_input)
        
        # 2. 通過PINN和LSTM分支獲取修正因子
        pinn_out = self.pinn_branch(static_input)
        lstm_out = self.lstm_branch(time_series_input)
        
        # 3. 融合修正因子 - 簡單加權平均
        pinn_factor = pinn_out['correction_factor']
        lstm_factor = lstm_out['correction_factor']
        combined_factor = (self.pinn_weight * pinn_factor + self.lstm_weight * lstm_factor)
        
        # 4. 應用修正因子到直接計算的delta_w
        delta_w = direct_delta_w * combined_factor
        delta_w = torch.clamp(delta_w, min=1e-8)  # 確保最小值
        
        # 5. 使用物理公式計算疲勞壽命 - 不添加任何校正因子
        a_coef = float(self.a_coefficient)
        b_coef = float(self.b_coefficient)
        nf_pred = a_coef * torch.pow(delta_w, b_coef)
        nf_pred = torch.clamp(nf_pred, min=10.0)  # 確保疲勞壽命下限
        
        # 6. 準備返回結果
        result = {
            'delta_w': delta_w,                      # 修正後的delta_w
            'direct_delta_w': direct_delta_w,        # 直接計算的delta_w
            'nf_pred': nf_pred,                      # 預測的疲勞壽命
            'pinn_factor': pinn_factor,              # PINN分支的修正因子
            'lstm_factor': lstm_factor,              # LSTM分支的修正因子
            'combined_factor': combined_factor       # 融合後的修正因子
        }
        
        # 可選輸出
        if return_features:
            result['pinn_features'] = pinn_out['features']
            result['lstm_features'] = lstm_out['features']
            result['attention_weights'] = lstm_out['attention_weights']
        
        return result
    
    def _calculate_direct_delta_w(self, time_series_input):
        """
        直接從時間序列數據計算delta_w物理量
        
        參數:
            time_series_input (torch.Tensor): 時間序列輸入，形狀為 (batch_size, time_steps, time_input_dim)
        
        返回:
            torch.Tensor: 直接計算的delta_w值
        """
        # 假設time_input_dim=2，分別是上下界面的應變能密度
        if time_series_input.shape[-1] >= 2:
            # 提取上下界面的數據
            up_interface = time_series_input[:, :, 0]  # (batch_size, time_steps)
            down_interface = time_series_input[:, :, 1]  # (batch_size, time_steps)
            
            # 計算差值 - 使用首末時間點
            up_delta = up_interface[:, -1] - up_interface[:, 0]
            down_delta = down_interface[:, -1] - down_interface[:, 0]
            
            # 上下界面加權平均
            direct_delta_w = 0.55 * up_delta + 0.45 * down_delta
        else:
            # 如果只有一個特徵，直接計算差值
            direct_delta_w = time_series_input[:, -1, 0] - time_series_input[:, 0, 0]
        
        # 使用物理校正因子 - 基於實驗數據確定的最佳值
        direct_delta_w = direct_delta_w * 0.4
        
        # 確保值為正
        direct_delta_w = torch.clamp(direct_delta_w, min=1e-8)
        
        return direct_delta_w

class HybridPINNLSTMModel(SimplifiedHybridModel):
    """
    保持與原始模型相同的類名，但使用簡化的實現
    """
    def __init__(self, 
                static_input_dim=5,
                time_input_dim=2,
                time_steps=4,
                pinn_hidden_dims=[32],
                lstm_hidden_size=32,
                lstm_num_layers=1,
                fusion_dim=16,
                dropout_rate=0.1,
                bidirectional=True,
                use_attention=True,
                use_physics_layer=True,
                physics_layer_trainable=False,
                use_batch_norm=True,
                pinn_weight_init=0.5,
                lstm_weight_init=0.5,
                a_coefficient=55.83,
                b_coefficient=-2.259,
                use_log_transform=True,
                ensemble_method='weighted',
                l2_reg=0.0005):
        """
        保持與原始模型兼容的初始化參數，但內部使用簡化的實現
        """
        # 取出主要參數，忽略不必要的參數
        hidden_dim = pinn_hidden_dims[0] if isinstance(pinn_hidden_dims, list) and len(pinn_hidden_dims) > 0 else 32
        
        # 調用簡化模型的初始化
        super(HybridPINNLSTMModel, self).__init__(
            static_input_dim=static_input_dim,
            time_input_dim=time_input_dim,
            time_steps=time_steps,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            pinn_weight=pinn_weight_init,
            lstm_weight=lstm_weight_init,
            a_coefficient=a_coefficient,
            b_coefficient=b_coefficient
        )
        
        # 保存其他參數以保持兼容性
        self.use_physics_layer = use_physics_layer
        self.use_log_transform = use_log_transform
        self.ensemble_method = ensemble_method
        self.l2_reg = l2_reg
        
        logger.info(f"初始化簡化的HybridPINNLSTMModel - 使用直接物理計算優先策略")

    def set_delta_w_theory(self, delta_w_theory):
        """
        設置理論的delta_w值，用於輔助訓練
        為了與原始模型兼容
        
        參數:
            delta_w_theory (torch.Tensor): 理論的delta_w值
        """
        self._delta_w_theory = delta_w_theory

# PINNLSTMTrainer類保持unchanged
class PINNLSTMTrainer:
    """
    混合PINN-LSTM模型專用訓練器
    提供基於delta_w為核心的特化訓練流程和損失計算
    """
    def __init__(self, model, optimizer, device, lambda_physics_init=0.1, 
                 lambda_physics_max=1.0, lambda_consistency_init=0.1, 
                 lambda_consistency_max=0.5, delta_w_weight_init=1.5,
                 delta_w_weight_max=3.0, lambda_ramp_epochs=50,
                 clip_grad_norm=1.0, scheduler=None, log_interval=10):
        """
        初始化PINNLSTMTrainer
        
        參數:
            model (HybridPINNLSTMModel): 混合模型
            optimizer (torch.optim.Optimizer): 優化器
            device (torch.device): 計算設備
            lambda_physics_init (float): 初始物理約束權重
            lambda_physics_max (float): 最大物理約束權重
            lambda_consistency_init (float): 初始一致性約束權重
            lambda_consistency_max (float): 最大一致性約束權重
            delta_w_weight_init (float): 初始delta_w預測權重
            delta_w_weight_max (float): 最大delta_w預測權重
            lambda_ramp_epochs (int): 權重由初始值增加到最大值的輪數
            clip_grad_norm (float): 梯度裁剪範數
            scheduler (object): 學習率調度器
            log_interval (int): 日誌輸出間隔
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.lambda_physics_init = lambda_physics_init
        self.lambda_physics_max = lambda_physics_max
        self.lambda_consistency_init = lambda_consistency_init
        self.lambda_consistency_max = lambda_consistency_max
        self.delta_w_weight_init = delta_w_weight_init
        self.delta_w_weight_max = delta_w_weight_max
        self.lambda_ramp_epochs = lambda_ramp_epochs
        self.clip_grad_norm = clip_grad_norm
        self.scheduler = scheduler
        self.log_interval = log_interval
        
        # 訓練歷史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'lambdas': [],
            'val_metrics': {}
        }
        
        # 計算物理係數
        self.a_coefficient = getattr(model, 'a_coefficient', torch.tensor(55.83)).item()
        self.b_coefficient = getattr(model, 'b_coefficient', torch.tensor(-2.259)).item()
        
        logger.info(f"初始化PINNLSTMTrainer - delta_w為核心的特化訓練")
        logger.info(f"物理約束權重: {lambda_physics_init} -> {lambda_physics_max}")
        logger.info(f"一致性約束權重: {lambda_consistency_init} -> {lambda_consistency_max}")
        logger.info(f"Delta_W預測權重: {delta_w_weight_init} -> {delta_w_weight_max}")
        logger.info(f"權重增長輪數: {lambda_ramp_epochs}")
    
    def _get_lambda_values(self, epoch):
        """計算當前輪次的損失權重值"""
        # 計算線性增長因子 (0->1)
        if epoch >= self.lambda_ramp_epochs:
            ramp_factor = 1.0
        else:
            ramp_factor = epoch / self.lambda_ramp_epochs
        
        # 計算當前輪次的lambda值
        lambda_physics = self.lambda_physics_init + (self.lambda_physics_max - self.lambda_physics_init) * ramp_factor
        lambda_consistency = self.lambda_consistency_init + (self.lambda_consistency_max - self.lambda_consistency_init) * ramp_factor
        delta_w_weight = self.delta_w_weight_init + (self.delta_w_weight_max - self.delta_w_weight_init) * ramp_factor
        
        return lambda_physics, lambda_consistency, delta_w_weight
    
    def _compute_loss(self, outputs, targets, lambda_physics, lambda_consistency, delta_w_weight):
        """
        計算混合損失 - 專注於delta_w的預測精度
        
        參數:
            outputs (dict): 模型輸出
            targets (torch.Tensor): 目標值 (真實疲勞壽命)
            lambda_physics (float): 物理約束權重
            lambda_consistency (float): 一致性約束權重
            delta_w_weight (float): delta_w預測權重
            
        返回:
            dict: 包含各部分損失的字典
        """
        # 確保目標值為正數且有適當的形狀
        targets = targets.clamp(min=10.0)
        
        # 獲取預測值
        predictions = outputs['nf_pred']
        delta_w = outputs['delta_w']
        direct_delta_w = outputs['direct_delta_w']
        
        # 1. 計算疲勞壽命預測損失 (對數空間和直接空間)
        # 對數空間損失
        log_predictions = torch.log10(predictions.clamp(min=1e-6))
        log_targets = torch.log10(targets.clamp(min=1e-6))
        log_mse = F.mse_loss(log_predictions, log_targets)
        
        # 直接空間損失 (相對誤差)
        rel_error = torch.abs(predictions - targets) / targets.clamp(min=1e-6)
        direct_mse = torch.mean(rel_error ** 2)
        
        # 組合疲勞壽命預測損失 (主要使用對數空間)
        nf_loss = 0.8 * log_mse + 0.2 * direct_mse
        
        # 2. 計算delta_w預測損失 (主要目標)
        # 從目標疲勞壽命計算理論delta_w
        # 使用固定的浮點數係數以確保穩定性
        a_coef = float(self.a_coefficient) if isinstance(self.a_coefficient, float) else float(self.a_coefficient)
        b_coef = float(self.b_coefficient) if isinstance(self.b_coefficient, float) else float(self.b_coefficient)
        
        # 使用物理模型求解理論delta_w
        delta_w_theory = torch.pow(targets / a_coef, 1.0 / b_coef)
        delta_w_theory = delta_w_theory.clamp(min=1e-8)
        
        # 計算delta_w預測損失 (對數空間)
        log_delta_w = torch.log10(delta_w.clamp(min=1e-8))
        log_delta_w_theory = torch.log10(delta_w_theory)
        delta_w_loss = F.mse_loss(log_delta_w, log_delta_w_theory)
        
        # 3. 計算修正因子損失 - 鼓勵修正因子接近1.0
        correction_factor = delta_w / direct_delta_w.clamp(min=1e-8)
        # 直接鼓勵修正因子接近理論值
        optimal_factor = delta_w_theory / direct_delta_w.clamp(min=1e-8)
        factor_loss = F.mse_loss(correction_factor, optimal_factor)
        
        # 4. 物理約束損失
        # 從delta_w計算疲勞壽命，使用明確的物理公式
        nf_from_delta_w = a_coef * torch.pow(delta_w, b_coef)
        nf_from_delta_w = nf_from_delta_w.clamp(min=10.0)
        
        # 物理約束損失 - 比較物理計算的nf與直接預測的nf
        # 使用對數空間計算
        log_nf_pred = torch.log10(predictions.clamp(min=1e-6))
        log_nf_physics = torch.log10(nf_from_delta_w.clamp(min=1e-6))
        physics_loss = F.mse_loss(log_nf_pred, log_nf_physics)
        
        # 5. 總損失 - 重點關注delta_w預測和物理約束
        total_loss = (
            0.05 * nf_loss +  # 弱化直接預測損失  
            delta_w_weight * delta_w_loss +  # delta_w預測權重
            0.5 * factor_loss +  # 中等程度權重的修正因子損失
            lambda_physics * physics_loss  # 物理約束損失
        )
        
        # 返回各部分損失
        return {
            'total_loss': total_loss,
            'nf_loss': nf_loss,
            'delta_w_loss': delta_w_loss,
            'factor_loss': factor_loss,
            'physics_loss': physics_loss
        }
    
    def train_epoch(self, train_loader, epoch, lambda_physics, lambda_consistency, delta_w_weight):
        """
        訓練一個輪次
        
        參數:
            train_loader (DataLoader): 訓練數據載入器
            epoch (int): 當前輪次
            lambda_physics (float): 物理約束權重
            lambda_consistency (float): 一致性約束權重
            delta_w_weight (float): delta_w預測權重
            
        返回:
            dict: 訓練結果
        """
        self.model.train()
        total_loss = 0.0
        loss_components = {'nf_loss': 0.0, 'delta_w_loss': 0.0, 
                          'factor_loss': 0.0, 'physics_loss': 0.0}
        num_batches = 0
        
        for batch_idx, (static_features, time_series, targets) in enumerate(train_loader):
            static_features = static_features.to(self.device)
            time_series = time_series.to(self.device)
            targets = targets.to(self.device)
            
            # 梯度歸零
            self.optimizer.zero_grad()
            
            # 前向傳播
            outputs = self.model(static_features, time_series)
            
            # 計算損失
            losses = self._compute_loss(outputs, targets, lambda_physics, 
                                     lambda_consistency, delta_w_weight)
            
            # 反向傳播
            losses['total_loss'].backward()
            
            # 梯度裁剪
            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            
            # 更新權重
            self.optimizer.step()
            
            # 累計損失
            total_loss += losses['total_loss'].item()
            for key, value in losses.items():
                if key in loss_components and isinstance(value, torch.Tensor):
                    loss_components[key] += value.item()
            
            num_batches += 1
        
        # 計算平均損失
        avg_loss = total_loss / num_batches
        avg_components = {key: value / num_batches for key, value in loss_components.items()}
        
        return {'loss': avg_loss, 'components': avg_components}
    
    def validate(self, val_loader, lambda_physics, lambda_consistency, delta_w_weight):
        """
        驗證模型
        
        參數:
            val_loader (DataLoader): 驗證數據載入器
            lambda_physics (float): 物理約束權重
            lambda_consistency (float): 一致性約束權重
            delta_w_weight (float): delta_w預測權重
            
        返回:
            dict: 驗證結果
        """
        self.model.eval()
        total_loss = 0.0
        loss_components = {'nf_loss': 0.0, 'delta_w_loss': 0.0, 
                          'factor_loss': 0.0, 'physics_loss': 0.0}
        num_batches = 0
        
        all_targets = []
        all_predictions = []
        all_delta_w = []
        all_delta_w_theory = []
        
        with torch.no_grad():
            for static_features, time_series, targets in val_loader:
                static_features = static_features.to(self.device)
                time_series = time_series.to(self.device)
                targets = targets.to(self.device)
                
                # 前向傳播
                outputs = self.model(static_features, time_series)
                
                # 計算損失
                losses = self._compute_loss(outputs, targets, lambda_physics, 
                                         lambda_consistency, delta_w_weight)
                
                # 累計損失
                total_loss += losses['total_loss'].item()
                for key, value in losses.items():
                    if key in loss_components and isinstance(value, torch.Tensor):
                        loss_components[key] += value.item()
                
                # 收集預測和目標
                all_targets.append(targets.cpu().numpy())
                all_predictions.append(outputs['nf_pred'].cpu().numpy())
                all_delta_w.append(outputs['delta_w'].cpu().numpy())
                
                # 計算理論delta_w
                delta_w_theory = torch.pow(targets / self.a_coefficient, 1.0 / self.b_coefficient)
                all_delta_w_theory.append(delta_w_theory.cpu().numpy())
                
                num_batches += 1
        
        # 計算平均損失
        avg_loss = total_loss / max(num_batches, 1)  # 避免除以零
        avg_components = {key: value / max(num_batches, 1) for key, value in loss_components.items()}
        
        # 合併預測和目標
        all_targets = np.concatenate(all_targets)
        all_predictions = np.concatenate(all_predictions)
        all_delta_w = np.concatenate(all_delta_w)
        all_delta_w_theory = np.concatenate(all_delta_w_theory)
        
        # 計算評估指標
        metrics = self._compute_metrics(all_targets, all_predictions, all_delta_w, all_delta_w_theory)
        
        return {
            'loss': avg_loss, 
            'components': avg_components,
            'metrics': metrics,
            'predictions': all_predictions,
            'targets': all_targets,
            'delta_w': all_delta_w,
            'delta_w_theory': all_delta_w_theory
        }
    
    def _compute_metrics(self, targets, predictions, delta_w, delta_w_theory):
        """
        計算評估指標
        
        參數:
            targets (numpy.ndarray): 目標值
            predictions (numpy.ndarray): 預測值
            delta_w (numpy.ndarray): 預測的delta_w
            delta_w_theory (numpy.ndarray): 理論delta_w
            
        返回:
            dict: 評估指標
        """
        try:
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            
            # 計算疲勞壽命的評估指標
            rmse = np.sqrt(mean_squared_error(targets, predictions))
            r2 = r2_score(targets, predictions)
            mae = mean_absolute_error(targets, predictions)
            
            # 對數空間的評估指標
            log_targets = np.log10(targets)
            log_predictions = np.log10(predictions)
            log_rmse = np.sqrt(mean_squared_error(log_targets, log_predictions))
            log_r2 = r2_score(log_targets, log_predictions)
            
            # 相對誤差
            rel_error = np.abs((targets - predictions) / targets) * 100
            mean_rel_error = np.mean(rel_error)
            median_rel_error = np.median(rel_error)
            
            # delta_w的評估指標
            delta_w_log_mse = mean_squared_error(np.log10(delta_w_theory), np.log10(delta_w))
            delta_w_rel_error = np.abs((delta_w_theory - delta_w) / delta_w_theory) * 100
            delta_w_mean_rel_error = np.mean(delta_w_rel_error)
            
            return {
                'rmse': rmse,
                'r2': r2,
                'mae': mae,
                'log_rmse': log_rmse,
                'log_r2': log_r2,
                'mean_rel_error': mean_rel_error,
                'median_rel_error': median_rel_error,
                'delta_w_log_mse': delta_w_log_mse,
                'delta_w_mean_rel_error': delta_w_mean_rel_error
            }
        except Exception as e:
            logger.error(f"計算評估指標時出錯: {str(e)}")
            return {}
    
    def train(self, train_loader, val_loader, epochs, early_stopping_patience=20,
             save_path=None, callbacks=None):
        """
        訓練模型
        
        參數:
            train_loader (DataLoader): 訓練數據載入器
            val_loader (DataLoader): 驗證數據載入器
            epochs (int): 訓練輪數
            early_stopping_patience (int): 早停耐心值
            save_path (str): 模型保存路徑
            callbacks (list): 回調函數列表
            
        返回:
            dict: 訓練歷史
        """
        callbacks = callbacks or []
        best_val_loss = float('inf')
        best_epoch = -1
        patience_counter = 0
        
        logger.info(f"開始訓練，總輪數: {epochs}")
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # 獲取當前輪次的lambda值
            lambda_physics, lambda_consistency, delta_w_weight = self._get_lambda_values(epoch)
            
            # 訓練一個輪次
            train_results = self.train_epoch(train_loader, epoch, lambda_physics, 
                                           lambda_consistency, delta_w_weight)
            
            # 驗證
            val_results = self.validate(val_loader, lambda_physics, 
                                      lambda_consistency, delta_w_weight)
            
            # 更新學習率
            if self.scheduler is not None:
                if hasattr(self.scheduler, 'step_with_metrics'):
                    self.scheduler.step_with_metrics(val_results['loss'])
                else:
                    self.scheduler.step()
            
            # 記錄訓練歷史
            self.history['train_loss'].append(train_results['loss'])
            self.history['val_loss'].append(val_results['loss'])
            self.history['lambdas'].append({
                'lambda_physics': lambda_physics,
                'lambda_consistency': lambda_consistency,
                'delta_w_weight': delta_w_weight
            })
            
            # 記錄評估指標
            for metric, value in val_results['metrics'].items():
                if metric not in self.history['val_metrics']:
                    self.history['val_metrics'][metric] = []
                self.history['val_metrics'][metric].append(value)
            
            # 輸出訓練信息
            if epoch % self.log_interval == 0 or epoch == epochs - 1:
                epoch_time = time.time() - epoch_start_time
                log_info = [
                    f"輪次 {epoch+1}/{epochs}",
                    f"訓練損失: {train_results['loss']:.6f}",
                    f"驗證損失: {val_results['loss']:.6f}",
                ]
                
                # 添加關鍵指標
                metrics = val_results['metrics']
                if 'rmse' in metrics:
                    log_info.append(f"RMSE: {metrics['rmse']:.4f}")
                if 'r2' in metrics:
                    log_info.append(f"R²: {metrics['r2']:.4f}")
                if 'delta_w_log_mse' in metrics:
                    log_info.append(f"Delta_W Log MSE: {metrics['delta_w_log_mse']:.4f}")
                
                # 添加損失組成部分
                log_info.append(f"Delta_W損失: {train_results['components'].get('delta_w_loss', 0):.4f}")
                log_info.append(f"物理損失: {train_results['components'].get('physics_loss', 0):.4f}")
                
                # 添加學習率
                current_lr = self.optimizer.param_groups[0]['lr']
                log_info.append(f"學習率: {current_lr:.6f}")
                
                log_info.append(f"時間: {epoch_time:.2f}秒")
                logger.info(" - ".join(log_info))
            
            # 執行回調
            for callback in callbacks:
                callback(epoch, {
                    'model': self.model,
                    'optimizer': self.optimizer,
                    'train_loss': train_results['loss'],
                    'val_loss': val_results['loss'],
                    'metrics': val_results['metrics'],
                    'epoch': epoch
                })
            
            # 檢查是否是最佳模型
            if val_results['loss'] < best_val_loss:
                best_val_loss = val_results['loss']
                best_epoch = epoch
                patience_counter = 0
                
                # 保存最佳模型
                if save_path:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'epoch': epoch,
                        'val_loss': best_val_loss,
                        'val_metrics': val_results['metrics']
                    }, save_path)
                    logger.info(f"保存最佳模型，輪次: {epoch+1}，驗證損失: {best_val_loss:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"早停觸發，在輪次 {epoch+1} 停止訓練")
                    break
        
        # 添加最佳模型信息
        self.history['best_val_loss'] = best_val_loss
        self.history['best_epoch'] = best_epoch
        self.history['epochs_trained'] = epoch + 1
        
        logger.info(f"訓練完成，最佳輪次: {best_epoch+1}，最佳驗證損失: {best_val_loss:.6f}")
        
        return self.history

if __name__ == "__main__":
    # 簡單的測試代碼
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 創建模型
    model = HybridPINNLSTMModel(
        static_input_dim=5,
        time_input_dim=2,
        time_steps=4,
        use_physics_layer=True
    )
    
    # 創建測試輸入
    batch_size = 10
    static_input = torch.rand(batch_size, 5)
    time_series = torch.rand(batch_size, 4, 2)
    
    # 前向傳播
    outputs = model(static_input, time_series, return_features=True)
    
    # 測試物理關係
    delta_w = outputs['delta_w']
    nf_pred = outputs['nf_pred']
    
    # 手動計算Nf
    a_coef = model.a_coefficient.item()
    b_coef = model.b_coefficient.item()
    manual_nf = a_coef * torch.pow(delta_w, b_coef)
    
    # 比較兩個結果
    print(f"物理係數: a={a_coef}, b={b_coef}")
    print(f"delta_w樣本: {delta_w[:5].detach().numpy()}")
    print(f"模型預測Nf: {nf_pred[:5].detach().numpy()}")
    print(f"手動計算Nf: {manual_nf[:5].detach().numpy()}")
    print(f"相對誤差: {torch.abs((nf_pred - manual_nf) / manual_nf * 100)[:5].detach().numpy()}%")
    
    assert torch.allclose(nf_pred, manual_nf, rtol=1e-3), "物理計算不一致!"
    print("測試通過: 物理計算一致性確認")