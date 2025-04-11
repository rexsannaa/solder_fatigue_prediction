#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
pinn.py - 物理資訊神經網絡模型
本模組實現了物理資訊神經網絡(PINN)，該網絡將物理知識融入神經網絡架構中，
用於處理銲錫接點的靜態結構參數，並引入物理約束以提高預測精度。

主要特點:
1. 多層全連接網絡處理結構參數特徵
2. 引入物理約束層，基於能量守恆原理
3. 增強模型對物理現象的理解和預測能力
4. 支援小樣本數據集的訓練
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)

class PhysicsLayer(nn.Module):
    """
    物理約束層
    實現銲錫接點疲勞壽命的物理模型約束
    基於疲勞壽命與非線性塑性應變能密度的關係: Nf=55.83⋅(ΔW)^(-2.259)
    """
    def __init__(self):
        super(PhysicsLayer, self).__init__()
        # 物理模型常數係數
        self.a = 55.83
        self.b = -2.259
        
    def forward(self, delta_w):
        """
        應用物理模型計算疲勞壽命
        
        參數:
            delta_w (torch.Tensor): 非線性塑性應變能密度變化量
            
        返回:
            torch.Tensor: 預測的疲勞壽命
        """
        # 確保輸入為正值 (物理上合理)
        delta_w = torch.clamp(delta_w, min=1e-6)
        
        # 應用物理模型: Nf = a * (ΔW)^b
        nf = self.a * torch.pow(delta_w, self.b)
        
        return nf

class PINNModel(nn.Module):
    """
    改進的物理資訊神經網絡(PINN)模型
    處理靜態結構參數特徵並應用物理約束
    """
    def __init__(self, input_dim=5, hidden_dims=[32, 16], output_dim=1, 
                 dropout_rate=0.2, use_physics_layer=True, physics_layer_trainable=False,
                 use_batch_norm=True, activation='relu', a_coefficient=55.83, b_coefficient=-2.259,
                 l2_reg=0.001):
        """初始化PINN模型"""
        super(PINNModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.use_physics_layer = use_physics_layer
        self.l2_reg = l2_reg
        
        # 選擇激活函數
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'selu':
            self.activation = nn.SELU()
        else:
            self.activation = nn.ReLU()
        
        # 構建特徵提取層
        feature_layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            feature_layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                feature_layers.append(nn.BatchNorm1d(hidden_dim))
                
            feature_layers.append(self.activation)
            
            if dropout_rate > 0:
                feature_layers.append(nn.Dropout(dropout_rate))
                
            prev_dim = hidden_dim
        
        # 特徵提取網絡
        self.feature_extractor = nn.Sequential(*feature_layers)
        
        # 改進：delta_w預測層 - 使用多層處理
        self.delta_w_layers = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.BatchNorm1d(hidden_dims[-1] // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dims[-1] // 2, 1)
        )
        
        # 物理約束層
        if use_physics_layer:
            self.physics_layer = PhysicsLayer(
                a=a_coefficient, 
                b=b_coefficient,
                trainable=physics_layer_trainable
            )
        else:
            # 如果不使用物理層，則直接預測疲勞壽命
            self.direct_predictor = nn.Linear(hidden_dims[-1], output_dim)
        
        # 初始化權重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化網絡權重 - 改進的初始化策略"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.weight.dim() >= 2:  # 檢查維度
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                else:
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    # 對於輸出層，初始化偏置使輸出在合理範圍
                    if m == self.delta_w_layers[-1]:  # delta_w輸出層
                        nn.init.constant_(m.bias, -3.0)  # exp(-3) ≈ 0.05，初始delta_w約為0.05
                    elif hasattr(self, 'direct_predictor') and m == self.direct_predictor:
                        nn.init.constant_(m.bias, 5.0)  # 初始預測值約為150
                    else:
                        nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        前向傳播 - 增強數值穩定性與物理一致性
        
        參數:
            x (torch.Tensor): 輸入特徵，形狀為 (batch_size, input_dim)
            
        返回:
            dict: 包含預測結果的字典:
                - 'nf_pred': 預測的疲勞壽命
                - 'delta_w': 預測的非線性塑性應變能密度變化量(如果使用物理層)
                - 'features': 提取的特徵表示
        """
        # 特徵提取
        features = self.feature_extractor(x)

        # 計算非線性塑性應變能密度變化量(ΔW) - 使用改進的計算方式
        delta_w_logits = self.delta_w_layers(features)  # 獲取對數空間輸出
        delta_w = torch.exp(delta_w_logits)  # 轉換為線性空間，確保為正值
        
        # 確保delta_w範圍合理 - 增強數值穩定性
        delta_w = torch.clamp(delta_w, min=1e-6, max=10.0)  # 限制在合理範圍內

        if self.use_physics_layer:
            # 使用物理層計算疲勞壽命
            nf_pred = self.physics_layer(delta_w)
        else:
            # 直接預測疲勞壽命
            nf_pred = torch.exp(self.direct_predictor(features))  # 使用exp確保為正值

        # 應用L2正則化 - 修改以確保l2_penalty是標量
        l2_penalty = 0.0
        if self.l2_reg > 0:
            for param in self.parameters():
                l2_penalty += torch.norm(param, 2)
            
            # 確保l2_penalty是標量
            if isinstance(l2_penalty, torch.Tensor) and l2_penalty.dim() > 0:
                l2_penalty = l2_penalty.mean()
            
            # 乘以正則化系數
            l2_penalty = l2_penalty * self.l2_reg

        # 確保預測值都是正數 - 增加最小閾值
        delta_w = torch.clamp(delta_w, min=1e-6)
        nf_pred = torch.clamp(nf_pred, min=10.0)  # 提高最小閾值到10

        return {
            'nf_pred': nf_pred.squeeze(-1),
            'delta_w': delta_w.squeeze(-1),
            'features': features,
            'l2_penalty': l2_penalty
        }
    
    def forward(self, x):
        """
        前向傳播
        
        參數:
            x (torch.Tensor): 輸入特徵，形狀為 (batch_size, input_dim)
            
        返回:
            dict: 包含預測結果的字典:
                - 'nf_pred': 預測的疲勞壽命
                - 'delta_w': 預測的非線性塑性應變能密度變化量(如果使用物理層)
        """
        # 特徵提取
        features = self.feature_extractor(x)
        
        # 計算非線性塑性應變能密度變化量(ΔW)
        delta_w = torch.exp(self.delta_w_predictor(features))  # 使用exp確保delta_w為正值
        
        if self.use_physics_layer:
            # 使用物理層計算疲勞壽命
            nf_pred = self.physics_layer(delta_w)
        else:
            # 直接預測疲勞壽命
            nf_pred = torch.exp(self.direct_predictor(features))  # 壽命通常為正值
        
        return {
            'nf_pred': nf_pred.squeeze(-1),
            'delta_w': delta_w.squeeze(-1)
        }
    
    def calculate_physics_loss(self, delta_w, nf_pred, nf_true, lambda_physics=1.0):
        """
        計算物理約束損失
        
        參數:
            delta_w (torch.Tensor): 預測的非線性塑性應變能密度變化量
            nf_pred (torch.Tensor): 預測的疲勞壽命
            nf_true (torch.Tensor): 真實的疲勞壽命
            lambda_physics (float): 物理約束權重
            
        返回:
            torch.Tensor: 物理約束損失
        """
        if not self.use_physics_layer:
            return torch.tensor(0.0, device=nf_pred.device)
        
        # 物理模型: Nf = a * (ΔW)^b
        a = self.physics_layer.a
        b = self.physics_layer.b
        
        # 從真實壽命推算的理論上的delta_w
        delta_w_theory = torch.pow(nf_true / a, 1 / b)
        
        # 物理損失: 預測的delta_w應與理論值一致
        physics_loss = F.mse_loss(delta_w, delta_w_theory)
        
        # 能量守恆原理: 預測的nf應符合物理模型
        predicted_nf_from_physics = a * torch.pow(delta_w, b)
        energy_loss = F.mse_loss(nf_pred, predicted_nf_from_physics)
        
        return lambda_physics * (physics_loss + energy_loss)

    def get_delta_w(self, x):
        """
        獲取預測的非線性塑性應變能密度變化量
        
        參數:
            x (torch.Tensor): 輸入特徵
            
        返回:
            torch.Tensor: 預測的非線性塑性應變能密度變化量
        """
        features = self.feature_extractor(x)
        delta_w = torch.exp(self.delta_w_predictor(features))
        return delta_w.squeeze(-1)


if __name__ == "__main__":
    # 簡單的測試代碼
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 創建一個小型PINN模型進行測試
    model = PINNModel(input_dim=5, hidden_dims=[32, 16], use_physics_layer=True)
    
    # 創建隨機輸入資料
    batch_size = 8
    x = torch.randn(batch_size, 5)
    
    # 前向傳播
    output = model(x)
    
    logger.info(f"模型輸出:")
    logger.info(f"  預測疲勞壽命形狀: {output['nf_pred'].shape}")
    logger.info(f"  預測應變能密度變化量形狀: {output['delta_w'].shape}")
    logger.info(f"  預測疲勞壽命範圍: [{output['nf_pred'].min().item()}, {output['nf_pred'].max().item()}]")
    logger.info(f"  預測應變能密度變化量範圍: [{output['delta_w'].min().item()}, {output['delta_w'].max().item()}]")