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
    基於疲勞壽命與非線性塑性應變能密度的關係: Nf=55.83*(ΔW)^(-2.259)
    """
    def __init__(self, a=55.83, b=-2.259, trainable=False):
        super(PhysicsLayer, self).__init__()
        # 物理模型常數係數
        if trainable:
            # 若允許訓練則把 a、b 設為可學習參數
            self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
            self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32))
        else:
            # 否則以 register_buffer 方式儲存成固定參數
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
                 use_batch_norm=True, activation='relu', a_coefficient=55.83,
                 b_coefficient=-2.259, l2_reg=0.001):
        """初始化PINN模型"""
        super(PINNModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.use_physics_layer = use_physics_layer
        self.l2_reg = l2_reg
        
        # 選擇激活函數
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU(),
            'selu': nn.SELU()
        }
        self.activation = activations.get(activation, nn.ReLU())

        # 構建特徵提取層
        feature_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            feature_layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                feature_layers.append(nn.BatchNorm1d(hidden_dim))
            feature_layers.append(self.activation)
            if dropout_rate > 0:
                feature_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*feature_layers)

        # delta_w預測層 - 使用多層處理
        self.delta_w_layers = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.BatchNorm1d(hidden_dims[-1] // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dims[-1] // 2, 1)
        )

        # 物理約束層或直接預測疲勞壽命
        if use_physics_layer:
            self.physics_layer = PhysicsLayer(a=a_coefficient, b=b_coefficient,
                                              trainable=physics_layer_trainable)
        else:
            self.direct_predictor = nn.Linear(hidden_dims[-1], output_dim)

        # 初始化權重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化網絡權重 - 改進的初始化策略"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 針對線性層
                if m.weight.dim() >= 2:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                else:
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    # 對於輸出層，初始化偏置
                    if m == list(self.delta_w_layers.modules())[-1]:  # delta_w輸出層
                        nn.init.constant_(m.bias, -3.0)  # exp(-3) ≈ 0.05
                    elif hasattr(self, 'direct_predictor') and m == self.direct_predictor:
                        nn.init.constant_(m.bias, 5.0)
                    else:
                        nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        前向傳播 - 專注於預測 delta_w，不直接計算 nf_pred

        參數:
            x (torch.Tensor): 輸入特徵，形狀為 (batch_size, input_dim)
            
        返回:
            dict: 
                - 'delta_w': 預測的非線性塑性應變能密度變化量
                - 'nf_pred': 疲勞壽命預測（基於物理層）
                - 'features': 特徵向量
                - 'l2_penalty': L2正則化懲罰項
        """
        # 特徵提取
        features = self.feature_extractor(x)

        # 預測 delta_w (log 空間)
        delta_w = torch.exp(self.delta_w_layers(features))
        delta_w = torch.clamp(delta_w, min=1e-6)

        if self.use_physics_layer:
            nf_pred = self.physics_layer(delta_w)
        else:
            # 若不使用物理層，則使用硬編碼的 a, b 計算 nf_pred
            a = getattr(self, 'a', 55.83)
            b = getattr(self, 'b', -2.259)
            nf_pred = a * torch.pow(delta_w, b)

        # 避免數值問題，稍微做個下界
        nf_pred = torch.clamp(nf_pred, min=10.0)

        # L2正則化
        l2_penalty = 0.0
        if self.l2_reg > 0:
            for param in self.parameters():
                l2_penalty += torch.norm(param, 2)
            # 若是Tensor且dim>0則需mean
            if isinstance(l2_penalty, torch.Tensor) and l2_penalty.dim() > 0:
                l2_penalty = l2_penalty.mean()
            l2_penalty = l2_penalty * self.l2_reg

        return {
            'delta_w': delta_w.squeeze(-1),
            'nf_pred': nf_pred.squeeze(-1),
            'features': features,
            'l2_penalty': l2_penalty
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
            # 未啟用物理層則不需計算物理約束
            return torch.tensor(0.0, device=nf_pred.device)

        # 物理模型: Nf = a * (ΔW)^b
        a = self.physics_layer.a
        b = self.physics_layer.b

        # 從真實壽命推得理論上的 delta_w
        delta_w_theory = torch.pow(nf_true / a, 1.0 / b)
        # 預測的 delta_w 與理論值差異
        physics_loss = F.mse_loss(delta_w, delta_w_theory)
        # 從預測的 delta_w 計算 nf
        predicted_nf_from_physics = a * torch.pow(delta_w, b)
        # 預測 nf 與 由 delta_w 算出的理論 nf 差異
        energy_loss = F.mse_loss(nf_pred, predicted_nf_from_physics)

        return lambda_physics * (physics_loss + energy_loss)

    def get_delta_w(self, x):
        """
        只獲取預測的非線性塑性應變能密度變化量
        
        參數:
            x (torch.Tensor): 輸入特徵
        返回:
            torch.Tensor: delta_w
        """
        features = self.feature_extractor(x)
        delta_w = torch.exp(self.delta_w_layers(features))
        return delta_w.squeeze(-1)


if __name__ == "__main__":
    # 簡單測試
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    model = PINNModel(input_dim=5, hidden_dims=[32, 16], use_physics_layer=True)
    batch_size = 8
    x = torch.randn(batch_size, 5)
    output = model(x)

    logger.info(f"模型輸出:")
    logger.info(f"  預測疲勞壽命形狀: {output['nf_pred'].shape}")
    logger.info(f"  預測應變能密度變化量形狀: {output['delta_w'].shape}")
    logger.info(f"  預測疲勞壽命範圍: [{output['nf_pred'].min().item()}, {output['nf_pred'].max().item()}]")
    logger.info(f"  預測應變能密度變化量範圍: [{output['delta_w'].min().item()}, {output['delta_w'].max().item()}]")
