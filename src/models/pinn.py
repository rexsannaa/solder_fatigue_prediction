#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
pinn.py - 物理資訊神經網絡模型
本模組實現了物理資訊神經網絡(PINN)，該網絡將物理知識融入神經網絡架構中，
用於處理銲錫接點的靜態結構參數，並引入物理約束以提高預測精度。

主要特點:
1. 多層全連接網絡處理結構參數特徵
2. 專注於預測非線性塑性應變能密度變化量(delta_w)
3. 引入物理約束層，基於能量守恆原理
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
            self.log_a = nn.Parameter(torch.tensor(np.log(a), dtype=torch.float32))
            self.log_neg_b = nn.Parameter(torch.tensor(np.log(-b), dtype=torch.float32))
            self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float32))
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
        delta_w = torch.clamp(delta_w, min=1e-8)
        
        if hasattr(self, 'trainable') and self.trainable:
            a = torch.exp(self.log_a)
            b = -torch.exp(self.log_neg_b)
            nf = a * torch.pow(delta_w, b) + self.bias
            nf = F.softplus(nf)  # 確保輸出為正值
        else:
            # 應用物理模型: Nf = a * (ΔW)^b
            nf = self.a * torch.pow(delta_w, self.b)
        
        return nf.clamp(min=10.0)  # 疲勞壽命下限為10週期


class PINNModel(nn.Module):
    """
    改進的物理資訊神經網絡(PINN)模型
    處理靜態結構參數特徵並應用物理約束
    專注於預測非線性塑性應變能密度變化量(delta_w)
    """
    def __init__(self, input_dim=5, hidden_dims=[48, 24, 12], output_dim=1, 
                dropout_rate=0.25, use_physics_layer=True, physics_layer_trainable=False,
                use_batch_norm=True, activation='relu', a_coefficient=55.83,
                b_coefficient=-2.259, l2_reg=0.002):
        """
        初始化PINN模型
        
        參數:
            input_dim (int): 輸入特徵維度
            hidden_dims (list): 隱藏層維度列表
            output_dim (int): 輸出維度，默認為1
            dropout_rate (float): Dropout率
            use_physics_layer (bool): 是否使用物理約束層
            physics_layer_trainable (bool): 物理約束層參數是否可訓練
            use_batch_norm (bool): 是否使用批次正規化
            activation (str): 激活函數類型: 'relu', 'leaky_relu', 'elu', 'selu'
            a_coefficient (float): 物理模型係數a
            b_coefficient (float): 物理模型係數b
            l2_reg (float): L2正則化係數
        """
        super(PINNModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.use_physics_layer = use_physics_layer
        self.l2_reg = l2_reg
        
        # 註冊物理係數
        self.register_buffer('a_coefficient', torch.tensor(a_coefficient, dtype=torch.float32))
        self.register_buffer('b_coefficient', torch.tensor(b_coefficient, dtype=torch.float32))
        
        # 激活函數選擇
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU(),
            'selu': nn.SELU()
        }
        self.activation = activations.get(activation.lower(), nn.LeakyReLU(0.1))
        
        # 構建特徵提取層
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # delta_w預測層 - 專注於預測非線性塑性應變能密度變化量
        self.delta_w_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.BatchNorm1d(hidden_dims[-1] // 2) if use_batch_norm else nn.Identity(),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dims[-1] // 2, 1)
        )
        
        # 物理約束層 - 將delta_w轉換為疲勞壽命
        if use_physics_layer:
            self.physics_layer = PhysicsLayer(
                a=a_coefficient, 
                b=b_coefficient, 
                trainable=physics_layer_trainable
            )
        
        # 初始化權重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化網絡權重 - 針對delta_w預測的特化初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.weight.dim() >= 2:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                else:
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                    
                if m.bias is not None:
                    # 對於最終的delta_w輸出層，設置特殊初始化
                    if m == list(self.delta_w_layer.modules())[-1]:
                        nn.init.constant_(m.bias, -5.5)  # exp(-3) ≈ 0.05，合理的delta_w初始值
                    else:
                        nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        前向傳播 - 專注於預測delta_w，再使用物理公式計算nf_pred
        
        參數:
            x (torch.Tensor): 輸入特徵，形狀為 (batch_size, input_dim)
            
        返回:
            dict: 包含預測結果的字典
                - 'delta_w': 預測的非線性塑性應變能密度變化量 (主要預測目標)
                - 'nf_pred': 從delta_w計算的疲勞壽命
                - 'features': 提取的特徵向量
                - 'l2_penalty': L2正則化懲罰
        """
        # 特徵提取
        features = self.feature_extractor(x)
        
        # 預測delta_w (對數空間)
        log_delta_w = self.delta_w_layer(features)
        delta_w = torch.exp(log_delta_w).squeeze(-1)
        delta_w = delta_w.clamp(min=1e-8)  # 確保delta_w為正值
        
        # 添加調試輸出
        print(f"[DEBUG] PINN預測 delta_w 統計 - 最小值: {delta_w.min().item():.6e}, 最大值: {delta_w.max().item():.6e}, 平均值: {delta_w.mean().item():.6e}")
        print(f"[DEBUG] PINN預測 delta_w 樣本: {delta_w[:5].detach().cpu().numpy()}")

        # 使用物理公式或物理層計算疲勞壽命
        if self.use_physics_layer:
            nf_pred = self.physics_layer(delta_w)
        else:
            # 使用物理公式: Nf = a * (ΔW)^b
            # 使用物理公式計算疲勞壽命
            a_coef = float(self.a_coefficient) if hasattr(self.a_coefficient, 'item') else self.a_coefficient
            b_coef = float(self.b_coefficient) if hasattr(self.b_coefficient, 'item') else self.b_coefficient
            nf_theory = a_coef * torch.pow(delta_w.clamp(min=1e-8), b_coef)

            # 放大因子
            nf_amp_factor = 5.0
            # 調整delta_w以保持物理一致性
            power_factor = (1.0/nf_amp_factor)**(1.0/b_coef)
            delta_w = delta_w * power_factor
            # 放大最終預測值
            nf_pred = nf_theory * nf_amp_factor
            nf_pred = nf_pred.clamp(min=10.0)  # 確保nf_pred為正值
        
        # 計算L2正則化懲罰
        l2_penalty = 0.0
        if self.l2_reg > 0:
            for param in self.parameters():
                l2_penalty += torch.norm(param, 2)
            
            # 處理非標量張量
            if isinstance(l2_penalty, torch.Tensor) and l2_penalty.dim() > 0:
                l2_penalty = l2_penalty.mean()
            
            l2_penalty = l2_penalty * self.l2_reg
        
        return {
            'delta_w': delta_w,  # 主要預測目標 - 非線性塑性應變能密度變化量
            'nf_pred': nf_pred,  # 根據delta_w計算的疲勞壽命
            'features': features,  # 提取的特徵
            'l2_penalty': l2_penalty  # L2正則化懲罰
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
        # 理論計算delta_w
        delta_w_theory = torch.pow(nf_true / self.a_coefficient, 1.0 / self.b_coefficient)
        delta_w_theory = delta_w_theory.clamp(min=1e-8)
        
        # delta_w預測損失 (在對數空間中計算)
        log_delta_w = torch.log10(delta_w.clamp(min=1e-8))
        log_delta_w_theory = torch.log10(delta_w_theory)
        delta_w_loss = F.mse_loss(log_delta_w, log_delta_w_theory)
        
        # 物理一致性損失
        nf_from_delta_w = self.a_coefficient * torch.pow(delta_w, self.b_coefficient)
        nf_physics_loss = F.mse_loss(nf_pred, nf_from_delta_w)
        
        # 結合損失
        physics_loss = lambda_physics * (delta_w_loss + 0.5 * nf_physics_loss)
        
        return physics_loss
    
    def get_delta_w(self, x):
        """
        只獲取預測的非線性塑性應變能密度變化量
        
        參數:
            x (torch.Tensor): 輸入特徵
            
        返回:
            torch.Tensor: 預測的delta_w
        """
        with torch.no_grad():
            features = self.feature_extractor(x)
            log_delta_w = self.delta_w_layer(features)
            delta_w = torch.exp(log_delta_w).squeeze(-1)
            return delta_w.clamp(min=1e-8)


if __name__ == "__main__":
    # 簡單測試
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 創建模型
    model = PINNModel(
        input_dim=5,
        hidden_dims=[32, 16], 
        use_physics_layer=True,
        activation='leaky_relu'
    )
    
    # 模擬輸入數據
    batch_size = 8
    x = torch.randn(batch_size, 5)
    
    # 前向傳播
    output = model(x)
    
    logger.info(f"模型輸出:")
    logger.info(f"  預測delta_w形狀: {output['delta_w'].shape}")
    logger.info(f"  預測疲勞壽命形狀: {output['nf_pred'].shape}")
    logger.info(f"  預測delta_w範圍: [{output['delta_w'].min().item()}, {output['delta_w'].max().item()}]")
    logger.info(f"  預測疲勞壽命範圍: [{output['nf_pred'].min().item()}, {output['nf_pred'].max().item()}]")