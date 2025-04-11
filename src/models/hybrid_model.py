#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
hybrid_model.py - 混合PINN-LSTM模型
本模組實現了結合物理信息神經網絡(PINN)和長短期記憶網絡(LSTM)的混合模型，
用於準確預測銲錫接點的疲勞壽命。

主要特點:
1. 使用PINN分支從靜態特徵中提取物理關係並強化物理約束
2. 使用LSTM分支從時間序列數據中提取動態特徵
3. 採用優化的特徵融合與損失平衡機制
4. 針對小樣本數據集(81筆)專門優化
5. 提供分階段訓練和物理約束驅動的預測流程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import math
import traceback

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
            a (float): 物理模型係數a
            b (float): 物理模型係數b
            trainable (bool): 是否允許係數可訓練(微調)
        """
        super(PhysicsLayer, self).__init__()
        
        # 初始化物理模型係數
        if trainable:
            # 使用對數參數化確保a為正值、b為負值
            self.log_a = nn.Parameter(torch.tensor(np.log(a), dtype=torch.float32))
            self.log_neg_b = nn.Parameter(torch.tensor(np.log(-b), dtype=torch.float32))
            # 添加額外的偏置參數，提高靈活性
            self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        else:
            self.register_buffer('a', torch.tensor(a, dtype=torch.float32))
            self.register_buffer('b', torch.tensor(b, dtype=torch.float32))
        
        self.trainable = trainable
    
    def forward(self, delta_w):
        """
        應用物理模型計算疲勞壽命
    
        參數:
            delta_w (torch.Tensor): 非線性塑性應變能密度變化量
        
        返回:
            torch.Tensor: 預測的疲勞壽命
        """
        # 確保delta_w為正值，使用更安全的最小值
        delta_w = torch.clamp(delta_w, min=1e-8)
    
        if self.trainable:
            # 從參數計算實際係數值
            a = torch.exp(self.log_a)
            b = -torch.exp(self.log_neg_b)  # 負指數        
            # 應用物理模型: Nf = a * (ΔW)^b
            nf = a * torch.pow(delta_w, b) + self.bias
            # 確保輸出為正值
            nf = F.softplus(nf)
        else:
            a = self.a
            b = self.b
            nf = a * torch.pow(delta_w, b)      
    
        # 確保輸出是正數且數值穩定
        nf = torch.clamp(nf, min=10.0)
    
        return nf


class AttentionLayer(nn.Module):
    """
    注意力機制層
    計算時間序列中不同時間步的重要性權重
    """
    def __init__(self, hidden_size):
        """
        初始化注意力層
        
        參數:
            hidden_size (int): 隱藏層大小
        """
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        
        # 注意力計算參數
        self.attention_weights = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, lstm_output, mask=None):
        """
        前向傳播
    
        參數:
            lstm_output (torch.Tensor): LSTM輸出，形狀為 (batch_size, seq_len, hidden_size)
            mask (torch.Tensor, optional): 用於遮蔽填充值的掩碼
        
        返回:
            tuple: (加權後的特徵向量, 注意力權重)
        """
        # 計算注意力分數
        attention_scores = self.attention_weights(lstm_output)  # (batch_size, seq_len, 1)
        attention_scores = attention_scores.squeeze(-1)  # (batch_size, seq_len)
    
        # 如果有掩碼，將填充位置的分數設為負無窮大
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
    
        # 應用softmax獲取注意力權重
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_len)
    
        # 將注意力權重應用於LSTM輸出
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch_size, 1, seq_len)
            lstm_output  # (batch_size, seq_len, hidden_size)
        )  # (batch_size, 1, hidden_size)
    
        context_vector = context_vector.squeeze(1)  # (batch_size, hidden_size)
    
        return context_vector, attention_weights


class PINNModel(nn.Module):
    """
    物理資訊神經網絡(PINN)模型
    處理靜態結構參數特徵並應用物理約束
    """
    def __init__(self, input_dim=5, hidden_dims=[32, 16], output_dim=1, 
                 dropout_rate=0.2, use_physics_layer=True, physics_layer_trainable=False,
                 use_batch_norm=True, activation='relu', a_coefficient=55.83, b_coefficient=-2.259,
                 l2_reg=0.001):
        """
        初始化PINN模型
        
        參數:
            input_dim (int): 輸入特徵維度
            hidden_dims (list): 隱藏層維度列表
            output_dim (int): 輸出維度
            dropout_rate (float): Dropout比率
            use_physics_layer (bool): 是否使用物理約束層
            physics_layer_trainable (bool): 物理層參數是否可訓練
            use_batch_norm (bool): 是否使用批次正規化
            activation (str): 激活函數類型
            a_coefficient (float): 物理模型係數a
            b_coefficient (float): 物理模型係數b
            l2_reg (float): L2正則化系數
        """
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
        
        # 能量密度變化量(ΔW)預測層
        # 使用log空間預測以確保輸出為正值
        self.delta_w_predictor = nn.Linear(hidden_dims[-1], 1)
        
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
        """初始化網絡權重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.weight.dim() >= 2:  # 檢查維度
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    # 對於輸出層，初始化偏置使輸出在合理範圍
                    if m.out_features == 1:  # 輸出層
                        nn.init.constant_(m.bias, 7.0)  # log(1000) ≈ 7.0，初始預測值約為1000
                    else:
                        nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        前向傳播

        參數:
            x (torch.Tensor): 輸入特徵，形狀為 (batch_size, input_dim)
        
        返回:
            dict: 包含預測結果的字典:
                - 'nf_pred': 預測的疲勞壽命
                - 'delta_w': 預測的非線性塑性應變能密度變化量(如果使用物理層)
                - 'features': 提取的特徵表示
        """
        # 提取特徵
        features = self.feature_extractor(x)

        # 計算非線性塑性應變能密度變化量(ΔW)
        delta_w = torch.exp(self.delta_w_predictor(features))  # 使用exp確保為正值

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

        # 確保預測值都是正數
        delta_w = torch.clamp(delta_w, min=1e-8)
        nf_pred = torch.clamp(nf_pred, min=1e-8)

        return {
            'nf_pred': nf_pred.squeeze(-1),
            'delta_w': delta_w.squeeze(-1),
            'features': features,
            'l2_penalty': l2_penalty
        }

    
class LSTMModel(nn.Module):
    """
    長短期記憶網絡模型
    專門用於處理銲錫接點的非線性塑性應變功時間序列資料
    """
    def __init__(self, input_dim=2, hidden_size=32, num_layers=1, output_dim=1,
                 bidirectional=True, dropout_rate=0.2, use_attention=True,
                 l2_reg=0.001):
        """
        初始化LSTM模型
        
        參數:
            input_dim (int): 輸入特徵維度，預設為2 (上下界面非線性塑性應變功)
            hidden_size (int): LSTM隱藏層大小
            num_layers (int): LSTM層數
            output_dim (int): 輸出維度，預設為1 (疲勞壽命)
            bidirectional (bool): 是否使用雙向LSTM
            dropout_rate (float): Dropout比率
            use_attention (bool): 是否使用注意力機制
            l2_reg (float): L2正則化系數
        """
        super(LSTMModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.l2_reg = l2_reg
        
        # LSTM層
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # 計算LSTM輸出維度
        lstm_output_dim = hidden_size * 2 if bidirectional else hidden_size
        
        # 注意力層
        if use_attention:
            self.attention = AttentionLayer(lstm_output_dim)
        
        # 全連接層，用於最終預測
        fc_layers = []
        fc_input_dim = lstm_output_dim
        fc_hidden_dims = [lstm_output_dim // 2]
        
        for hidden_dim in fc_hidden_dims:
            fc_layers.append(nn.Linear(fc_input_dim, hidden_dim))
            fc_layers.append(nn.BatchNorm1d(hidden_dim))
            fc_layers.append(nn.ReLU())
            if dropout_rate > 0:
                fc_layers.append(nn.Dropout(dropout_rate))
            fc_input_dim = hidden_dim
        
        self.fc_layers = nn.Sequential(*fc_layers)
        self.output_layer = nn.Linear(fc_input_dim, output_dim)
        
        # 初始化權重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化網絡權重"""
        for name, param in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    if param.dim() >= 2:  # 檢查維度
                        nn.init.xavier_uniform_(param.data)
                    else:
                        nn.init.uniform_(param.data, -0.1, 0.1)
                elif 'weight_hh' in name:
                    if param.dim() >= 2:  # 檢查維度
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.uniform_(param.data, -0.1, 0.1)
                elif 'bias' in name:
                    nn.init.zeros_(param.data)
            elif 'attention_weights' in name:
                if param.dim() >= 2:  # 檢查維度
                    nn.init.xavier_uniform_(param.data)
                else:
                    # 處理一維參數
                    nn.init.uniform_(param.data, -0.1, 0.1)
            elif 'linear' in name and 'weight' in name:
                if param.dim() >= 2:  # 檢查維度
                    nn.init.xavier_uniform_(param.data)
                else:
                    # 處理一維參數
                    nn.init.uniform_(param.data, -0.1, 0.1)
            elif 'linear' in name and 'bias' in name:
                nn.init.zeros_(param.data)
    
    def forward(self, x, return_attention=False):
        """
        前向傳播

        參數:
            x (torch.Tensor): 輸入時間序列，形狀為 (batch_size, seq_len, input_dim)
            return_attention (bool): 是否返回注意力權重
        
        返回:
            dict: 包含預測結果的字典:
                - 'output': 預測的疲勞壽命
                - 'features': 提取的時序特徵
                - 'attention_weights': 注意力權重 (如果使用注意力機制且return_attention=True)
        """
        # LSTM前向傳播
        lstm_output, (hidden, cell) = self.lstm(x)
        # lstm_output形狀: (batch_size, seq_len, hidden_size*2 if bidirectional else hidden_size)

        # 獲取特徵向量
        if self.use_attention:
            # 使用注意力機制
            context_vector, attention_weights = self.attention(lstm_output)
        else:
            # 使用最後一個時間步的輸出
            if self.bidirectional:
                # 如果是雙向LSTM，合併前向和後向的最後隱藏狀態
                last_forward = hidden[-2, :, :]
                last_backward = hidden[-1, :, :]
                context_vector = torch.cat((last_forward, last_backward), dim=1)
            else:
                context_vector = hidden[-1, :, :]
            attention_weights = None

        # 全連接層處理
        # 使用log空間預測，確保輸出為正值
        fc_output = self.fc_layers(context_vector)
        output = torch.exp(self.output_layer(fc_output))

        # 確保預測值是正數
        output = torch.clamp(output, min=1e-8)

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

        result = {
            'output': output.squeeze(-1),
            'features': context_vector,
            'l2_penalty': l2_penalty
        }

        if return_attention and attention_weights is not None:
            result['attention_weights'] = attention_weights

        return result

class FeatureFusionLayer(nn.Module):
    """
    特徵融合層
    融合PINN和LSTM分支提取的特徵，並透過注意力機制處理特徵的重要性
    """
    def __init__(self, pinn_feature_dim, lstm_feature_dim, fusion_dim=32, 
                 dropout_rate=0.2, use_batch_norm=True):
        """
        初始化特徵融合層
        
        參數:
            pinn_feature_dim (int): PINN分支特徵維度
            lstm_feature_dim (int): LSTM分支特徵維度
            fusion_dim (int): 融合後的特徵維度
            dropout_rate (float): Dropout比率
            use_batch_norm (bool): 是否使用批次正規化
        """
        super(FeatureFusionLayer, self).__init__()
        
        self.pinn_feature_dim = pinn_feature_dim
        self.lstm_feature_dim = lstm_feature_dim
        self.fusion_dim = fusion_dim
        
        # 門控機制用於動態調整兩個分支的重要性
        self.gate_network = nn.Sequential(
            nn.Linear(pinn_feature_dim + lstm_feature_dim, 2),
            nn.Softmax(dim=1)
        )
        
        # 特徵投影層
        self.pinn_projection = nn.Linear(pinn_feature_dim, fusion_dim)
        self.lstm_projection = nn.Linear(lstm_feature_dim, fusion_dim)
        
        # 融合後的特徵處理
        fusion_layers = []
        fusion_layers.append(nn.Linear(fusion_dim, fusion_dim))
        
        if use_batch_norm:
            fusion_layers.append(nn.BatchNorm1d(fusion_dim))
            
        fusion_layers.append(nn.ReLU())
        
        if dropout_rate > 0:
            fusion_layers.append(nn.Dropout(dropout_rate))
        
        self.fusion_layers = nn.Sequential(*fusion_layers)
        
    def forward(self, pinn_features, lstm_features):
        """
        前向傳播
    
        參數:
            pinn_features (torch.Tensor): PINN分支特徵
            lstm_features (torch.Tensor): LSTM分支特徵
        
        返回:
            tuple: (融合特徵, 門控權重)
        """
        # 計算特徵融合的門控權重
        combined_features = torch.cat([pinn_features, lstm_features], dim=1)
        gate_weights = self.gate_network(combined_features)
    
        # 投影特徵到相同的空間
        pinn_projected = self.pinn_projection(pinn_features)
        lstm_projected = self.lstm_projection(lstm_features)
    
        # 加權融合
        fused_features = (
            gate_weights[:, 0].unsqueeze(1) * pinn_projected + 
            gate_weights[:, 1].unsqueeze(1) * lstm_projected
        )
    
        # 進一步處理融合特徵
        output_features = self.fusion_layers(fused_features)
    
        return output_features, gate_weights
    

class HybridPINNLSTMModel(nn.Module):
    """
    改進的混合PINN-LSTM模型
    結合物理信息神經網絡和長短期記憶網絡的優勢，專為小樣本數據集優化
    """
    def __init__(self, 
                 static_input_dim=5,
                 time_input_dim=2,
                 time_steps=4,
                 pinn_hidden_dims=[64, 32, 16],  # 修改點: 增加模型容量
                 lstm_hidden_size=64,            # 修改點: 增加LSTM隱藏層大小
                 lstm_num_layers=2,              # 修改點: 增加LSTM層數
                 fusion_dim=32,                  # 修改點: 增加融合層維度
                 dropout_rate=0.1,               # 修改點: 減少dropout以減少資訊損失
                 bidirectional=True,
                 use_attention=True,
                 use_physics_layer=True,
                 physics_layer_trainable=True,   # 修改點: 允許物理層參數微調
                 use_batch_norm=True,
                 pinn_weight_init=0.8,           # 修改點: 增加PINN分支初始權重
                 lstm_weight_init=0.2,           # 修改點: 減少LSTM分支初始權重
                 a_coefficient=55.83,
                 b_coefficient=-2.259,
                 use_log_transform=True,
                 ensemble_method='weighted',     # 可考慮改為'deep_fusion'
                 l2_reg=0.0005):                 # 修改點: 減少L2正則化強度
        super(HybridPINNLSTMModel, self).__init__()
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
            physics_layer_trainable (bool): 物理層參數是否可訓練
            use_batch_norm (bool): 是否使用批次正規化
            pinn_weight_init (float): PINN分支初始權重
            lstm_weight_init (float): LSTM分支初始權重
            a_coefficient (float): 物理模型係數a
            b_coefficient (float): 物理模型係數b
            use_log_transform (bool): 是否使用對數變換
            ensemble_method (str): 集成方法
            l2_reg (float): L2正則化系數
        """
        super(HybridPINNLSTMModel, self).__init__()
        
        self.static_input_dim = static_input_dim
        self.time_input_dim = time_input_dim
        self.time_steps = time_steps
        self.use_physics_layer = use_physics_layer
        self.use_log_transform = use_log_transform
        self.ensemble_method = ensemble_method
        self.l2_reg = l2_reg
        self.a_coefficient = a_coefficient
        self.b_coefficient = b_coefficient
        
        # 1. PINN分支，處理靜態結構參數
        self.pinn_branch = PINNModel(
            input_dim=static_input_dim,
            hidden_dims=pinn_hidden_dims,
            output_dim=1,
            dropout_rate=dropout_rate,
            use_physics_layer=use_physics_layer,
            physics_layer_trainable=physics_layer_trainable,
            use_batch_norm=use_batch_norm,
            a_coefficient=a_coefficient,
            b_coefficient=b_coefficient,
            l2_reg=l2_reg
        )
        
        # 2. LSTM分支，處理時間序列資料
        self.lstm_branch = LSTMModel(
            input_dim=time_input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            output_dim=1,
            bidirectional=bidirectional,
            dropout_rate=dropout_rate,
            use_attention=use_attention,
            l2_reg=l2_reg
        )
        
        # 3. 分支權重 - 可訓練參數
        if ensemble_method == 'weighted':
            # 初始化分支權重 - 使用sigmoid確保正值並在0-1之間，且和為1
            weight_param = torch.tensor(
                [math.log(pinn_weight_init / (1 - pinn_weight_init))], 
                dtype=torch.float32
            )
            self.branch_weight_param = nn.Parameter(weight_param)
        
        # 4. 特徵融合層 (用於'gate'和'deep_fusion'方法)
        if ensemble_method in ['gate', 'deep_fusion']:
            # 計算PINN分支最後隱藏層維度
            pinn_feature_dim = pinn_hidden_dims[-1]
            
            # 計算LSTM分支最後隱藏層維度
            lstm_feature_dim = lstm_hidden_size * 2 if bidirectional else lstm_hidden_size
            
            self.fusion_layer = FeatureFusionLayer(
                pinn_feature_dim=pinn_feature_dim,
                lstm_feature_dim=lstm_feature_dim,
                fusion_dim=fusion_dim,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm
            )
            
            # 對於deep_fusion方法，添加輸出層
            if ensemble_method == 'deep_fusion':
                self.output_layer = nn.Sequential(
                    nn.Linear(fusion_dim, fusion_dim // 2),
                    nn.ReLU(),
                    nn.Linear(fusion_dim // 2, 1)
                )
        # 修改點: 添加對數變換的縮放因子，以處理大範圍值
        self.log_scale = nn.Parameter(torch.tensor([2.5], dtype=torch.float32))  # 初始值更大，縮放更顯著
        # 修改點: 添加最終輸出校正層
        self.output_correction = nn.Sequential(
            nn.Linear(1, 32),            # 增加層寬度提升表達能力
            nn.BatchNorm1d(32),          # 添加批標準化提高穩定性
            nn.LeakyReLU(0.1),           # 使用LeakyReLU激活改善訓練
            nn.Dropout(0.1),             # 輕微Dropout防止過擬合
            nn.Linear(32, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 1)
        )


        # 初始化權重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        初始化網絡權重 - 改進初始化策略
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.weight.dim() >= 2:  # 檢查維度
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                else:
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                    
                if m.bias is not None:
                    # ==== 關鍵修改：對於輸出層，初始化偏置使輸出在合理範圍 ====
                    if m.out_features == 1:  # 輸出層
                        nn.init.constant_(m.bias, 5.0)  # log(150) ≈ 5.0，初始預測值約為150
                    else:
                        nn.init.zeros_(m.bias)
                        
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def get_branch_weights(self):
        """獲取分支權重"""
        if self.ensemble_method == 'weighted':
            # 使用sigmoid轉換參數，獲得PINN分支權重
            pinn_weight = torch.sigmoid(self.branch_weight_param)
            # LSTM分支權重為補值
            lstm_weight = 1 - pinn_weight
            # 返回一個包含兩個元素的張量
            return torch.tensor([pinn_weight.item(), lstm_weight.item()], device=pinn_weight.device)
    
        # 如果不是加權方法，返回預設權重 [0.5, 0.5]
        return torch.tensor([0.5, 0.5], device=self.branch_weight_param.device if hasattr(self, 'branch_weight_param') else torch.device('cpu'))
    
    def calculate_loss(self, outputs, targets, lambda_physics=0.5, lambda_consistency=0.1):
        """
        計算混合損失，增強在對數空間的損失權重

        參數:
            outputs (dict): 模型輸出
            targets (torch.Tensor): 目標疲勞壽命
            lambda_physics (float): 物理約束損失權重
            lambda_consistency (float): 分支間一致性損失權重

        返回:
            dict: 包含各部分損失的字典
        """
        # 確保targets和predictions都是正值，提高安全性
        targets = torch.clamp(targets, min=10.0)  # 提高最小閾值到10
        predictions = torch.clamp(outputs['nf_pred'], min=10.0)  # 提高最小閾值到10

        # 獲取對數空間的目標值和預測值
        log_targets = torch.log10(targets)  # 使用以10為底的對數，更直觀
        log_predictions = torch.log10(predictions)

        # 1. 主要預測損失 - 主要使用對數空間損失
        # 常規空間的均方誤差
        mse_loss = F.mse_loss(predictions, targets)

        # 對數空間的均方誤差 (更適合跨度大的數據)
        log_mse_loss = F.mse_loss(log_predictions, log_targets)

        # 結合損失，95%使用對數空間損失
        pred_loss = 0.05 * mse_loss + 0.95 * log_mse_loss
        
        # 針對大值區域的特殊處理 - 增強大值區域權重
        large_value_mask = log_targets > 6.0  # 閾值對應約1100循環
        if torch.any(large_value_mask):
            large_value_loss = F.mse_loss(
                log_predictions[large_value_mask], 
                log_targets[large_value_mask]
            )
            pred_loss = pred_loss + large_value_loss * 2.0  # 增加大值區域權重

        # 2. 物理約束損失
        if self.use_physics_layer and 'delta_w' in outputs:
            # 確保delta_w是正值
            delta_w = torch.clamp(outputs['delta_w'], min=1e-6)  # 提高最小閾值
        
            # 從真實壽命推算的理論上的delta_w (使用物理公式)
            delta_w_theory = torch.pow(targets / self.a_coefficient, 1/self.b_coefficient)
            delta_w_theory = torch.clamp(delta_w_theory, min=1e-6)  # 提高最小閾值
        
            # 使用對數空間的損失，對於大範圍值更穩定
            log_delta_w = torch.log10(delta_w)
            log_delta_w_theory = torch.log10(delta_w_theory)
            delta_w_loss = F.mse_loss(log_delta_w, log_delta_w_theory)
        
            # 物理預測一致性損失 - 強制PINN分支符合物理模型
            nf_physics = self.a_coefficient * torch.pow(delta_w, self.b_coefficient)
            nf_physics = torch.clamp(nf_physics, min=10.0)  # 提高最小閾值到10
        
            log_nf_physics = torch.log10(nf_physics)
            log_pinn_pred = torch.log10(torch.clamp(outputs['pinn_nf_pred'], min=10.0))  # 提高最小閾值到10
            physics_consistency_loss = F.mse_loss(log_pinn_pred, log_nf_physics)
        
            # 整合物理損失 - 增加物理一致性權重
            physics_loss = 0.5 * delta_w_loss + 0.5 * physics_consistency_loss  # 物理一致性權重從0.4增加到0.5
        
            # 新增: 直接監督總輸出符合物理模型 (這是確保最終預測遵循物理模型的關鍵)
            log_nf_from_physics = torch.log10(nf_physics)
            direct_physics_loss = F.mse_loss(log_predictions, log_nf_from_physics)
            physics_loss = physics_loss + 0.7 * direct_physics_loss  # 大幅提高直接物理監督權重
        
            # 針對大值區域的物理約束強化
            if torch.any(large_value_mask):
                large_delta_w_loss = F.mse_loss(
                    log_delta_w[large_value_mask],
                    log_delta_w_theory[large_value_mask]
                )
                physics_loss = physics_loss + large_delta_w_loss * 1.5  # 增強大值區域物理約束
        else:
            physics_loss = torch.tensor(0.0, device=targets.device)
            delta_w_loss = torch.tensor(0.0, device=targets.device)
            physics_consistency_loss = torch.tensor(0.0, device=targets.device)
            direct_physics_loss = torch.tensor(0.0, device=targets.device)

        # 3. 分支間一致性損失
        if 'pinn_nf_pred' in outputs and 'lstm_nf_pred' in outputs:
            pinn_pred = torch.clamp(outputs['pinn_nf_pred'], min=10.0)  # 提高最小閾值到10
            lstm_pred = torch.clamp(outputs['lstm_nf_pred'], min=10.0)  # 提高最小閾值到10
        
            log_pinn_pred = torch.log10(pinn_pred)
            log_lstm_pred = torch.log10(lstm_pred)
        
            consistency_loss = F.mse_loss(log_pinn_pred, log_lstm_pred)
        else:
            consistency_loss = torch.tensor(0.0, device=targets.device)

        # 4. L2正則化損失
        l2_loss = outputs.get('l2_penalty', torch.tensor(0.0, device=targets.device))
        
        # 確保l2_loss是標量
        if isinstance(l2_loss, torch.Tensor) and l2_loss.dim() > 0:
            l2_loss = l2_loss.mean()

        # 5. 自適應物理約束權重 - 大幅增強物理約束權重
        rel_error = torch.abs((targets - predictions) / targets)
        mean_rel_error = torch.mean(rel_error)
        # 當誤差較大時，更加依賴物理模型
        adaptive_lambda_physics = lambda_physics * (5.0 + torch.clamp(mean_rel_error, 0.0, 5.0))

        # 6. 輸出偏移損失 - 鼓勵輸出遠離零
        zero_avoidance_loss = torch.mean(1.0 / (predictions + 1e-3))

        # 7. 總損失 - 物理約束權重更大
        total_loss = (
            pred_loss + 
            adaptive_lambda_physics * physics_loss + 
            lambda_consistency * consistency_loss +
            l2_loss +
            0.1 * zero_avoidance_loss
        )

        # 確保所有返回的損失都是標量
        def ensure_scalar(tensor):
            if isinstance(tensor, torch.Tensor) and tensor.dim() > 0:
                return tensor.mean()
            return tensor

        # 建立損失字典，確保所有值都是標量
        loss_dict = {
            'total_loss': ensure_scalar(total_loss),
            'pred_loss': ensure_scalar(pred_loss),
            'physics_loss': ensure_scalar(physics_loss),
            'consistency_loss': ensure_scalar(consistency_loss),
            'mse_loss': ensure_scalar(mse_loss),
            'log_mse_loss': ensure_scalar(log_mse_loss),
            'l2_loss': ensure_scalar(l2_loss),
            'rel_error': ensure_scalar(mean_rel_error),
            'zero_avoidance_loss': ensure_scalar(zero_avoidance_loss)
        }
        
        # 添加物理損失組件（如果可用）
        if self.use_physics_layer and 'delta_w' in outputs:
            loss_dict.update({
                'delta_w_loss': ensure_scalar(delta_w_loss),
                'physics_consistency_loss': ensure_scalar(physics_consistency_loss),
                'direct_physics_loss': ensure_scalar(direct_physics_loss)
            })

        return loss_dict
    
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
        pinn_features = pinn_output['features']
        pinn_l2_penalty = pinn_output.get('l2_penalty', torch.tensor(0.0, device=static_features.device))

        # 2. LSTM分支前向傳播
        lstm_output = self.lstm_branch(time_series, return_attention=return_features)
        lstm_nf_pred = lstm_output['output']
        lstm_features = lstm_output['features']
        lstm_l2_penalty = lstm_output.get('l2_penalty', torch.tensor(0.0, device=static_features.device))
        
        # 3. 根据集成方法合并预测结果
        # 確保預測值均為正數且不為零
        pinn_nf_pred = torch.clamp(pinn_nf_pred, min=10.0)
        lstm_nf_pred = torch.clamp(lstm_nf_pred, min=10.0)
        
        # 初始化結果字典
        result = {}
        
        if self.use_physics_layer and hasattr(self, 'a_coefficient') and hasattr(self, 'b_coefficient'):
            # 直接使用物理模型計算最終預測，完全跳過融合步驟
            # 物理模型: Nf = a * (delta_w)^b
            physics_nf = self.a_coefficient * torch.pow(pinn_delta_w, self.b_coefficient)
            
            # 使用物理計算結果作為最終預測，這是關鍵修改
            nf_pred = physics_nf
            
            # 將物理計算的結果添加到結果中
            result['physics_nf'] = physics_nf
        else:
            # 如果未啟用物理層，則回退到原有的融合邏輯
            if self.ensemble_method == 'weighted':
                # 獲取分支權重
                branch_weights = self.get_branch_weights()
                
                # 確保預測值的維度一致以進行加權
                if pinn_nf_pred.dim() != lstm_nf_pred.dim() or pinn_nf_pred.shape != lstm_nf_pred.shape:
                    if pinn_nf_pred.dim() < lstm_nf_pred.dim():
                        pinn_nf_pred = pinn_nf_pred.view(lstm_nf_pred.shape)
                    elif lstm_nf_pred.dim() < pinn_nf_pred.dim():
                        lstm_nf_pred = lstm_nf_pred.view(pinn_nf_pred.shape)
                
                # 使用對數空間加權平均
                log_pinn_pred = torch.log(pinn_nf_pred)
                log_lstm_pred = torch.log(lstm_nf_pred)
                log_nf_pred = branch_weights[0] * log_pinn_pred + branch_weights[1] * log_lstm_pred
                
                # 轉回線性空間
                nf_pred = torch.exp(log_nf_pred)
                
                # 融合權重
                fusion_weights = torch.tensor([branch_weights[0].item(), 
                                            branch_weights[1].item()], 
                                            device=static_features.device)
                
                # 添加融合權重到結果中
                result['fusion_weights'] = fusion_weights
                
            elif self.ensemble_method == 'gate':
                # 使用特徵融合層進行融合，並使用門控機制
                fused_features, gate_weights = self.fusion_layer(pinn_features, lstm_features)
                
                # 確保預測值的維度一致以進行加權
                if pinn_nf_pred.dim() != lstm_nf_pred.dim() or pinn_nf_pred.shape != lstm_nf_pred.shape:
                    if pinn_nf_pred.dim() < lstm_nf_pred.dim():
                        pinn_nf_pred = pinn_nf_pred.view(lstm_nf_pred.shape)
                    elif lstm_nf_pred.dim() < pinn_nf_pred.dim():
                        lstm_nf_pred = lstm_nf_pred.view(pinn_nf_pred.shape)
                
                # 加權平均兩個分支的預測
                nf_pred = gate_weights[:, 0].unsqueeze(-1) * pinn_nf_pred + gate_weights[:, 1].unsqueeze(-1) * lstm_nf_pred
                
                # 融合權重
                fusion_weights = gate_weights.mean(dim=0)  # 取平均為整體權重
                
                # 定義dynamic_weights為gate_weights.T，便於合併
                batch_dynamic_weights = gate_weights.t()  # 形狀為(2, batch_size)
                
                # 添加到結果中
                result['fusion_weights'] = fusion_weights
                result['dynamic_weights'] = batch_dynamic_weights
                
            else:  # deep_fusion
                # 使用特徵融合層進行深度融合
                fused_features, gate_weights = self.fusion_layer(pinn_features, lstm_features)
                
                # 使用輸出層進行預測
                if self.use_log_transform:
                    # 使用對數空間進行預測，確保預測值為正
                    nf_pred = torch.exp(self.output_layer(fused_features)).squeeze(-1)
                else:
                    # 直接預測，使用softplus確保正值
                    nf_pred = F.softplus(self.output_layer(fused_features)).squeeze(-1)
                
                # 融合權重
                fusion_weights = gate_weights.mean(dim=0)  # 取平均為整體權重
                
                # 定義dynamic_weights為gate_weights.T，便於合併
                batch_dynamic_weights = gate_weights.t()  # 形狀為(2, batch_size)
                
                # 添加到結果中
                result['fusion_weights'] = fusion_weights
                result['dynamic_weights'] = batch_dynamic_weights
                
                if return_features:
                    result['fused_features'] = fused_features
        
        # 確保最終預測值在合理範圍內
        nf_pred = torch.clamp(nf_pred, min=10.0, max=1e5)
        
        # 添加主要預測結果到結果字典
        result['nf_pred'] = nf_pred
        result['pinn_nf_pred'] = pinn_nf_pred
        result['lstm_nf_pred'] = lstm_nf_pred
        
        # 確保l2_penalty是標量值
        total_l2_penalty = pinn_l2_penalty + lstm_l2_penalty
        if isinstance(total_l2_penalty, torch.Tensor) and total_l2_penalty.dim() > 0:
            total_l2_penalty = total_l2_penalty.mean()
        result['l2_penalty'] = total_l2_penalty
        
        # 添加delta_w到結果中
        if self.use_physics_layer:
            result['delta_w'] = torch.clamp(pinn_delta_w, min=1e-8)
        
        # 添加特征到结果中
        if return_features:
            result['pinn_features'] = pinn_features
            result['lstm_features'] = lstm_features
            if 'attention_weights' in lstm_output:
                result['attention_weights'] = lstm_output['attention_weights']
        
        return result

class PINNLSTMTrainer:
    """
    PINN-LSTM混合模型訓練器
    提供分階段訓練、物理約束動態調整等功能
    """
    def __init__(self, model, optimizer, device, 
                 lambda_physics_init=0.1, lambda_physics_max=1.0,
                 lambda_consistency_init=0.05, lambda_consistency_max=0.3,
                 lambda_ramp_epochs=50, clip_grad_norm=1.0,
                 scheduler=None, log_interval=10):
        """
        初始化PINN-LSTM訓練器
        
        參數:
            model (HybridPINNLSTMModel): 混合模型
            optimizer (torch.optim.Optimizer): 優化器
            device (torch.device): 計算設備
            lambda_physics_init (float): 物理約束初始權重
            lambda_physics_max (float): 物理約束最大權重
            lambda_consistency_init (float): 一致性損失初始權重
            lambda_consistency_max (float): 一致性損失最大權重
            lambda_ramp_epochs (int): 達到最大權重的輪數
            clip_grad_norm (float): 梯度裁剪範數
            scheduler (torch.optim.lr_scheduler): 學習率調度器
            log_interval (int): 日誌輸出間隔
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.lambda_physics_init = lambda_physics_init
        self.lambda_physics_max = lambda_physics_max
        self.lambda_consistency_init = lambda_consistency_init
        self.lambda_consistency_max = lambda_consistency_max
        self.lambda_ramp_epochs = lambda_ramp_epochs
        self.clip_grad_norm = clip_grad_norm
        self.scheduler = scheduler
        self.log_interval = log_interval
        
        # 初始化訓練記錄
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = {}
        self.val_metrics = {}
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        # 初始化損失權重
        self.lambda_physics = lambda_physics_init
        self.lambda_consistency = lambda_consistency_init
    
    def update_loss_weights(self, epoch):
        """
        更新損失權重
        
        參數:
            epoch (int): 當前訓練輪次
        """
        if self.lambda_ramp_epochs <= 0:
            return
            
        progress = min(epoch / self.lambda_ramp_epochs, 1.0)
        self.lambda_physics = self.lambda_physics_init + (self.lambda_physics_max - self.lambda_physics_init) * progress
        self.lambda_consistency = self.lambda_consistency_init + (self.lambda_consistency_max - self.lambda_consistency_init) * progress
        
        logger.debug(f"輪次 {epoch}: 物理約束權重 = {self.lambda_physics:.4f}, 一致性約束權重 = {self.lambda_consistency:.4f}")
    
    def train_epoch(self, train_loader):
        """
        訓練一個輪次
        
        參數:
            train_loader (DataLoader): 訓練資料載入器
            
        返回:
            dict: 包含訓練損失和指標的字典
        """
        self.model.train()
        epoch_losses = {'total': 0.0, 'pred': 0.0, 'physics': 0.0, 'consistency': 0.0}
        num_batches = len(train_loader)
        all_targets = []
        all_predictions = []
        
        # 使用 tqdm 顯示進度條（如果可用）
        try:
            from tqdm import tqdm
            pbar = tqdm(enumerate(train_loader), total=num_batches, desc="Training", leave=False)
        except ImportError:
            pbar = enumerate(train_loader)
        
        for batch_idx, (static_features, time_series, targets) in pbar:
            # 將資料移至設備
            static_features = static_features.to(self.device)
            time_series = time_series.to(self.device)
            targets = targets.to(self.device)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向傳播
            outputs = self.model(static_features, time_series)
            
            # 計算損失
            losses = self.model.calculate_loss(
                outputs, targets, 
                lambda_physics=self.lambda_physics,
                lambda_consistency=self.lambda_consistency
            )
            
            loss = losses['total_loss']
            
            # 反向傳播
            loss.backward()
            
            # 梯度裁剪
            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            
            # 參數更新
            self.optimizer.step()
            
            # 累計損失
            for k, v in losses.items():
                if k.endswith('_loss'):
                    name = k.replace('_loss', '')
                    if name in epoch_losses:
                        epoch_losses[name] += v.item()
                    else:
                        epoch_losses[name] = v.item()
                        
            # 收集預測和目標，用於計算指標
            all_predictions.append(outputs['nf_pred'].detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())
            
            # 更新進度條
            if hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({'loss': loss.item()})
        
        # 計算平均損失
        for k in epoch_losses:
            epoch_losses[k] /= num_batches
        
        # 計算訓練指標
        if all_predictions and all_targets:
            all_predictions = np.concatenate(all_predictions)
            all_targets = np.concatenate(all_targets)
            train_metrics = self._calculate_metrics(all_predictions, all_targets)
        else:
            train_metrics = {}
            
        return {'losses': epoch_losses, 'metrics': train_metrics}
    
    def evaluate(self, val_loader):
        """
        評估模型
        
        參數:
            val_loader (DataLoader): 驗證資料載入器
            
        返回:
            tuple: (平均損失, 評估指標字典, 預測結果, 目標值)
        """
        self.model.eval()
        val_losses = {'total': 0.0, 'pred': 0.0, 'physics': 0.0, 'consistency': 0.0}
        all_outputs = []
        all_targets = []
        scalar_values = defaultdict(list)  # 用於收集標量值
        
        with torch.no_grad():
            for static_features, time_series, targets in val_loader:
                # 將資料移至設備
                static_features = static_features.to(self.device)
                time_series = time_series.to(self.device)
                targets = targets.to(self.device)
                
                # 前向傳播
                outputs = self.model(static_features, time_series, return_features=True)
                
                # 計算損失
                losses = self.model.calculate_loss(
                    outputs, targets, 
                    lambda_physics=self.lambda_physics,
                    lambda_consistency=self.lambda_consistency
                )
                
                # 累計損失
                for k, v in losses.items():
                    if k.endswith('_loss'):
                        name = k.replace('_loss', '')
                        if name in val_losses:
                            val_losses[name] += v.item()
                        else:
                            val_losses[name] = v.item()
                
                # 收集預測和目標
                all_targets.append(targets.cpu().numpy())
                
                # 分離標量值和張量值
                batch_outputs = {}
                for key, value in outputs.items():
                    if isinstance(value, torch.Tensor):
                        if value.ndim == 0 or key == 'l2_penalty':
                            # 為標量值，記錄其值
                            scalar_values[key].append(float(value.cpu().item()))
                        else:
                            # 為張量，保存至batch_outputs
                            batch_outputs[key] = value.cpu().numpy()
                    else:
                        # 非張量值
                        batch_outputs[key] = value
                
                all_outputs.append(batch_outputs)
        
        # 計算平均損失
        num_batches = len(val_loader)
        for k in val_losses:
            val_losses[k] /= num_batches
        
        # 合併預測和目標
        all_predictions = np.concatenate([o['nf_pred'] for o in all_outputs])
        all_targets = np.concatenate(all_targets)
        
        # 計算指標
        metrics = self._calculate_metrics(all_predictions, all_targets)
        
        # 合併所有輸出
        merged_outputs = {}
        
        # 處理批次輸出
        # 找出所有鍵
        all_keys = set()
        for output in all_outputs:
            all_keys.update(output.keys())
        
        # 合併每個鍵的數據
        for key in all_keys:
            values = [o[key] for o in all_outputs if key in o]
            if not values:
                continue
                
            if key == 'dynamic_weights':
                # 特殊處理 dynamic_weights
                try:
                    merged_outputs[key] = np.vstack(values)
                except Exception as e:
                    logger.warning(f"合併 dynamic_weights 時出錯: {str(e)}")
                    merged_outputs[key] = values
            elif all(isinstance(val, np.ndarray) for val in values):
                try:
                    merged_outputs[key] = np.concatenate(values)
                except Exception as e:
                    logger.warning(f"合併輸出 {key} 時出錯: {str(e)}")
                    merged_outputs[key] = values
            else:
                merged_outputs[key] = values
        
        # 處理標量值，計算平均值
        for key, values in scalar_values.items():
            if values:
                merged_outputs[key] = np.mean(values)
        
        # 添加預測和目標
        merged_outputs['predictions'] = all_predictions
        merged_outputs['targets'] = all_targets
        
        return val_losses, metrics, merged_outputs
    
    def _calculate_metrics(self, predictions, targets):
        """
        計算評估指標
        
        參數:
            predictions (np.ndarray): 預測值
            targets (np.ndarray): 真實值
            
        返回:
            dict: 包含評估指標的字典
        """
        try:
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            
            # 計算基本指標
            mse = mean_squared_error(targets, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(targets, predictions)
            mae = mean_absolute_error(targets, predictions)
            
            # 計算相對誤差
            rel_error = np.abs((targets - predictions) / (targets + 1e-8)) * 100
            mean_rel_error = np.mean(rel_error)
            median_rel_error = np.median(rel_error)
            
            # 計算對數空間的指標
            log_targets = np.log(targets + 1e-8)
            log_predictions = np.log(predictions + 1e-8)
            log_mse = mean_squared_error(log_targets, log_predictions)
            log_rmse = np.sqrt(log_mse)
            
            return {
                'rmse': rmse,
                'r2': r2,
                'mae': mae,
                'mean_rel_error': mean_rel_error,
                'median_rel_error': median_rel_error,
                'log_rmse': log_rmse
            }
        except Exception as e:
            logger.error(f"計算指標時出錯: {str(e)}")
            return {}
    
    def train(self, train_loader, val_loader, epochs, early_stopping_patience=20,
             save_path=None, callbacks=None):
        """
        訓練模型
        
        參數:
            train_loader (DataLoader): 訓練資料載入器
            val_loader (DataLoader): 驗證資料載入器
            epochs (int): 訓練輪數
            early_stopping_patience (int): 早停耐心值
            save_path (str): 模型保存路徑
            callbacks (list): 回調函數列表
            
        返回:
            dict: 訓練歷史記錄
        """
        callbacks = callbacks or []
        best_val_loss = float('inf')
        patience_counter = 0
        history = {
            'train_losses': [],
            'val_losses': [],
            'train_metrics': {},
            'val_metrics': {},
            'best_val_loss': float('inf')
        }
        
        # 初始化指標記錄
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = {}
        self.val_metrics = {}
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # 更新損失權重
            self.update_loss_weights(epoch)
            
            # 訓練一個輪次
            train_results = self.train_epoch(train_loader)
            train_losses = train_results['losses']
            train_metrics = train_results.get('metrics', {})
            
            self.train_losses.append(train_losses)
            
            # 更新訓練指標記錄
            for k, v in train_metrics.items():
                if k not in self.train_metrics:
                    self.train_metrics[k] = []
                self.train_metrics[k].append(v)
            
            # 評估
            val_losses, val_metrics, val_outputs = self.evaluate(val_loader)
            self.val_losses.append(val_losses)
            
            # 更新驗證指標記錄
            for k, v in val_metrics.items():
                if k not in self.val_metrics:
                    self.val_metrics[k] = []
                self.val_metrics[k].append(v)
            
            # 更新學習率
            if self.scheduler is not None:
                if hasattr(self.scheduler, 'step_with_metrics'):
                    self.scheduler.step_with_metrics(val_losses['total'])
                else:
                    self.scheduler.step()
            
            # 輸出日誌
            if epoch % self.log_interval == 0 or epoch == epochs - 1:
                curr_lr = self.optimizer.param_groups[0]['lr']
                log_msg = (f"輪次 {epoch+1}/{epochs} - "
                          f"訓練損失: {train_losses['total']:.4f}, "
                          f"驗證損失: {val_losses['total']:.4f}, "
                          f"RMSE: {val_metrics.get('rmse', 0):.4f}, "
                          f"R²: {val_metrics.get('r2', 0):.4f}, "
                          f"學習率: {curr_lr:.6f}")
                logger.info(log_msg)
            
            # 早停檢查
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                patience_counter = 0
                
                # 保存最佳模型
                if save_path:
                    self._save_model(save_path, val_losses, val_metrics)
                
                # 保存最佳模型狀態
                self.best_val_loss = best_val_loss
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                
                # 更新歷史記錄
                history['best_val_loss'] = best_val_loss
            else:
                patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    logger.info(f"早停觸發，在輪次 {epoch+1} 停止訓練")
                    break
            
            # 執行回調函數
            for callback in callbacks:
                callback(epoch, {
                    'model': self.model,
                    'optimizer': self.optimizer,
                    'train_loss': train_losses['total'],
                    'val_loss': val_losses['total'],
                    'metrics': val_metrics,
                    'epoch': epoch
                })
        
        # 更新歷史記錄
        history['train_losses'] = self.train_losses
        history['val_losses'] = self.val_losses
        history['train_metrics'] = self.train_metrics
        history['val_metrics'] = self.val_metrics
        
        # 恢復最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return history
    
    def _save_model(self, path, val_losses, val_metrics):
        """
        保存模型
        
        參數:
            path (str): 保存路徑
            val_losses (dict): 驗證損失
            val_metrics (dict): 驗證指標
        """
        # 確保目錄存在
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存模型
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_losses': val_losses,
            'val_metrics': val_metrics,
            'epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'lambda_physics': self.lambda_physics,
            'lambda_consistency': self.lambda_consistency,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }, path)
        
        logger.info(f"模型已保存至 {path}")
    
    def load_model(self, path):
        """
        載入模型
        
        參數:
            path (str): 模型路徑
            
        返回:
            dict: 包含模型指標的字典
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.lambda_physics = checkpoint.get('lambda_physics', self.lambda_physics)
        self.lambda_consistency = checkpoint.get('lambda_consistency', self.lambda_consistency)
        
        # 載入訓練記錄
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        if 'train_metrics' in checkpoint:
            self.train_metrics = checkpoint['train_metrics']
        if 'val_metrics' in checkpoint:
            self.val_metrics = checkpoint['val_metrics']
        
        logger.info(f"模型已從 {path} 載入")
        
        return {
            'val_losses': checkpoint.get('val_losses', {}),
            'val_metrics': checkpoint.get('val_metrics', {})
        }