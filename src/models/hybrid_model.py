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

def _l2_penalty(parameters):
    """輔助函數：計算所有參數的 L2 範數總和"""
    return sum(p.norm(2) for p in parameters)
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
            self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float32))
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
        delta_w = delta_w.clamp(min=1e-8)
        if self.trainable:
            a = torch.exp(self.log_a)
            b = -torch.exp(self.log_neg_b)
            nf = a * delta_w.pow(b) + self.bias
            nf = F.softplus(nf)
        else:
            nf = self.a * delta_w.pow(self.b)
        nf = nf.clamp(min=10.0)
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
        scores = self.attention_weights(lstm_output).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1)
        return context, weights
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
        """
        super(PINNModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.use_physics_layer = use_physics_layer
        self.l2_reg = l2_reg
        
        self.register_buffer('a', torch.tensor(a_coefficient, dtype=torch.float32))
        self.register_buffer('b', torch.tensor(b_coefficient, dtype=torch.float32))
        
        # 簡化激活函數選擇
        activation_dict = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU(),
            'selu': nn.SELU()
        }
        self.activation = activation_dict.get(activation.lower(), nn.ReLU())
        
        # 構建特徵提取層
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity())
            prev_dim = hidden_dim
        self.feature_extractor = nn.Sequential(*layers)
        self.delta_w_predictor = nn.Linear(hidden_dims[-1], 1)
        
        if use_physics_layer:
            self.physics_layer = PhysicsLayer(a=a_coefficient, b=b_coefficient, trainable=physics_layer_trainable)
        else:
            self.direct_predictor = nn.Linear(hidden_dims[-1], output_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化網絡權重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.weight.dim() >= 2:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    if m.out_features == 1:
                        nn.init.constant_(m.bias, 7.0)
                    else:
                        nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        前向傳播 - 修改版：專注於預測delta_w，不直接計算nf_pred
        
        參數:
            x (torch.Tensor): 輸入特徵，形狀為 (batch_size, input_dim)
            
        返回:
            dict: 包含預測結果的字典
        """
        features = self.feature_extractor(x)
        delta_w = torch.exp(self.delta_w_predictor(features)).clamp(min=1e-6)
        if self.use_physics_layer:
            nf_pred = self.physics_layer(delta_w)
        else:
            nf_pred = self.a * delta_w.pow(self.b)
        nf_pred = nf_pred.clamp(min=10.0)
        l2 = self.l2_reg * _l2_penalty(self.parameters())
        return {
            'delta_w': delta_w.squeeze(-1),
            'nf_pred': nf_pred.squeeze(-1),
            'features': features,
            'l2_penalty': l2
        }
class LSTMModel(nn.Module):
    """
    長短期記憶網絡模型 - 修改版
    專門用於處理銲錫接點的非線性塑性應變功時間序列資料，預測delta_w
    """
    def __init__(self, input_dim=2, hidden_size=32, num_layers=1, output_dim=1,
                 bidirectional=True, dropout_rate=0.2, use_attention=True,
                 l2_reg=0.001, a_coefficient=55.83, b_coefficient=-2.259):
        """
        初始化LSTM模型
        """
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.l2_reg = l2_reg
        self.register_buffer('a', torch.tensor(a_coefficient, dtype=torch.float32))
        self.register_buffer('b', torch.tensor(b_coefficient, dtype=torch.float32))
        self.register_buffer('a_coefficient', torch.tensor(a_coefficient, dtype=torch.float32))
        self.register_buffer('b_coefficient', torch.tensor(b_coefficient, dtype=torch.float32))
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        lstm_out_dim = hidden_size * 2 if bidirectional else hidden_size
        
        # 注意力層
        if use_attention:
            self.attention = AttentionLayer(lstm_out_dim)
        
        # 全連接層 (使用生成式方式)
        fc_layers = []
        fc_in = lstm_out_dim
        for hidden in [lstm_out_dim // 2]:
            fc_layers.extend([
                nn.Linear(fc_in, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
            ])
            fc_in = hidden
        self.fc_layers = nn.Sequential(*fc_layers)
        
        self.delta_w_layer = nn.Linear(fc_in, 1)
        self.output_layer = nn.Linear(fc_in, output_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化網絡權重"""
        for name, param in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data) if param.dim() >= 2 else nn.init.uniform_(param.data, -0.1, 0.1)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data) if param.dim() >= 2 else nn.init.uniform_(param.data, -0.1, 0.1)
                elif 'bias' in name:
                    nn.init.zeros_(param.data)
            elif 'attention_weights' in name:
                nn.init.xavier_uniform_(param.data) if param.dim() >= 2 else nn.init.uniform_(param.data, -0.1, 0.1)
            elif 'delta_w_layer' in name and 'bias' in name:
                nn.init.constant_(param.data, -3.0)
            elif 'linear' in name and 'weight' in name:
                nn.init.xavier_uniform_(param.data) if param.dim() >= 2 else nn.init.uniform_(param.data, -0.1, 0.1)
            elif 'linear' in name and 'bias' in name:
                nn.init.zeros_(param.data)
    
    def forward(self, x, return_attention=False):
        """
        前向傳播 - 修改版：專注於預測delta_w
        參數:
            x (torch.Tensor): 輸入時間序列，形狀為 (batch_size, seq_len, input_dim)
            return_attention (bool): 是否返回注意力權重
        返回:
            dict: 包含預測結果的字典
        """
        lstm_out, (hidden, cell) = self.lstm(x)
        if self.use_attention:
            context, attn = self.attention(lstm_out)
        else:
            if self.bidirectional:
                context = torch.cat((hidden[-2], hidden[-1]), dim=1)
            else:
                context = hidden[-1]
            attn = None
        
        fc_out = self.fc_layers(context)
        delta_w = torch.exp(self.delta_w_layer(fc_out)).clamp(min=1e-6)
        nf_pred = self.a_coefficient * delta_w.pow(self.b_coefficient)
        output = torch.exp(self.output_layer(fc_out))
        delta_w = delta_w.clamp(min=1e-6)
        output = output.clamp(min=10.0)
        nf_pred = nf_pred.clamp(min=10.0)
        
        l2 = self.l2_reg * _l2_penalty(self.parameters())
        result = {
            'delta_w': delta_w.squeeze(-1),
            'nf_pred': nf_pred.squeeze(-1),
            'output': output.squeeze(-1),
            'features': context,
            'l2_penalty': l2
        }
        if return_attention and attn is not None:
            result['attention_weights'] = attn
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
        # 融合後進一步處理層
        layers = [nn.Linear(fusion_dim, fusion_dim)]
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(fusion_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity())
        self.fusion_layers = nn.Sequential(*layers)
        
    def forward(self, pinn_features, lstm_features):
        """
        前向傳播
        """
        combined = torch.cat([pinn_features, lstm_features], dim=1)
        gate = self.gate_network(combined)
        pinn_proj = self.pinn_projection(pinn_features)
        lstm_proj = self.lstm_projection(lstm_features)
        fused = gate[:, 0].unsqueeze(1) * pinn_proj + gate[:, 1].unsqueeze(1) * lstm_proj
        output = self.fusion_layers(fused)
        return output, gate
class HybridPINNLSTMModel(nn.Module):
    """
    改進的混合PINN-LSTM模型
    結合物理信息神經網絡和長短期記憶網絡的優勢，專為小樣本數據集優化
    """
    def __init__(self, 
                 static_input_dim=5,
                 time_input_dim=2,
                 time_steps=4,
                 pinn_hidden_dims=[64, 32, 16],
                 lstm_hidden_size=64,
                 lstm_num_layers=2,
                 fusion_dim=32,
                 dropout_rate=0.1,
                 bidirectional=True,
                 use_attention=True,
                 use_physics_layer=True,
                 physics_layer_trainable=True,
                 use_batch_norm=True,
                 pinn_weight_init=0.8,
                 lstm_weight_init=0.2,
                 a_coefficient=55.83,
                 b_coefficient=-2.259,
                 use_log_transform=True,
                 ensemble_method='weighted',
                 l2_reg=0.0005):
        """
        初始化混合PINN-LSTM模型 - 修改版：明確強調模型的目標是預測delta_w
        """
        super(HybridPINNLSTMModel, self).__init__()
        self.static_input_dim = static_input_dim
        self.time_input_dim = time_input_dim
        self.time_steps = time_steps
        self.use_physics_layer = use_physics_layer
        self.use_log_transform = use_log_transform
        self.ensemble_method = ensemble_method
        self.l2_reg = l2_reg
        
        self.register_buffer('a_coefficient', torch.tensor(a_coefficient, dtype=torch.float32))
        self.register_buffer('b_coefficient', torch.tensor(b_coefficient, dtype=torch.float32))
        
        # PINN分支：預測靜態特徵下的delta_w
        self.pinn_branch = PINNModel(
            input_dim=static_input_dim,
            hidden_dims=pinn_hidden_dims,
            output_dim=1,
            dropout_rate=dropout_rate,
            use_physics_layer=False,
            physics_layer_trainable=physics_layer_trainable,
            use_batch_norm=use_batch_norm,
            a_coefficient=a_coefficient,
            b_coefficient=b_coefficient,
            l2_reg=l2_reg
        )
        # LSTM分支：處理時間序列資料
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
        # 分支權重 (加權融合)：若使用weighted方法
        if ensemble_method == 'weighted':
            weight_param = torch.tensor([math.log(pinn_weight_init / (1 - pinn_weight_init))], dtype=torch.float32)
            self.branch_weight_param = nn.Parameter(weight_param)
        # 如果使用 'gate' 或 'deep_fusion'，加入特徵融合層
        if ensemble_method in ['gate', 'deep_fusion']:
            pinn_feat_dim = pinn_hidden_dims[-1]
            lstm_feat_dim = lstm_hidden_size * 2 if bidirectional else lstm_hidden_size
            self.fusion_layer = FeatureFusionLayer(
                pinn_feature_dim=pinn_feat_dim,
                lstm_feature_dim=lstm_feat_dim,
                fusion_dim=fusion_dim,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm
            )
        # 物理層，用於將delta_w轉換為疲勞壽命
        if use_physics_layer:
            self.physics_layer = PhysicsLayer(a=a_coefficient, b=b_coefficient, trainable=physics_layer_trainable)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        初始化網絡權重 - 改進初始化策略
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.weight.dim() >= 2:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                else:
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    if m.out_features == 1:
                        nn.init.constant_(m.bias, 5.0)
                    else:
                        nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def get_branch_weights(self):
        """獲取分支權重"""
        if self.ensemble_method == 'weighted':
            pinn_w = torch.sigmoid(self.branch_weight_param)
            lstm_w = 1 - pinn_w
            return torch.tensor([pinn_w.item(), lstm_w.item()], device=pinn_w.device)
        return torch.tensor([0.5, 0.5], device=self.branch_weight_param.device if hasattr(self, 'branch_weight_param') else torch.device('cpu'))
    
    def calculate_loss(self, outputs, targets, lambda_physics=0.5, lambda_consistency=0.1):
        """
        計算混合損失 - 修改版：主要針對delta_w的預測精度
        
        參數:
            outputs (dict): 模型輸出
            targets (torch.Tensor): 目標疲勞壽命
            lambda_physics (float): 物理約束損失權重
            lambda_consistency (float): 分支間一致性損失權重
            
        返回:
            dict: 包含各部分損失的字典
        """
        targets = targets.clamp(min=10.0)
        predictions = outputs['nf_pred'].clamp(min=10.0)
        log_targets = torch.log10(targets)
        log_predictions = torch.log10(predictions)
        mse_loss = F.mse_loss(predictions, targets)
        log_mse_loss = F.mse_loss(log_predictions, log_targets)
        pred_loss = 0.05 * mse_loss + 0.95 * log_mse_loss
        
        large_mask = log_targets > 6.0  # 約1000循環以上
        if torch.any(large_mask):
            large_loss = F.mse_loss(log_predictions[large_mask], log_targets[large_mask])
            pred_loss += 2.0 * large_loss
        
        if 'delta_w' in outputs:
            delta_w_theory = torch.pow(targets / self.a_coefficient, 1/self.b_coefficient).clamp(min=1e-6)
            delta_w_pred = outputs['delta_w'].clamp(min=1e-6)
            delta_w_loss = F.mse_loss(torch.log10(delta_w_pred), torch.log10(delta_w_theory))
        else:
            delta_w_loss = 0.0
        
        total_loss = pred_loss + lambda_physics * delta_w_loss
        return {
            'total_loss': total_loss,
            'pred_loss': pred_loss,
            'delta_w_loss': delta_w_loss
        }
    
    def forward(self, static_input, time_series_input):
        """
        前向傳播 - 結合PINN與LSTM分支
        參數:
            static_input (torch.Tensor): 靜態特徵，形狀為 (batch_size, static_input_dim)
            time_series_input (torch.Tensor): 時間序列資料，形狀為 (batch_size, time_steps, time_input_dim)
        返回:
            dict: 包含綜合預測結果的字典
        """
        pinn_out = self.pinn_branch(static_input)
        lstm_out = self.lstm_branch(time_series_input)
        
        if self.ensemble_method == 'weighted':
            weights = self.get_branch_weights()
            delta_w = weights[0] * pinn_out['delta_w'] + weights[1] * lstm_out['delta_w']
        elif self.ensemble_method in ['gate', 'deep_fusion']:
            fused_feat, _ = self.fusion_layer(pinn_out['features'], lstm_out['features'])
            # 使用新線性層對融合特徵進行映射
            delta_w = torch.exp(nn.Linear(fused_feat.size(1), 1).to(fused_feat.device)(fused_feat)).squeeze(-1)
        else:
            delta_w = 0.5 * pinn_out['delta_w'] + 0.5 * lstm_out['delta_w']
        
        if self.use_physics_layer:
            nf_pred = self.physics_layer(delta_w.clamp(min=1e-6))
        else:
            nf_pred = self.a_coefficient * delta_w.pow(self.b_coefficient)
        nf_pred = nf_pred.clamp(min=10.0)
        
        l2 = self.l2_reg * _l2_penalty(self.parameters())
        return {
            'delta_w': delta_w,
            'nf_pred': nf_pred,
            'l2_penalty': l2,
            'branch_weights': self.get_branch_weights() if self.ensemble_method == 'weighted' else None
        }
