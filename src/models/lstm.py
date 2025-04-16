#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
lstm.py - 長短期記憶網絡模型
本模組實現了長短期記憶網絡(LSTM)，專門用於處理銲錫接點非線性塑性應變功的時間序列資料，
捕捉其中的時序特徵和動態變化模式，專注於預測非線性塑性應變能密度變化量(delta_w)。

主要特點:
1. 雙向LSTM層提取時間序列特徵
2. 專注於預測非線性塑性應變能密度變化量(delta_w)
3. 注意力機制突出關鍵時間步的重要性
4. 適應小樣本資料集的時序特徵萃取
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)

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


class LSTMModel(nn.Module):
    """
    長短期記憶網絡模型 - 專注於預測delta_w
    專門用於處理銲錫接點的非線性塑性應變功時間序列資料
    """
    def __init__(self, input_dim=2, hidden_size=48, num_layers=2, output_dim=1,
                bidirectional=True, dropout_rate=0.25, use_attention=True,
                l2_reg=0.002, a_coefficient=55.83, b_coefficient=-2.259):
        """
        初始化LSTM模型
        
        參數:
            input_dim (int): 輸入特徵維度，預設為2 (上下界面非線性塑性應變功)
            hidden_size (int): LSTM隱藏層大小
            num_layers (int): LSTM層數
            output_dim (int): 輸出維度，預設為1
            bidirectional (bool): 是否使用雙向LSTM
            dropout_rate (float): Dropout比率
            use_attention (bool): 是否使用注意力機制
            l2_reg (float): L2正則化係數
            a_coefficient (float): 物理模型係數a
            b_coefficient (float): 物理模型係數b
        """
        super(LSTMModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.l2_reg = l2_reg
        
        # 註冊物理係數
        self.register_buffer('a_coefficient', torch.tensor(a_coefficient, dtype=torch.float32))
        self.register_buffer('b_coefficient', torch.tensor(b_coefficient, dtype=torch.float32))
        
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
        
        # 全連接層，用於特徵處理
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
        
        # 非線性塑性應變能密度變化量(delta_w)預測層
        # 使用對數空間預測，確保輸出為正值
        self.delta_w_layer = nn.Linear(fc_input_dim, 1)
        
        # 初始化權重
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
                nn.init.constant_(param.data, -5.5)  # 初始偏置值為對數空間中的-3，exp(-3)≈0.05
            elif 'linear' in name and 'weight' in name:
                nn.init.xavier_uniform_(param.data) if param.dim() >= 2 else nn.init.uniform_(param.data, -0.1, 0.1)
            elif 'linear' in name and 'bias' in name:
                nn.init.zeros_(param.data)
    
    def forward(self, x, return_attention=False):
        """
        前向傳播 - 專注於預測delta_w，再通過物理公式計算nf_pred
        
        參數:
            x (torch.Tensor): 輸入時間序列，形狀為 (batch_size, seq_len, input_dim)
            return_attention (bool): 是否返回注意力權重
        
        返回:
            dict: 包含預測結果的字典:
                - 'delta_w': 預測的非線性塑性應變能密度變化量 (主要預測目標)
                - 'nf_pred': 根據delta_w計算的疲勞壽命
                - 'features': 提取的時序特徵
                - 'l2_penalty': L2正則化懲罰
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
        fc_output = self.fc_layers(context_vector)
        
        # 預測delta_w - 使用對數空間確保輸出為正值
        log_delta_w = self.delta_w_layer(fc_output)
        delta_w = torch.exp(log_delta_w).squeeze(-1)
        delta_w = delta_w.clamp(min=1e-8)  # 確保delta_w為正值
        
        # 使用物理公式計算疲勞壽命
        a_coef = float(self.a_coefficient) if hasattr(self.a_coefficient, 'item') else self.a_coefficient
        b_coef = float(self.b_coefficient) if hasattr(self.b_coefficient, 'item') else self.b_coefficient
        nf_pred = a_coef * torch.pow(delta_w.clamp(min=1e-8), b_coef)
        nf_pred = nf_pred.clamp(min=10.0)  # 確保疲勞壽命不會太小

        # 計算L2正則化懲罰
        l2_penalty = 0.0
        if self.l2_reg > 0:
            for param in self.parameters():
                l2_penalty += torch.norm(param, 2)
            
            # 確保l2_penalty是標量
            if isinstance(l2_penalty, torch.Tensor) and l2_penalty.dim() > 0:
                l2_penalty = l2_penalty.mean()
            
            # 乘以正則化系數
            l2_penalty = l2_penalty * self.l2_reg

        # 準備輸出結果
        result = {
            'delta_w': delta_w,  # 主要預測目標 - 非線性塑性應變能密度變化量
            'nf_pred': nf_pred,  # 根據delta_w計算的疲勞壽命
            'features': context_vector,  # 提取的時序特徵
            'l2_penalty': l2_penalty  # L2正則化懲罰
        }

        # 如果需要返回注意力權重
        if return_attention and attention_weights is not None:
            result['attention_weights'] = attention_weights

        return result
    
    def get_time_features(self, x):
        """
        獲取時間特徵向量
        
        參數:
            x (torch.Tensor): 輸入時間序列
            
        返回:
            torch.Tensor: 時間特徵向量
        """
        with torch.no_grad():
            lstm_output, (hidden, cell) = self.lstm(x)
            
            if self.use_attention:
                context_vector, _ = self.attention(lstm_output)
            else:
                if self.bidirectional:
                    last_forward = hidden[-2, :, :]
                    last_backward = hidden[-1, :, :]
                    context_vector = torch.cat((last_forward, last_backward), dim=1)
                else:
                    context_vector = hidden[-1, :, :]
            
            return context_vector


if __name__ == "__main__":
    # 簡單的測試代碼
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 創建一個小型LSTM模型進行測試
    model = LSTMModel(
        input_dim=2, 
        hidden_size=64, 
        num_layers=2, 
        bidirectional=True,
        use_attention=True
    )
    
    # 創建隨機輸入資料，模擬4個時間步的上下界面非線性塑性應變功
    batch_size = 8
    seq_len = 4
    input_dim = 2
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # 前向傳播
    output = model(x, return_attention=True)
    
    logger.info(f"模型輸出:")
    logger.info(f"  預測delta_w形狀: {output['delta_w'].shape}")
    logger.info(f"  預測疲勞壽命形狀: {output['nf_pred'].shape}")
    logger.info(f"  特徵向量形狀: {output['features'].shape}")
    if 'attention_weights' in output:
        logger.info(f"  注意力權重形狀: {output['attention_weights'].shape}")
        logger.info(f"  注意力權重總和: {output['attention_weights'].sum(dim=1)}")  # 應為每個樣本總和為1
    logger.info(f"  預測delta_w範圍: [{output['delta_w'].min().item()}, {output['delta_w'].max().item()}]")
    logger.info(f"  預測疲勞壽命範圍: [{output['nf_pred'].min().item()}, {output['nf_pred'].max().item()}]")