#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
lstm.py - 長短期記憶網絡模型
本模組實現了長短期記憶網絡(LSTM)，專門用於處理銲錫接點非線性塑性應變功的時間序列資料，
捕捉其中的時序特徵和動態變化模式，為疲勞壽命預測提供時序資訊。

主要特點:
1. 雙向LSTM層提取時間序列特徵
2. 注意力機制突出關鍵時間步的重要性
3. 適應小樣本資料集的時序特徵萃取
4. 提供多種時序特徵輸出模式
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
    長短期記憶網絡模型
    專門用於處理銲錫接點的非線性塑性應變功時間序列資料
    """
    def __init__(self, input_dim=2, hidden_size=64, num_layers=2, output_dim=1,
                 bidirectional=True, dropout_rate=0.2, use_attention=True):
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
        """
        super(LSTMModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # LSTM層數
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
        fc_hidden_dims = [lstm_output_dim // 2, lstm_output_dim // 4]
        
        for hidden_dim in fc_hidden_dims:
            fc_layers.append(nn.Linear(fc_input_dim, hidden_dim))
            fc_layers.append(nn.BatchNorm1d(hidden_dim))
            fc_layers.append(nn.ReLU())
            if dropout_rate > 0:
                fc_layers.append(nn.Dropout(dropout_rate))
            fc_input_dim = hidden_dim
        
        self.fc_layers = nn.Sequential(*fc_layers)
        self.output_layer = nn.Linear(fc_input_dim, output_dim)
        
        logger.info(f"初始化LSTMModel，輸入維度: {input_dim}, 隱藏層大小: {hidden_size}, "
                  f"LSTM層數: {num_layers}, 雙向: {bidirectional}, 使用注意力: {use_attention}")
        
        # 初始化權重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化網絡權重"""
        for name, param in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    if param.dim() >= 2:
                        nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    if param.dim() >= 2:
                        nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.zeros_(param.data)
            elif 'attention_weights' in name:
                if param.dim() >= 2:
                    nn.init.xavier_uniform_(param.data)
                else:
                    # 處理一維參數
                    nn.init.uniform_(param.data, -0.1, 0.1)
            elif 'linear' in name and 'weight' in name:
                if param.dim() >= 2:
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
                - 'last_hidden': 最後時間步的隱藏狀態
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
        output = torch.exp(self.output_layer(fc_output))  # 使用exp確保壽命為正
        
        result = {
            'output': output.squeeze(-1),
            'last_hidden': context_vector
        }
        
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
            result = self.forward(x)
            return result['last_hidden']


if __name__ == "__main__":
    # 簡單的測試代碼
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 創建一個小型LSTM模型進行測試
    model = LSTMModel(input_dim=2, hidden_size=32, num_layers=1, use_attention=True)
    
    # 創建隨機輸入資料，模擬4個時間步的上下界面非線性塑性應變功
    batch_size = 8
    seq_len = 4
    input_dim = 2
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # 前向傳播
    output = model(x, return_attention=True)
    
    logger.info(f"模型輸出:")
    logger.info(f"  預測疲勞壽命形狀: {output['output'].shape}")
    logger.info(f"  隱藏層特徵形狀: {output['last_hidden'].shape}")
    if 'attention_weights' in output:
        logger.info(f"  注意力權重形狀: {output['attention_weights'].shape}")
        logger.info(f"  注意力權重總和: {output['attention_weights'].sum(dim=1)}")  # 應為每個樣本總和為1
    logger.info(f"  預測疲勞壽命範圍: [{output['output'].min().item()}, {output['output'].max().item()}]")