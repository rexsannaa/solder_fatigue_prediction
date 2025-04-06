#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
hybrid_model_patch.py - 修復混合模型 AttentionLayer 問題的補丁檔案

使用方法:
1. 將此檔案保存為 hybrid_model_patch.py
2. 執行 python hybrid_model_patch.py
3. 接著執行 python scripts/train.py

此補丁會備份原始文件並應用必要的修改
"""

import os
import sys
import re
import shutil
import datetime

# 設定日誌輸出
def log(message):
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

# 主要檔案路徑
HYBRID_MODEL_PATH = "src/models/hybrid_model.py"

# 備份原始檔案
def backup_file(file_path):
    backup_path = f"{file_path}.bak.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    try:
        shutil.copy2(file_path, backup_path)
        log(f"已備份原始檔案至 {backup_path}")
        return True
    except Exception as e:
        log(f"備份檔案失敗: {e}")
        return False

# 修復 AttentionLayer 類別
def fix_attention_layer(content):
    # 尋找 AttentionLayer 類別定義
    attention_layer_pattern = r"class\s+AttentionLayer\s*\(\s*nn\.Module\s*\):(.*?)def\s+forward\s*\(\s*self,\s*([^)]*)\):(.*?)return\s+([^,\n]*),\s*([^\n]*)"
    
    # 使用正則表達式來匹配 forward 方法
    matches = re.findall(attention_layer_pattern, content, re.DOTALL)
    
    if not matches:
        log("無法找到 AttentionLayer 類別的 forward 方法，請手動修改檔案。")
        return content
    
    # 取得匹配的內容
    class_content, params, method_body, return_val1, return_val2 = matches[0]
    
    # 檢查是否需要修改
    if "static_features" in params and "time_series" in params:
        log("發現需要修改的 AttentionLayer.forward 方法")
        
        # 新的 forward 方法實現
        new_forward = '''    def forward(self, lstm_output, mask=None):
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
        
        return context_vector, attention_weights'''
        
        # 使用新的 forward 方法替換原始方法
        new_class_content = re.sub(
            r"def\s+forward\s*\(\s*self,\s*([^)]*)\):(.*?)return\s+([^,\n]*),\s*([^\n]*)",
            new_forward,
            "class AttentionLayer(nn.Module):" + class_content,
            flags=re.DOTALL
        )
        
        # 替換整個類別定義
        content = content.replace("class AttentionLayer(nn.Module):" + class_content, new_class_content)
        log("已修改 AttentionLayer.forward 方法")
    else:
        log("AttentionLayer.forward 方法看起來已經正確，不需要修改")
    
    return content

# 修復 LSTMModel 類別的 forward 方法
def fix_lstm_model_forward(content):
    # 尋找 LSTMModel 類別中的 forward 方法
    lstm_forward_pattern = r"class\s+LSTMModel\s*\(\s*nn\.Module\s*\):.*?def\s+forward\s*\(\s*self,\s*([^)]*)\):(.*?)if\s+self\.use_attention:(.*?)context_vector,\s*attention_weights\s*=\s*self\.attention\s*\(([^)]*)\)"
    
    matches = re.findall(lstm_forward_pattern, content, re.DOTALL)
    
    if not matches:
        log("無法找到 LSTMModel 類別的 forward 方法中對 attention 的調用，請手動修改檔案。")
        return content
    
    # 取得匹配的內容
    params, method_body, attention_section, attention_params = matches[0]
    
    # 檢查是否需要修改
    if "static_features" in attention_params or "time_series" in attention_params:
        log("發現需要修改的 LSTMModel.forward 方法中對 attention 的調用")
        
        # 新的調用方式
        new_attention_call = "            # 使用注意力機制\n            context_vector, attention_weights = self.attention(lstm_output)"
        
        # 使用新的調用方式替換原始調用
        new_content = re.sub(
            r"if\s+self\.use_attention:(.*?)context_vector,\s*attention_weights\s*=\s*self\.attention\s*\(([^)]*)\)",
            f"if self.use_attention:{attention_section.split('context_vector')[0]}{new_attention_call}",
            content,
            flags=re.DOTALL
        )
        
        if new_content != content:
            content = new_content
            log("已修改 LSTMModel.forward 方法中對 attention 的調用")
        else:
            log("無法通過正則表達式替換 LSTMModel.forward 方法中對 attention 的調用，請手動修改檔案。")
    else:
        log("LSTMModel.forward 方法中對 attention 的調用看起來已經正確，不需要修改")
    
    return content

# 應用補丁
def apply_patch():
    # 檢查檔案是否存在
    if not os.path.exists(HYBRID_MODEL_PATH):
        log(f"錯誤: 找不到檔案 {HYBRID_MODEL_PATH}")
        return False
    
    # 備份原始檔案
    if not backup_file(HYBRID_MODEL_PATH):
        log("無法繼續修改，備份失敗")
        return False
    
    # 讀取檔案內容
    try:
        with open(HYBRID_MODEL_PATH, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        log(f"讀取檔案失敗: {e}")
        return False
    
    # 應用修改
    content = fix_attention_layer(content)
    content = fix_lstm_model_forward(content)
    
    # 寫入修改後的內容
    try:
        with open(HYBRID_MODEL_PATH, "w", encoding="utf-8") as f:
            f.write(content)
        log(f"已成功修改檔案 {HYBRID_MODEL_PATH}")
        return True
    except Exception as e:
        log(f"寫入檔案失敗: {e}")
        return False

# 手動編輯說明
def manual_edit_instructions():
    print("\n" + "="*80)
    print("手動編輯說明".center(80))
    print("="*80)
    print("如果自動修補失敗，請手動修改 src/models/hybrid_model.py 檔案:")
    print("\n1. 修改 AttentionLayer 類別的 forward 方法:")
    print("   將:\n   def forward(self, static_features, time_series, ...): ...")
    print("   改為:\n   def forward(self, lstm_output, mask=None): ...")
    print("\n2. 修改 LSTMModel 類別的 forward 方法中對 attention 的調用:")
    print("   將:\n   context_vector, attention_weights = self.attention(static_features, time_series, ...)")
    print("   改為:\n   context_vector, attention_weights = self.attention(lstm_output)")
    print("\n3. 檢查 HybridPINNLSTMModel 類別中所有對 attention 的調用，確保調用方式一致")
    print("="*80 + "\n")

# 主函數
def main():
    log("開始修補 PINN-LSTM 混合模型的 AttentionLayer 問題")
    
    success = apply_patch()
    
    if success:
        log("補丁已成功應用，現在可以嘗試運行 python scripts/train.py")
    else:
        log("補丁應用失敗，請嘗試手動修改檔案")
        manual_edit_instructions()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())