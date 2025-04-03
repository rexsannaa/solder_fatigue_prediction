#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
debug_evaluate.py - 調試版模型評估腳本
用於排查銲錫接點疲勞壽命預測模型評估中的問題
"""

import os
import sys
import argparse
import yaml
import logging
import traceback
import torch
import numpy as np
import time

# 設置基本日誌配置
logging.basicConfig(
    level=logging.DEBUG,  # 使用DEBUG級別獲取最詳細的日誌
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("debug_evaluate.log")
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description="調試銲錫接點疲勞壽命預測模型評估")
    parser.add_argument("--model-path", type=str, required=True,
                        help="模型檢查點路徑")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml",
                        help="模型配置文件路徑")
    parser.add_argument("--model-type", type=str, default="hybrid",
                        help="模型類型: hybrid, pinn, lstm")
    
    return parser.parse_args()

def load_config(config_path):
    """載入YAML配置文件"""
    logger.info(f"嘗試載入配置文件: {config_path}")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"成功載入配置文件")
        return config
    except Exception as e:
        logger.error(f"載入配置文件失敗: {str(e)}")
        raise

def check_model_file(model_path):
    """檢查模型文件"""
    logger.info(f"檢查模型文件: {model_path}")
    
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        return False
    
    try:
        # 嘗試載入模型
        logger.info(f"嘗試載入模型檢查點")
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        logger.info(f"成功載入模型檢查點")
        
        # 檢查檢查點內容
        if "model_state_dict" in checkpoint:
            logger.info("檢查點包含model_state_dict")
            keys_count = len(checkpoint["model_state_dict"].keys())
            logger.info(f"模型狀態字典包含 {keys_count} 個鍵")
        else:
            logger.info("檢查點不包含model_state_dict，嘗試直接載入")
            keys_count = len(checkpoint.keys())
            logger.info(f"檢查點包含 {keys_count} 個頂級鍵")
            
        # 打印部分鍵名以便檢查
        if "model_state_dict" in checkpoint:
            some_keys = list(checkpoint["model_state_dict"].keys())[:5]
        else:
            some_keys = list(checkpoint.keys())[:5]
        logger.info(f"部分鍵名: {some_keys}")
        
        return True
    except Exception as e:
        logger.error(f"載入模型時出錯: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def try_import_modules():
    """嘗試導入必要的模組"""
    logger.info("嘗試導入必要模組")
    
    try:
        logger.info("導入數據處理模組")
        sys.path.insert(0, os.getcwd())
        from src.data.preprocess import load_data, standardize_features
        logger.info("成功導入數據處理模組")
        
        logger.info("導入模型模組")
        from src.models.hybrid_model import HybridPINNLSTMModel
        from src.models.pinn import PINNModel
        from src.models.lstm import LSTMModel
        logger.info("成功導入模型模組")
        
        logger.info("導入工具模組")
        from src.utils.metrics import evaluate_model
        logger.info("成功導入工具模組")
        
        return True
    except Exception as e:
        logger.error(f"導入模組失敗: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def main():
    """主函數"""
    logger.info("=== 開始調試評估腳本 ===")
    start_time = time.time()
    
    try:
        # 解析命令行參數
        args = parse_args()
        logger.info(f"命令行參數: {args}")
        
        # 檢查模型文件
        if not check_model_file(args.model_path):
            logger.error("模型文件檢查失敗，終止執行")
            return
        
        # 載入配置文件
        try:
            config = load_config(args.config)
        except Exception as e:
            logger.error(f"載入配置文件失敗: {str(e)}")
            return
        
        # 嘗試導入必要模組
        if not try_import_modules():
            logger.error("導入必要模組失敗，終止執行")
            return
        
        # 設定計算設備
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用計算設備: {device}")
        
        # 顯示Python路徑
        logger.info(f"Python搜索路徑:")
        for p in sys.path:
            logger.info(f"  {p}")
        
        logger.info("調試評估腳本執行成功!")
    except Exception as e:
        logger.error(f"調試過程中發生錯誤: {str(e)}")
        logger.error(traceback.format_exc())
    
    end_time = time.time()
    logger.info(f"=== 調試完成，耗時: {end_time - start_time:.2f}秒 ===")

if __name__ == "__main__":
    main()