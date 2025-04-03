#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
predict.py - 模型預測腳本
本腳本用於使用訓練好的模型對銲錫接點進行疲勞壽命預測。
可以處理單個樣本預測或批量預測，並提供詳細的預測結果分析。

主要功能:
1. 載入訓練好的模型
2. 對新的結構參數和時間序列資料進行預測
3. 分析預測結果並產生視覺化圖表
4. 驗證預測結果是否符合物理約束
5. 批次處理多個輸入檔案
"""

import os
import sys
import argparse
import yaml
import logging
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# 確保可以導入專案模組
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# 導入專案模組
from src.data.preprocess import load_data, standardize_features, prepare_time_series
from src.models.hybrid_model import HybridPINNLSTMModel
from src.models.pinn import PINNModel
from src.models.lstm import LSTMModel
from src.utils.metrics import evaluate_model
from src.utils.visualization import plot_prediction_vs_true, visualize_model_results
from src.utils.physics import validate_physical_constraints, nf_from_delta_w

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, "logs", "predict.log"))
    ]
)
logger = logging.getLogger(__name__)

# 確保日誌目錄存在
os.makedirs(os.path.join(project_root, "logs"), exist_ok=True)


def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description="使用訓練好的模型進行銲錫接點疲勞壽命預測")
    parser.add_argument("--model-path", type=str, required=True,
                        help="模型檢查點路徑")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml",
                        help="模型配置文件路徑")
    parser.add_argument("--input", type=str,
                        help="輸入資料檔案路徑（CSV格式）")
    parser.add_argument("--output-dir", type=str, default="outputs/predictions",
                        help="預測結果輸出目錄")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"],
                        help="計算設備")
    parser.add_argument("--model-type", type=str, choices=["hybrid", "pinn", "lstm"],
                        default="hybrid", help="模型類型: hybrid, pinn, lstm")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="批次大小")
    parser.add_argument("--enforce-physics", action="store_true",
                        help="強制預測結果符合物理約束")
    parser.add_argument("--visualize", action="store_true",
                        help="產生視覺化結果")
    parser.add_argument("--single-sample", action="store_true",
                        help="僅預測單個樣本（交互式輸入）")
    
    return parser.parse_args()


def load_config(config_path):
    """載入YAML配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_model(model_path, config, model_type="hybrid", device=None):
    """
    載入已訓練的模型
    
    參數:
        model_path (str): 模型檢查點路徑
        config (dict): 模型配置
        model_type (str): 模型類型: hybrid, pinn, lstm
        device (torch.device): 計算設備
        
    返回:
        torch.nn.Module: 載入的模型
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 根據模型類型創建模型實例
    if model_type == "hybrid":
        model = HybridPINNLSTMModel(
            static_input_dim=len(config["model"]["input"]["static_features"]),
            time_input_dim=len(config["model"]["input"]["time_series_features"]),
            time_steps=config["model"]["input"]["time_steps"],
            pinn_hidden_dims=config["model"]["pinn"]["hidden_layers"],
            lstm_hidden_size=config["model"]["lstm"]["hidden_size"],
            lstm_num_layers=config["model"]["lstm"]["num_layers"],
            fusion_dim=config["model"]["fusion"]["fusion_dim"],
            dropout_rate=config["model"]["pinn"]["dropout_rate"],
            bidirectional=config["model"]["lstm"]["bidirectional"],
            use_attention=config["model"]["lstm"]["use_attention"],
            use_physics_layer=config["model"]["pinn"]["use_physics_layer"]
        )
    elif model_type == "pinn":
        model = PINNModel(
            input_dim=len(config["model"]["input"]["static_features"]),
            hidden_dims=config["model"]["pinn"]["hidden_layers"],
            dropout_rate=config["model"]["pinn"]["dropout_rate"],
            use_physics_layer=config["model"]["pinn"]["use_physics_layer"]
        )
    elif model_type == "lstm":
        model = LSTMModel(
            input_dim=len(config["model"]["input"]["time_series_features"]),
            hidden_size=config["model"]["lstm"]["hidden_size"],
            num_layers=config["model"]["lstm"]["num_layers"],
            bidirectional=config["model"]["lstm"]["bidirectional"],
            dropout_rate=config["model"]["lstm"]["dropout_rate"],
            use_attention=config["model"]["lstm"]["use_attention"]
        )
    else:
        raise ValueError(f"不支援的模型類型: {model_type}")
    
    # 載入模型權重
    checkpoint = torch.load(model_path, map_location=device)
    
    # 處理不同格式的檢查點
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    # 將模型移至指定設備並設為評估模式
    model = model.to(device)
    model.eval()
    
    return model


def preprocess_input_data(input_df, config, scalers=None):
    """
    預處理輸入資料
    
    參數:
        input_df (pandas.DataFrame): 輸入資料
        config (dict): 模型配置
        scalers (dict, optional): 包含特徵縮放器的字典
        
    返回:
        tuple: (靜態特徵, 時間序列, 縮放器)
    """
    # 檢查輸入資料是否包含所有必要的欄位
    static_features = config["model"]["input"]["static_features"]
    time_series_features = config["model"]["input"]["time_series_features"]
    
    # 檢查靜態特徵
    for feature in static_features:
        if feature not in input_df.columns:
            raise ValueError(f"輸入資料缺少靜態特徵: {feature}")
    
    # 檢查時間序列特徵（基本形式，如"NLPLWK_up"和"NLPLWK_down"）
    for feature in time_series_features:
        # 查找包含該特徵的所有列
        matching_columns = [col for col in input_df.columns if feature in col]
        if not matching_columns:
            raise ValueError(f"輸入資料缺少時間序列特徵: {feature}")
    
    # 提取靜態特徵
    X = input_df[static_features].values
    
    # 標準化靜態特徵（如果提供了縮放器）
    if scalers is not None and "feature_scaler" in scalers:
        X_scaled = scalers["feature_scaler"].transform(X)
    else:
        # 否則，使用簡單的最小-最大縮放
        X_scaled = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0) + 1e-8)
    
    # 預處理時間序列特徵
    # 首先確定時間步數
    time_steps = config["model"]["input"]["time_steps"]
    
    # 尋找時間序列列
    up_columns = [col for col in input_df.columns if "NLPLWK_up" in col]
    down_columns = [col for col in input_df.columns if "NLPLWK_down" in col]
    
    # 確保時間序列列的數量正確
    if len(up_columns) != time_steps or len(down_columns) != time_steps:
        raise ValueError(f"時間序列列數量不匹配: 上界面 {len(up_columns)}, 下界面 {len(down_columns)}, 預期 {time_steps}")
    
    # 按時間順序排序時間序列列
    up_columns.sort()
    down_columns.sort()
    
    # 創建時間序列資料
    n_samples = len(input_df)
    time_series_data = np.zeros((n_samples, time_steps, 2))
    
    for i in range(time_steps):
        time_series_data[:, i, 0] = input_df[up_columns[i]].values
        time_series_data[:, i, 1] = input_df[down_columns[i]].values
    
    # 標準化時間序列資料
    for feature_idx in range(2):
        feature_data = time_series_data[:, :, feature_idx]
        feature_mean = np.mean(feature_data)
        feature_std = np.std(feature_data)
        time_series_data[:, :, feature_idx] = (feature_data - feature_mean) / (feature_std + 1e-8)
    
    return X_scaled, time_series_data, scalers


def make_predictions(model, X, time_series, config, device, enforce_physics=False):
    """
    使用模型進行預測
    
    參數:
        model (torch.nn.Module): 模型
        X (numpy.ndarray): 靜態特徵
        time_series (numpy.ndarray): 時間序列
        config (dict): 模型配置
        device (torch.device): 計算設備
        enforce_physics (bool): 是否強制預測結果符合物理約束
        
    返回:
        dict: 預測結果
    """
    # 將輸入轉換為PyTorch張量
    X_tensor = torch.FloatTensor(X).to(device)
    time_series_tensor = torch.FloatTensor(time_series).to(device)
    
    # 將模型設為評估模式
    model.eval()
    
    # 獲取物理模型參數
    a = config["model"]["physics"]["a_coefficient"]
    b = config["model"]["physics"]["b_coefficient"]
    
    # 設定批次大小
    batch_size = 32
    
    # 初始化結果
    all_outputs = {"predictions": []}
    
    # 分批處理大型資料集
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            X_batch = X_tensor[i:i+batch_size]
            time_series_batch = time_series_tensor[i:i+batch_size]
            
            # 根據模型類型進行預測
            if isinstance(model, HybridPINNLSTMModel):
                outputs = model(X_batch, time_series_batch, return_features=True)
                
                # 收集各分支預測結果
                batch_predictions = outputs["nf_pred"].cpu().numpy()
                
                # 收集其他輸出
                for key, value in outputs.items():
                    if key not in all_outputs:
                        all_outputs[key] = []
                    
                    if isinstance(value, torch.Tensor):
                        all_outputs[key].append(value.cpu().numpy())
                    else:
                        all_outputs[key].append(value)
                
            elif isinstance(model, PINNModel):
                outputs = model(X_batch)
                batch_predictions = outputs["nf_pred"].cpu().numpy()
                
                # 收集其他輸出
                for key, value in outputs.items():
                    if key not in all_outputs:
                        all_outputs[key] = []
                    
                    if isinstance(value, torch.Tensor):
                        all_outputs[key].append(value.cpu().numpy())
                    else:
                        all_outputs[key].append(value)
                
            elif isinstance(model, LSTMModel):
                outputs = model(time_series_batch)
                batch_predictions = outputs["output"].cpu().numpy()
                
                # 收集其他輸出
                for key, value in outputs.items():
                    if key not in all_outputs:
                        all_outputs[key] = []
                    
                    if isinstance(value, torch.Tensor):
                        all_outputs[key].append(value.cpu().numpy())
                    else:
                        all_outputs[key].append(value)
            else:
                raise ValueError(f"不支援的模型類型: {type(model)}")
            
            # 如果啟用物理約束，強制預測結果符合物理模型
            if enforce_physics and "delta_w" in outputs:
                delta_w_batch = outputs["delta_w"].cpu().numpy()
                physics_nf = nf_from_delta_w(delta_w_batch, a, b)
                batch_predictions = physics_nf
            
            all_outputs["predictions"].append(batch_predictions)
    
    # 合併批次結果
    for key in all_outputs:
        if isinstance(all_outputs[key][0], np.ndarray):
            all_outputs[key] = np.concatenate(all_outputs[key])
    
    return all_outputs


def create_input_from_user():
    """
    從用戶交互式輸入創建單個樣本資料
    
    返回:
        pandas.DataFrame: 包含單個樣本的資料框
    """
    print("\n===== 請輸入銲錫接點結構參數 =====")
    
    # 結構參數輸入
    die = float(input("Die (晶片高度, μm, 150-250): ") or "200")
    stud = float(input("Stud (銅高度, μm, 60-80): ") or "70")
    mold = float(input("Mold (環氧樹脂, μm, 55-75): ") or "65")
    pcb = float(input("PCB (基板厚度, mm, 0.6-1.0): ") or "0.8")
    warpage = float(input("Unit warpage (翹曲變形量, μm): ") or "10")
    
    print("\n===== 請輸入非線性塑性應變功時間序列資料 =====")
    print("(注意: 四個時間步3600s, 7200s, 10800s, 14400s)")
    
    # 時間序列資料輸入
    up_3600 = float(input("NLPLWK_up_3600: ") or "0.005")
    up_7200 = float(input("NLPLWK_up_7200: ") or "0.010")
    up_10800 = float(input("NLPLWK_up_10800: ") or "0.015")
    up_14400 = float(input("NLPLWK_up_14400: ") or "0.020")
    
    down_3600 = float(input("NLPLWK_down_3600: ") or "0.004")
    down_7200 = float(input("NLPLWK_down_7200: ") or "0.008")
    down_10800 = float(input("NLPLWK_down_10800: ") or "0.012")
    down_14400 = float(input("NLPLWK_down_14400: ") or "0.016")
    
    # 創建資料框
    data = {
        "Die": [die],
        "stud": [stud],
        "mold": [mold],
        "PCB": [pcb],
        "Unit_warpage": [warpage],
        "NLPLWK_up_3600": [up_3600],
        "NLPLWK_up_7200": [up_7200],
        "NLPLWK_up_10800": [up_10800],
        "NLPLWK_up_14400": [up_14400],
        "NLPLWK_down_3600": [down_3600],
        "NLPLWK_down_7200": [down_7200],
        "NLPLWK_down_10800": [down_10800],
        "NLPLWK_down_14400": [down_14400]
    }
    
    return pd.DataFrame(data)


def save_prediction_results(results, input_df, output_dir, config, visualize=True):
    """
    保存預測結果
    
    參數:
        results (dict): 預測結果
        input_df (pandas.DataFrame): 輸入資料
        output_dir (str): 輸出目錄
        config (dict): 模型配置
        visualize (bool): 是否產生視覺化結果
    """
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 獲取預測結果
    predictions = results["predictions"]
    
    # 將預測結果添加到輸入資料中
    output_df = input_df.copy()
    output_df["Nf_pred"] = predictions
    
    # 如果有delta_w，也添加到輸出
    if "delta_w" in results:
        output_df["Delta_W"] = results["delta_w"]
    
    # 如果是混合模型，添加各分支預測結果
    if "pinn_nf_pred" in results:
        output_df["Nf_pred_PINN"] = results["pinn_nf_pred"]
    
    if "lstm_nf_pred" in results:
        output_df["Nf_pred_LSTM"] = results["lstm_nf_pred"]
    
    # 保存預測結果
    output_file = os.path.join(output_dir, "predictions.csv")
    output_df.to_csv(output_file, index=False)
    logger.info(f"預測結果已保存到: {output_file}")
    
    # 如果有真實值，評估預測性能
    if "Nf_pred (cycles)" in input_df.columns:
        targets = input_df["Nf_pred (cycles)"].values
        results["targets"] = targets
        
        # 評估預測性能
        metrics = evaluate_model(targets, predictions, model_name="Model", verbose=True)
        
        # 保存評估指標
        metrics_file = os.path.join(output_dir, "metrics.csv")
        pd.DataFrame([metrics]).to_csv(metrics_file, index=False)
        logger.info(f"評估指標已保存到: {metrics_file}")
    
    # 如果需要，產生視覺化結果
    if visualize:
        visualizations_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(visualizations_dir, exist_ok=True)
        
        # 產生視覺化結果
        visualization_paths = visualize_model_results(results, output_dir=visualizations_dir)
        logger.info(f"視覺化結果已保存到: {visualizations_dir}")
    
    # 驗證物理約束條件
    if "delta_w" in results and "predictions" in results:
        a = config["model"]["physics"]["a_coefficient"]
        b = config["model"]["physics"]["b_coefficient"]
        
        passed, residuals, violated = validate_physical_constraints(
            results["delta_w"], results["predictions"], 
            a=a, b=b, threshold=20.0, verbose=True
        )
        
        if not passed:
            logger.warning(f"物理約束驗證失敗, 有 {len(violated)} 個樣本違反約束")
            
            # 保存違反約束的樣本
            if len(violated) > 0:
                violated_samples = output_df.iloc[violated].copy()
                violated_file = os.path.join(output_dir, "violated_samples.csv")
                violated_samples.to_csv(violated_file, index=False)
                logger.warning(f"違反物理約束的樣本已保存到: {violated_file}")
        else:
            logger.info("所有預測結果都滿足物理約束條件")
    
    return output_file


def main():
    """主函數"""
    # 確保輸出目錄存在
    os.makedirs(os.path.join(project_root, "outputs", "predictions"), exist_ok=True)
    
    # 解析命令行參數
    args = parse_args()
    
    # 載入配置文件
    logger.info(f"載入配置文件: {args.config}")
    config = load_config(args.config)
    
    # 設定計算設備
    device_name = args.device if args.device else config.get("device", "cuda")
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    logger.info(f"使用計算設備: {device}")
    
    # 載入模型
    logger.info(f"載入模型: {args.model_path}")
    model = load_model(args.model_path, config, model_type=args.model_type, device=device)
    
    # 設定輸出目錄
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 交互式輸入模式
    if args.single_sample:
        logger.info("進入單樣本交互式預測模式")
        
        # 從用戶輸入創建樣本
        input_df = create_input_from_user()
        
        logger.info("預處理輸入資料...")
        X, time_series, _ = preprocess_input_data(input_df, config)
        
        logger.info("進行預測...")
        results = make_predictions(model, X, time_series, config, device, 
                                 enforce_physics=args.enforce_physics)
        
        # 顯示預測結果
        prediction = results["predictions"][0]
        logger.info(f"\n預測結果: Nf = {prediction:.2f} 循環")
        
        # 如果有delta_w，也顯示
        if "delta_w" in results:
            delta_w = results["delta_w"][0]
            logger.info(f"非線性塑性應變能密度變化量 (ΔW): {delta_w:.6f}")
        
        # 如果是混合模型，顯示各分支預測
        if "pinn_nf_pred" in results:
            pinn_pred = results["pinn_nf_pred"][0]
            logger.info(f"PINN分支預測: Nf = {pinn_pred:.2f} 循環")
        
        if "lstm_nf_pred" in results:
            lstm_pred = results["lstm_nf_pred"][0]
            logger.info(f"LSTM分支預測: Nf = {lstm_pred:.2f} 循環")
        
        # 詢問是否保存結果
        save_result = input("\n是否保存預測結果? (y/n): ").lower() == 'y'
        if save_result:
            output_file = save_prediction_results(
                results, input_df, output_dir, config, visualize=args.visualize
            )
            logger.info(f"預測結果已保存到: {output_file}")
    
    # 批次預測模式
    elif args.input:
        logger.info(f"載入輸入資料: {args.input}")
        input_df = load_data(args.input)
        
        logger.info(f"資料集大小: {len(input_df)} 樣本")
        
        logger.info("預處理輸入資料...")
        X, time_series, _ = preprocess_input_data(input_df, config)
        
        logger.info("進行預測...")
        results = make_predictions(model, X, time_series, config, device,
                                 enforce_physics=args.enforce_physics)
        
        logger.info("保存預測結果...")
        output_file = save_prediction_results(
            results, input_df, output_dir, config, visualize=args.visualize
        )
        
        logger.info(f"預測完成，結果已保存到: {output_file}")
    
    else:
        logger.error("請指定輸入資料文件路徑 (--input) 或使用單樣本模式 (--single-sample)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"執行過程中發生錯誤: {str(e)}")
        sys.exit(1)