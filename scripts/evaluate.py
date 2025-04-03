#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
complete_evaluate.py - 完整的模型評估腳本
本腳本用於評估銲錫接點疲勞壽命預測模型的性能，提供詳細的評估指標和視覺化結果。
"""

import os
import sys
import argparse
import yaml
import logging
import traceback
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# 確保可以導入專案模組
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, "logs", "complete_evaluate.log"))
    ]
)
logger = logging.getLogger(__name__)

# 確保日誌目錄存在
os.makedirs(os.path.join(project_root, "logs"), exist_ok=True)


def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description="評估銲錫接點疲勞壽命預測模型")
    parser.add_argument("--model-path", type=str, required=True,
                        help="模型檢查點路徑")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml",
                        help="模型配置文件路徑")
    parser.add_argument("--data", type=str, default="data/raw/Training_data_warpage_final_20250321_v1.2.csv",
                        help="評估數據路徑")
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation",
                        help="評估結果輸出目錄")
    parser.add_argument("--model-type", type=str, choices=["hybrid", "pinn", "lstm"],
                        default="hybrid", help="模型類型: hybrid, pinn, lstm")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"],
                        help="計算設備，覆蓋配置文件中的設定")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="批次大小")
    parser.add_argument("--visualize", action="store_true",
                        help="產生視覺化結果")
    parser.add_argument("--debug", action="store_true",
                        help="啟用除錯模式")
    
    return parser.parse_args()


def load_config(config_path):
    """載入YAML配置文件"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"成功載入配置文件: {config_path}")
        return config
    except Exception as e:
        logger.error(f"載入配置文件失敗: {str(e)}")
        raise


def load_model(model_path, config, model_type="hybrid", device=None):
    """載入模型"""
    try:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 導入模型類
        from src.models.hybrid_model import HybridPINNLSTMModel
        from src.models.pinn import PINNModel
        from src.models.lstm import LSTMModel
        
        # 根據模型類型創建模型實例
        if model_type == "hybrid":
            logger.info("創建混合PINN-LSTM模型")
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
            logger.info("創建PINN模型")
            model = PINNModel(
                input_dim=len(config["model"]["input"]["static_features"]),
                hidden_dims=config["model"]["pinn"]["hidden_layers"],
                dropout_rate=config["model"]["pinn"]["dropout_rate"],
                use_physics_layer=config["model"]["pinn"]["use_physics_layer"]
            )
        elif model_type == "lstm":
            logger.info("創建LSTM模型")
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
        logger.info(f"從 {model_path} 載入模型權重")
        checkpoint = torch.load(model_path, map_location=device)
        
        # 處理不同格式的檢查點
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("從model_state_dict載入模型權重")
        else:
            model.load_state_dict(checkpoint)
            logger.info("直接從檢查點載入模型權重")
        
        # 將模型移至指定設備並設為評估模式
        model = model.to(device)
        model.eval()
        
        logger.info(f"模型載入成功，移至 {device} 設備")
        
        return model
    except Exception as e:
        logger.error(f"載入模型失敗: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def load_data(data_path, config):
    """載入評估數據"""
    try:
        from src.data.preprocess import load_data as load_data_func
        from src.data.preprocess import standardize_features, prepare_time_series
        
        logger.info(f"從 {data_path} 載入數據")
        df = load_data_func(data_path)
        
        # 取得特徵和目標列
        feature_cols = config["model"]["input"]["static_features"]
        target_col = "Nf_pred (cycles)"
        
        logger.info(f"標準化特徵: {feature_cols}")
        X, feature_scaler, y = standardize_features(df, feature_cols, target_col)
        
        logger.info("準備時間序列數據")
        time_series_data = prepare_time_series(df)
        
        logger.info(f"數據載入完成: {len(df)} 個樣本")
        
        return {
            "X": X,
            "time_series": time_series_data,
            "y": y,
            "feature_scaler": feature_scaler,
            "df_original": df
        }
    except Exception as e:
        logger.error(f"載入數據失敗: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def create_dataloaders(X, time_series, y, batch_size=16):
    """創建數據載入器"""
    try:
        # 轉換為PyTorch張量
        X_tensor = torch.FloatTensor(X)
        time_series_tensor = torch.FloatTensor(time_series)
        y_tensor = torch.FloatTensor(y)
        
        # 創建數據集
        dataset = torch.utils.data.TensorDataset(X_tensor, time_series_tensor, y_tensor)
        
        # 創建數據載入器
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        logger.info(f"創建數據載入器成功: {len(dataset)} 個樣本，批次大小 {batch_size}")
        
        return dataloader
    except Exception as e:
        logger.error(f"創建數據載入器失敗: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def evaluate_model_performance(model, dataloader, device, model_type="hybrid"):
    """評估模型性能"""
    try:
        # 初始化結果收集
        all_predictions = []
        all_targets = []
        model_outputs = {}
        
        # 進行評估
        model.eval()
        with torch.no_grad():
            for batch_idx, (X_batch, time_series_batch, y_batch) in enumerate(dataloader):
                # 將數據移到設備
                X_batch = X_batch.to(device)
                time_series_batch = time_series_batch.to(device)
                y_batch = y_batch.to(device)
                
                # 根據模型類型進行預測
                if model_type == "hybrid":
                    outputs = model(X_batch, time_series_batch, return_features=True)
                    predictions = outputs["nf_pred"]
                    # 收集其他輸出
                    for key in outputs:
                        if key not in model_outputs:
                            model_outputs[key] = []
                        model_outputs[key].append(outputs[key].cpu().numpy())
                elif model_type == "pinn":
                    outputs = model(X_batch)
                    predictions = outputs["nf_pred"]
                    # 收集其他輸出
                    for key in outputs:
                        if key not in model_outputs:
                            model_outputs[key] = []
                        model_outputs[key].append(outputs[key].cpu().numpy())
                elif model_type == "lstm":
                    outputs = model(time_series_batch)
                    predictions = outputs["output"]
                    # 收集其他輸出
                    for key in outputs:
                        if key not in model_outputs:
                            model_outputs[key] = []
                        model_outputs[key].append(outputs[key].cpu().numpy())
                else:
                    raise ValueError(f"不支援的模型類型: {model_type}")
                
                # 收集預測和目標
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        
        # 合併批次結果
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        
        # 合併模型輸出
        for key in model_outputs:
            model_outputs[key] = np.concatenate(model_outputs[key])
        
        # 添加預測和目標到模型輸出
        model_outputs["predictions"] = all_predictions
        model_outputs["targets"] = all_targets
        
        logger.info(f"評估完成: {len(all_predictions)} 個預測")
        
        return model_outputs
    except Exception as e:
        logger.error(f"評估模型性能失敗: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def compute_metrics(predictions, targets):
    """計算評估指標"""
    try:
        from src.utils.metrics import evaluate_model as compute_eval_metrics
        
        # 計算評估指標
        metrics = compute_eval_metrics(
            targets, predictions, model_name="Model", verbose=True, return_dataframe=True
        )
        
        return metrics
    except Exception as e:
        logger.error(f"計算評估指標失敗: {str(e)}")
        logger.error(traceback.format_exc())
        
        # 如果指標計算失敗，回退到基本指標計算
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        r2 = r2_score(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        
        logger.info(f"使用基本指標計算: RMSE={rmse:.4f}, R²={r2:.4f}, MAE={mae:.4f}")
        
        return pd.DataFrame({
            "model": ["Model"],
            "rmse": [rmse],
            "r2": [r2],
            "mae": [mae]
        })


def visualize_results(results, output_dir, config=None):
    """生成視覺化結果"""
    try:
        from src.utils.visualization import (
            plot_prediction_vs_true, 
            visualize_model_results
        )
        
        # 創建視覺化目錄
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # 繪製預測vs真實值圖
        logger.info("繪製預測vs真實值圖")
        fig = plot_prediction_vs_true(
            results["targets"], results["predictions"], 
            model_name="Model",
            save_path=os.path.join(vis_dir, "prediction_vs_true.png")
        )
        plt.close(fig)
        
        # 生成詳細視覺化結果
        logger.info("生成詳細視覺化結果")
        vis_paths = visualize_model_results(results, output_dir=vis_dir)
        
        logger.info(f"視覺化結果保存到: {vis_dir}")
        
        return vis_paths
    except Exception as e:
        logger.error(f"生成視覺化結果失敗: {str(e)}")
        logger.error(traceback.format_exc())
        return []


def validate_physics(results, config):
    """驗證物理約束"""
    try:
        from src.utils.physics import validate_physical_constraints
        
        if "delta_w" in results and results["delta_w"] is not None:
            logger.info("驗證物理約束")
            
            # 獲取物理模型參數
            a = config["model"]["physics"]["a_coefficient"]
            b = config["model"]["physics"]["b_coefficient"]
            
            # 驗證物理約束
            passed, residuals, violated = validate_physical_constraints(
                results["delta_w"], results["predictions"], 
                a=a, b=b, threshold=20.0, verbose=True
            )
            
            if passed:
                logger.info("所有預測結果都滿足物理約束條件")
            else:
                logger.warning(f"物理約束驗證失敗, 有 {len(violated)} 個樣本違反約束")
            
            return {
                "passed": passed,
                "residuals": residuals,
                "violated": violated
            }
        else:
            logger.info("結果中沒有delta_w，跳過物理約束驗證")
            return None
    except Exception as e:
        logger.error(f"驗證物理約束失敗: {str(e)}")
        logger.error(traceback.format_exc())
        return None


def save_results(results, metrics, physics_validation, output_dir, data_df=None):
    """保存評估結果"""
    try:
        # 創建輸出目錄
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存預測結果
        predictions_df = pd.DataFrame({
            "Targets": results["targets"],
            "Predictions": results["predictions"]
        })
        
        # 如果有原始數據，添加其他列
        if data_df is not None:
            for col in data_df.columns:
                if col not in predictions_df.columns:
                    predictions_df[col] = data_df[col].values
        
        # 保存預測結果
        predictions_path = os.path.join(output_dir, "predictions.csv")
        predictions_df.to_csv(predictions_path, index=False)
        logger.info(f"預測結果保存到: {predictions_path}")
        
        # 保存評估指標
        metrics_path = os.path.join(output_dir, "metrics.csv")
        metrics.to_csv(metrics_path, index=False)
        logger.info(f"評估指標保存到: {metrics_path}")
        
        # 如果有物理驗證結果，保存違反約束的樣本
        if physics_validation is not None and not physics_validation["passed"]:
            violated_idx = physics_validation["violated"]
            if len(violated_idx) > 0 and data_df is not None:
                violated_df = data_df.iloc[violated_idx].copy()
                violated_df["Prediction"] = results["predictions"][violated_idx]
                violated_df["DeltaW"] = results["delta_w"][violated_idx]
                violated_df["Residual"] = physics_validation["residuals"][violated_idx]
                
                violated_path = os.path.join(output_dir, "violated_samples.csv")
                violated_df.to_csv(violated_path, index=False)
                logger.info(f"違反物理約束的樣本保存到: {violated_path}")
        
        return True
    except Exception as e:
        logger.error(f"保存評估結果失敗: {str(e)}")
        logger.error(traceback.format_exc())
        return False


def main():
    """主函數"""
    start_time = time.time()
    logger.info("=== 開始模型評估 ===")
    
    try:
        # 解析命令行參數
        args = parse_args()
        logger.info(f"命令行參數: {args}")
        
        # 如果啟用除錯模式，調整日誌級別
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("除錯模式已啟用")
        
        # 載入配置文件
        config = load_config(args.config)
        
        # 設定計算設備
        device_name = args.device if args.device else config.get("device", "cuda")
        device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
        logger.info(f"使用計算設備: {device}")
        
        # 載入模型
        model = load_model(args.model_path, config, model_type=args.model_type, device=device)
        
        # 載入評估數據
        data_dict = load_data(args.data, config)
        
        # 創建數據載入器
        dataloader = create_dataloaders(
            data_dict["X"], data_dict["time_series"], data_dict["y"], 
            batch_size=args.batch_size
        )
        
        # 評估模型性能
        logger.info("開始評估模型性能")
        results = evaluate_model_performance(model, dataloader, device, model_type=args.model_type)
        
        # 計算評估指標
        logger.info("計算評估指標")
        metrics = compute_metrics(results["predictions"], results["targets"])
        
        # 驗證物理約束
        physics_validation = validate_physics(results, config)
        
        # 創建輸出目錄
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存評估結果
        save_results(results, metrics, physics_validation, output_dir, data_dict["df_original"])
        
        # 生成視覺化結果
        if args.visualize:
            logger.info("生成視覺化結果")
            visualize_results(results, output_dir, config)
        
        end_time = time.time()
        logger.info(f"=== 評估完成，耗時: {end_time - start_time:.2f}秒 ===")
        
        return 0
    except Exception as e:
        logger.error(f"評估過程中發生錯誤: {str(e)}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("評估被用戶中斷")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"未捕獲的錯誤: {str(e)}")
        sys.exit(1)