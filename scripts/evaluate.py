#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
evaluate.py - 模型評估腳本
本腳本用於詳細評估銲錫接點疲勞壽命預測的混合PINN-LSTM模型性能。
不同於一般預測，評估腳本專注於全面分析模型的預測性能，驗證物理約束條件，
並針對不同結構參數組合進行敏感性分析。

主要功能:
1. 全面評估模型效能，包括RMSE、R²、MAE、相對誤差等多種指標
2. 物理約束驗證，確保預測結果符合能量密度法疲勞壽命模型關係
3. 特徵重要性分析，評估不同結構參數對預測結果的影響
4. 對比多個模型性能，包括混合模型與單一分支模型
5. 產生詳細評估報告和視覺化結果
"""

import os
import sys
import argparse
import yaml
import logging
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance

# 確保可以導入專案模組
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# 導入專案模組
from src.data.preprocess import process_pipeline
from src.data.dataset import create_dataloaders
from src.models.hybrid_model import HybridPINNLSTMModel
from src.models.pinn import PINNModel
from src.models.lstm import LSTMModel
from src.training.trainer import Trainer, EarlyStopping
from src.utils.metrics import (
    evaluate_model, compare_models, calculate_rmse, calculate_r2, 
    calculate_mae, calculate_relative_error, plot_metrics_comparison
)
from src.utils.visualization import (
    plot_prediction_vs_true, plot_parameter_impact, 
    plot_physical_constraint_validation, visualize_model_results,
    plot_attention_weights, plot_feature_importance
)
from src.utils.physics import (
    validate_physical_constraints, calculate_delta_w, nf_from_delta_w
)

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, "logs", "evaluate.log"))
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
    parser.add_argument("--data", type=str,
                        help="資料檔案路徑，覆蓋配置文件中的設定")
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation",
                        help="評估結果輸出目錄")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"],
                        help="計算設備，覆蓋配置文件中的設定")
    parser.add_argument("--model-type", type=str, choices=["hybrid", "pinn", "lstm"],
                        default="hybrid", help="模型類型: hybrid, pinn, lstm")
    parser.add_argument("--batch-size", type=int,
                        help="批次大小，覆蓋配置文件中的設定")
    parser.add_argument("--cross-validation", action="store_true",
                        help="使用交叉驗證評估模型")
    parser.add_argument("--n-splits", type=int, default=5,
                        help="交叉驗證的折數")
    parser.add_argument("--feature-importance", action="store_true",
                        help="進行特徵重要性分析")
    parser.add_argument("--sensitivity-analysis", action="store_true",
                        help="進行敏感性分析")
    parser.add_argument("--compare-models", action="store_true",
                        help="比較不同模型的性能")
    parser.add_argument("--compare-model-paths", type=str, nargs="+",
                        help="要比較的模型路徑列表")
    parser.add_argument("--compare-model-names", type=str, nargs="+",
                        help="要比較的模型名稱列表")
    parser.add_argument("--report", action="store_true",
                        help="生成詳細評估報告")
    
    return parser.parse_args()


def load_config(config_path):
    """載入YAML配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_model(model_path, config, model_type="hybrid", device=None):
    """
    載入模型
    
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


def evaluate_standard(model, data_dict, config, device, output_dir, batch_size=16, model_name=None):
    """
    標準評估
    
    參數:
        model (torch.nn.Module): 模型
        data_dict (dict): 資料字典
        config (dict): 配置
        device (torch.device): 計算設備
        output_dir (str): 輸出目錄
        batch_size (int): 批次大小
        model_name (str): 模型名稱
        
    返回:
        dict: 評估結果
    """
    # 創建評估目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 創建資料載入器
    dataloaders = create_dataloaders(
        data_dict["X_train"], data_dict["X_val"], data_dict["X_test"],
        data_dict["time_series_train"], data_dict["time_series_val"], data_dict["time_series_test"],
        data_dict["y_train"], data_dict["y_val"], data_dict["y_test"],
        batch_size=batch_size
    )
    
    # 創建訓練器 (僅用於評估)
    trainer = Trainer(model, None, None, device=device)
    
    # 評估訓練集
    train_loss, train_metrics, train_predictions, train_targets = trainer.evaluate(
        dataloaders["train_loader"], return_predictions=True
    )
    
    # 評估驗證集
    val_loss, val_metrics, val_predictions, val_targets = trainer.evaluate(
        dataloaders["val_loader"], return_predictions=True
    )
    
    # 評估測試集
    test_loss, test_metrics, test_predictions, test_targets = trainer.evaluate(
        dataloaders["test_loader"], return_predictions=True
    )
    
    # 獲取詳細預測結果
    results = trainer.predict(dataloaders["test_loader"])
    
    # 記錄評估指標
    logger.info("評估指標:")
    logger.info(f"訓練集 - 損失: {train_loss:.6f}, RMSE: {train_metrics['rmse']:.6f}, R²: {train_metrics['r2_score']:.6f}")
    logger.info(f"驗證集 - 損失: {val_loss:.6f}, RMSE: {val_metrics['rmse']:.6f}, R²: {val_metrics['r2_score']:.6f}")
    logger.info(f"測試集 - 損失: {test_loss:.6f}, RMSE: {test_metrics['rmse']:.6f}, R²: {test_metrics['r2_score']:.6f}")
    
    # 保存評估指標
    metrics_df = pd.DataFrame({
        "dataset": ["train", "validation", "test"],
        "loss": [train_loss, val_loss, test_loss],
        "rmse": [train_metrics["rmse"], val_metrics["rmse"], test_metrics["rmse"]],
        "r2": [train_metrics["r2_score"], val_metrics["r2_score"], test_metrics["r2_score"]],
        "mae": [train_metrics["mae"], val_metrics["mae"], test_metrics["mae"]],
        "mape": [train_metrics.get("mape", 0), val_metrics.get("mape", 0), test_metrics.get("mape", 0)]
    })
    metrics_df.to_csv(os.path.join(output_dir, "evaluation_metrics.csv"), index=False)
    
    # 詳細評估測試集
    model_name_str = model_name if model_name else config["model"]["name"]
    detailed_metrics = evaluate_model(
        test_targets, test_predictions, model_name=model_name_str, verbose=True, return_dataframe=True
    )
    detailed_metrics.to_csv(os.path.join(output_dir, "detailed_metrics.csv"), index=False)
    
    # 物理約束驗證
    if "delta_w" in results:
        a = config["model"]["physics"]["a_coefficient"]
        b = config["model"]["physics"]["b_coefficient"]
        
        passed, residuals, violated = validate_physical_constraints(
            results["delta_w"], results["predictions"], 
            a=a, b=b, threshold=20.0, verbose=True
        )
        
        # 保存物理約束驗證結果
        physics_validation = {
            "passed": passed,
            "mean_residual": np.mean(residuals),
            "max_residual": np.max(residuals),
            "violations_count": len(violated),
            "violations_percentage": len(violated) / len(residuals) * 100 if len(residuals) > 0 else 0
        }
        
        pd.DataFrame([physics_validation]).to_csv(os.path.join(output_dir, "physics_validation.csv"), index=False)
    
    # 創建視覺化目錄
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # 產生預測對比圖
    fig = plot_prediction_vs_true(
        test_targets, test_predictions, 
        model_name=model_name_str,
        save_path=os.path.join(vis_dir, "prediction_vs_true.png")
    )
    plt.close(fig)
    
    # 產生詳細視覺化結果
    results["targets"] = test_targets
    visualization_paths = visualize_model_results(results, output_dir=vis_dir)
    
    # 收集所有評估結果
    evaluation_results = {
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss,
        "train_predictions": train_predictions,
        "train_targets": train_targets,
        "val_predictions": val_predictions,
        "val_targets": val_targets,
        "test_predictions": test_predictions,
        "test_targets": test_targets,
        "detailed_predictions": results
    }
    
    if "delta_w" in results:
        evaluation_results["physics_validation"] = physics_validation
    
    return evaluation_results


def evaluate_cross_validation(model_path, data_dict, config, device, output_dir, 
                             n_splits=5, batch_size=16, model_type="hybrid"):
    """
    交叉驗證評估
    
    參數:
        model_path (str): 模型檢查點路徑
        data_dict (dict): 資料字典
        config (dict): 配置
        device (torch.device): 計算設備
        output_dir (str): 輸出目錄
        n_splits (int): 交叉驗證的折數
        batch_size (int): 批次大小
        model_type (str): 模型類型
        
    返回:
        dict: 交叉驗證結果
    """
    # 創建交叉驗證目錄
    cv_dir = os.path.join(output_dir, "cross_validation")
    os.makedirs(cv_dir, exist_ok=True)
    
    # 收集所有特徵和標籤
    X = np.vstack([data_dict["X_train"], data_dict["X_val"], data_dict["X_test"]])
    time_series = np.vstack([data_dict["time_series_train"], data_dict["time_series_val"], data_dict["time_series_test"]])
    y = np.concatenate([data_dict["y_train"], data_dict["y_val"], data_dict["y_test"]])
    
    # 初始化KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # 初始化結果收集
    cv_results = {
        "fold": [],
        "rmse": [],
        "r2": [],
        "mae": [],
        "mape": [],
        "rel_error_mean": []
    }
    
    all_predictions = []
    all_targets = []
    
    # 進行交叉驗證
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        logger.info(f"開始第 {fold+1} 折評估...")
        
        # 分割資料
        X_train, X_test = X[train_idx], X[test_idx]
        time_series_train, time_series_test = time_series[train_idx], time_series[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 創建資料載入器
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(time_series_train),
            torch.FloatTensor(y_train)
        )
        test_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_test),
            torch.FloatTensor(time_series_test),
            torch.FloatTensor(y_test)
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        
        # 加載模型
        model = load_model(model_path, config, model_type=model_type, device=device)
        
        # 創建訓練器
        trainer = Trainer(model, None, None, device=device)
        
        # 評估測試集
        _, test_metrics, test_predictions, test_targets = trainer.evaluate(
            test_loader, return_predictions=True
        )
        
        # 記錄評估指標
        cv_results["fold"].append(fold + 1)
        cv_results["rmse"].append(test_metrics["rmse"])
        cv_results["r2"].append(test_metrics["r2_score"])
        cv_results["mae"].append(test_metrics["mae"])
        cv_results["mape"].append(test_metrics.get("mape", 0))
        
        # 計算相對誤差
        rel_error = calculate_relative_error(test_targets, test_predictions)
        cv_results["rel_error_mean"].append(rel_error["mean"])
        
        # 收集預測和目標
        all_predictions.append(test_predictions)
        all_targets.append(test_targets)
        
        # 保存本折評估指標
        fold_dir = os.path.join(cv_dir, f"fold_{fold+1}")
        os.makedirs(fold_dir, exist_ok=True)
        
        # 保存本折詳細評估指標
        detailed_metrics = evaluate_model(
            test_targets, test_predictions, 
            model_name=f"Fold {fold+1}", 
            verbose=False, return_dataframe=True
        )
        detailed_metrics.to_csv(os.path.join(fold_dir, "metrics.csv"), index=False)
        
        # 產生本折預測對比圖
        fig = plot_prediction_vs_true(
            test_targets, test_predictions, 
            model_name=f"Fold {fold+1}",
            save_path=os.path.join(fold_dir, "prediction_vs_true.png")
        )
        plt.close(fig)
        
        logger.info(f"完成第 {fold+1} 折評估 - RMSE: {test_metrics['rmse']:.6f}, R²: {test_metrics['r2_score']:.6f}")
    
    # 計算平均指標
    cv_results["mean_rmse"] = np.mean(cv_results["rmse"])
    cv_results["std_rmse"] = np.std(cv_results["rmse"])
    cv_results["mean_r2"] = np.mean(cv_results["r2"])
    cv_results["std_r2"] = np.std(cv_results["r2"])
    cv_results["mean_mae"] = np.mean(cv_results["mae"])
    cv_results["std_mae"] = np.std(cv_results["mae"])
    cv_results["mean_mape"] = np.mean(cv_results["mape"])
    cv_results["std_mape"] = np.std(cv_results["mape"])
    cv_results["mean_rel_error"] = np.mean(cv_results["rel_error_mean"])
    cv_results["std_rel_error"] = np.std(cv_results["rel_error_mean"])
    
    # 保存交叉驗證結果
    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv(os.path.join(cv_dir, "cv_results.csv"), index=False)
    
    # 記錄交叉驗證摘要
    logger.info("\n交叉驗證摘要:")
    logger.info(f"平均 RMSE: {cv_results['mean_rmse']:.6f} ± {cv_results['std_rmse']:.6f}")
    logger.info(f"平均 R²: {cv_results['mean_r2']:.6f} ± {cv_results['std_r2']:.6f}")
    logger.info(f"平均 MAE: {cv_results['mean_mae']:.6f} ± {cv_results['std_mae']:.6f}")
    logger.info(f"平均 相對誤差: {cv_results['mean_rel_error']:.2f}% ± {cv_results['std_rel_error']:.2f}%")
    
    # 產生交叉驗證摘要圖
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    
    # RMSE圖
    ax[0, 0].bar(range(1, n_splits + 1), cv_results["rmse"])
    ax[0, 0].axhline(y=cv_results["mean_rmse"], color='r', linestyle='-')
    ax[0, 0].set_xlabel('Fold')
    ax[0, 0].set_ylabel('RMSE')
    ax[0, 0].set_title('RMSE across folds')
    ax[0, 0].grid(True)
    
    # R²圖
    ax[0, 1].bar(range(1, n_splits + 1), cv_results["r2"])
    ax[0, 1].axhline(y=cv_results["mean_r2"], color='r', linestyle='-')
    ax[0, 1].set_xlabel('Fold')
    ax[0, 1].set_ylabel('R²')
    ax[0, 1].set_title('R² across folds')
    ax[0, 1].grid(True)
    
    # MAE圖
    ax[1, 0].bar(range(1, n_splits + 1), cv_results["mae"])
    ax[1, 0].axhline(y=cv_results["mean_mae"], color='r', linestyle='-')
    ax[1, 0].set_xlabel('Fold')
    ax[1, 0].set_ylabel('MAE')
    ax[1, 0].set_title('MAE across folds')
    ax[1, 0].grid(True)
    
    # 相對誤差圖
    ax[1, 1].bar(range(1, n_splits + 1), cv_results["rel_error_mean"])
    ax[1, 1].axhline(y=cv_results["mean_rel_error"], color='r', linestyle='-')
    ax[1, 1].set_xlabel('Fold')
    ax[1, 1].set_ylabel('Mean Relative Error (%)')
    ax[1, 1].set_title('Relative Error across folds')
    ax[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(cv_dir, "cv_summary.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 合併所有預測和目標
    all_predictions_flat = np.concatenate(all_predictions)
    all_targets_flat = np.concatenate(all_targets)
    
    # 產生整體預測對比圖
    fig = plot_prediction_vs_true(
        all_targets_flat, all_predictions_flat, 
        model_name="Cross Validation",
        save_path=os.path.join(cv_dir, "overall_prediction_vs_true.png")
    )
    plt.close(fig)
    
    return {
        "cv_results": cv_results,
        "all_predictions": all_predictions_flat,
        "all_targets": all_targets_flat
    }


def analyze_feature_importance(model, data_dict, config, device, output_dir, batch_size=16, model_type="hybrid"):
    """
    特徵重要性分析
    
    參數:
        model (torch.nn.Module): 模型
        data_dict (dict): 資料字典
        config (dict): 配置
        device (torch.device): 計算設備
        output_dir (str): 輸出目錄
        batch_size (int): 批次大小
        
    返回:
        dict: 特徵重要性分析結果
    """
    # 創建特徵重要性目錄
    importance_dir = os.path.join(output_dir, "feature_importance")
    os.makedirs(importance_dir, exist_ok=True)
    
    # 獲取靜態特徵名稱
    feature_names = config["model"]["input"]["static_features"]
    
    # 創建評估函數
    def evaluate_model_with_features(X, time_series, y):
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            time_series_tensor = torch.FloatTensor(time_series).to(device)
            
            if isinstance(model, HybridPINNLSTMModel):
                outputs = model(X_tensor, time_series_tensor)
                predictions = outputs["nf_pred"].cpu().numpy()
            elif isinstance(model, PINNModel):
                outputs = model(X_tensor)
                predictions = outputs["nf_pred"].cpu().numpy()
            elif isinstance(model, LSTMModel):
                outputs = model(time_series_tensor)
                predictions = outputs["output"].cpu().numpy()
            else:
                raise ValueError(f"不支援的模型類型: {type(model)}")
            
            # 計算RMSE
            rmse = calculate_rmse(y, predictions)
            r2 = calculate_r2(y, predictions)
            
            return rmse, r2, predictions
    
    # 獲取測試資料
    X_test = data_dict["X_test"]
    time_series_test = data_dict["time_series_test"]
    y_test = data_dict["y_test"]
    
    # 1. 留一法分析 (每次移除一個特徵)
    logger.info("進行特徵重要性分析 - 留一法...")
    leave_one_out_results = {
        "feature": [],
        "rmse": [],
        "r2": [],
        "rmse_change": [],
        "r2_change": []
    }
    
    # 首先評估完整模型
    baseline_rmse, baseline_r2, baseline_pred = evaluate_model_with_features(
        X_test, time_series_test, y_test
    )
    
    # 對每個特徵進行留一法分析
    for i, feature in enumerate(feature_names):
        # 創建移除一個特徵的資料
        X_test_modified = X_test.copy()
        # 將特徵設為零以模擬移除
        X_test_modified[:, i] = 0
        
        # 評估修改後的資料
        mod_rmse, mod_r2, _ = evaluate_model_with_features(
            X_test_modified, time_series_test, y_test
        )
        
        # 計算指標變化
        rmse_change = ((mod_rmse - baseline_rmse) / baseline_rmse) * 100
        r2_change = ((baseline_r2 - mod_r2) / baseline_r2) * 100 if baseline_r2 > 0 else 0
        
        # 記錄結果
        leave_one_out_results["feature"].append(feature)
        leave_one_out_results["rmse"].append(mod_rmse)
        leave_one_out_results["r2"].append(mod_r2)
        leave_one_out_results["rmse_change"].append(rmse_change)
        leave_one_out_results["r2_change"].append(r2_change)
    
    # 保存留一法分析結果
    loo_df = pd.DataFrame(leave_one_out_results)
    loo_df.to_csv(os.path.join(importance_dir, "leave_one_out_importance.csv"), index=False)
    
    # 繪製特徵重要性條形圖
    fig = plot_feature_importance(
        feature_names, leave_one_out_results["rmse_change"], 
        model_name="留一法分析", 
        save_path=os.path.join(importance_dir, "leave_one_out_importance.png")
    )
    plt.close(fig)
    
    # 2. 排列重要性分析
    logger.info("進行特徵重要性分析 - 排列重要性...")
    
    # 定義預測函數
    def predict_func(X):
        X_tensor = torch.FloatTensor(X).to(device)
        time_series_tensor = torch.FloatTensor(np.repeat(time_series_test, len(X) // len(time_series_test) + 1, axis=0)[:len(X)]).to(device)
        
        with torch.no_grad():
            if isinstance(model, HybridPINNLSTMModel):
                outputs = model(X_tensor, time_series_tensor)
                predictions = outputs["nf_pred"].cpu().numpy()
            elif isinstance(model, PINNModel):
                outputs = model(X_tensor)
                predictions = outputs["nf_pred"].cpu().numpy()
            else:
                # 對於其他模型類型，使用默認策略
                outputs = model(X_tensor, time_series_tensor)
                predictions = outputs["nf_pred"].cpu().numpy() if "nf_pred" in outputs else outputs.cpu().numpy()
                
        return predictions
    
    try:
        # 計算排列重要性
        perm_importance = permutation_importance(
            predict_func, X_test, y_test, 
            n_repeats=10, random_state=42, n_jobs=-1
        )
        
        # 整理排列重要性結果
        perm_importance_results = {
            "feature": feature_names,
            "importance_mean": perm_importance.importances_mean,
            "importance_std": perm_importance.importances_std
        }
        
        # 保存排列重要性結果
        perm_df = pd.DataFrame(perm_importance_results)
        perm_df.to_csv(os.path.join(importance_dir, "permutation_importance.csv"), index=False)
        
        # 繪製排列重要性條形圖
        fig = plot_feature_importance(
            feature_names, perm_importance.importances_mean, 
            model_name="排列重要性分析", 
            save_path=os.path.join(importance_dir, "permutation_importance.png")
        )
        plt.close(fig)
    except Exception as e:
        logger.warning(f"排列重要性分析失敗: {str(e)}")
    
    # 3. 參數影響分析
    logger.info("進行參數影響分析...")
    
    # 為每個特徵產生影響曲線
    for i, feature in enumerate(feature_names):
        try:
            # 獲取特徵範圍
            feature_min = np.min(data_dict["X_train"][:, i])
            feature_max = np.max(data_dict["X_train"][:, i])
            
            # 創建一組變化的特徵值
            feature_range = np.linspace(feature_min, feature_max, 20)
            
            # 初始化結果收集
            impact_results = {
                "feature_value": feature_range,
                "prediction": []
            }
            
            # 對每個特徵值進行預測
            for value in feature_range:
                # 創建修改後的資料
                X_modified = np.tile(np.mean(X_test, axis=0), (1, 1))
                X_modified[0, i] = value
                
                # 預測
                predictions = predict_func(X_modified)
                impact_results["prediction"].append(predictions[0])
            
            # 保存參數影響結果
            impact_df = pd.DataFrame(impact_results)
            impact_df.to_csv(os.path.join(importance_dir, f"{feature}_impact.csv"), index=False)
            
            # 繪製參數影響曲線
            fig = plot_parameter_impact(
                impact_results["feature_value"], 
                impact_results["prediction"], 
                parameter_name=feature,
                save_path=os.path.join(importance_dir, f"{feature}_impact.png")
            )
            plt.close(fig)
        except Exception as e:
            logger.warning(f"參數 {feature} 影響分析失敗: {str(e)}")
    
    return {
        "leave_one_out": leave_one_out_results,
        "permutation_importance": perm_importance_results if 'perm_importance_results' in locals() else None
    }