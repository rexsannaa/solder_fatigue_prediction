#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
train.py - 模型訓練腳本
本腳本用於訓練銲錫接點疲勞壽命預測的混合PINN-LSTM模型。
它整合了資料預處理、模型構建、訓練和評估的完整流程。

主要功能:
1. 載入並解析配置文件
2. 準備訓練、驗證和測試資料
3. 構建混合PINN-LSTM模型
4. 訓練模型並記錄結果
5. 評估模型性能
6. 保存模型和視覺化結果
"""

import os
import sys
import argparse
import yaml
import logging
import time
import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# 確保可以導入專案模組
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# 導入專案模組
from src.data.preprocess import process_pipeline
from src.data.dataset import create_dataloaders
from src.models.hybrid_model import HybridPINNLSTMModel, PINNLSTMTrainer
from src.models.pinn import PINNModel
from src.models.lstm import LSTMModel
from src.training.losses import get_loss_function, AdaptiveHybridLoss
from src.training.trainer import Trainer, EarlyStopping, LearningRateScheduler
from src.training.callbacks import create_default_callbacks, AdaptiveCallbacks
from src.utils.metrics import evaluate_model, compare_models
from src.utils.visualization import plot_prediction_vs_true, visualize_model_results
from src.utils.physics import validate_physical_constraints

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, "logs", f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
    ]
)
logger = logging.getLogger(__name__)

# 確保日誌目錄存在
os.makedirs(os.path.join(project_root, "logs"), exist_ok=True)


def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description="訓練銲錫接點疲勞壽命預測模型")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml",
                        help="模型配置文件路徑")
    parser.add_argument("--train-config", type=str, default="configs/train_config.yaml",
                        help="訓練配置文件路徑")
    parser.add_argument("--data", type=str, 
                        help="資料文件路徑，覆蓋配置文件中的設定")
    parser.add_argument("--output-dir", type=str,
                        help="輸出目錄，覆蓋配置文件中的設定")
    parser.add_argument("--epochs", type=int,
                        help="訓練輪數，覆蓋配置文件中的設定")
    parser.add_argument("--batch-size", type=int,
                        help="批次大小，覆蓋配置文件中的設定")
    parser.add_argument("--lr", type=float,
                        help="學習率，覆蓋配置文件中的設定")
    parser.add_argument("--seed", type=int,
                        help="隨機種子，覆蓋配置文件中的設定")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"],
                        help="計算設備，覆蓋配置文件中的設定")
    parser.add_argument("--model-type", type=str, choices=["hybrid", "pinn", "lstm"],
                        default="hybrid", help="模型類型: hybrid, pinn, lstm")
    parser.add_argument("--no-physics", action="store_true",
                        help="禁用物理約束")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="僅評估模型，不進行訓練")
    parser.add_argument("--model-path", type=str,
                        help="要評估的模型路徑，與--evaluate-only一起使用")
    
    return parser.parse_args()


def load_config(config_path):
    """載入YAML配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def set_random_seed(seed):
    """設定隨機種子以確保結果可重現"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_model(config, model_type="hybrid", use_physics=True):
    """
    創建模型
    
    參數:
        config (dict): 模型配置
        model_type (str): 模型類型: hybrid, pinn, lstm
        use_physics (bool): 是否使用物理約束
    
    返回:
        torch.nn.Module: 創建的模型
    """
    if model_type == "hybrid":
        # 創建混合PINN-LSTM模型
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
            use_physics_layer=use_physics and config["model"]["pinn"]["use_physics_layer"]
        )
    elif model_type == "pinn":
        # 只創建PINN模型
        model = PINNModel(
            input_dim=len(config["model"]["input"]["static_features"]),
            hidden_dims=config["model"]["pinn"]["hidden_layers"],
            dropout_rate=config["model"]["pinn"]["dropout_rate"],
            use_physics_layer=use_physics and config["model"]["pinn"]["use_physics_layer"]
        )
    elif model_type == "lstm":
        # 只創建LSTM模型
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
    
    return model


def create_optimizer(model, config):
    """
    創建優化器
    
    參數:
        model (torch.nn.Module): 模型
        config (dict): 配置
    
    返回:
        torch.optim.Optimizer: 優化器
    """
    optimizer_config = config["training"]["optimizer"]
    lr = optimizer_config["learning_rate"]
    weight_decay = optimizer_config["weight_decay"]
    
    if optimizer_config["name"].lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_config["name"].lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_config["name"].lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        logger.warning(f"不支援的優化器類型: {optimizer_config['name']}，使用Adam")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    return optimizer


def create_loss_function(config, use_physics=True):
    """
    創建損失函數
    
    參數:
        config (dict): 配置
        use_physics (bool): 是否使用物理約束
    
    返回:
        torch.nn.Module: 損失函數
    """
    loss_config = config["training"]["loss"]
    loss_type = loss_config["type"]
    
    # 獲取物理模型參數
    a = config["model"]["physics"]["a_coefficient"]
    b = config["model"]["physics"]["b_coefficient"]
    
    # 創建損失函數
    if loss_type == "adaptive":
        loss_fn = AdaptiveHybridLoss(
            initial_lambda_physics=loss_config["initial_lambda_physics"] if use_physics else 0.0,
            max_lambda_physics=loss_config["max_lambda_physics"] if use_physics else 0.0,
            initial_lambda_consistency=loss_config["initial_lambda_consistency"],
            max_lambda_consistency=loss_config["max_lambda_consistency"],
            epochs_to_max=loss_config["epochs_to_max"],
            a=a,
            b=b
        )
    else:
        # 使用一般混合損失或其他損失
        lambda_physics = 0.1 if use_physics else 0.0
        loss_fn = get_loss_function(loss_type, lambda_physics=lambda_physics, 
                                  lambda_consistency=0.1, a=a, b=b)
    
    return loss_fn


def create_lr_scheduler(optimizer, config):
    """
    創建學習率調度器
    
    參數:
        optimizer (torch.optim.Optimizer): 優化器
        config (dict): 配置
    
    返回:
        LearningRateScheduler: 學習率調度器
    """
    scheduler_config = config["training"]["lr_scheduler"]
    
    if not scheduler_config:
        return None
    
    scheduler_type = scheduler_config["name"]
    
    if scheduler_type == "step":
        step_size = scheduler_config.get("step_size", 10)
        gamma = scheduler_config.get("gamma", 0.1)
        scheduler = LearningRateScheduler(optimizer, mode="step", step_size=step_size, gamma=gamma)
    elif scheduler_type == "exp":
        gamma = scheduler_config.get("gamma", 0.95)
        scheduler = LearningRateScheduler(optimizer, mode="exp", gamma=gamma)
    elif scheduler_type == "cosine":
        T_max = scheduler_config.get("T_max", 100)
        min_lr = scheduler_config.get("min_lr", 0)
        scheduler = LearningRateScheduler(optimizer, mode="cosine", T_max=T_max, min_lr=min_lr)
    elif scheduler_type == "plateau":
        factor = scheduler_config.get("factor", 0.1)
        patience = scheduler_config.get("patience", 10)
        min_lr = scheduler_config.get("min_lr", 0)
        scheduler = LearningRateScheduler(optimizer, mode="plateau", factor=factor, 
                                       patience=patience, min_lr=min_lr)
    else:
        logger.warning(f"不支援的學習率調度器類型: {scheduler_type}，不使用學習率調度")
        scheduler = None
    
    return scheduler


def train_model(model, dataloaders, config, device, output_dir, use_physics=True):
    """
    訓練模型
    
    參數:
        model (torch.nn.Module): 模型
        dataloaders (dict): 資料載入器
        config (dict): 配置
        device (torch.device): 計算設備
        output_dir (str): 輸出目錄
        use_physics (bool): 是否使用物理約束
    
    返回:
        dict: 訓練歷史記錄
    """
    # 創建優化器
    optimizer = create_optimizer(model, config)
    
    # 創建損失函數
    loss_fn = create_loss_function(config, use_physics)
    
    # 創建學習率調度器
    lr_scheduler = create_lr_scheduler(optimizer, config)
    
    # 創建回調函數
    callbacks = create_default_callbacks(
        model_name=config["model"]["name"],
        output_dir=output_dir,
        epochs=config["training"]["epochs"]
    )
    
    # 創建早停機制
    early_stopping_config = config["training"]["early_stopping"]
    early_stopping = EarlyStopping(
        patience=early_stopping_config["patience"],
        min_delta=early_stopping_config["min_delta"],
        mode=early_stopping_config["mode"],
        verbose=True,
        save_path=os.path.join(output_dir, "models", "best_model.pt")
    )
    
    # 設定梯度裁剪範數
    clip_grad_norm = config["training"]["clip_grad_norm"]
    
    # 創建訓練器
    trainer = Trainer(model, optimizer, loss_fn, device=device, lr_scheduler=lr_scheduler)
    trainer.clip_grad_norm = clip_grad_norm
    
    # 開始訓練
    logger.info(f"開始訓練 - 模型: {config['model']['name']}, 輪次: {config['training']['epochs']}")
    start_time = time.time()
    
    history = trainer.train(
        train_loader=dataloaders["train_loader"],
        val_loader=dataloaders["val_loader"],
        epochs=config["training"]["epochs"],
        early_stopping=early_stopping,
        verbose=True,
        eval_interval=1,
        save_path=os.path.join(output_dir, "models", "last_model.pt"),
        callbacks=callbacks
    )
    
    train_time = time.time() - start_time
    logger.info(f"訓練完成 - 耗時: {train_time:.2f}秒, 最佳驗證損失: {history['best_val_loss']:.6f}")
    
    # 繪製訓練曲線
    trainer.plot_losses(figsize=(12, 6), save_path=os.path.join(output_dir, "plots", "training_loss.png"))
    
    # 如果有指標歷史記錄，繪製指標曲線
    for metric_name in trainer.metrics_history:
        trainer.plot_metrics(metric_name, figsize=(10, 6), 
                            save_path=os.path.join(output_dir, "plots", f"{metric_name}_history.png"))
    
    return history


def evaluate_and_visualize(model, dataloaders, output_dir, device, config):
    """
    評估模型並產生視覺化結果
    
    參數:
        model (torch.nn.Module): 模型
        dataloaders (dict): 資料載入器
        output_dir (str): 輸出目錄
        device (torch.device): 計算設備
        config (dict): 配置
    """
    # 創建評估目錄
    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    # 將模型移至評估模式
    model.eval()
    
    # 獲取測試資料載入器
    test_loader = dataloaders["test_loader"]
    
    # 創建訓練器 (僅用於評估)
    trainer = Trainer(model, None, None, device=device)
    
    # 進行評估
    logger.info("評估模型性能...")
    _, metrics, predictions, targets = trainer.evaluate(test_loader, return_predictions=True)
    
    # 輸出評估指標
    logger.info("測試集評估指標:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.6f}")
    
    # 保存評估指標
    with open(os.path.join(eval_dir, "metrics.txt"), "w") as f:
        for metric_name, metric_value in metrics.items():
            f.write(f"{metric_name}: {metric_value:.6f}\n")
    
    # 繪製預測對比圖
    logger.info("生成預測對比圖...")
    fig = plot_prediction_vs_true(
        targets, predictions, 
        model_name=config["model"]["name"],
        save_path=os.path.join(eval_dir, "prediction_vs_true.png")
    )
    plt.close(fig)
    
    # 使用模型進行預測並獲取所有輸出
    logger.info("使用模型進行詳細預測...")
    results = trainer.predict(test_loader)
    
    # 產生視覺化結果
    logger.info("生成視覺化結果...")
    vis_dir = os.path.join(eval_dir, "visualizations")
    visualization_paths = visualize_model_results(results, output_dir=vis_dir)
    
    # 驗證物理約束條件
    if "delta_w" in results and results["delta_w"] is not None:
        logger.info("驗證物理約束條件...")
        a = config["model"]["physics"]["a_coefficient"]
        b = config["model"]["physics"]["b_coefficient"]
        
        passed, residuals, violated = validate_physical_constraints(
            results["delta_w"], results["predictions"], 
            a=a, b=b, threshold=20.0, verbose=True
        )
        
        if not passed:
            logger.warning(f"物理約束驗證失敗, 有 {len(violated)} 個樣本違反約束")
        else:
            logger.info("所有預測結果都滿足物理約束條件")
    
    logger.info(f"評估和視覺化完成，結果保存在: {eval_dir}")


def main():
    """主函數"""
    # 解析命令行參數
    args = parse_args()
    
    # 載入配置文件
    logger.info(f"載入配置文件: {args.config}")
    config = load_config(args.config)
    
    # 載入訓練配置文件
    logger.info(f"載入訓練配置文件: {args.train_config}")
    train_config = load_config(args.train_config)
    
    # 更新配置（根據命令行參數覆蓋）
    if args.data:
        config["paths"]["data"] = args.data
    
    if args.output_dir:
        config["paths"]["output_dir"] = args.output_dir
    
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    
    if args.lr:
        config["training"]["optimizer"]["learning_rate"] = args.lr
    
    if args.seed:
        config["random_seed"] = args.seed
    
    if args.device:
        config["device"] = args.device
    
    # 設定隨機種子
    set_random_seed(config["random_seed"])
    
    # 設定計算設備
    device = torch.device(config["device"] if torch.cuda.is_available() and config["device"] == "cuda" else "cpu")
    logger.info(f"使用計算設備: {device}")
    
    # 創建輸出目錄
    output_dir = config["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    
    # 保存配置副本
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(output_dir, f"config_{timestamp}.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    with open(os.path.join(output_dir, f"train_config_{timestamp}.yaml"), "w") as f:
        yaml.dump(train_config, f, default_flow_style=False)
    
    # 載入並預處理資料
    logger.info(f"載入資料: {config['paths']['data']}")
    data_dict = process_pipeline(
        config["paths"]["data"],
        test_size=config["training"]["data_split"]["test_size"],
        val_size=config["training"]["data_split"]["val_size"],
        random_state=config["training"]["data_split"]["random_seed"]
    )
    
    # 創建資料載入器
    batch_size = config["training"]["batch_size"]
    logger.info(f"創建資料載入器，批次大小: {batch_size}")
    dataloaders = create_dataloaders(
        data_dict["X_train"], data_dict["X_val"], data_dict["X_test"],
        data_dict["time_series_train"], data_dict["time_series_val"], data_dict["time_series_test"],
        data_dict["y_train"], data_dict["y_val"], data_dict["y_test"],
        batch_size=batch_size,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"]
    )
    
    # 記錄資料集大小
    logger.info(f"資料集大小 - 訓練: {len(data_dict['y_train'])}, "
               f"驗證: {len(data_dict['y_val'])}, 測試: {len(data_dict['y_test'])}")
    
    # 如果只是評估模型
    if args.evaluate_only:
        if args.model_path:
            logger.info(f"載入模型進行評估: {args.model_path}")
            model_path = args.model_path
            
            # 創建模型
            model = create_model(config, model_type=args.model_type, use_physics=not args.no_physics)
            
            # 載入模型權重
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            
            # 評估模型
            evaluate_and_visualize(model, dataloaders, output_dir, device, config)
        else:
            logger.error("評估模式需要指定模型路徑 (--model-path)")
        
        return
    
    # 創建模型
    logger.info(f"創建模型: {args.model_type}")
    model = create_model(config, model_type=args.model_type, use_physics=not args.no_physics)
    model.to(device)
    
    # 輸出模型資訊
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型參數總數: {total_params:,}")
    
    # 訓練模型
    history = train_model(model, dataloaders, config, device, output_dir, use_physics=not args.no_physics)
    
    # 評估模型
    logger.info("訓練完成，評估模型性能...")
    evaluate_and_visualize(model, dataloaders, output_dir, device, config)
    
    # 比較不同分支的性能
    if args.model_type == "hybrid":
        logger.info("比較不同分支的預測性能...")
        
        # 載入最佳模型
        best_model_path = os.path.join(output_dir, "models", "best_model.pt")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
        
        # 創建訓練器
        trainer = Trainer(model, None, None, device=device)
        
        # 進行預測並獲取各分支結果
        results = trainer.predict(dataloaders["test_loader"])
        
        # 評估各分支性能
        if "targets" in results and results["targets"] is not None:
            targets = results["targets"]
            
            # 混合模型預測
            hybrid_pred = results["predictions"]
            hybrid_metrics = evaluate_model(targets, hybrid_pred, model_name="Hybrid", verbose=True)
            
            # PINN分支預測
            if "pinn_nf_pred" in results:
                pinn_pred = results["pinn_nf_pred"]
                pinn_metrics = evaluate_model(targets, pinn_pred, model_name="PINN", verbose=True)
            
            # LSTM分支預測
            if "lstm_nf_pred" in results:
                lstm_pred = results["lstm_nf_pred"]
                lstm_metrics = evaluate_model(targets, lstm_pred, model_name="LSTM", verbose=True)
            
            # 比較並保存結果
            branch_metrics = [hybrid_metrics]
            if "pinn_nf_pred" in results:
                branch_metrics.append(pinn_metrics)
            if "lstm_nf_pred" in results:
                branch_metrics.append(lstm_metrics)
            
            # 輸出比較結果
            comparison = compare_models(branch_metrics, sort_by="rmse", ascending=True)
            logger.info("\n分支性能比較:")
            logger.info(comparison)
            
            # 保存比較結果
            comparison.to_csv(os.path.join(output_dir, "evaluation", "branch_comparison.csv"), index=False)
    
    logger.info(f"所有任務完成，結果保存在: {output_dir}")


if __name__ == "__main__":
    # 確保輸出目錄存在
    os.makedirs(os.path.join(project_root, "outputs", "models"), exist_ok=True)
    os.makedirs(os.path.join(project_root, "outputs", "plots"), exist_ok=True)
    os.makedirs(os.path.join(project_root, "logs"), exist_ok=True)
    
    try:
        main()
    except Exception as e:
        logger.exception(f"執行過程中發生錯誤: {str(e)}")
        sys.exit(1)