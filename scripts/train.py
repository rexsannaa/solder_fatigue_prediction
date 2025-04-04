#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
improved_train.py - 優化的模型訓練腳本
本腳本用於訓練銲錫接點疲勞壽命預測的混合PINN-LSTM模型，
整合了改進的預處理、物理知識驅動的資料增強和分階段訓練策略。

主要改進:
1. 整合物理知識驅動的資料增強
2. 支援分階段訓練策略，優化微調流程
3. 實現自適應損失權重調整
4. 加強模型評估與物理約束驗證
5. 針對小樣本數據集優化訓練過程
"""

import os
import sys
import argparse
import yaml
import logging
import time
import datetime
import numpy as np
import pandas as pd
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
from src.training.callbacks import AdaptiveCallbacks
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
    parser = argparse.ArgumentParser(description="銲錫接點疲勞壽命預測模型訓練")
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
    parser.add_argument("--no-augmentation", action="store_true",
                        help="禁用資料增強")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="僅評估模型，不進行訓練")
    parser.add_argument("--model-path", type=str,
                        help="要評估的模型路徑，與--evaluate-only一起使用")
    parser.add_argument("--stage", type=str, choices=["all", "warmup", "main", "finetune"],
                        default="all", help="訓練階段: all, warmup, main, finetune")
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


def set_random_seed(seed):
    """設定隨機種子以確保結果可重現"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"已設定隨機種子: {seed}")


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
    # 獲取物理模型參數
    a_coefficient = config["model"]["physics"]["a_coefficient"]
    b_coefficient = config["model"]["physics"]["b_coefficient"]
    
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
            use_physics_layer=use_physics and config["model"]["pinn"]["use_physics_layer"],
            physics_layer_trainable=config["model"]["pinn"].get("physics_layer_trainable", False),
            use_batch_norm=config["model"]["pinn"].get("use_batch_norm", True),
            pinn_weight_init=config["model"]["fusion"].get("pinn_weight_init", 0.7),
            lstm_weight_init=config["model"]["fusion"].get("lstm_weight_init", 0.3),
            a_coefficient=a_coefficient,
            b_coefficient=b_coefficient,
            use_log_transform=config["model"].get("use_log_transform", True),
            ensemble_method=config["model"]["fusion"].get("ensemble_method", "weighted"),
            l2_reg=config["training"].get("l2_reg", 0.001)
        )
        logger.info(f"創建混合PINN-LSTM模型，使用物理約束: {use_physics}，融合方法: {config['model']['fusion'].get('ensemble_method', 'weighted')}")
    elif model_type == "pinn":
        # 只創建PINN模型
        model = PINNModel(
            input_dim=len(config["model"]["input"]["static_features"]),
            hidden_dims=config["model"]["pinn"]["hidden_layers"],
            dropout_rate=config["model"]["pinn"]["dropout_rate"],
            use_physics_layer=use_physics and config["model"]["pinn"]["use_physics_layer"],
            physics_layer_trainable=config["model"]["pinn"].get("physics_layer_trainable", False),
            use_batch_norm=config["model"]["pinn"].get("use_batch_norm", True),
            activation=config["model"]["pinn"].get("activation", "relu"),
            a_coefficient=a_coefficient,
            b_coefficient=b_coefficient,
            l2_reg=config["training"].get("l2_reg", 0.001)
        )
        logger.info(f"創建PINN模型，使用物理約束: {use_physics}")
    elif model_type == "lstm":
        # 只創建LSTM模型
        model = LSTMModel(
            input_dim=len(config["model"]["input"]["time_series_features"]),
            hidden_size=config["model"]["lstm"]["hidden_size"],
            num_layers=config["model"]["lstm"]["num_layers"],
            bidirectional=config["model"]["lstm"]["bidirectional"],
            dropout_rate=config["model"]["lstm"]["dropout_rate"],
            use_attention=config["model"]["lstm"]["use_attention"],
            l2_reg=config["training"].get("l2_reg", 0.001)
        )
        logger.info(f"創建LSTM模型，使用注意力機制: {config['model']['lstm']['use_attention']}")
    else:
        raise ValueError(f"不支援的模型類型: {model_type}")
    
    return model


def prepare_training_config(config, train_config, stage="all", start_epoch=0):
    """
    根據訓練階段準備訓練配置
    
    參數:
        config (dict): 模型配置
        train_config (dict): 訓練配置
        stage (str): 訓練階段: all, warmup, main, finetune
        start_epoch (int): 起始輪次
        
    返回:
        dict: 訓練配置
    """
    # 獲取分階段訓練配置
    training_strategy = train_config.get("training_strategy", {})
    stages = training_strategy.get("stages", [])
    
    # 如果沒有分階段配置或選擇了'all'，使用默認配置
    if not stages or stage == "all":
        return {
            "epochs": config["training"]["epochs"],
            "learning_rate": config["training"]["optimizer"]["learning_rate"],
            "lambda_physics": config["training"]["loss"].get("initial_lambda_physics", 0.1),
            "lambda_consistency": config["training"]["loss"].get("initial_lambda_consistency", 0.1),
            "batch_size": config["training"]["batch_size"],
            "clip_grad_norm": config["training"].get("clip_grad_norm", 1.0),
            "description": "完整訓練"
        }
    
    # 根據指定階段獲取配置
    for stage_config in stages:
        if stage_config["name"] == stage:
            # 計算學習率
            base_lr = config["training"]["optimizer"]["learning_rate"]
            lr_factor = stage_config.get("learning_rate_factor", 1.0)
            
            # 獲取物理約束和一致性權重
            if stage == "warmup":
                lambda_physics = stage_config.get("lambda_physics", 0.01)
                lambda_consistency = stage_config.get("lambda_consistency", 0.01)
            elif stage == "main":
                lambda_physics = stage_config.get("lambda_physics_start", 0.05)
                lambda_consistency = stage_config.get("lambda_consistency_start", 0.05)
            elif stage == "finetune":
                lambda_physics = stage_config.get("lambda_physics", 0.5)
                lambda_consistency = stage_config.get("lambda_consistency", 0.3)
            
            return {
                "epochs": stage_config["epochs"],
                "learning_rate": base_lr * lr_factor,
                "lambda_physics": lambda_physics,
                "lambda_consistency": lambda_consistency,
                "batch_size": config["training"]["batch_size"],
                "clip_grad_norm": config["training"].get("clip_grad_norm", 1.0),
                "description": stage_config.get("description", f"{stage}階段訓練")
            }
    
    # 如果找不到指定階段，使用默認配置
    logger.warning(f"找不到指定訓練階段: {stage}，使用默認配置")
    return {
        "epochs": config["training"]["epochs"],
        "learning_rate": config["training"]["optimizer"]["learning_rate"],
        "lambda_physics": config["training"]["loss"].get("initial_lambda_physics", 0.1),
        "lambda_consistency": config["training"]["loss"].get("initial_lambda_consistency", 0.1),
        "batch_size": config["training"]["batch_size"],
        "clip_grad_norm": config["training"].get("clip_grad_norm", 1.0),
        "description": "默認訓練"
    }


def create_optimizer(model, config, stage_config=None):
    """
    創建優化器
    
    參數:
        model (torch.nn.Module): 模型
        config (dict): 配置
        stage_config (dict, optional): 階段配置
        
    返回:
        torch.optim.Optimizer: 優化器
    """
    optimizer_config = config["training"]["optimizer"]
    
    # 使用階段配置中的學習率（如果有）
    if stage_config and "learning_rate" in stage_config:
        lr = stage_config["learning_rate"]
    else:
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
    
    logger.info(f"創建{optimizer_config['name']}優化器，學習率: {lr}，權重衰減: {weight_decay}")
    return optimizer


def create_lr_scheduler(optimizer, config, stage_config=None, total_epochs=None):
    """
    創建學習率調度器
    
    參數:
        optimizer (torch.optim.Optimizer): 優化器
        config (dict): 配置
        stage_config (dict, optional): 階段配置
        total_epochs (int, optional): 總訓練輪數
        
    返回:
        LearningRateScheduler: 學習率調度器
    """
    scheduler_config = config["training"]["lr_scheduler"]
    
    if not scheduler_config:
        return None
    
    # 使用階段特定的輪數（如果有）
    if stage_config and "epochs" in stage_config:
        epochs = stage_config["epochs"]
    elif total_epochs:
        epochs = total_epochs
    else:
        epochs = config["training"]["epochs"]
    
    scheduler_type = scheduler_config["name"]
    
    if scheduler_type == "step":
        step_size = scheduler_config.get("step_size", 10)
        gamma = scheduler_config.get("gamma", 0.1)
        scheduler = LearningRateScheduler(optimizer, mode="step", step_size=step_size, gamma=gamma)
        logger.info(f"創建步進式學習率調度器，步長: {step_size}，衰減因子: {gamma}")
    elif scheduler_type == "exp":
        gamma = scheduler_config.get("gamma", 0.95)
        scheduler = LearningRateScheduler(optimizer, mode="exp", gamma=gamma)
        logger.info(f"創建指數式學習率調度器，衰減因子: {gamma}")
    elif scheduler_type == "cosine":
        T_max = scheduler_config.get("T_max", epochs)
        min_lr = scheduler_config.get("min_lr", 0)
        scheduler = LearningRateScheduler(optimizer, mode="cosine", T_max=T_max, min_lr=min_lr)
        logger.info(f"創建餘弦退火學習率調度器，週期: {T_max}，最小學習率: {min_lr}")
    elif scheduler_type == "plateau":
        factor = scheduler_config.get("factor", 0.1)
        patience = scheduler_config.get("patience", 10)
        min_lr = scheduler_config.get("min_lr", 0)
        scheduler = LearningRateScheduler(optimizer, mode="plateau", factor=factor, 
                                       patience=patience, min_lr=min_lr)
        logger.info(f"創建平原式學習率調度器，衰減因子: {factor}，耐心值: {patience}，最小學習率: {min_lr}")
    else:
        logger.warning(f"不支援的學習率調度器類型: {scheduler_type}，不使用學習率調度")
        scheduler = None
    
    return scheduler


def get_pinn_lstm_trainer(model, optimizer, config, device, 
                         lambda_physics=0.1, lambda_consistency=0.1, 
                         clip_grad_norm=1.0, scheduler=None):
    """
    創建PINN-LSTM訓練器
    
    參數:
        model (HybridPINNLSTMModel): 混合模型
        optimizer (torch.optim.Optimizer): 優化器
        config (dict): 配置
        device (torch.device): 計算設備
        lambda_physics (float): 物理約束損失權重
        lambda_consistency (float): 一致性損失權重
        clip_grad_norm (float): 梯度裁剪範數
        scheduler (LearningRateScheduler): 學習率調度器
        
    返回:
        PINNLSTMTrainer: PINN-LSTM訓練器
    """
    # 獲取損失配置
    loss_config = config["training"]["loss"]
    
    # 創建專用訓練器
    trainer = PINNLSTMTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        lambda_physics_init=lambda_physics,
        lambda_physics_max=loss_config.get("max_lambda_physics", lambda_physics * 5),
        lambda_consistency_init=lambda_consistency,
        lambda_consistency_max=loss_config.get("max_lambda_consistency", lambda_consistency * 3),
        lambda_ramp_epochs=loss_config.get("epochs_to_max", 50),
        clip_grad_norm=clip_grad_norm,
        scheduler=scheduler,
        log_interval=config.get("debug", {}).get("verbose", 1)
    )
    
    logger.info(f"創建PINN-LSTM專用訓練器，"
               f"初始物理約束權重: {lambda_physics}，"
               f"初始一致性約束權重: {lambda_consistency}，"
               f"梯度裁剪範數: {clip_grad_norm}")
    
    return trainer


def train_hybrid_model_with_stages(model, dataloaders, config, train_config, device, output_dir, 
                                  use_physics=True, stages=["warmup", "main", "finetune"]):
    """
    使用分階段策略訓練混合模型
    
    參數:
        model (HybridPINNLSTMModel): 混合模型
        dataloaders (dict): 資料載入器
        config (dict): 模型配置
        train_config (dict): 訓練配置
        device (torch.device): 計算設備
        output_dir (str): 輸出目錄
        use_physics (bool): 是否使用物理約束
        stages (list): 要執行的訓練階段列表
        
    返回:
        dict: 訓練歷史記錄
    """
    all_history = {}
    
    # 定義階段訓練記錄檔案
    stage_log_file = os.path.join(output_dir, "stage_training_log.txt")
    with open(stage_log_file, "w") as f:
        f.write(f"分階段訓練日誌 - 開始時間: {datetime.datetime.now()}\n")
        f.write(f"模型: {config['model']['name']}, 使用物理約束: {use_physics}\n")
        f.write("="*50 + "\n")
    
    current_epoch = 0
    
    # 遍歷每個訓練階段
    for stage in stages:
        logger.info("="*50)
        logger.info(f"開始 {stage} 階段訓練")
        
        # 獲取階段配置
        stage_config = prepare_training_config(config, train_config, stage=stage, start_epoch=current_epoch)
        logger.info(f"階段描述: {stage_config['description']}")
        logger.info(f"訓練輪數: {stage_config['epochs']}")
        logger.info(f"學習率: {stage_config['learning_rate']}")
        logger.info(f"物理約束權重: {stage_config['lambda_physics']}")
        logger.info(f"一致性約束權重: {stage_config['lambda_consistency']}")
        
        # 創建階段特定的輸出目錄
        stage_dir = os.path.join(output_dir, f"stage_{stage}")
        os.makedirs(stage_dir, exist_ok=True)
        
        # 創建階段優化器
        optimizer = create_optimizer(model, config, stage_config)
        
        # 創建階段學習率調度器
        scheduler = create_lr_scheduler(optimizer, config, stage_config)
        
        # 創建專用訓練器
        trainer = get_pinn_lstm_trainer(
            model=model,
            optimizer=optimizer,
            config=config,
            device=device,
            lambda_physics=stage_config["lambda_physics"] if use_physics else 0.0,
            lambda_consistency=stage_config["lambda_consistency"],
            clip_grad_norm=stage_config["clip_grad_norm"],
            scheduler=scheduler
        )
        
        # 創建階段回調函數
        callbacks = AdaptiveCallbacks.create_callbacks(
            model=model,
            dataset_size=len(dataloaders["train_loader"].dataset),
            epochs=stage_config["epochs"],
            output_dir=stage_dir,
            use_tensorboard=True,
            use_progress_bar=True,
            use_early_stopping=True,
            patience=config["training"]["early_stopping"]["patience"],
            monitor="val_loss",
            mode="min",
            save_freq=5
        )
        
        # 訓練階段
        start_time = time.time()
        
        stage_history = trainer.train(
            train_loader=dataloaders["train_loader"],
            val_loader=dataloaders["val_loader"],
            epochs=stage_config["epochs"],
            early_stopping_patience=config["training"]["early_stopping"]["patience"],
            save_path=os.path.join(stage_dir, "models", "best_model.pt"),
            callbacks=callbacks
        )
        
        train_time = time.time() - start_time
        
        # 記錄階段訓練結果
        with open(stage_log_file, "a") as f:
            f.write(f"\n階段: {stage}\n")
            f.write(f"描述: {stage_config['description']}\n")
            f.write(f"訓練輪數: {stage_config['epochs']}\n")
            f.write(f"訓練時間: {train_time:.2f}秒\n")
            f.write(f"最佳驗證損失: {stage_history['best_val_loss']:.6f}\n")
            if 'val_metrics' in stage_history and 'rmse' in stage_history['val_metrics']:
                f.write(f"RMSE: {stage_history['val_metrics']['rmse'][-1]:.4f}\n")
            if 'val_metrics' in stage_history and 'r2' in stage_history['val_metrics']:
                f.write(f"R²: {stage_history['val_metrics']['r2'][-1]:.4f}\n")
            f.write("-"*40 + "\n")
        
        # 將階段歷史添加到總歷史
        all_history[stage] = stage_history
        
        # 更新當前輪次
        current_epoch += stage_config["epochs"]
        
        logger.info(f"{stage} 階段訓練完成 - 耗時: {train_time:.2f}秒, 最佳驗證損失: {stage_history['best_val_loss']:.6f}")
    
    # 最終模型保存
    final_model_path = os.path.join(output_dir, "models", "final_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'training_history': all_history,
        'final_epoch': current_epoch
    }, final_model_path)
    
    logger.info(f"分階段訓練完成，最終模型已保存至: {final_model_path}")
    
    return all_history


def evaluate_model_performance(model, data_loader, device, config, output_dir):
    """
    評估模型性能
    
    參數:
        model (torch.nn.Module): 模型
        data_loader (DataLoader): 數據載入器
        device (torch.device): 計算設備
        config (dict): 配置
        output_dir (str): 輸出目錄
        
    返回:
        dict: 評估結果
    """
    # 創建評估目錄
    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    logger.info("開始評估模型性能...")
    
    # 設定模型為評估模式
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_outputs = {}
    
    # 收集預測結果
    with torch.no_grad():
        for batch_data in data_loader:
            if len(batch_data) == 3:
                static_features, time_series, targets = batch_data
                static_features = static_features.to(device)
                time_series = time_series.to(device)
                targets = targets.to(device)
                
                if isinstance(model, HybridPINNLSTMModel):
                    outputs = model(static_features, time_series, return_features=True)
                    predictions = outputs["nf_pred"]
                    
                    # 收集其他輸出
                    for key, value in outputs.items():
                        if key not in all_outputs:
                            all_outputs[key] = []
                        
                        if isinstance(value, torch.Tensor):
                            all_outputs[key].append(value.cpu().numpy())
                        else:
                            all_outputs[key].append(value)
                elif isinstance(model, PINNModel):
                    outputs = model(static_features)
                    predictions = outputs["nf_pred"]
                    
                    # 收集其他輸出
                    for key, value in outputs.items():
                        if key not in all_outputs:
                            all_outputs[key] = []
                        
                        if isinstance(value, torch.Tensor):
                            all_outputs[key].append(value.cpu().numpy())
                        else:
                            all_outputs[key].append(value)
                elif isinstance(model, LSTMModel):
                    outputs = model(time_series)
                    predictions = outputs["output"]
                    
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
                
                # 收集預測和目標
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
    
    # 合併批次結果
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    # 合併模型輸出
    for key in all_outputs:
        if all_outputs[key] and isinstance(all_outputs[key][0], np.ndarray):
            all_outputs[key] = np.concatenate(all_outputs[key])
    
    # 添加預測和目標到輸出
    all_outputs["predictions"] = all_predictions
    all_outputs["targets"] = all_targets
    
    # 計算評估指標
    metrics = evaluate_model(all_targets, all_predictions, model_name=config["model"]["name"], verbose=True)
    
    # 保存評估指標
    metrics_path = os.path.join(eval_dir, "metrics.csv")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    logger.info(f"評估指標已保存至: {metrics_path}")
    
    # 產生視覺化結果
    logger.info("生成視覺化結果...")
    vis_dir = os.path.join(eval_dir, "visualizations")
    visualization_paths = visualize_model_results(all_outputs, output_dir=vis_dir)
    
    # 驗證物理約束
    if "delta_w" in all_outputs and all_outputs["delta_w"] is not None:
        a_coefficient = config["model"]["physics"]["a_coefficient"]
        b_coefficient = config["model"]["physics"]["b_coefficient"]
        
        logger.info("驗證物理約束...")
        physics_result = validate_physical_constraints(
            all_outputs["delta_w"], all_predictions, 
            a=a_coefficient, b=b_coefficient, 
            threshold=20.0, verbose=True
        )
        
        # 保存物理約束驗證結果
        physics_result_path = os.path.join(eval_dir, "physics_validation.txt")
        with open(physics_result_path, "w") as f:
            f.write(f"物理約束驗證結果:\n")
            f.write(f"通過: {physics_result[0]}\n")
            f.write(f"違反約束的樣本數: {len(physics_result[2])}\n")
            f.write(f"最大相對殘差: {np.max(physics_result[1]):.2f}%\n")
            f.write(f"平均相對殘差: {np.mean(physics_result[1]):.2f}%\n")
            f.write(f"中位相對殘差: {np.median(physics_result[1]):.2f}%\n")
    
    logger.info("模型評估完成")
    
    return {
        "metrics": metrics,
        "outputs": all_outputs
    }


if __name__ == "__main__":
    import argparse
    import yaml
    import torch
    import os
    from src.data.dataset import load_dataset
    from src.training.trainer import get_pinn_lstm_trainer, train_hybrid_model_with_stages
    from src.models.hybrid_model import HybridPINNLSTMModel
    from src.training.losses import HybridLoss

    parser = argparse.ArgumentParser(description="Train Hybrid PINN-LSTM model")
    parser.add_argument("--config", type=str, required=True, help="Path to model config YAML")
    parser.add_argument("--train-config", type=str, required=True, help="Path to training config YAML")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV data file")
    args = parser.parse_args()

    # 載入 config
    with open(args.config, "r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f)
    with open(args.train_config, "r", encoding="utf-8") as f:
        train_config = yaml.safe_load(f)

    # 設定設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 載入資料
    dataloaders = load_dataset(
        args.data,
        static_features=model_config["model"]["input"]["static_features"],
        time_series_features=model_config["model"]["input"]["time_series_features"],
        target_feature=model_config["model"]["input"]["target"],
        batch_size=model_config["training"]["batch_size"],
        sequence_length=model_config["model"].get("sequence_length", 10),
        num_workers=0
    )

    # 初始化模型
    model = HybridPINNLSTMModel(
        static_input_dim=len(model_config["model"]["input"]["static_features"]),
        temporal_input_dim=len(model_config["model"]["input"]["time_series_features"]),
        pinn_hidden_layers=model_config["model"]["pinn"]["hidden_layers"],
        lstm_hidden_size=model_config["model"]["lstm"]["hidden_size"],
        lstm_num_layers=model_config["model"]["lstm"]["num_layers"],
        dropout_rate=model_config["model"]["fusion"]["dropout_rate"],
        use_attention=model_config["model"]["lstm"]["use_attention"],
        fusion_type=model_config["model"]["fusion"]["type"],
        ensemble_method=model_config["model"]["fusion"]["ensemble_method"],
        l2_reg=model_config["training"].get("l2_reg", 0.001)
    ).to(device)

    # 訓練
    output_dir = "outputs/HybridPINNLSTM_run"
    os.makedirs(output_dir, exist_ok=True)

    history = train_hybrid_model_with_stages(
        model=model,
        dataloaders=dataloaders,
        config=model_config,
        train_config=train_config,
        device=device,
        output_dir=output_dir,
        use_physics=model_config["model"].get("use_physics", True),
        stages=["warmup", "main", "finetune"]
    )
