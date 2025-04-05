#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
train.py - 模型訓練腳本
本腳本用於訓練銲錫接點疲勞壽命預測的混合PINN-LSTM模型，
整合了改進的預處理、物理知識驅動的資料增強和分階段訓練策略。

主要特點:
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
import traceback
import matplotlib.pyplot as plt
from pathlib import Path

# 確保可以導入專案模組
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# 導入專案模組
from src.data.preprocess import process_pipeline
from src.data.dataset import create_dataloaders
from src.models.hybrid_model import HybridPINNLSTMModel
from src.models.pinn import PINNModel
from src.models.lstm import LSTMModel
from src.training.losses import get_loss_function, AdaptiveHybridLoss
from src.training.trainer import (
    Trainer, 
    EarlyStopping, 
    LearningRateScheduler, 
    prepare_training_config, 
    create_optimizer, 
    create_lr_scheduler, 
    train_hybrid_model_with_stages
)
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
# 建立完整的輸出目錄結構
os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "evaluation"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)

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


def apply_data_augmentation(data_dict, train_config, config):
    """
    應用資料增強
    
    參數:
        data_dict (dict): 包含分割後資料的字典
        train_config (dict): 訓練配置
        config (dict): 模型配置
        
    返回:
        dict: 包含增強後資料的字典
    """
    # 檢查是否需要資料增強
    augmentation_config = train_config.get("training_strategy", {}).get("data_augmentation", {})
    if not augmentation_config.get("enabled", False):
        logger.info("資料增強已禁用")
        return data_dict
    
    try:
        # 導入資料增強模組
        from src.data.dataset import augment_training_data
        
        logger.info("應用資料增強...")
        
        # 應用資料增強
        augmented_dict = augment_training_data(
            X_train=data_dict["X_train"],
            time_series_train=data_dict["time_series_train"],
            y_train=data_dict["y_train"],
            synthetic_samples=augmentation_config.get("synthetic_samples", 20),
            noise_level=augmentation_config.get("noise_level", 0.05),
            physics_guided=augmentation_config.get("physics_guided", True),
            a_coefficient=config["model"]["physics"]["a_coefficient"],
            b_coefficient=config["model"]["physics"]["b_coefficient"]
        )
        
        # 更新訓練資料
        data_dict["X_train"] = augmented_dict["X_train"]
        data_dict["time_series_train"] = augmented_dict["time_series_train"]
        data_dict["y_train"] = augmented_dict["y_train"]
        
        logger.info(f"資料增強完成，增強後的訓練集大小: {len(data_dict['X_train'])} 樣本")
        
    except ImportError:
        logger.warning("資料增強模組未找到，跳過資料增強")
    except Exception as e:
        logger.error(f"資料增強時發生錯誤: {str(e)}")
        logger.warning("使用原始資料繼續訓練")
    
    return data_dict


def train_model(model, data_dict, config, train_config, args, device, output_dir):
    """
    訓練模型
    
    參數:
        model (torch.nn.Module): 模型
        data_dict (dict): 包含資料的字典
        config (dict): 模型配置
        train_config (dict): 訓練配置
        args (argparse.Namespace): 命令行參數
        device (torch.device): 計算設備
        output_dir (str): 輸出目錄
        
    返回:
        dict: 訓練歷史記錄
    """
    # 創建資料載入器
    batch_size = args.batch_size if args.batch_size else config["training"]["batch_size"]
    dataloaders = create_dataloaders(
        X_train=data_dict["X_train"],
        X_val=data_dict["X_val"],
        X_test=data_dict["X_test"],
        time_series_train=data_dict["time_series_train"],
        time_series_val=data_dict["time_series_val"],
        time_series_test=data_dict["time_series_test"],
        y_train=data_dict["y_train"],
        y_val=data_dict["y_val"],
        y_test=data_dict["y_test"],
        batch_size=batch_size,
        num_workers=0,
        use_weighted_sampler=config["training"].get("use_weighted_sampler", False),
        augmentation=not args.no_augmentation
    )
    
    # 根據模型類型和訓練階段選擇訓練策略
    if args.stage != "all" and args.model_type == "hybrid":
        # 使用分階段訓練 (僅限混合模型)
        stages = [args.stage] if args.stage != "all" else ["warmup", "main", "finetune"]
        
        # 分階段訓練
        history = train_hybrid_model_with_stages(
            model=model,
            dataloaders=dataloaders,
            config=config,
            train_config=train_config,
            device=device,
            output_dir=output_dir,
            use_physics=not args.no_physics,
            stages=stages
        )
    else:
        # 使用標準訓練流程
        # 獲取訓練配置
        train_config_dict = prepare_training_config(config, train_config, stage=args.stage)
        
        # 創建優化器
        optimizer = create_optimizer(model, config, train_config_dict)
        
        # 創建損失函數
        loss_type = config["training"]["loss"]["type"]
        loss_function = get_loss_function(
            loss_type=loss_type,
            lambda_physics=train_config_dict["lambda_physics"] if not args.no_physics else 0.0,
            lambda_consistency=train_config_dict["lambda_consistency"],
            a=config["model"]["physics"]["a_coefficient"],
            b=config["model"]["physics"]["b_coefficient"],
            log_space=config["model"].get("use_log_transform", True),
            relative_error_weight=0.3,
            l2_reg=config["training"].get("l2_reg", 0.001)
        )
        
        # 創建學習率調度器
        scheduler = create_lr_scheduler(optimizer, config, train_config_dict)
        
        # 創建早停機制
        early_stopping = EarlyStopping(
            patience=config["training"]["early_stopping"]["patience"],
            min_delta=config["training"]["early_stopping"]["min_delta"],
            verbose=True,
            mode=config["training"]["early_stopping"]["mode"]
        )
        
        # 創建回調函數
        callbacks = AdaptiveCallbacks.create_callbacks(
            model=model,
            dataset_size=len(dataloaders["train_loader"].dataset),
            epochs=train_config_dict["epochs"],
            output_dir=output_dir,
            use_tensorboard=True,
            use_progress_bar=True,
            use_early_stopping=True,
            patience=config["training"]["early_stopping"]["patience"],
            monitor="val_loss",
            mode="min"
        )
        
        # 創建訓練器
        trainer = Trainer(
            model=model,
            criterion=loss_function,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
            clip_grad_norm=train_config_dict["clip_grad_norm"],
            log_interval=config["debug"]["verbose"] if "debug" in config else 10,
        )
        
        # 訓練模型
        logger.info(f"開始訓練，總輪數: {train_config_dict['epochs']}")
        history = trainer.train(
            train_loader=dataloaders["train_loader"],
            val_loader=dataloaders["val_loader"],
            epochs=train_config_dict["epochs"],
            early_stopping=early_stopping,
            callbacks=callbacks,
            save_path=os.path.join(output_dir, "models", "best_model.pt")
        )
    
    # 保存最終模型
    final_model_path = os.path.join(output_dir, "models", "final_model.pt")
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history
    }, final_model_path)
    
    logger.info(f"訓練完成，最終模型已保存至: {final_model_path}")
    
    return history


def evaluate_trained_model(model, data_dict, device, output_dir):
    """
    評估訓練好的模型
    
    參數:
        model (torch.nn.Module): 模型
        data_dict (dict): 包含資料的字典
        device (torch.device): 計算設備
        output_dir (str): 輸出目錄
        
    返回:
        dict: 評估結果
    """
    # 創建測試資料載入器
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(data_dict["X_test"]),
        torch.FloatTensor(data_dict["time_series_test"]),
        torch.FloatTensor(data_dict["y_test"])
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False
    )
    
    # 創建評估目錄
    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    # 模型評估
    model.eval()
    all_predictions = []
    all_targets = []
    all_outputs = {}
    
    with torch.no_grad():
        for static_features, time_series, targets in test_loader:
            # 將資料移至設備
            static_features = static_features.to(device)
            time_series = time_series.to(device)
            targets = targets.to(device)
            
            # 前向傳播
            if hasattr(model, 'forward') and 'static_features' in model.forward.__code__.co_varnames:
                # 混合模型
                outputs = model(static_features, time_series, return_features=True)
                predictions = outputs["nf_pred"]
                
                # 收集輸出
                for key, value in outputs.items():
                    if key not in all_outputs:
                        all_outputs[key] = []
                    
                    if isinstance(value, torch.Tensor):
                        all_outputs[key].append(value.cpu().numpy())
                    else:
                        all_outputs[key].append(value)
            else:
                # 單一模型
                if isinstance(model, PINNModel):
                    outputs = model(static_features)
                    predictions = outputs["nf_pred"]
                else:  # LSTM模型
                    outputs = model(time_series)
                    predictions = outputs["output"]
                
                # 收集輸出
                for key, value in outputs.items():
                    if key not in all_outputs:
                        all_outputs[key] = []
                    
                    if isinstance(value, torch.Tensor):
                        all_outputs[key].append(value.cpu().numpy())
                    else:
                        all_outputs[key].append(value)
            
            # 收集預測和目標
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # 合併結果
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    # 合併所有輸出
    for key in all_outputs:
        if isinstance(all_outputs[key][0], np.ndarray):
            all_outputs[key] = np.concatenate(all_outputs[key])
    
    # 添加預測和目標到輸出
    all_outputs["predictions"] = all_predictions
    all_outputs["targets"] = all_targets
    
    # 計算評估指標
    metrics = evaluate_model(all_targets, all_predictions, model_name=type(model).__name__, verbose=True)
    
    # 保存評估結果
    metrics_path = os.path.join(eval_dir, "metrics.csv")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    logger.info(f"評估指標已保存至: {metrics_path}")
    
    # 產生視覺化結果
    logger.info("生成視覺化結果...")
    vis_dir = os.path.join(eval_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    visualize_model_results(all_outputs, output_dir=vis_dir)
    
    logger.info(f"評估完成，結果已保存至: {eval_dir}")
    
    return {
        "metrics": metrics,
        "predictions": all_predictions,
        "targets": all_targets,
        "outputs": all_outputs
    }


def main():
    """主函數"""
    # 解析命令行參數
    args = parse_args()
    
    # 設定除錯模式
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("除錯模式已啟用")
    
    # 載入配置文件
    config = load_config(args.config)
    train_config = load_config(args.train_config)
    
    # 設定隨機種子
    random_seed = args.seed if args.seed is not None else config.get("random_seed", 42)
    set_random_seed(random_seed)
    
    # 設定計算設備
    device_name = args.device if args.device else config.get("device", "cuda")
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    logger.info(f"使用計算設備: {device}")
    
    # 設定輸出目錄
    output_dir = args.output_dir if args.output_dir else os.path.join(
        config.get("paths", {}).get("output_dir", "outputs"),
        f"{args.model_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"輸出目錄: {output_dir}")
    
    # 準備資料
    data_path = args.data if args.data else config["paths"]["data"]
    feature_cols = config["model"]["input"]["static_features"]
    target_col = "Nf_pred (cycles)"
    time_series_prefix = config["model"]["input"]["time_series_features"]
    time_points = [3600, 7200, 10800, 14400]  # 明確指定時間點

    logger.info(f"載入資料: {data_path}")
    data_dict = process_pipeline(
        filepath=data_path,
        feature_cols=feature_cols,
        target_col=target_col,
        time_series_prefix=time_series_prefix,
        time_points=time_points,  # 傳入時間點
        test_size=config["training"]["data_split"]["test_size"],
        val_size=config["training"]["data_split"]["val_size"],
        random_state=config["training"]["data_split"]["random_seed"]
    )
    
    logger.info(f"資料分割完成, 訓練集: {len(data_dict['X_train'])} 樣本, "
               f"驗證集: {len(data_dict['X_val'])} 樣本, "
               f"測試集: {len(data_dict['X_test'])} 樣本")
    
    # 如果需要，應用資料增強
    if not args.no_augmentation:
        data_dict = apply_data_augmentation(data_dict, train_config, config)
    
    # 根據模式選擇操作
    if args.evaluate_only:
        # 僅評估模式
        if args.model_path is None:
            logger.error("使用評估模式需要提供模型路徑 (--model-path)")
            return
        
        # 載入模型
        logger.info(f"載入模型: {args.model_path}")
        model = create_model(config, model_type=args.model_type, use_physics=not args.no_physics)
        
        # 將模型移至設備
        model = model.to(device)
        
        # 載入模型權重
        checkpoint = torch.load(args.model_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        # 評估模型
        evaluate_trained_model(model, data_dict, device, output_dir)
    else:
        # 訓練模式
        # 創建模型
        model = create_model(config, model_type=args.model_type, use_physics=not args.no_physics)
        
        # 將模型移至設備
        model = model.to(device)
        
        # 訓練模型
        train_model(model, data_dict, config, train_config, args, device, output_dir)
        
        # 評估模型
        evaluate_trained_model(model, data_dict, device, output_dir)
    
    logger.info("任務完成")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("用戶中斷訓練")
    except Exception as e:
        logger.exception(f"執行過程中發生錯誤: {str(e)}")
        logger.error(f"詳細錯誤追蹤: {traceback.format_exc()}")
        sys.exit(1)

