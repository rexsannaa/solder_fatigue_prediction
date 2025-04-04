class PINNLSTMTrainer:
    """
    PINN-LSTM混合模型訓練器
    提供分階段訓練、物理約束動態調整等功能
    """
    def __init__(self, model, optimizer, device, 
                 lambda_physics_init=0.1, lambda_physics_max=1.0,
                 lambda_consistency_init=0.05, lambda_consistency_max=0.3,
                 lambda_ramp_epochs=50, clip_grad_norm=1.0,
                 scheduler=None, log_interval=10):
        """
        初始化PINN-LSTM訓練器
        
        參數:
            model (HybridPINNLSTMModel): 混合模型
            optimizer (torch.optim.Optimizer): 優化器
            device (torch.device): 計算設備
            lambda_physics_init (float): 物理約束初始權重
            lambda_physics_max (float): 物理約束最大權重
            lambda_consistency_init (float): 一致性損失初始權重
            lambda_consistency_max (float): 一致性損失最大權重
            lambda_ramp_epochs (int): 達到最大權重的輪數
            clip_grad_norm (float): 梯度裁剪範數
            scheduler (torch.optim.lr_scheduler): 學習率調度器
            log_interval (int): 日誌輸出間隔
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.lambda_physics_init = lambda_physics_init
        self.lambda_physics_max = lambda_physics_max
        self.lambda_consistency_init = lambda_consistency_init
        self.lambda_consistency_max = lambda_consistency_max
        self.lambda_ramp_epochs = lambda_ramp_epochs
        self.clip_grad_norm = clip_grad_norm
        self.scheduler = scheduler
        self.log_interval = log_interval
        
        # 初始化訓練記錄
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = {}
        self.val_metrics = {}
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        # 初始化損失權重
        self.lambda_physics = lambda_physics_init
        self.lambda_consistency = lambda_consistency_init
    
    def update_loss_weights(self, epoch):
        """
        更新損失權重
        
        參數:
            epoch (int): 當前訓練輪次
        """
        if self.lambda_ramp_epochs <= 0:
            return
            
        progress = min(epoch / self.lambda_ramp_epochs, 1.0)
        self.lambda_physics = self.lambda_physics_init + (self.lambda_physics_max - self.lambda_physics_init) * progress
        self.lambda_consistency = self.lambda_consistency_init + (self.lambda_consistency_max - self.lambda_consistency_init) * progress
        
        logger.debug(f"輪次 {epoch}: 物理約束權重 = {self.lambda_physics:.4f}, 一致性約束權重 = {self.lambda_consistency:.4f}")
    
    def train_epoch(self, train_loader):
        """
        訓練一個輪次
        
        參數:
            train_loader (DataLoader): 訓練資料載入器
            
        返回:
            dict: 包含訓練損失和指標的字典
        """
        self.model.train()
        epoch_losses = {'total': 0.0, 'pred': 0.0, 'physics': 0.0, 'consistency': 0.0}
        num_batches = len(train_loader)
        all_targets = []
        all_predictions = []
        
        # 使用 tqdm 顯示進度條（如果可用）
        try:
            from tqdm import tqdm
            pbar = tqdm(enumerate(train_loader), total=num_batches, desc="Training", leave=False)
        except ImportError:
            pbar = enumerate(train_loader)
        
        for batch_idx, (static_features, time_series, targets) in pbar:
            # 將資料移至設備
            static_features = static_features.to(self.device)
            time_series = time_series.to(self.device)
            targets = targets.to(self.device)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向傳播
            outputs = self.model(static_features, time_series)
            
            # 計算損失
            losses = self.model.calculate_loss(
                outputs, targets, 
                lambda_physics=self.lambda_physics,
                lambda_consistency=self.lambda_consistency
            )
            
            loss = losses['total_loss']
            
            # 反向傳播
            loss.backward()
            
            # 梯度裁剪
            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            
            # 參數更新
            self.optimizer.step()
            
            # 累計損失
            for k, v in losses.items():
                if k.endswith('_loss'):
                    name = k.replace('_loss', '')
                    if name in epoch_losses:
                        epoch_losses[name] += v.item()
                    else:
                        epoch_losses[name] = v.item()
                        
            # 收集預測和目標，用於計算指標
            all_predictions.append(outputs['nf_pred'].detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())
            
            # 更新進度條
            if hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({'loss': loss.item()})
        
        # 計算平均損失
        for k in epoch_losses:
            epoch_losses[k] /= num_batches
        
        # 計算訓練指標
        if all_predictions and all_targets:
            all_predictions = np.concatenate(all_predictions)
            all_targets = np.concatenate(all_targets)
            train_metrics = self._calculate_metrics(all_predictions, all_targets)
        else:
            train_metrics = {}
            
        return {'losses': epoch_losses, 'metrics': train_metrics}
    
    def evaluate(self, val_loader):
        """
        評估模型
        
        參數:
            val_loader (DataLoader): 驗證資料載入器
            
        返回:
            tuple: (平均損失, 評估指標字典, 預測結果, 目標值)
        """
        self.model.eval()
        val_losses = {'total': 0.0, 'pred': 0.0, 'physics': 0.0, 'consistency': 0.0}
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for static_features, time_series, targets in val_loader:
                # 將資料移至設備
                static_features = static_features.to(self.device)
                time_series = time_series.to(self.device)
                targets = targets.to(self.device)
                
                # 前向傳播
                outputs = self.model(static_features, time_series, return_features=True)
                
                # 計算損失
                losses = self.model.calculate_loss(
                    outputs, targets, 
                    lambda_physics=self.lambda_physics,
                    lambda_consistency=self.lambda_consistency
                )
                
                # 累計損失
                for k, v in losses.items():
                    if k.endswith('_loss'):
                        name = k.replace('_loss', '')
                        if name in val_losses:
                            val_losses[name] += v.item()
                        else:
                            val_losses[name] = v.item()
                
                # 收集預測和目標
                all_outputs.append(outputs)
                all_targets.append(targets.cpu().numpy())
        
        # 計算平均損失
        num_batches = len(val_loader)
        for k in val_losses:
            val_losses[k] /= num_batches
        
        # 合併預測和目標
        all_predictions = torch.cat([o['nf_pred'].cpu() for o in all_outputs]).numpy()
        all_targets = np.concatenate(all_targets)
        
        # 計算指標
        metrics = self._calculate_metrics(all_predictions, all_targets)
        
        # 合併所有輸出的特徵
        merged_outputs = {}
        for key in all_outputs[0].keys():
            if isinstance(all_outputs[0][key], torch.Tensor):
                merged_outputs[key] = torch.cat([o[key].cpu() for o in all_outputs]).numpy()
        
        merged_outputs['predictions'] = all_predictions
        merged_outputs['targets'] = all_targets
        
        return val_losses, metrics, merged_outputs
    
    def _calculate_metrics(self, predictions, targets):
        """
        計算評估指標
        
        參數:
            predictions (np.ndarray): 預測值
            targets (np.ndarray): 真實值
            
        返回:
            dict: 包含評估指標的字典
        """
        try:
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            
            # 計算基本指標
            mse = mean_squared_error(targets, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(targets, predictions)
            mae = mean_absolute_error(targets, predictions)
            
            # 計算相對誤差
            rel_error = np.abs((targets - predictions) / (targets + 1e-8)) * 100
            mean_rel_error = np.mean(rel_error)
            median_rel_error = np.median(rel_error)
            
            # 計算對數空間的指標
            log_targets = np.log(targets + 1e-8)
            log_predictions = np.log(predictions + 1e-8)
            log_mse = mean_squared_error(log_targets, log_predictions)
            log_rmse = np.sqrt(log_mse)
            
            return {
                'rmse': rmse,
                'r2': r2,
                'mae': mae,
                'mean_rel_error': mean_rel_error,
                'median_rel_error': median_rel_error,
                'log_rmse': log_rmse
            }
        except Exception as e:
            logger.error(f"計算指標時出錯: {str(e)}")
            return {}
    
    def train(self, train_loader, val_loader, epochs, early_stopping_patience=20,
             save_path=None, callbacks=None):
        """
        訓練模型
        
        參數:
            train_loader (DataLoader): 訓練資料載入器
            val_loader (DataLoader): 驗證資料載入器
            epochs (int): 訓練輪數
            early_stopping_patience (int): 早停耐心值
            save_path (str): 模型保存路徑
            callbacks (list): 回調函數列表
            
        返回:
            dict: 訓練歷史記錄
        """
        callbacks = callbacks or []
        best_val_loss = float('inf')
        patience_counter = 0
        history = {
            'train_losses': [],
            'val_losses': [],
            'train_metrics': {},
            'val_metrics': {},
            'best_val_loss': float('inf')
        }
        
        # 初始化指標記錄
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = {}
        self.val_metrics = {}
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # 更新損失權重
            self.update_loss_weights(epoch)
            
            # 訓練一個輪次
            train_results = self.train_epoch(train_loader)
            train_losses = train_results['losses']
            train_metrics = train_results.get('metrics', {})
            
            self.train_losses.append(train_losses)
            
            # 更新訓練指標記錄
            for k, v in train_metrics.items():
                if k not in self.train_metrics:
                    self.train_metrics[k] = []
                self.train_metrics[k].append(v)
            
            # 評估
            val_losses, val_metrics, val_outputs = self.evaluate(val_loader)
            self.val_losses.append(val_losses)
            
            # 更新驗證指標記錄
            for k, v in val_metrics.items():
                if k not in self.val_metrics:
                    self.val_metrics[k] = []
                self.val_metrics[k].append(v)
            
            # 更新學習率
            if self.scheduler is not None:
                if hasattr(self.scheduler, 'step_with_metrics'):
                    self.scheduler.step_with_metrics(val_losses['total'])
                else:
                    self.scheduler.step()
            
            # 輸出日誌
            if epoch % self.log_interval == 0 or epoch == epochs - 1:
                curr_lr = self.optimizer.param_groups[0]['lr']
                log_msg = (f"輪次 {epoch+1}/{epochs} - "
                          f"訓練損失: {train_losses['total']:.4f}, "
                          f"驗證損失: {val_losses['total']:.4f}, "
                          f"RMSE: {val_metrics.get('rmse', 0):.4f}, "
                          f"R²: {val_metrics.get('r2', 0):.4f}, "
                          f"學習率: {curr_lr:.6f}")
                logger.info(log_msg)
            
            # 早停檢查
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                patience_counter = 0
                
                # 保存最佳模型
                if save_path:
                    self._save_model(save_path, val_losses, val_metrics)
                
                # 保存最佳模型狀態
                self.best_val_loss = best_val_loss
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                
                # 更新歷史記錄
                history['best_val_loss'] = best_val_loss
            else:
                patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    logger.info(f"早停觸發，在輪次 {epoch+1} 停止訓練")
                    break
            
            # 執行回調函數
            for callback in callbacks:
                callback(epoch, {
                    'model': self.model,
                    'optimizer': self.optimizer,
                    'train_loss': train_losses['total'],
                    'val_loss': val_losses['total'],
                    'metrics': val_metrics,
                    'epoch': epoch
                })
        
        # 更新歷史記錄
        history['train_losses'] = self.train_losses
        history['val_losses'] = self.val_losses
        history['train_metrics'] = self.train_metrics
        history['val_metrics'] = self.val_metrics
        
        # 恢復最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return history
    
    def _save_model(self, path, val_losses, val_metrics):
        """
        保存模型
        
        參數:
            path (str): 保存路徑
            val_losses (dict): 驗證損失
            val_metrics (dict): 驗證指標
        """
        # 確保目錄存在
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存模型
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_losses': val_losses,
            'val_metrics': val_metrics,
            'epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'lambda_physics': self.lambda_physics,
            'lambda_consistency': self.lambda_consistency,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }, path)
        
        logger.info(f"模型已保存至 {path}")
    
    def load_model(self, path):
        """
        載入模型
        
        參數:
            path (str): 模型路徑
            
        返回:
            dict: 包含模型指標的字典
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.lambda_physics = checkpoint.get('lambda_physics', self.lambda_physics)
        self.lambda_consistency = checkpoint.get('lambda_consistency', self.lambda_consistency)
        
        # 載入訓練記錄
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        if 'train_metrics' in checkpoint:
            self.train_metrics = checkpoint['train_metrics']
        if 'val_metrics' in checkpoint:
            self.val_metrics = checkpoint['val_metrics']
        
        logger.info(f"模型已從 {path} 載入")
        
        return {
            'val_losses': checkpoint.get('val_losses', {}),
            'val_metrics': checkpoint.get('val_metrics', {})
        }