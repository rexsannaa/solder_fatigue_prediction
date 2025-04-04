
import torch
import torch.nn as nn
import torch.optim as optim
import logging

logger = logging.getLogger(__name__)

class EarlyStopping:
    def __init__(self, patience=10, verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info("觸發早停機制")

class LearningRateScheduler:
    def __init__(self, optimizer, step_size=10, gamma=0.1):
        self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    def step(self):
        self.scheduler.step()

class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler=None, early_stopping=None, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.device = device

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        logger.info(f"本 epoch 平均訓練損失: {avg_loss}")
        return avg_loss

    def validate_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        logger.info(f"本 epoch 平均驗證損失: {avg_loss}")
        return avg_loss
