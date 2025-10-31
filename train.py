"""
Training pipeline for the Financial Risk Model.
Handles training loop, validation, and model checkpointing.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import json

from models.hybrid import HybridRiskModel
from config import Config, set_seed
from utils import (
    calculate_metrics, save_checkpoint, load_checkpoint,
    plot_confusion_matrix, plot_training_history, ensure_dir
)


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
            
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class Trainer:
    """Trainer class for the hybrid risk model."""
    
    def __init__(self,
                 model: HybridRiskModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: Optional[DataLoader] = None,
                 config: Config = Config):
        """
        Initialize trainer.
        
        Args:
            model: The hybrid risk model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader (optional)
            config: Configuration object
        """
        self.model = model.to(config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = config.DEVICE
        
        # Loss function
        if config.LOSS_FUNCTION == 'CrossEntropy':
            if config.CLASS_WEIGHTS is not None:
                weights = torch.FloatTensor(config.CLASS_WEIGHTS).to(self.device)
                self.criterion = nn.CrossEntropyLoss(weight=weights)
            else:
                self.criterion = nn.CrossEntropyLoss()
        elif config.LOSS_FUNCTION == 'BCE':
            self.criterion = nn.BCEWithLogitsLoss()
        elif config.LOSS_FUNCTION == 'MSE':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss function: {config.LOSS_FUNCTION}")
        
        # Optimizer
        if config.OPTIMIZER == 'Adam':
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=config.LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY
            )
        elif config.OPTIMIZER == 'AdamW':
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=config.LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY
            )
        elif config.OPTIMIZER == 'SGD':
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=config.LEARNING_RATE,
                momentum=0.9,
                weight_decay=config.WEIGHT_DECAY
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.OPTIMIZER}")
        
        # Learning rate scheduler
        if config.SCHEDULER == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=config.SCHEDULER_PATIENCE,
                factor=config.SCHEDULER_FACTOR
            )
        elif config.SCHEDULER == 'CosineAnnealing':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.NUM_EPOCHS
            )
        elif config.SCHEDULER == 'StepLR':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
        else:
            self.scheduler = None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.EARLY_STOPPING_PATIENCE,
            mode='min'
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Tensorboard writer
        ensure_dir(config.LOG_DIR)
        self.writer = SummaryWriter(log_dir=config.LOG_DIR)
        
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.NUM_EPOCHS}')
        for batch_idx, (temporal_data, graph_data, labels) in enumerate(pbar):
            # Move data to device
            temporal_data = temporal_data.to(self.device)
            labels = labels.to(self.device)
            graph_data = graph_data.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(temporal_data, graph_data)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.GRADIENT_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.GRADIENT_CLIP
                )
            
            self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Track loss
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total
            })
            
            # Log to tensorboard
            if batch_idx % self.config.LOG_INTERVAL == 0:
                step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), step)
                self.writer.add_scalar('Train/Accuracy', 100. * correct / total, step)
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, epoch: int, loader: Optional[DataLoader] = None) -> Tuple[float, float, Dict]:
        """
        Validate the model.
        
        Args:
            epoch: Current epoch number
            loader: Data loader to use (default: validation loader)
            
        Returns:
            Tuple of (average loss, accuracy, metrics dict)
        """
        if loader is None:
            loader = self.val_loader
        
        self.model.eval()
        total_loss = 0.0
        all_labels = []
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for temporal_data, graph_data, labels in loader:
                # Move data to device
                temporal_data = temporal_data.to(self.device)
                labels = labels.to(self.device)
                graph_data = graph_data.to(self.device)
                
                # Forward pass
                outputs = self.model(temporal_data, graph_data)
                loss = self.criterion(outputs, labels)
                
                # Track loss
                total_loss += loss.item()
                
                # Get predictions and probabilities
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                # Store results
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        avg_loss = total_loss / len(loader)
        
        # Calculate metrics
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        
        metrics = calculate_metrics(all_labels, all_predictions, all_probabilities)
        
        # Log to tensorboard
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        self.writer.add_scalar('Val/Accuracy', metrics['accuracy'] * 100, epoch)
        self.writer.add_scalar('Val/F1', metrics['f1'], epoch)
        if 'auc_roc' in metrics:
            self.writer.add_scalar('Val/AUC-ROC', metrics['auc_roc'], epoch)
        
        return avg_loss, metrics['accuracy'] * 100, metrics
    
    def train(self) -> Dict[str, List[float]]:
        """
        Complete training loop.
        
        Returns:
            Training history dictionary
        """
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.NUM_EPOCHS):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, val_metrics = self.validate(epoch)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print epoch summary
            print(f'\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'  Val Metrics: {val_metrics}')
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'  Learning Rate: {current_lr:.6f}')
                self.writer.add_scalar('Train/LearningRate', current_lr, epoch)
            
            # Save checkpoint if best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(
                    self.config.CHECKPOINT_DIR,
                    'best_model.pth'
                )
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics,
                    checkpoint_path
                )
                print(f'  Saved best model to {checkpoint_path}')
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config.SAVE_CHECKPOINT_EVERY == 0:
                checkpoint_path = os.path.join(
                    self.config.CHECKPOINT_DIR,
                    f'checkpoint_epoch_{epoch+1}.pth'
                )
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics,
                    checkpoint_path
                )
            
            # Early stopping
            if self.early_stopping(val_loss):
                print(f'\nEarly stopping triggered at epoch {epoch+1}')
                break
        
        self.writer.close()
        print('\nTraining complete!')
        
        return self.history
    
    def evaluate(self, save_dir: str = None) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            save_dir: Directory to save evaluation results
            
        Returns:
            Dictionary of test metrics
        """
        if self.test_loader is None:
            print("No test loader provided")
            return {}
        
        if save_dir is None:
            save_dir = self.config.RESULTS_DIR
        
        print("\nEvaluating on test set...")
        _, test_acc, test_metrics = self.validate(0, self.test_loader)
        
        print(f"Test Accuracy: {test_acc:.2f}%")
        print(f"Test Metrics: {test_metrics}")
        
        # Get predictions for visualization
        self.model.eval()
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for temporal_data, graph_data, labels in self.test_loader:
                temporal_data = temporal_data.to(self.device)
                labels = labels.to(self.device)
                graph_data = graph_data.to(self.device)
                
                outputs = self.model(temporal_data, graph_data)
                _, predicted = outputs.max(1)
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        
        # Save confusion matrix
        ensure_dir(save_dir)
        cm_path = os.path.join(save_dir, 'confusion_matrix.png')
        plot_confusion_matrix(all_labels, all_predictions, cm_path)
        
        # Save metrics
        metrics_path = os.path.join(save_dir, 'test_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=4)
        
        print(f"Results saved to {save_dir}")
        
        return test_metrics
