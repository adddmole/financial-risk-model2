"""
Training module for Financial Risk Model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils import set_seed, get_device, save_model, plot_training_history


class Trainer:
    """
    Trainer class for financial risk model
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        learning_rate=0.001,
        weight_decay=1e-5,
        device=None
    ):
        """
        Initialize trainer
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            device: Device to use (cuda or cpu)
        """
        self.device = device if device else get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metric': [],
            'val_metric': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        predictions = []
        targets = []
        
        for batch in tqdm(self.train_loader, desc='Training'):
            # Get data
            features = batch['features'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Create adjacency matrix from graph data
            if 'graph' in batch:
                adj_matrix = batch['graph'].to(self.device)
            else:
                # Create identity matrix if no graph provided
                batch_size = features.size(0)
                num_nodes = features.size(-1)
                adj_matrix = torch.eye(num_nodes).unsqueeze(0).repeat(batch_size, 1, 1)
                adj_matrix = adj_matrix.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features, adj_matrix).squeeze()
            
            # Calculate loss
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            predictions.extend(outputs.detach().cpu().numpy())
            targets.extend(labels.detach().cpu().numpy())
        
        avg_loss = total_loss / len(self.train_loader)
        mae = mean_absolute_error(targets, predictions)
        
        return avg_loss, mae
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Get data
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Create adjacency matrix
                if 'graph' in batch:
                    adj_matrix = batch['graph'].to(self.device)
                else:
                    batch_size = features.size(0)
                    num_nodes = features.size(-1)
                    adj_matrix = torch.eye(num_nodes).unsqueeze(0).repeat(batch_size, 1, 1)
                    adj_matrix = adj_matrix.to(self.device)
                
                # Forward pass
                outputs = self.model(features, adj_matrix).squeeze()
                
                # Calculate loss
                loss = self.criterion(outputs, labels)
                
                # Track metrics
                total_loss += loss.item()
                predictions.extend(outputs.cpu().numpy())
                targets.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        mae = mean_absolute_error(targets, predictions)
        
        return avg_loss, mae
    
    def train(self, epochs=50, early_stopping_patience=10):
        """
        Train the model
        
        Args:
            epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
        
        Returns:
            Training history
        """
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_loss, train_mae = self.train_epoch()
            
            # Validate
            val_loss, val_mae = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metric'].append(train_mae)
            self.history['val_metric'].append(val_mae)
            
            print(f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_model(self.model, 'best_model.pt')
                self.patience_counter = 0
                print("Model improved and saved!")
            else:
                self.patience_counter += 1
                print(f"No improvement. Patience: {self.patience_counter}/{early_stopping_patience}")
            
            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Plot training history
        plot_training_history(self.history)
        
        return self.history


def evaluate_model(model, test_loader, device=None):
    """
    Evaluate model on test set
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to use
    
    Returns:
        Dictionary of metrics and predictions
    """
    if device is None:
        device = get_device()
    
    model = model.to(device)
    model.eval()
    
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            
            if 'graph' in batch:
                adj_matrix = batch['graph'].to(device)
            else:
                batch_size = features.size(0)
                num_nodes = features.size(-1)
                adj_matrix = torch.eye(num_nodes).unsqueeze(0).repeat(batch_size, 1, 1)
                adj_matrix = adj_matrix.to(device)
            
            outputs = model(features, adj_matrix).squeeze()
            
            predictions.extend(outputs.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Calculate metrics
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, predictions)
    
    metrics = {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }
    
    return metrics, predictions, targets
