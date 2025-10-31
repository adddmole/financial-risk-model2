"""
Training module for the financial risk model.
Contains Transformer, GNN, and combined model architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, Dict
import config
from utils import calculate_metrics


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for time series data.
    """
    
    def __init__(self, n_features: int, d_model: int, nhead: int, 
                 num_layers: int, dim_feedforward: int, dropout: float = 0.1):
        super(TransformerEncoder, self).__init__()
        
        self.embedding = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.d_model = d_model
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, sequence_length, n_features)
            
        Returns:
            Encoded tensor (batch_size, d_model)
        """
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        # Global average pooling
        x = torch.mean(x, dim=1)  # (batch_size, d_model)
        return x


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)


class GNNLayer(nn.Module):
    """
    Graph Neural Network layer.
    """
    
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super(GNNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features (batch_size, n_nodes, in_dim)
            adj: Adjacency matrix (n_nodes, n_nodes)
            
        Returns:
            Updated node features (batch_size, n_nodes, out_dim)
        """
        # Aggregate neighbor features
        x = torch.matmul(adj, x)  # (batch_size, n_nodes, in_dim)
        x = self.linear(x)  # (batch_size, n_nodes, out_dim)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class GNNEncoder(nn.Module):
    """
    Graph Neural Network encoder.
    """
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, 
                 num_layers: int, dropout: float = 0.1):
        super(GNNEncoder, self).__init__()
        
        self.layers = nn.ModuleList()
        # First layer
        self.layers.append(GNNLayer(in_dim, hidden_dim, dropout))
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GNNLayer(hidden_dim, hidden_dim, dropout))
        # Last layer
        self.layers.append(GNNLayer(hidden_dim, out_dim, dropout))
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features (batch_size, n_nodes, in_dim)
            adj: Adjacency matrix (n_nodes, n_nodes)
            
        Returns:
            Graph embedding (batch_size, out_dim)
        """
        for layer in self.layers:
            x = layer(x, adj)
        # Global mean pooling
        x = torch.mean(x, dim=1)  # (batch_size, out_dim)
        return x


class FinancialRiskModel(nn.Module):
    """
    Combined financial risk model with Transformer and GNN.
    """
    
    def __init__(self):
        super(FinancialRiskModel, self).__init__()
        
        # Transformer for time series
        self.transformer = TransformerEncoder(
            n_features=config.N_FEATURES,
            d_model=config.TRANSFORMER_D_MODEL,
            nhead=config.TRANSFORMER_NHEAD,
            num_layers=config.TRANSFORMER_NUM_LAYERS,
            dim_feedforward=config.TRANSFORMER_DIM_FEEDFORWARD,
            dropout=config.TRANSFORMER_DROPOUT
        )
        
        # GNN for graph data
        self.gnn = GNNEncoder(
            in_dim=config.N_FEATURES,
            hidden_dim=config.GNN_HIDDEN_DIM,
            out_dim=config.GNN_OUTPUT_DIM,
            num_layers=config.GNN_NUM_LAYERS,
            dropout=config.GNN_DROPOUT
        )
        
        # Combined layers
        combined_dim = config.TRANSFORMER_D_MODEL + config.GNN_OUTPUT_DIM
        self.fc1 = nn.Linear(combined_dim, config.HIDDEN_DIM)
        self.fc2 = nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2)
        self.fc3 = nn.Linear(config.HIDDEN_DIM // 2, config.OUTPUT_DIM)
        self.dropout = nn.Dropout(config.TRANSFORMER_DROPOUT)
        
    def forward(self, time_series: torch.Tensor, graph_features: torch.Tensor,
                adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            time_series: Time series data (batch_size, seq_len, n_features)
            graph_features: Graph features (batch_size, n_nodes, feature_dim)
            adj_matrix: Adjacency matrix (n_nodes, n_nodes)
            
        Returns:
            Risk predictions (batch_size, 1)
        """
        # Transformer encoding
        ts_encoding = self.transformer(time_series)  # (batch_size, d_model)
        
        # GNN encoding
        graph_encoding = self.gnn(graph_features, adj_matrix)  # (batch_size, gnn_out_dim)
        
        # Combine encodings
        combined = torch.cat([ts_encoding, graph_encoding], dim=1)
        
        # MLP layers
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                criterion: nn.Module, device: torch.device) -> float:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    
    for time_series, graph_features, adj_matrix, targets in dataloader:
        time_series = time_series.to(device)
        graph_features = graph_features.to(device)
        adj_matrix = adj_matrix.to(device)
        targets = targets.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(time_series, graph_features, adj_matrix)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
             device: torch.device) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate the model.
    
    Args:
        model: Model to evaluate
        dataloader: Evaluation data loader
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        Tuple of (average_loss, metrics_dict)
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for time_series, graph_features, adj_matrix, targets in dataloader:
            time_series = time_series.to(device)
            graph_features = graph_features.to(device)
            adj_matrix = adj_matrix.to(device)
            targets = targets.to(device).unsqueeze(1)
            
            outputs = model(time_series, graph_features, adj_matrix)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            all_predictions.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
    
    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(np.array(all_predictions), np.array(all_targets))
    
    return avg_loss, metrics


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                num_epochs: int, learning_rate: float, device: torch.device,
                patience: int = 10) -> Tuple[nn.Module, Dict]:
    """
    Train the model with early stopping.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to train on
        patience: Early stopping patience
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, 
                                 weight_decay=config.WEIGHT_DECAY)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    print("Starting training...")
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Val R2: {val_metrics['r2']:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }
    
    return model, history
