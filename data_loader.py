"""
Data loader for financial risk model.
Handles data loading, preprocessing, and batch generation.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
import config


class FinancialDataset(Dataset):
    """
    Custom Dataset for financial time series data.
    """
    
    def __init__(self, time_series: np.ndarray, graph_features: np.ndarray, 
                 adjacency_matrix: np.ndarray, targets: np.ndarray):
        """
        Initialize the dataset.
        
        Args:
            time_series: Time series data (N, sequence_length, n_features)
            graph_features: Graph node features (N, n_nodes, feature_dim)
            adjacency_matrix: Graph adjacency matrix (n_nodes, n_nodes)
            targets: Target risk scores (N,)
        """
        self.time_series = torch.FloatTensor(time_series)
        self.graph_features = torch.FloatTensor(graph_features)
        self.adjacency_matrix = torch.FloatTensor(adjacency_matrix)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self) -> int:
        return len(self.time_series)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (self.time_series[idx], 
                self.graph_features[idx], 
                self.adjacency_matrix,
                self.targets[idx])


def generate_synthetic_data(n_samples: int, sequence_length: int, n_features: int,
                           n_nodes: int, node_feature_dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic financial data for demonstration.
    
    Args:
        n_samples: Number of samples
        sequence_length: Length of time series
        n_features: Number of time series features
        n_nodes: Number of graph nodes
        node_feature_dim: Dimension of node features
        
    Returns:
        Tuple of (time_series, graph_features, adjacency_matrix, targets)
    """
    # Generate time series data
    time_series = np.random.randn(n_samples, sequence_length, n_features)
    
    # Add some temporal patterns
    for i in range(n_samples):
        trend = np.linspace(0, 1, sequence_length).reshape(-1, 1)
        time_series[i] += trend * np.random.randn(1, n_features)
    
    # Generate graph features
    graph_features = np.random.randn(n_samples, n_nodes, node_feature_dim)
    
    # Generate adjacency matrix (same for all samples)
    adjacency_matrix = np.random.rand(n_nodes, n_nodes)
    adjacency_matrix = (adjacency_matrix > 0.8).astype(float)  # Sparse graph
    adjacency_matrix = (adjacency_matrix + adjacency_matrix.T) / 2  # Make symmetric
    np.fill_diagonal(adjacency_matrix, 0)  # Remove self-loops
    
    # Generate targets (risk scores based on data characteristics)
    targets = np.zeros(n_samples)
    for i in range(n_samples):
        # Risk based on volatility and graph connectivity
        volatility = np.std(time_series[i])
        graph_activity = np.mean(graph_features[i])
        targets[i] = 0.5 * volatility + 0.3 * graph_activity + 0.2 * np.random.randn()
    
    # Normalize targets to [0, 1] range
    targets = (targets - targets.min()) / (targets.max() - targets.min() + 1e-8)
    
    return time_series, graph_features, adjacency_matrix, targets


def split_data(time_series: np.ndarray, graph_features: np.ndarray, 
               adjacency_matrix: np.ndarray, targets: np.ndarray,
               train_ratio: float = 0.7, val_ratio: float = 0.15, 
               test_ratio: float = 0.15) -> Tuple:
    """
    Split data into train, validation, and test sets.
    
    Args:
        time_series: Time series data
        graph_features: Graph features
        adjacency_matrix: Adjacency matrix
        targets: Target values
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    n_samples = len(time_series)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    # Split data
    train_ts = time_series[:n_train]
    train_gf = graph_features[:n_train]
    train_targets = targets[:n_train]
    
    val_ts = time_series[n_train:n_train+n_val]
    val_gf = graph_features[n_train:n_train+n_val]
    val_targets = targets[n_train:n_train+n_val]
    
    test_ts = time_series[n_train+n_val:]
    test_gf = graph_features[n_train+n_val:]
    test_targets = targets[n_train+n_val:]
    
    return (train_ts, train_gf, adjacency_matrix, train_targets,
            val_ts, val_gf, adjacency_matrix, val_targets,
            test_ts, test_gf, adjacency_matrix, test_targets)


def get_data_loaders(train_ratio: float = 0.7, val_ratio: float = 0.15, 
                     test_ratio: float = 0.15, batch_size: int = 32,
                     n_samples: int = 1000) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        batch_size: Batch size
        n_samples: Total number of samples to generate
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Generate data
    time_series, graph_features, adjacency_matrix, targets = generate_synthetic_data(
        n_samples=n_samples,
        sequence_length=config.SEQUENCE_LENGTH,
        n_features=config.N_FEATURES,
        n_nodes=config.N_GRAPH_NODES,
        node_feature_dim=config.N_FEATURES
    )
    
    # Split data
    split_results = split_data(time_series, graph_features, adjacency_matrix, targets,
                               train_ratio, val_ratio, test_ratio)
    
    train_ts, train_gf, train_adj, train_targets = split_results[:4]
    val_ts, val_gf, val_adj, val_targets = split_results[4:8]
    test_ts, test_gf, test_adj, test_targets = split_results[8:]
    
    # Create datasets
    train_dataset = FinancialDataset(train_ts, train_gf, train_adj, train_targets)
    val_dataset = FinancialDataset(val_ts, val_gf, val_adj, val_targets)
    test_dataset = FinancialDataset(test_ts, test_gf, test_adj, test_targets)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
