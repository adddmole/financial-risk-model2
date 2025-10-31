"""
Utility functions for the financial risk model.
"""

import torch
import numpy as np
import random
import json
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple


def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_model(model: torch.nn.Module, path: str):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        path: Save path
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model: torch.nn.Module, path: str, device: torch.device):
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        path: Load path
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Model loaded from {path}")
    return model


def save_results(results: Dict[str, Any], path: str):
    """
    Save results to JSON file.
    
    Args:
        results: Dictionary of results
        path: Save path
    """
    with open(path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {path}")


def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        
    Returns:
        Dictionary of metrics
    """
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))
    
    # Calculate R-squared
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2)
    }


def plot_training_history(train_losses: list, val_losses: list, save_path: str):
    """
    Plot training and validation loss history.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Training history plot saved to {save_path}")


def create_adjacency_matrix(n_nodes: int, edge_prob: float = 0.1) -> torch.Tensor:
    """
    Create a random adjacency matrix for GNN.
    
    Args:
        n_nodes: Number of nodes
        edge_prob: Probability of edge between nodes
        
    Returns:
        Adjacency matrix
    """
    adj_matrix = torch.rand(n_nodes, n_nodes) < edge_prob
    adj_matrix = adj_matrix.float()
    # Make symmetric and remove self-loops
    adj_matrix = (adj_matrix + adj_matrix.t()) / 2
    adj_matrix.fill_diagonal_(0)
    return adj_matrix


def normalize_adjacency(adj_matrix: torch.Tensor) -> torch.Tensor:
    """
    Normalize adjacency matrix for GNN.
    
    Args:
        adj_matrix: Adjacency matrix
        
    Returns:
        Normalized adjacency matrix
    """
    # Add self-loops
    adj_matrix = adj_matrix + torch.eye(adj_matrix.size(0))
    # Calculate degree matrix
    degree = torch.sum(adj_matrix, dim=1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
    # Normalize
    degree_matrix = torch.diag(degree_inv_sqrt)
    normalized_adj = torch.mm(torch.mm(degree_matrix, adj_matrix), degree_matrix)
    return normalized_adj
