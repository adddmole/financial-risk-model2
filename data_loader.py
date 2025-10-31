"""
Data loading and preprocessing for Financial Risk Model
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os
from config import (
    DATA_FILE, PROCESSED_DATA_FILE, TRAIN_RATIO, 
    VAL_RATIO, TEST_RATIO, RANDOM_SEED
)


class FinancialDataset(Dataset):
    """Custom dataset for financial data"""
    
    def __init__(self, features, labels, graph_data=None):
        """
        Args:
            features: Sequential features for Transformer (batch, seq_len, features)
            labels: Target labels
            graph_data: Graph structure for GNN (optional)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.graph_data = graph_data
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sample = {
            'features': self.features[idx],
            'label': self.labels[idx]
        }
        if self.graph_data is not None:
            sample['graph'] = self.graph_data[idx]
        return sample


def generate_synthetic_data(num_samples=1000, num_features=10, seq_length=30):
    """
    Generate synthetic financial data for demonstration
    
    Args:
        num_samples: Number of samples
        num_features: Number of features per timestep
        seq_length: Length of time series
    
    Returns:
        features, labels, adjacency_matrix
    """
    np.random.seed(RANDOM_SEED)
    
    # Generate time series features
    features = np.random.randn(num_samples, seq_length, num_features)
    
    # Add some trend and patterns
    for i in range(num_samples):
        trend = np.linspace(0, 1, seq_length).reshape(-1, 1)
        features[i] += trend * np.random.randn(1, num_features) * 0.5
        
        # Add seasonality
        seasonality = np.sin(np.linspace(0, 4*np.pi, seq_length)).reshape(-1, 1)
        features[i] += seasonality * np.random.randn(1, num_features) * 0.3
    
    # Generate labels (risk scores between 0 and 1)
    # Risk is influenced by mean, variance, and trend of features
    mean_features = features.mean(axis=(1, 2))
    var_features = features.var(axis=(1, 2))
    trend_features = (features[:, -1, :] - features[:, 0, :]).mean(axis=1)
    
    labels = (
        0.3 * (mean_features - mean_features.mean()) / mean_features.std() +
        0.3 * (var_features - var_features.mean()) / var_features.std() +
        0.4 * (trend_features - trend_features.mean()) / trend_features.std()
    )
    
    # Normalize to [0, 1]
    labels = (labels - labels.min()) / (labels.max() - labels.min())
    
    # Generate adjacency matrix for GNN (correlation-based)
    # Create graph connections based on feature correlations
    adjacency_matrices = []
    for i in range(num_samples):
        # Use last timestep for graph construction
        last_step = features[i, -1, :]
        corr_matrix = np.abs(np.corrcoef(last_step.reshape(1, -1).T))
        # Threshold to create sparse graph
        adj = (corr_matrix > 0.5).astype(float)
        np.fill_diagonal(adj, 1)
        adjacency_matrices.append(adj)
    
    adjacency_matrices = np.array(adjacency_matrices)
    
    return features, labels, adjacency_matrices


def preprocess_data(features, labels, scaler=None):
    """
    Preprocess features using standardization
    
    Args:
        features: Raw features
        labels: Labels
        scaler: Existing scaler (optional)
    
    Returns:
        preprocessed_features, labels, scaler
    """
    original_shape = features.shape
    # Reshape to 2D for scaling
    features_2d = features.reshape(-1, features.shape[-1])
    
    if scaler is None:
        scaler = StandardScaler()
        features_2d = scaler.fit_transform(features_2d)
    else:
        features_2d = scaler.transform(features_2d)
    
    # Reshape back
    features = features_2d.reshape(original_shape)
    
    return features, labels, scaler


def load_and_split_data(batch_size=32):
    """
    Load or generate data and split into train/val/test sets
    
    Returns:
        train_loader, val_loader, test_loader, scaler, graph_data
    """
    # Check if processed data exists
    if os.path.exists(PROCESSED_DATA_FILE):
        print(f"Loading processed data from {PROCESSED_DATA_FILE}")
        with open(PROCESSED_DATA_FILE, 'rb') as f:
            data = pickle.load(f)
            features = data['features']
            labels = data['labels']
            adjacency_matrices = data['adjacency_matrices']
            scaler = data['scaler']
    else:
        print("Generating synthetic data...")
        features, labels, adjacency_matrices = generate_synthetic_data()
        
        # Preprocess
        features, labels, scaler = preprocess_data(features, labels)
        
        # Save processed data
        os.makedirs(os.path.dirname(PROCESSED_DATA_FILE), exist_ok=True)
        with open(PROCESSED_DATA_FILE, 'wb') as f:
            pickle.dump({
                'features': features,
                'labels': labels,
                'adjacency_matrices': adjacency_matrices,
                'scaler': scaler
            }, f)
        print(f"Processed data saved to {PROCESSED_DATA_FILE}")
    
    # Split data
    train_size = int(len(features) * TRAIN_RATIO)
    val_size = int(len(features) * VAL_RATIO)
    
    indices = np.arange(len(features))
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(indices)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    # Create datasets
    train_dataset = FinancialDataset(
        features[train_idx], 
        labels[train_idx],
        adjacency_matrices[train_idx]
    )
    val_dataset = FinancialDataset(
        features[val_idx], 
        labels[val_idx],
        adjacency_matrices[val_idx]
    )
    test_dataset = FinancialDataset(
        features[test_idx], 
        labels[test_idx],
        adjacency_matrices[test_idx]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    print(f"Data split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, scaler, adjacency_matrices


def get_sample_for_shap(dataset, num_samples=100):
    """Get sample data for SHAP analysis"""
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    samples = []
    for idx in indices:
        samples.append(dataset[idx])
    return samples
