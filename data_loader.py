"""
Data loader for financial risk modeling.
Handles time series data and graph structure creation.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from typing import Tuple, List, Optional, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import networkx as nx

from config import Config


class FinancialDataset(Dataset):
    """Dataset for financial time series and graph data."""
    
    def __init__(self, 
                 temporal_data: np.ndarray,
                 graph_data: Data,
                 labels: np.ndarray,
                 entity_ids: Optional[np.ndarray] = None):
        """
        Initialize Financial Dataset.
        
        Args:
            temporal_data: Temporal features (num_samples, seq_len, num_features)
            graph_data: PyTorch Geometric Data object
            labels: Target labels (num_samples,)
            entity_ids: Optional entity identifiers (num_samples,)
        """
        self.temporal_data = torch.FloatTensor(temporal_data)
        self.graph_data = graph_data
        self.labels = torch.LongTensor(labels)
        self.entity_ids = entity_ids
        
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Data, torch.Tensor]:
        """Get a single data sample."""
        return self.temporal_data[idx], self.graph_data, self.labels[idx]


def create_correlation_graph(features: np.ndarray, 
                            threshold: float = 0.7,
                            method: str = 'pearson') -> Tuple[np.ndarray, np.ndarray]:
    """
    Create graph edges based on feature correlations.
    
    Args:
        features: Feature matrix (num_nodes, num_features)
        threshold: Correlation threshold for edge creation
        method: Correlation method ('pearson' or 'spearman')
        
    Returns:
        Tuple of (edge_index, edge_weights)
    """
    # Calculate correlation matrix
    if method == 'pearson':
        corr_matrix = np.corrcoef(features)
    elif method == 'spearman':
        from scipy.stats import spearmanr
        corr_matrix, _ = spearmanr(features, axis=1)
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    
    # Get edges above threshold
    edges = []
    edge_weights = []
    
    num_nodes = features.shape[0]
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if abs(corr_matrix[i, j]) >= threshold:
                edges.append([i, j])
                edges.append([j, i])  # Add both directions for undirected graph
                edge_weights.append(abs(corr_matrix[i, j]))
                edge_weights.append(abs(corr_matrix[i, j]))
    
    if len(edges) == 0:
        # If no edges found, create self-loops
        edges = [[i, i] for i in range(num_nodes)]
        edge_weights = [1.0] * num_nodes
    
    edge_index = np.array(edges).T
    edge_weights = np.array(edge_weights)
    
    return edge_index, edge_weights


def create_knn_graph(features: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create k-nearest neighbors graph.
    
    Args:
        features: Feature matrix (num_nodes, num_features)
        k: Number of nearest neighbors
        
    Returns:
        Tuple of (edge_index, edge_weights)
    """
    from sklearn.neighbors import kneighbors_graph
    
    # Build KNN graph
    knn_graph = kneighbors_graph(features, k, mode='distance', include_self=False)
    
    # Convert to edge list
    edges = []
    edge_weights = []
    
    rows, cols = knn_graph.nonzero()
    for i, j in zip(rows, cols):
        edges.append([i, j])
        # Convert distance to similarity (inverse distance)
        distance = knn_graph[i, j]
        similarity = 1.0 / (1.0 + distance)
        edge_weights.append(similarity)
    
    edge_index = np.array(edges).T
    edge_weights = np.array(edge_weights)
    
    return edge_index, edge_weights


def load_and_preprocess_data(data_path: str,
                            sequence_length: int = 60,
                            num_features: int = 10,
                            graph_method: str = 'correlation',
                            **kwargs) -> Tuple[np.ndarray, Data, np.ndarray]:
    """
    Load and preprocess financial data.
    
    Args:
        data_path: Path to CSV file
        sequence_length: Length of time series sequences
        num_features: Number of temporal features
        graph_method: Method for graph construction ('correlation', 'knn', or 'predefined')
        **kwargs: Additional parameters for graph construction
        
    Returns:
        Tuple of (temporal_data, graph_data, labels)
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Extract temporal features, node features, and labels
    # Expected columns: entity_id, time_step, feature_1, ..., feature_n, node_feature_1, ..., node_feature_m, label
    
    # Get unique entities
    if 'entity_id' in df.columns:
        entity_ids = df['entity_id'].unique()
        num_entities = len(entity_ids)
    else:
        # If no entity_id, treat each row as a separate entity
        num_entities = len(df) // sequence_length
        entity_ids = np.arange(num_entities)
    
    # Initialize arrays
    temporal_data_list = []
    labels_list = []
    node_features_list = []
    
    # Process each entity
    for entity_id in entity_ids:
        if 'entity_id' in df.columns:
            entity_df = df[df['entity_id'] == entity_id].sort_values('time_step')
        else:
            # Take sequence_length rows per entity
            start_idx = int(entity_id) * sequence_length
            end_idx = start_idx + sequence_length
            entity_df = df.iloc[start_idx:end_idx]
        
        # Get temporal features
        temporal_cols = [col for col in df.columns if col.startswith('feature_')]
        if len(temporal_cols) == 0:
            # Use numeric columns except specific ones
            exclude_cols = ['entity_id', 'time_step', 'label']
            temporal_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
            temporal_cols = temporal_cols[:num_features]
        
        temporal_features = entity_df[temporal_cols].values
        
        # Pad or truncate to sequence_length
        if len(temporal_features) < sequence_length:
            padding = np.zeros((sequence_length - len(temporal_features), len(temporal_cols)))
            temporal_features = np.vstack([temporal_features, padding])
        else:
            temporal_features = temporal_features[:sequence_length]
        
        temporal_data_list.append(temporal_features)
        
        # Get label (use the last time step's label or the most common)
        if 'label' in df.columns:
            label = entity_df['label'].iloc[-1] if len(entity_df) > 0 else 0
        else:
            label = 0  # Default label if not provided
        labels_list.append(label)
        
        # Get node features (aggregate temporal features or use separate columns)
        node_feature_cols = [col for col in df.columns if col.startswith('node_feature_')]
        if len(node_feature_cols) > 0:
            node_features = entity_df[node_feature_cols].mean().values
        else:
            # Use mean of temporal features as node features
            node_features = temporal_features.mean(axis=0)
        
        node_features_list.append(node_features)
    
    # Convert to arrays
    temporal_data = np.array(temporal_data_list)  # (num_entities, seq_len, num_features)
    labels = np.array(labels_list)
    node_features = np.array(node_features_list)  # (num_entities, num_node_features)
    
    # Normalize temporal data
    original_shape = temporal_data.shape
    temporal_data_flat = temporal_data.reshape(-1, temporal_data.shape[-1])
    scaler_temporal = StandardScaler()
    temporal_data_flat = scaler_temporal.fit_transform(temporal_data_flat)
    temporal_data = temporal_data_flat.reshape(original_shape)
    
    # Normalize node features
    scaler_node = StandardScaler()
    node_features = scaler_node.fit_transform(node_features)
    
    # Create graph
    if graph_method == 'correlation':
        threshold = kwargs.get('threshold', Config.CORRELATION_THRESHOLD)
        edge_index, edge_weights = create_correlation_graph(node_features, threshold)
    elif graph_method == 'knn':
        k = kwargs.get('k', Config.KNN_K)
        edge_index, edge_weights = create_knn_graph(node_features, k)
    elif graph_method == 'predefined':
        # Load predefined edges from file
        edge_file = kwargs.get('edge_file', None)
        if edge_file and os.path.exists(edge_file):
            edges_df = pd.read_csv(edge_file)
            edge_index = edges_df[['source', 'target']].values.T
            edge_weights = edges_df['weight'].values if 'weight' in edges_df.columns else np.ones(len(edges_df))
        else:
            # Fallback to correlation
            edge_index, edge_weights = create_correlation_graph(node_features, 0.5)
    else:
        raise ValueError(f"Unknown graph construction method: {graph_method}")
    
    # Create PyTorch Geometric Data object
    graph_data = Data(
        x=torch.FloatTensor(node_features),
        edge_index=torch.LongTensor(edge_index),
        edge_attr=torch.FloatTensor(edge_weights).unsqueeze(-1)
    )
    
    return temporal_data, graph_data, labels


def create_data_loaders(temporal_data: np.ndarray,
                       graph_data: Data,
                       labels: np.ndarray,
                       batch_size: int = 32,
                       train_split: float = 0.7,
                       val_split: float = 0.15,
                       test_split: float = 0.15,
                       random_seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        temporal_data: Temporal features
        graph_data: Graph data
        labels: Target labels
        batch_size: Batch size
        train_split: Training set ratio
        val_split: Validation set ratio
        test_split: Test set ratio
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Split data
    num_samples = len(labels)
    indices = np.arange(num_samples)
    
    # First split: train + val vs test
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_split,
        random_state=random_seed,
        stratify=labels
    )
    
    # Second split: train vs val
    val_ratio = val_split / (train_split + val_split)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio,
        random_state=random_seed,
        stratify=labels[train_val_idx]
    )
    
    # Create datasets
    train_dataset = FinancialDataset(
        temporal_data[train_idx],
        graph_data,
        labels[train_idx]
    )
    
    val_dataset = FinancialDataset(
        temporal_data[val_idx],
        graph_data,
        labels[val_idx]
    )
    
    test_dataset = FinancialDataset(
        temporal_data[test_idx],
        graph_data,
        labels[test_idx]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader
