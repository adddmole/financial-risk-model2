"""
Graph Neural Network models for entity relationship modeling.
Implements GCN, GAT, and GraphSAGE architectures.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from typing import Optional


class GNNModel(nn.Module):
    """
    Graph Neural Network for modeling financial entity relationships.
    Supports multiple GNN architectures: GCN, GAT, GraphSAGE.
    """
    
    def __init__(self,
                 num_node_features: int,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 gnn_type: str = 'GAT',
                 num_heads: int = 4):
        """
        Initialize GNN model.
        
        Args:
            num_node_features: Number of input node features
            hidden_dim: Hidden dimension size
            num_layers: Number of GNN layers
            dropout: Dropout rate
            gnn_type: Type of GNN ('GCN', 'GAT', 'GraphSAGE')
            num_heads: Number of attention heads (for GAT)
        """
        super(GNNModel, self).__init__()
        
        self.num_node_features = num_node_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type
        
        # Build GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        if gnn_type == 'GCN':
            self.convs.append(GCNConv(num_node_features, hidden_dim))
        elif gnn_type == 'GAT':
            self.convs.append(GATConv(num_node_features, hidden_dim, heads=num_heads, concat=False))
        elif gnn_type == 'GraphSAGE':
            self.convs.append(SAGEConv(num_node_features, hidden_dim))
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
        
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            if gnn_type == 'GCN':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == 'GAT':
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False))
            elif gnn_type == 'GraphSAGE':
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass of GNN.
        
        Args:
            data: PyTorch Geometric Data object containing:
                - x: Node features (num_nodes, num_node_features)
                - edge_index: Edge indices (2, num_edges)
                - batch: Batch assignment (num_nodes,)
                
        Returns:
            Node embeddings of shape (num_nodes, hidden_dim)
        """
        x, edge_index = data.x, data.edge_index
        
        # Apply GNN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout_layer(x)
        
        return x
    
    def get_graph_embedding(self, data: Data, pooling: str = 'mean') -> torch.Tensor:
        """
        Get graph-level embedding using global pooling.
        
        Args:
            data: PyTorch Geometric Data object
            pooling: Pooling method ('mean' or 'max')
            
        Returns:
            Graph embedding of shape (batch_size, hidden_dim)
        """
        # Get node embeddings
        node_embeddings = self.forward(data)
        
        # Global pooling
        if pooling == 'mean':
            graph_embedding = global_mean_pool(node_embeddings, data.batch)
        elif pooling == 'max':
            graph_embedding = global_max_pool(node_embeddings, data.batch)
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
        
        return graph_embedding


class GNNClassifier(nn.Module):
    """
    Standalone GNN classifier for graph-based financial data.
    Can be used independently without the hybrid architecture.
    """
    
    def __init__(self,
                 num_node_features: int,
                 num_classes: int,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 gnn_type: str = 'GAT',
                 num_heads: int = 4,
                 pooling: str = 'mean'):
        """
        Initialize GNN classifier.
        
        Args:
            num_node_features: Number of input node features
            num_classes: Number of output classes
            hidden_dim: Hidden dimension size
            num_layers: Number of GNN layers
            dropout: Dropout rate
            gnn_type: Type of GNN ('GCN', 'GAT', 'GraphSAGE')
            num_heads: Number of attention heads (for GAT)
            pooling: Global pooling method ('mean' or 'max')
        """
        super(GNNClassifier, self).__init__()
        
        # GNN encoder
        self.gnn = GNNModel(
            num_node_features=num_node_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            gnn_type=gnn_type,
            num_heads=num_heads
        )
        
        self.pooling = pooling
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Get graph embedding
        graph_embedding = self.gnn.get_graph_embedding(data, self.pooling)
        
        # Classify
        logits = self.classifier(graph_embedding)
        
        return logits


class EdgePredictor(nn.Module):
    """
    Edge prediction module for dynamic graph construction.
    Can be used to learn edge weights or predict new edges.
    """
    
    def __init__(self, node_dim: int, hidden_dim: int = 64):
        """
        Initialize edge predictor.
        
        Args:
            node_dim: Dimension of node embeddings
            hidden_dim: Hidden dimension for edge prediction
        """
        super(EdgePredictor, self).__init__()
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, node_embeddings: torch.Tensor, 
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Predict edge weights.
        
        Args:
            node_embeddings: Node embeddings (num_nodes, node_dim)
            edge_index: Edge indices (2, num_edges)
            
        Returns:
            Edge weights (num_edges,)
        """
        # Get source and target node embeddings
        src_embeddings = node_embeddings[edge_index[0]]
        dst_embeddings = node_embeddings[edge_index[1]]
        
        # Concatenate embeddings
        edge_features = torch.cat([src_embeddings, dst_embeddings], dim=1)
        
        # Predict edge weights
        edge_weights = self.edge_mlp(edge_features).squeeze(-1)
        
        return edge_weights
