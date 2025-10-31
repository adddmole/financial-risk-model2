"""
Hybrid model combining Transformer and GNN for financial risk prediction.
Fuses temporal and graph-based representations for comprehensive risk assessment.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import Optional, Tuple

from .transformer import TransformerModel
from .gnn import GNNModel


class AttentionFusion(nn.Module):
    """Attention-based fusion module for combining Transformer and GNN outputs."""
    
    def __init__(self, transformer_dim: int, gnn_dim: int, hidden_dim: int):
        """
        Initialize attention fusion module.
        
        Args:
            transformer_dim: Dimension of transformer output
            gnn_dim: Dimension of GNN output
            hidden_dim: Hidden dimension for fusion
        """
        super(AttentionFusion, self).__init__()
        
        # Project both inputs to same dimension
        self.transformer_proj = nn.Linear(transformer_dim, hidden_dim)
        self.gnn_proj = nn.Linear(gnn_dim, hidden_dim)
        
        # Attention weights
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1)
        )
    
    def forward(self, transformer_out: torch.Tensor, 
                gnn_out: torch.Tensor) -> torch.Tensor:
        """
        Fuse transformer and GNN outputs using attention.
        
        Args:
            transformer_out: Transformer output (batch_size, transformer_dim)
            gnn_out: GNN output (batch_size, gnn_dim)
            
        Returns:
            Fused representation (batch_size, hidden_dim)
        """
        # Project to same dimension
        t_proj = self.transformer_proj(transformer_out)
        g_proj = self.gnn_proj(gnn_out)
        
        # Compute attention weights
        combined = torch.cat([t_proj, g_proj], dim=1)
        weights = self.attention(combined)  # (batch_size, 2)
        
        # Apply attention weights
        t_weighted = t_proj * weights[:, 0:1]
        g_weighted = g_proj * weights[:, 1:2]
        
        # Combine
        fused = t_weighted + g_weighted
        
        return fused


class HybridRiskModel(nn.Module):
    """
    Hybrid model combining Transformer for temporal patterns and GNN for entity relationships.
    Provides comprehensive risk assessment using both time series and graph structure.
    """
    
    def __init__(self,
                 # Temporal parameters
                 num_temporal_features: int,
                 sequence_length: int,
                 transformer_dim: int = 128,
                 transformer_heads: int = 8,
                 transformer_layers: int = 4,
                 transformer_ff_dim: int = 512,
                 transformer_dropout: float = 0.1,
                 # Graph parameters
                 num_node_features: int = 20,
                 gnn_hidden_dim: int = 128,
                 gnn_layers: int = 3,
                 gnn_dropout: float = 0.1,
                 gnn_type: str = 'GAT',
                 gnn_heads: int = 4,
                 # Fusion parameters
                 fusion_method: str = 'concat',
                 hybrid_hidden_dim: int = 256,
                 hybrid_dropout: float = 0.2,
                 # Output parameters
                 num_classes: int = 2):
        """
        Initialize Hybrid Risk Model.
        
        Args:
            num_temporal_features: Number of temporal features per time step
            sequence_length: Length of temporal sequence
            transformer_dim: Transformer hidden dimension
            transformer_heads: Number of transformer attention heads
            transformer_layers: Number of transformer layers
            transformer_ff_dim: Transformer feed-forward dimension
            transformer_dropout: Transformer dropout rate
            num_node_features: Number of graph node features
            gnn_hidden_dim: GNN hidden dimension
            gnn_layers: Number of GNN layers
            gnn_dropout: GNN dropout rate
            gnn_type: Type of GNN ('GCN', 'GAT', 'GraphSAGE')
            gnn_heads: Number of GNN attention heads (for GAT)
            fusion_method: Method to fuse outputs ('concat' or 'attention')
            hybrid_hidden_dim: Hidden dimension for fusion layer
            hybrid_dropout: Dropout rate for hybrid layers
            num_classes: Number of output classes
        """
        super(HybridRiskModel, self).__init__()
        
        self.fusion_method = fusion_method
        self.num_classes = num_classes
        
        # Transformer for temporal data
        self.transformer = TransformerModel(
            num_features=num_temporal_features,
            d_model=transformer_dim,
            nhead=transformer_heads,
            num_layers=transformer_layers,
            dim_feedforward=transformer_ff_dim,
            dropout=transformer_dropout,
            max_seq_length=sequence_length
        )
        
        # GNN for graph data
        self.gnn = GNNModel(
            num_node_features=num_node_features,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_layers,
            dropout=gnn_dropout,
            gnn_type=gnn_type,
            num_heads=gnn_heads
        )
        
        # Fusion layer
        if fusion_method == 'concat':
            fusion_input_dim = transformer_dim + gnn_hidden_dim
            self.fusion = nn.Identity()
        elif fusion_method == 'attention':
            fusion_input_dim = hybrid_hidden_dim
            self.fusion = AttentionFusion(transformer_dim, gnn_hidden_dim, hybrid_hidden_dim)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, hybrid_hidden_dim),
            nn.ReLU(),
            nn.Dropout(hybrid_dropout),
            nn.BatchNorm1d(hybrid_hidden_dim),
            nn.Linear(hybrid_hidden_dim, hybrid_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(hybrid_dropout),
            nn.Linear(hybrid_hidden_dim // 2, num_classes)
        )
    
    def forward(self, temporal_data: torch.Tensor, 
                graph_data: Data,
                temporal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the hybrid model.
        
        Args:
            temporal_data: Temporal input (batch_size, seq_len, num_features)
            graph_data: PyTorch Geometric Data object
            temporal_mask: Optional mask for temporal data
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Get transformer embedding
        transformer_embedding = self.transformer.get_sequence_embedding(
            temporal_data, temporal_mask
        )  # (batch_size, transformer_dim)
        
        # Get GNN embedding
        gnn_embedding = self.gnn.get_graph_embedding(graph_data)  # (batch_size, gnn_hidden_dim)
        
        # Ensure embeddings have the same batch size
        batch_size = temporal_data.size(0)
        if gnn_embedding.size(0) != batch_size:
            # Replicate GNN embedding if needed (for cases where graph is shared)
            gnn_embedding = gnn_embedding.repeat(batch_size, 1)
        
        # Fuse embeddings
        if self.fusion_method == 'concat':
            fused = torch.cat([transformer_embedding, gnn_embedding], dim=1)
        elif self.fusion_method == 'attention':
            fused = self.fusion(transformer_embedding, gnn_embedding)
        
        # Classification
        logits = self.classifier(fused)
        
        return logits
    
    def get_embeddings(self, temporal_data: torch.Tensor,
                      graph_data: Data,
                      temporal_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get intermediate embeddings for analysis.
        
        Args:
            temporal_data: Temporal input
            graph_data: Graph data
            temporal_mask: Optional temporal mask
            
        Returns:
            Tuple of (transformer_embedding, gnn_embedding, fused_embedding)
        """
        # Get transformer embedding
        transformer_embedding = self.transformer.get_sequence_embedding(
            temporal_data, temporal_mask
        )
        
        # Get GNN embedding
        gnn_embedding = self.gnn.get_graph_embedding(graph_data)
        
        # Ensure same batch size
        batch_size = temporal_data.size(0)
        if gnn_embedding.size(0) != batch_size:
            gnn_embedding = gnn_embedding.repeat(batch_size, 1)
        
        # Fuse embeddings
        if self.fusion_method == 'concat':
            fused = torch.cat([transformer_embedding, gnn_embedding], dim=1)
        elif self.fusion_method == 'attention':
            fused = self.fusion(transformer_embedding, gnn_embedding)
        
        return transformer_embedding, gnn_embedding, fused
    
    def predict_proba(self, temporal_data: torch.Tensor,
                     graph_data: Data,
                     temporal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get probability predictions.
        
        Args:
            temporal_data: Temporal input
            graph_data: Graph data
            temporal_mask: Optional temporal mask
            
        Returns:
            Probabilities of shape (batch_size, num_classes)
        """
        logits = self.forward(temporal_data, graph_data, temporal_mask)
        probs = F.softmax(logits, dim=1)
        return probs
