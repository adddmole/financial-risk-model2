"""
Hybrid model combining Transformer and GNN for financial risk prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import TransformerModel
from .gnn import GNNModel


class HybridRiskModel(nn.Module):
    """
    Hybrid model combining Transformer (temporal) and GNN (relational)
    
    Args:
        transformer_config: Configuration dict for Transformer
        gnn_config: Configuration dict for GNN
        fusion_dim: Dimension for feature fusion
        output_dim: Final output dimension (1 for risk score)
    """
    
    def __init__(
        self,
        transformer_config,
        gnn_config,
        fusion_dim=64,
        output_dim=1
    ):
        super(HybridRiskModel, self).__init__()
        
        # Extract input dimensions
        input_dim = transformer_config.get('input_dim', gnn_config['node_features'])
        
        # Transformer for temporal modeling
        self.transformer = TransformerModel(
            input_dim=input_dim,
            d_model=transformer_config['d_model'],
            nhead=transformer_config['nhead'],
            num_layers=transformer_config['num_layers'],
            dim_feedforward=transformer_config['dim_feedforward'],
            dropout=transformer_config['dropout'],
            output_dim=fusion_dim
        )
        
        # GNN for relational modeling
        self.gnn = GNNModel(
            node_features=gnn_config['node_features'],
            hidden_channels=gnn_config['hidden_channels'],
            num_layers=gnn_config['num_layers'],
            dropout=gnn_config['dropout'],
            output_dim=fusion_dim
        )
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Output layer for risk prediction
        self.output_layer = nn.Linear(fusion_dim // 2, output_dim)
        
    def forward(self, features, adj_matrix):
        """
        Forward pass
        
        Args:
            features: Sequential features (batch, seq_len, feature_dim)
            adj_matrix: Adjacency matrix (batch, num_nodes, num_nodes)
        
        Returns:
            Risk score (batch, 1)
        """
        # Transformer encoding (temporal patterns)
        transformer_out = self.transformer(features)  # (batch, fusion_dim)
        
        # Extract node features from last timestep for GNN
        # Use last timestep features as node features
        node_features = features[:, -1, :]  # (batch, feature_dim)
        
        # Expand to create node features for graph
        # Assume each feature dimension is a node
        batch_size = features.size(0)
        num_nodes = node_features.size(1)
        
        # Reshape to (batch, num_nodes, 1) and expand
        node_features = node_features.unsqueeze(-1)  # (batch, num_nodes, 1)
        
        # Pad or adjust if needed to match node_features dimension
        if node_features.size(1) != adj_matrix.size(1):
            # Interpolate or pad
            if node_features.size(1) < adj_matrix.size(1):
                padding = torch.zeros(
                    batch_size, 
                    adj_matrix.size(1) - node_features.size(1), 
                    1,
                    device=features.device
                )
                node_features = torch.cat([node_features, padding], dim=1)
            else:
                node_features = node_features[:, :adj_matrix.size(1), :]
        
        # GNN encoding (relational patterns)
        gnn_out = self.gnn(node_features, adj_matrix)  # (batch, fusion_dim)
        
        # Concatenate features
        combined = torch.cat([transformer_out, gnn_out], dim=-1)  # (batch, fusion_dim*2)
        
        # Fusion
        fused = self.fusion(combined)  # (batch, fusion_dim//2)
        
        # Output prediction
        output = self.output_layer(fused)  # (batch, 1)
        output = torch.sigmoid(output)  # Risk score in [0, 1]
        
        return output
    
    def get_embeddings(self, features, adj_matrix):
        """
        Get intermediate embeddings for SHAP analysis
        
        Returns:
            Dictionary of embeddings
        """
        with torch.no_grad():
            transformer_out = self.transformer(features)
            
            node_features = features[:, -1, :].unsqueeze(-1)
            batch_size = features.size(0)
            
            if node_features.size(1) != adj_matrix.size(1):
                if node_features.size(1) < adj_matrix.size(1):
                    padding = torch.zeros(
                        batch_size, 
                        adj_matrix.size(1) - node_features.size(1), 
                        1,
                        device=features.device
                    )
                    node_features = torch.cat([node_features, padding], dim=1)
                else:
                    node_features = node_features[:, :adj_matrix.size(1), :]
            
            gnn_out = self.gnn(node_features, adj_matrix)
            
            return {
                'transformer': transformer_out,
                'gnn': gnn_out
            }
