"""
Graph Neural Network model for financial relationship modeling
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """
    Simple Graph Convolution Layer
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """
        Forward pass
        
        Args:
            x: Node features (batch, num_nodes, in_features)
            adj: Adjacency matrix (batch, num_nodes, num_nodes)
        
        Returns:
            Updated node features (batch, num_nodes, out_features)
        """
        # Linear transformation: x @ W
        support = torch.matmul(x, self.weight)  # (batch, num_nodes, out_features)
        
        # Graph convolution: A @ (x @ W)
        output = torch.matmul(adj, support)  # (batch, num_nodes, out_features)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class GNNModel(nn.Module):
    """
    Graph Neural Network for financial relationship modeling
    
    Args:
        node_features: Number of features per node
        hidden_channels: Hidden dimension size
        num_layers: Number of GNN layers
        dropout: Dropout rate
        output_dim: Output dimension
    """
    
    def __init__(
        self,
        node_features=10,
        hidden_channels=64,
        num_layers=3,
        dropout=0.1,
        output_dim=64
    ):
        super(GNNModel, self).__init__()
        
        self.node_features = node_features
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GraphConvolution(node_features, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GraphConvolution(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Last layer
        self.convs.append(GraphConvolution(hidden_channels, output_dim))
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x, adj):
        """
        Forward pass
        
        Args:
            x: Node features (batch, num_nodes, node_features)
            adj: Adjacency matrix (batch, num_nodes, num_nodes)
        
        Returns:
            Graph embedding (batch, output_dim)
        """
        # Apply GNN layers
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, adj)
            # Reshape for batch norm
            batch_size, num_nodes, features = x.shape
            x = x.transpose(1, 2)  # (batch, features, num_nodes)
            x = self.batch_norms[i](x)
            x = x.transpose(1, 2)  # (batch, num_nodes, features)
            x = F.relu(x)
            x = self.dropout_layer(x)
        
        # Last layer without batch norm
        x = self.convs[-1](x, adj)
        
        # Global mean pooling over nodes
        x = x.mean(dim=1)  # (batch, output_dim)
        
        return x
