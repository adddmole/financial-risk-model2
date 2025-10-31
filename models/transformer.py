"""
Transformer model for temporal financial data processing.
Implements multi-head attention mechanism for time series pattern recognition.
"""
import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """Positional encoding for time series data."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Dimension of the model
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    Transformer model for financial time series analysis.
    Uses multi-head self-attention to capture temporal patterns.
    """
    
    def __init__(self, 
                 num_features: int,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 max_seq_length: int = 5000):
        """
        Initialize Transformer model.
        
        Args:
            num_features: Number of input features
            d_model: Dimension of the model
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
        """
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        self.num_features = num_features
        
        # Input projection
        self.input_projection = nn.Linear(num_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Use batch_first format
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the Transformer model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Project input to d_model dimensions
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Scale by sqrt(d_model) as per "Attention is All You Need"
        x = x * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Apply layer normalization
        x = self.norm(x)
        
        return x
    
    def get_sequence_embedding(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get sequence-level embedding by averaging over time.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)
            mask: Optional attention mask
            
        Returns:
            Sequence embedding of shape (batch_size, d_model)
        """
        # Get transformer output
        output = self.forward(x, mask)  # (batch_size, seq_len, d_model)
        
        # Average pooling over sequence length
        if mask is not None:
            # Apply mask before averaging
            mask_expanded = (~mask).unsqueeze(-1).float()
            output = output * mask_expanded
            seq_embedding = output.sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            seq_embedding = output.mean(dim=1)  # (batch_size, d_model)
        
        return seq_embedding


class TransformerClassifier(nn.Module):
    """
    Standalone Transformer classifier for temporal financial data.
    Can be used independently without the hybrid architecture.
    """
    
    def __init__(self,
                 num_features: int,
                 num_classes: int,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 max_seq_length: int = 5000):
        """
        Initialize Transformer classifier.
        
        Args:
            num_features: Number of input features
            num_classes: Number of output classes
            d_model: Dimension of the model
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
        """
        super(TransformerClassifier, self).__init__()
        
        # Transformer encoder
        self.transformer = TransformerModel(
            num_features=num_features,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_length=max_seq_length
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)
            mask: Optional attention mask
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Get sequence embedding
        seq_embedding = self.transformer.get_sequence_embedding(x, mask)
        
        # Classify
        logits = self.classifier(seq_embedding)
        
        return logits
