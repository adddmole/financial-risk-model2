"""
Configuration module for Financial Risk Model.
Contains all hyperparameters and settings for the model.
"""
import torch
from typing import Dict, Any


class Config:
    """Configuration class for model hyperparameters and training settings."""
    
    # Device configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data parameters
    SEQUENCE_LENGTH = 60  # Time steps for temporal features
    NUM_FEATURES = 10  # Number of input features per time step
    NUM_NODE_FEATURES = 20  # Number of features per graph node
    NUM_CLASSES = 2  # Binary classification (default/no default)
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    # Transformer parameters
    TRANSFORMER_DIM = 128  # Hidden dimension for transformer
    TRANSFORMER_HEADS = 8  # Number of attention heads
    TRANSFORMER_LAYERS = 4  # Number of transformer layers
    TRANSFORMER_FF_DIM = 512  # Feed-forward dimension
    TRANSFORMER_DROPOUT = 0.1
    
    # GNN parameters
    GNN_HIDDEN_DIM = 128  # Hidden dimension for GNN
    GNN_LAYERS = 3  # Number of GNN layers
    GNN_HEADS = 4  # Number of attention heads (for GAT)
    GNN_DROPOUT = 0.1
    GNN_TYPE = 'GAT'  # Options: 'GCN', 'GAT', 'GraphSAGE'
    
    # Hybrid model parameters
    HYBRID_HIDDEN_DIM = 256
    HYBRID_DROPOUT = 0.2
    FUSION_METHOD = 'concat'  # Options: 'concat', 'attention'
    
    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0001
    NUM_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 10
    GRADIENT_CLIP = 1.0
    
    # Optimizer parameters
    OPTIMIZER = 'AdamW'  # Options: 'Adam', 'AdamW', 'SGD'
    SCHEDULER = 'ReduceLROnPlateau'  # Options: 'ReduceLROnPlateau', 'CosineAnnealing', 'StepLR'
    SCHEDULER_PATIENCE = 5
    SCHEDULER_FACTOR = 0.5
    
    # Loss function
    LOSS_FUNCTION = 'CrossEntropy'  # Options: 'CrossEntropy', 'BCE', 'MSE', 'FocalLoss'
    CLASS_WEIGHTS = None  # Set to list for imbalanced datasets
    
    # Graph construction parameters
    GRAPH_CONSTRUCTION_METHOD = 'correlation'  # Options: 'correlation', 'knn', 'threshold', 'predefined'
    CORRELATION_THRESHOLD = 0.7
    KNN_K = 10
    
    # Paths
    DATA_DIR = './data'
    RESULTS_DIR = './results'
    CHECKPOINT_DIR = './checkpoints'
    LOG_DIR = './logs'
    
    # SHAP configuration
    SHAP_SAMPLE_SIZE = 100  # Number of samples for SHAP explanation
    SHAP_MAX_DISPLAY = 20  # Maximum features to display in SHAP plots
    
    # Logging
    LOG_INTERVAL = 10  # Log every N batches
    SAVE_CHECKPOINT_EVERY = 5  # Save checkpoint every N epochs
    
    # Random seed for reproducibility
    RANDOM_SEED = 42
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            key: value for key, value in vars(cls).items()
            if not key.startswith('_') and not callable(value)
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
    
    @classmethod
    def print_config(cls) -> None:
        """Print current configuration."""
        print("=" * 50)
        print("Configuration Settings")
        print("=" * 50)
        for key, value in cls.to_dict().items():
            print(f"{key}: {value}")
        print("=" * 50)


# Set random seeds for reproducibility
def set_seed(seed: int = Config.RANDOM_SEED) -> None:
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
