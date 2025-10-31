"""
Configuration file for Financial Risk Model
"""
import os

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Data configuration
DATA_FILE = os.path.join(DATA_DIR, 'financial_data.csv')
PROCESSED_DATA_FILE = os.path.join(DATA_DIR, 'processed_data.pkl')
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Model configuration
RANDOM_SEED = 42
DEVICE = 'cpu'  # 'cuda' if GPU available

# Transformer configuration
TRANSFORMER_CONFIG = {
    'd_model': 64,
    'nhead': 4,
    'num_layers': 2,
    'dim_feedforward': 128,
    'dropout': 0.1,
    'seq_length': 30,
}

# GNN configuration
GNN_CONFIG = {
    'hidden_channels': 64,
    'num_layers': 3,
    'dropout': 0.1,
    'node_features': 10,
}

# Hybrid model configuration
HYBRID_CONFIG = {
    'output_dim': 1,  # Binary classification or regression
    'fusion_dim': 64,
}

# Training configuration
TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'early_stopping_patience': 10,
}

# SHAP configuration
SHAP_CONFIG = {
    'num_samples': 100,
    'max_display': 20,
}

# Risk thresholds
RISK_THRESHOLDS = {
    'low': 0.3,
    'medium': 0.6,
    'high': 1.0,
}
