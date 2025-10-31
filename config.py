"""
Configuration file for the financial risk model.
Contains all hyperparameters and settings for the Transformer, GNN, and SHAP models.
"""

import torch

# General settings
RANDOM_SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data settings
DATA_DIR = 'data/'
MODELS_DIR = 'models/'
RESULTS_DIR = 'results/'
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Feature settings
N_FEATURES = 20  # Number of input features
SEQUENCE_LENGTH = 30  # Time series sequence length
N_GRAPH_NODES = 50  # Number of nodes in the graph (e.g., stocks, entities)

# Transformer settings
TRANSFORMER_D_MODEL = 128  # Dimension of transformer model
TRANSFORMER_NHEAD = 8  # Number of attention heads
TRANSFORMER_NUM_LAYERS = 4  # Number of transformer layers
TRANSFORMER_DIM_FEEDFORWARD = 512  # Dimension of feedforward network
TRANSFORMER_DROPOUT = 0.1

# GNN settings
GNN_HIDDEN_DIM = 64  # Hidden dimension for GNN
GNN_OUTPUT_DIM = 32  # Output dimension for GNN
GNN_NUM_LAYERS = 3  # Number of GNN layers
GNN_DROPOUT = 0.1

# Combined model settings
HIDDEN_DIM = 128  # Combined hidden dimension
OUTPUT_DIM = 1  # Output dimension (risk score)

# Training settings
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
PATIENCE = 10  # Early stopping patience
WEIGHT_DECAY = 1e-5

# SHAP settings
SHAP_BACKGROUND_SIZE = 100  # Number of background samples for SHAP
SHAP_TEST_SIZE = 50  # Number of test samples for SHAP explanation

# Model checkpoint
CHECKPOINT_PATH = f'{MODELS_DIR}best_model.pt'
RESULTS_PATH = f'{RESULTS_DIR}results.json'
SHAP_PLOT_PATH = f'{RESULTS_DIR}shap_explanations.png'
