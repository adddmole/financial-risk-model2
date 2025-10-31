# Financial Risk Model with Transformer, GNN, and SHAP

A comprehensive financial risk modeling system combining Transformer architecture for temporal pattern recognition, Graph Neural Networks (GNN) for entity relationship modeling, and SHAP (SHapley Additive exPlanations) for interpretability.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Quick Start with JupyterLab

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/adddmole/financial-risk-model2.git
cd financial-risk-model2

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install JupyterLab
pip install jupyterlab
```

### 2. Launch JupyterLab

```bash
# Start JupyterLab
jupyter lab
```

This will open JupyterLab in your web browser at `http://localhost:8888`

### 3. Run the Tutorial

1. In JupyterLab, navigate to `tutorial.ipynb`
2. Click on the file to open it
3. Run cells sequentially by pressing `Shift + Enter` or clicking the ▶️ button
4. Follow the step-by-step instructions in the notebook

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [JupyterLab Tutorial](#jupyterlab-tutorial)
  - [Command Line Interface](#command-line-interface)
- [Data Format](#data-format)
- [Model Architecture](#model-architecture)
- [Configuration](#configuration)
- [Results and Evaluation](#results-and-evaluation)
- [SHAP Explanations](#shap-explanations)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **Hybrid Architecture**: Combines Transformer and GNN for comprehensive risk assessment
- **Temporal Modeling**: Multi-head attention for time series pattern recognition
- **Graph Modeling**: Capture entity relationships using GCN, GAT, or GraphSAGE
- **Interpretability**: SHAP values for explainable predictions
- **Flexible Graph Construction**: Multiple methods (correlation, KNN, predefined)
- **Production-Ready**: Modular design, type hints, logging, and error handling
- **Easy to Use**: JupyterLab tutorial and command-line interface
- **GPU Support**: Automatic device detection and optimization

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Financial Risk Model                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐         ┌──────────────────┐          │
│  │   Transformer    │         │       GNN        │          │
│  │   (Temporal)     │         │   (Graph)        │          │
│  │                  │         │                  │          │
│  │ • Multi-head     │         │ • GCN/GAT/SAGE   │          │
│  │   Attention      │         │ • Node Embeddings│          │
│  │ • Positional     │         │ • Edge Features  │          │
│  │   Encoding       │         │ • Graph Pooling  │          │
│  └────────┬─────────┘         └────────┬─────────┘          │
│           │                            │                     │
│           └──────────┬─────────────────┘                     │
│                      ▼                                       │
│            ┌──────────────────┐                              │
│            │  Fusion Layer    │                              │
│            │  (Concat/Attn)   │                              │
│            └────────┬─────────┘                              │
│                     ▼                                        │
│            ┌──────────────────┐                              │
│            │  Classification  │                              │
│            │      Head        │                              │
│            └────────┬─────────┘                              │
│                     ▼                                        │
│            ┌──────────────────┐                              │
│            │  Risk Prediction │                              │
│            │   + SHAP Values  │                              │
│            └──────────────────┘                              │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
financial-risk-model2/
├── models/
│   ├── __init__.py
│   ├── transformer.py      # Transformer model for temporal data
│   ├── gnn.py              # GNN models (GCN, GAT, GraphSAGE)
│   └── hybrid.py           # Hybrid model combining both
├── data/
│   ├── README.md           # Data format documentation
│   ├── generate_sample_data.py  # Generate synthetic data
│   └── sample_data.csv     # Sample dataset (generated)
├── results/
│   └── README.md           # Results documentation
├── checkpoints/            # Model checkpoints (created during training)
├── logs/                   # Training logs (created during training)
├── config.py               # Configuration and hyperparameters
├── data_loader.py          # Data loading and graph construction
├── fix_financial.py        # Financial data preprocessing
├── train.py                # Training pipeline
├── main.py                 # Main execution script
├── utils.py                # Utility functions and SHAP integration
├── tutorial.ipynb          # JupyterLab tutorial notebook
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)

### Step-by-Step Installation

```bash
# 1. Clone the repository
git clone https://github.com/adddmole/financial-risk-model2.git
cd financial-risk-model2

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install PyTorch (choose based on your CUDA version)
# For CUDA 11.8:
pip install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
pip install torch==2.0.1 torchvision torchaudio

# 4. Install PyTorch Geometric
pip install torch-geometric==2.3.1

# 5. Install other dependencies
pip install -r requirements.txt

# 6. Install JupyterLab (for interactive use)
pip install jupyterlab
```

### Verify Installation

```python
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

## 📖 Usage

### JupyterLab Tutorial (Recommended for Beginners)

The easiest way to get started is using the interactive JupyterLab tutorial:

```bash
# Launch JupyterLab
jupyter lab

# Open tutorial.ipynb in the browser
# Run cells step-by-step with Shift+Enter
```

The tutorial covers:
1. ✅ Setup and installation verification
2. ✅ Generating sample financial data
3. ✅ Data exploration and visualization
4. ✅ Loading and preprocessing data
5. ✅ Creating the hybrid model
6. ✅ Training the model
7. ✅ Evaluation and metrics
8. ✅ Making predictions
9. ✅ SHAP explanations for interpretability
10. ✅ Using custom data

### Command Line Interface

For advanced users or production deployment:

#### 1. Generate Sample Data

```bash
# Generate synthetic financial data
python data/generate_sample_data.py
```

#### 2. Preprocess Data (Optional)

```bash
# Preprocess financial data with technical indicators
python main.py --mode preprocess \
    --data-path ./data/raw_data.csv \
    --output-path ./data/processed_data.csv
```

#### 3. Train Model

```bash
# Train with default settings
python main.py --mode train --data-path ./data/sample_data.csv

# Train with custom settings
python main.py --mode train \
    --data-path ./data/sample_data.csv \
    --batch-size 32 \
    --epochs 50 \
    --lr 0.001 \
    --graph-method correlation \
    --correlation-threshold 0.7
```

#### 4. Evaluate Model

```bash
# Evaluate trained model
python main.py --mode evaluate \
    --data-path ./data/sample_data.csv \
    --load-checkpoint ./checkpoints/best_model.pth \
    --explain  # Generate SHAP explanations
```

#### 5. Make Predictions

```bash
# Predict on new data
python main.py --mode predict \
    --data-path ./data/new_data.csv \
    --load-checkpoint ./checkpoints/best_model.pth \
    --save-dir ./results
```

## 📊 Data Format

Your data should be in CSV format with the following structure:

### Required Columns

| Column | Description | Type |
|--------|-------------|------|
| `entity_id` | Unique identifier for each entity | int |
| `time_step` | Time step index (0, 1, 2, ...) | int |
| `feature_1` ... `feature_n` | Temporal features | float |
| `label` | Risk label (0: low risk, 1: high risk) | int |

### Optional Columns

| Column | Description | Type |
|--------|-------------|------|
| `node_feature_1` ... `node_feature_m` | Graph node features | float |
| `price` | Asset price (for technical indicators) | float |
| `volume` | Trading volume | float |
| `returns` | Returns | float |
| `volatility` | Volatility measure | float |

### Example

```csv
entity_id,time_step,feature_1,feature_2,price,volume,label
0,0,0.5,1.2,100.0,1000000,0
0,1,0.6,1.1,101.5,1100000,0
0,2,0.4,1.3,99.8,950000,0
1,0,-0.2,0.8,50.0,500000,1
1,1,-0.3,0.9,49.5,480000,1
```

See `data/README.md` for detailed specifications.

## 🧠 Model Architecture

### Transformer Component

- **Input**: Temporal features (batch_size, sequence_length, num_features)
- **Positional Encoding**: Sine/cosine encoding for time series
- **Multi-Head Attention**: Captures temporal dependencies
- **Feed-Forward Networks**: Non-linear transformations
- **Output**: Sequence embedding (batch_size, transformer_dim)

### GNN Component

- **Input**: Node features and edge indices
- **Graph Convolution**: GCN, GAT, or GraphSAGE layers
- **Node Embeddings**: Learn entity representations
- **Global Pooling**: Aggregate to graph-level embedding
- **Output**: Graph embedding (batch_size, gnn_dim)

### Fusion and Classification

- **Fusion Methods**:
  - Concatenation: Simple concatenation of embeddings
  - Attention: Learned attention weights for combining
- **Classification Head**: Multi-layer perceptron with dropout
- **Output**: Risk probabilities (batch_size, num_classes)

## ⚙️ Configuration

Edit `config.py` to customize hyperparameters:

```python
# Model parameters
TRANSFORMER_DIM = 128           # Transformer hidden dimension
TRANSFORMER_HEADS = 8           # Number of attention heads
TRANSFORMER_LAYERS = 4          # Number of transformer layers
GNN_HIDDEN_DIM = 128           # GNN hidden dimension
GNN_LAYERS = 3                 # Number of GNN layers
GNN_TYPE = 'GAT'               # GNN type: 'GCN', 'GAT', 'GraphSAGE'

# Training parameters
BATCH_SIZE = 32                # Batch size
LEARNING_RATE = 0.001          # Learning rate
NUM_EPOCHS = 100               # Number of epochs
EARLY_STOPPING_PATIENCE = 10   # Early stopping patience

# Data parameters
SEQUENCE_LENGTH = 60           # Time series sequence length
NUM_FEATURES = 10              # Number of temporal features
GRAPH_CONSTRUCTION_METHOD = 'correlation'  # Graph construction
CORRELATION_THRESHOLD = 0.7    # Correlation threshold for edges
```

## 📈 Results and Evaluation

After training, results are saved in the `results/` directory:

### Metrics

- **Accuracy**: Overall prediction accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the ROC curve

### Visualizations

- `training_history.png`: Training and validation curves
- `confusion_matrix.png`: Confusion matrix heatmap
- `shap_summary.png`: SHAP feature importance
- `shap_waterfall.png`: SHAP explanation for individual predictions

## 🔍 SHAP Explanations

SHAP (SHapley Additive exPlanations) provides interpretable predictions:

```python
from utils import SHAPExplainer

# Create explainer
explainer = SHAPExplainer(model, background_data, device)

# Explain predictions
shap_values, base_values = explainer.explain_instance(temporal_features, graph_data)

# Visualize
explainer.plot_summary(shap_values, features, save_path='shap_summary.png')
explainer.plot_waterfall(shap_values, features, instance_idx=0, save_path='shap_waterfall.png')
```

## 💡 Examples

### Example 1: Quick Training in Jupyter

```python
from config import Config, set_seed
from data_loader import load_and_preprocess_data, create_data_loaders
from models.hybrid import HybridRiskModel
from train import Trainer

# Set seed
set_seed(42)

# Load data
temporal_data, graph_data, labels = load_and_preprocess_data(
    data_path='./data/sample_data.csv',
    sequence_length=60,
    num_features=10
)

# Create data loaders
train_loader, val_loader, test_loader = create_data_loaders(
    temporal_data, graph_data, labels, batch_size=32
)

# Create model
model = HybridRiskModel(
    num_temporal_features=10,
    sequence_length=60,
    num_node_features=graph_data.x.shape[1],
    num_classes=2
)

# Train
trainer = Trainer(model, train_loader, val_loader, test_loader)
history = trainer.train()

# Evaluate
metrics = trainer.evaluate()
```

### Example 2: Using Different Graph Construction Methods

```python
# Correlation-based graph
temporal_data, graph_data, labels = load_and_preprocess_data(
    data_path='./data/sample_data.csv',
    graph_method='correlation',
    threshold=0.7
)

# KNN-based graph
temporal_data, graph_data, labels = load_and_preprocess_data(
    data_path='./data/sample_data.csv',
    graph_method='knn',
    k=10
)

# Predefined graph
temporal_data, graph_data, labels = load_and_preprocess_data(
    data_path='./data/sample_data.csv',
    graph_method='predefined',
    edge_file='./data/edges.csv'
)
```

### Example 3: Custom Model Configuration

```python
# Create model with custom architecture
model = HybridRiskModel(
    num_temporal_features=20,
    sequence_length=100,
    transformer_dim=256,
    transformer_heads=16,
    transformer_layers=6,
    gnn_hidden_dim=256,
    gnn_layers=4,
    gnn_type='GraphSAGE',
    fusion_method='attention',
    num_classes=3  # Multi-class classification
)
```

## 🎓 Tutorial Workflow in JupyterLab

1. **Open JupyterLab**: Run `jupyter lab` in terminal
2. **Open Tutorial**: Click on `tutorial.ipynb`
3. **Run Setup**: Execute first cells to verify installation
4. **Generate Data**: Create synthetic financial data
5. **Explore Data**: Visualize time series and distributions
6. **Train Model**: Run training loop with progress bars
7. **Evaluate**: View metrics and confusion matrix
8. **Interpret**: Generate and visualize SHAP explanations
9. **Experiment**: Modify parameters and re-run

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 📚 References

- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Transformer Architecture
- [Graph Attention Networks](https://arxiv.org/abs/1710.10903) - GAT
- [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874) - SHAP

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Happy Risk Modeling! 🚀📊**