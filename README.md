# Financial Risk Model

A comprehensive financial risk assessment model combining Transformer networks for time series analysis, Graph Neural Networks (GNN) for relational data, and SHAP (SHapley Additive exPlanations) for model interpretability.

## Features

- **Transformer Encoder**: Processes time series financial data with attention mechanisms
- **Graph Neural Network**: Captures relationships between financial entities
- **Combined Architecture**: Fuses temporal and relational information for risk prediction
- **SHAP Explanations**: Provides interpretable feature importance for model decisions
- **End-to-End Pipeline**: From data loading to model training and evaluation

## Project Structure

```
financial-risk-model2/
├── data/                    # Data storage directory
├── models/                  # Trained model checkpoints
├── results/                 # Training results and visualizations
├── config.py               # Configuration and hyperparameters
├── data_loader.py          # Data loading and preprocessing
├── fix_financial.py        # Financial data cleaning and fixing utilities
├── train.py                # Model architectures and training logic
├── utils.py                # Helper functions and utilities
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/adddmole/financial-risk-model2.git
cd financial-risk-model2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the complete pipeline:
```bash
python main.py
```

This will:
1. Generate synthetic financial data (or load your own data)
2. Train the Transformer + GNN model
3. Evaluate on test set
4. Generate SHAP explanations
5. Save results and visualizations

## Configuration

Edit `config.py` to customize:
- Model hyperparameters (transformer layers, GNN dimensions, etc.)
- Training settings (batch size, learning rate, epochs)
- Data parameters (sequence length, number of features)
- SHAP settings

## Model Architecture

### Transformer Encoder
- Processes time series data with multi-head self-attention
- Captures temporal dependencies and patterns
- Positional encoding for sequence information

### Graph Neural Network
- Learns from graph-structured financial data
- Aggregates information from connected entities
- Multiple GNN layers for deep feature extraction

### Combined Model
- Fuses Transformer and GNN outputs
- Multi-layer perceptron for final risk prediction
- Dropout for regularization

## Output

After training, you'll find:
- `models/best_model.pt` - Best model checkpoint
- `results/results.json` - Evaluation metrics
- `results/training_history.png` - Loss curves
- `results/shap_explanations.png` - Feature importance

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Pandas
- SciPy
- Matplotlib
- SHAP
- scikit-learn

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.