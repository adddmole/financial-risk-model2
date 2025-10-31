# Financial Risk Model 2

A comprehensive financial risk assessment model combining **Transformer**, **Graph Neural Networks (GNN)**, and **SHAP** for explainable AI.

## 🏗️ Project Structure

```
financial-risk-model2/
├── data/                    # Data directory
│   ├── financial_data.csv   # Raw financial data
│   └── processed_data.pkl   # Preprocessed data
├── models/                  # Model architecture modules
│   ├── __init__.py
│   ├── transformer.py       # Transformer model for temporal patterns
│   ├── gnn.py              # Graph Neural Network for relational patterns
│   └── hybrid.py           # Hybrid model combining Transformer and GNN
├── results/                 # Output directory
│   ├── best_model.pt       # Saved model checkpoints
│   ├── training_history.png # Training visualization
│   ├── shap_summary.png    # SHAP explanations
│   └── predictions.csv     # Model predictions
├── config.py               # Configuration parameters
├── data_loader.py          # Data loading and preprocessing
├── fix_financial.py        # Data quality and fixing utilities
├── main.py                 # Main entry point
├── train.py                # Training logic
├── utils.py                # Utility functions
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Features

- **Hybrid Architecture**: Combines Transformer (temporal patterns) and GNN (relational patterns)
- **Financial Risk Prediction**: Predicts risk scores from 0 (low risk) to 1 (high risk)
- **Explainable AI**: Uses SHAP for model interpretability
- **Data Quality Tools**: Automated data cleaning and fixing utilities
- **Comprehensive Training**: Includes early stopping, learning rate scheduling, and validation

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy, Pandas, Scikit-learn
- Matplotlib, SHAP

Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Usage

### Training the Model

Train the financial risk model from scratch:

```bash
python main.py --mode train
```

### Full Pipeline (Train + Explain)

Run training and generate SHAP explanations:

```bash
python main.py --mode all
```

### Evaluate Existing Model

Evaluate a trained model on test data:

```bash
python main.py --mode evaluate --model-path results/best_model.pt
```

### Generate SHAP Explanations

Generate explainability plots for a trained model:

```bash
python main.py --mode explain --model-path results/best_model.pt
```

## 🔧 Configuration

Edit `config.py` to customize:

- **Model architecture**: Transformer layers, GNN layers, dimensions
- **Training parameters**: Batch size, learning rate, epochs
- **Data splits**: Train/validation/test ratios
- **Risk thresholds**: Risk categorization levels

## 📊 Model Architecture

### Transformer Component
- Captures temporal patterns in financial time series
- Multi-head attention mechanism
- Positional encoding for sequence modeling

### GNN Component
- Models relationships between financial features
- Graph convolution layers
- Captures inter-feature correlations

### Hybrid Fusion
- Combines Transformer and GNN embeddings
- Multi-layer fusion network
- Outputs risk score in [0, 1]

## 🛠️ Data Quality Tools

The `fix_financial.py` module provides:
- Missing value detection and imputation
- Outlier detection and handling
- Data validation
- Feature normalization

Example usage:
```python
from fix_financial import FinancialDataFixer

fixer = FinancialDataFixer()
fixed_data, report = fixer.fix_all(raw_data)
```

## 📈 Output

The model generates:
1. **Trained model**: Saved in `results/best_model.pt`
2. **Training plots**: Loss and metric curves
3. **SHAP explanations**: Feature importance visualizations
4. **Predictions CSV**: Test set predictions with true labels

## 🧪 Testing

Run the data fixer test:
```bash
python fix_financial.py
```

## 📝 Notes

- The system generates synthetic financial data if no real data is provided
- Model automatically uses GPU if available, otherwise CPU
- Early stopping prevents overfitting
- All random seeds are set for reproducibility

## 🤝 Contributing

Contributions are welcome! Please ensure code follows the existing structure and includes documentation.

## 📄 License

MIT License