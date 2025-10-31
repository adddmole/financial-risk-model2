# JupyterLab Quick Start Guide

This guide will help you get started with the Financial Risk Model in JupyterLab.

## 📋 Prerequisites

- Python 3.8 or higher installed
- Basic knowledge of Python and Jupyter notebooks
- (Optional) CUDA-capable GPU for faster training

## 🚀 Step-by-Step Setup

### Step 1: Install Dependencies

Open a terminal and run:

```bash
# Navigate to the project directory
cd financial-risk-model2

# Create a virtual environment (recommended)
python -m venv venv

# Activate the virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Install JupyterLab
pip install jupyterlab

# Install additional visualization tools (optional)
pip install ipywidgets widgetsnbextension
```

### Step 2: Launch JupyterLab

```bash
# Start JupyterLab
jupyter lab
```

This will:
1. Start the JupyterLab server
2. Open your default web browser to `http://localhost:8888`
3. Display the JupyterLab interface

### Step 3: Open the Tutorial Notebook

In the JupyterLab interface:

1. Look at the file browser on the left side
2. Click on `tutorial.ipynb` to open it
3. The notebook will open in a new tab

## 📖 Using the Tutorial Notebook

### Understanding the Interface

```
┌─────────────────────────────────────────────────────────┐
│  JupyterLab Interface                                    │
├─────────────┬───────────────────────────────────────────┤
│             │  Notebook Toolbar                          │
│  File       │  ▶ ⏸ ⏹ ↻ + [Code ▾]                      │
│  Browser    ├───────────────────────────────────────────┤
│             │  # Financial Risk Model Tutorial           │
│  📁 folder  │                                            │
│  📄 file.py │  Cell 1: Markdown                          │
│  📓 note.py │  This is a markdown cell for text          │
│             │                                            │
│             │  Cell 2: Code                              │
│             │  [ ]: import torch                         │
│             │       print("Hello")                       │
│             │                                            │
│             │  Output:                                   │
│             │  Hello                                     │
└─────────────┴───────────────────────────────────────────┘
```

### Running Cells

There are three ways to run cells:

1. **Shift + Enter**: Run current cell and move to the next
2. **Ctrl + Enter**: Run current cell and stay on it
3. **Click ▶ button**: Run the selected cell

**Best Practice**: Run cells sequentially from top to bottom.

### Cell Types

- **Code Cells**: Contains Python code (marked with `[ ]`)
- **Markdown Cells**: Contains formatted text and documentation
- **Output Cells**: Shows results, plots, or errors

## 📝 Tutorial Sections Explained

### Section 1: Setup and Imports

```python
import torch
from config import Config
# ... other imports
```

**What it does**: 
- Imports all necessary libraries
- Checks if GPU is available
- Sets up the environment

**Expected output**: 
```
PyTorch version: 2.0.1
CUDA available: True/False
Device: cuda:0 or cpu
```

### Section 2: Generate Sample Data

```python
from data.generate_sample_data import generate_sample_data
df = generate_sample_data(num_entities=50, sequence_length=60)
```

**What it does**:
- Creates synthetic financial time series data
- Saves to `./data/sample_data.csv`
- Generates 50 entities with 60 time steps each

**Expected output**:
```
Generated sample data with 50 entities and 60 time steps
Total rows: 3000
Label distribution:
0    30  (low risk)
1    20  (high risk)
```

### Section 3: Data Exploration

```python
df = pd.read_csv('./data/sample_data.csv')
# Visualization code...
```

**What it does**:
- Loads the generated data
- Shows price trajectories for low-risk and high-risk entities
- Visualizes the differences between risk classes

**Expected output**: Two plots showing price movements

### Section 4: Load and Preprocess

```python
temporal_data, graph_data, labels = load_and_preprocess_data(
    data_path='./data/sample_data.csv',
    sequence_length=60,
    num_features=10
)
```

**What it does**:
- Loads raw data
- Creates temporal sequences
- Constructs graph structure based on correlations
- Normalizes features

**Expected output**:
```
Temporal data shape: (50, 60, 10)
  - Samples: 50
  - Sequence length: 60
  - Features: 10
Graph data:
  - Nodes: 50
  - Node features: 20
  - Edges: 156
```

### Section 5: Create Model

```python
model = HybridRiskModel(
    num_temporal_features=10,
    sequence_length=60,
    # ... other parameters
)
```

**What it does**:
- Initializes the Transformer + GNN hybrid model
- Sets up all layers and connections
- Moves model to GPU if available

**Expected output**:
```
Model created successfully!
Total parameters: 1,234,567
Trainable parameters: 1,234,567
```

### Section 6: Train Model

```python
trainer = Trainer(model, train_loader, val_loader, test_loader)
history = trainer.train()
```

**What it does**:
- Trains the model for the specified epochs
- Shows progress bars for each epoch
- Validates after each epoch
- Saves the best model

**Expected output**: Progress bars and training metrics
```
Epoch 1/20: 100%|████████| 12/12 [00:05<00:00, loss=0.6543, acc=65.71%]

Epoch 1/20:
  Train Loss: 0.6543, Train Acc: 65.71%
  Val Loss: 0.5876, Val Acc: 70.00%
  ...
```

### Section 7: Evaluate Model

```python
test_metrics = trainer.evaluate(save_dir='./results')
```

**What it does**:
- Evaluates model on test set
- Calculates metrics (accuracy, precision, recall, F1, AUC-ROC)
- Generates confusion matrix
- Saves results

**Expected output**:
```
Test Set Results:
==================================================
Accuracy: 85.00%
PRECISION: 0.8421
RECALL: 0.8500
F1: 0.8460
AUC_ROC: 0.9123
```

### Section 8: Make Predictions

```python
# Get predictions on test set
model.eval()
with torch.no_grad():
    outputs = model(temporal_data, graph_data)
    predictions = outputs.argmax(dim=1)
```

**What it does**:
- Uses trained model to predict risk
- Shows prediction probabilities
- Visualizes prediction confidence

**Expected output**: DataFrame with predictions and probabilities

### Section 9: SHAP Explanations

```python
explainer = SHAPExplainer(model, background_data, device)
shap_values, base_values = explainer.explain_instance(test_data, graph_data)
```

**What it does**:
- Generates SHAP explanations for predictions
- Shows which features are most important
- Creates summary and waterfall plots

**Expected output**: SHAP visualization plots

### Section 10: Custom Data

Shows you how to use your own data with the model.

## 🔧 Common Operations in JupyterLab

### Saving Your Work

- **Auto-save**: JupyterLab auto-saves every 2 minutes
- **Manual save**: `Ctrl + S` or `Cmd + S` (Mac)
- **Checkpoint**: File > Save Notebook As...

### Restarting the Kernel

If something goes wrong:

1. Click **Kernel** in the menu bar
2. Select **Restart Kernel and Clear All Outputs**
3. Run all cells again from the top

### Viewing Variables

Use the Variable Inspector:

1. Click the 🔍 icon in the right sidebar
2. Or go to View > Activate Command Palette
3. Type "Show Variables" and press Enter

### Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Run cell and advance | Shift + Enter |
| Run cell | Ctrl + Enter |
| Insert cell above | A |
| Insert cell below | B |
| Delete cell | D, D (press D twice) |
| Change to Markdown | M |
| Change to Code | Y |
| Save notebook | Ctrl + S |
| Command mode | Esc |
| Edit mode | Enter |

## 🐛 Troubleshooting

### Issue 1: Module not found

**Error**: `ModuleNotFoundError: No module named 'torch'`

**Solution**:
```bash
# Make sure you're in the right environment
pip install -r requirements.txt
```

### Issue 2: CUDA out of memory

**Error**: `RuntimeError: CUDA out of memory`

**Solution**:
```python
# Reduce batch size in config
Config.BATCH_SIZE = 8  # or smaller
Config.NUM_EPOCHS = 10  # fewer epochs for testing
```

### Issue 3: Kernel dies during training

**Error**: Kernel crashes or becomes unresponsive

**Solution**:
```python
# Reduce model size
Config.TRANSFORMER_LAYERS = 2
Config.GNN_LAYERS = 2
Config.BATCH_SIZE = 4
```

### Issue 4: Slow training

**Solution**:
```python
# Reduce data size for testing
df = generate_sample_data(
    num_entities=20,  # Smaller dataset
    sequence_length=30
)
Config.NUM_EPOCHS = 10
```

## 💡 Tips for Success

### 1. Start Small
- Begin with small datasets (20-30 entities)
- Use fewer epochs (10-20) for initial testing
- Gradually increase as you understand the system

### 2. Monitor Training
- Watch the training progress bars
- Check if loss is decreasing
- Validate accuracy on validation set

### 3. Experiment
- Try different graph construction methods
- Adjust hyperparameters in `config.py`
- Compare results with different architectures

### 4. Save Your Work
- Save trained models: `torch.save(model.state_dict(), 'my_model.pth')`
- Export results to CSV for later analysis
- Take screenshots of important visualizations

### 5. Use GPU if Available
```python
# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## 📊 Expected Timeline

Here's how long each section typically takes:

| Section | Time | Notes |
|---------|------|-------|
| Setup | 2-3 min | One-time only |
| Generate Data | <1 min | Quick |
| Data Exploration | 1-2 min | Visual inspection |
| Load & Preprocess | 1-2 min | Depends on data size |
| Create Model | <1 min | Fast |
| Train Model | 5-15 min | Depends on GPU/CPU and epochs |
| Evaluate | 1-2 min | Quick |
| Predictions | 1-2 min | Quick |
| SHAP Explanations | 2-5 min | Computation-intensive |
| **Total** | **15-35 min** | First run |

## 🎯 Next Steps After Tutorial

1. **Try with your own data**:
   - Prepare your CSV file following the format
   - Modify `Config.NUM_FEATURES` if needed
   - Run the pipeline

2. **Experiment with hyperparameters**:
   - Edit `config.py`
   - Try different graph construction methods
   - Compare performance

3. **Deploy your model**:
   - Save the best model
   - Use it for predictions on new data
   - Integrate with your application

4. **Learn more**:
   - Read the detailed README.md
   - Check model architecture in source code
   - Explore SHAP documentation

## 🆘 Getting Help

- **Check the README.md**: Comprehensive documentation
- **Look at comments**: Code is well-commented
- **Run small examples**: Test individual components
- **GitHub Issues**: Report bugs or ask questions

## ✅ Checklist for Success

- [ ] Installed all dependencies
- [ ] Launched JupyterLab successfully
- [ ] Opened tutorial.ipynb
- [ ] Generated sample data
- [ ] Ran all cells without errors
- [ ] Trained a model successfully
- [ ] Viewed evaluation metrics
- [ ] Generated SHAP explanations
- [ ] Understood the workflow
- [ ] Ready to use custom data

---

**You're all set! Enjoy building financial risk models! 🚀📈**
