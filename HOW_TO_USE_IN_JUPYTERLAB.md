# How to Use the Financial Risk Model in JupyterLab

Welcome! This guide will walk you through using the Financial Risk Model system in JupyterLab, step by step.

## 🎯 What You'll Learn

By the end of this guide, you'll be able to:
- ✅ Set up the environment in JupyterLab
- ✅ Generate and explore financial data
- ✅ Train a hybrid Transformer+GNN model
- ✅ Evaluate model performance
- ✅ Generate interpretable predictions with SHAP
- ✅ Use your own data

## 📦 Installation (Do This First!)

### Option 1: Quick Setup (Recommended)

Open a terminal in JupyterLab (File > New > Terminal) and run:

```bash
# Navigate to the project directory
cd /path/to/financial-risk-model2

# Install all dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

### Option 2: Step-by-Step Setup

If the quick setup doesn't work, install packages one by one:

```bash
# Install PyTorch (choose CPU or GPU version)
# For CPU:
pip install torch==2.0.1 torchvision torchaudio

# For GPU (CUDA 11.8):
pip install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
pip install torch-geometric==2.3.1
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.1+cpu.html

# Install other dependencies
pip install numpy pandas scikit-learn matplotlib seaborn shap tqdm tensorboard networkx scipy
```

## 📓 Using the Tutorial Notebook

### 1. Open the Tutorial

In JupyterLab:
1. Click on `tutorial.ipynb` in the file browser (left sidebar)
2. The notebook will open in the main area
3. You should see multiple cells with code and explanations

### 2. Run Your First Cell

Click on the first code cell and press **Shift + Enter** or click the ▶️ button.

This cell imports libraries and checks your setup:

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

**Expected Output:**
```
PyTorch version: 2.0.1
CUDA available: True (or False if no GPU)
Device: cuda:0 (or cpu)
```

### 3. Generate Sample Data

Run the data generation cell:

```python
from data.generate_sample_data import generate_sample_data

df = generate_sample_data(
    num_entities=50,
    sequence_length=60,
    num_features=10,
    output_path='./data/sample_data.csv'
)
```

**What this does:**
- Creates synthetic financial time series data
- Generates 50 companies/entities
- Each has 60 time steps of data
- Includes 10 features per time step
- Saves to `./data/sample_data.csv`

**Expected Output:**
```
Generated sample data with 50 entities and 60 time steps
Total rows: 3000
Label distribution:
0    30  (low risk entities)
1    20  (high risk entities)
Saved to: ./data/sample_data.csv
```

### 4. Explore the Data

The next cells visualize the data:

```python
# Visualize price trajectories
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
# ... plotting code ...
plt.show()
```

You'll see:
- **Left plot**: Price trajectories for low-risk entities (more stable)
- **Right plot**: Price trajectories for high-risk entities (more volatile)

### 5. Load and Preprocess Data

This step creates the graph structure:

```python
temporal_data, graph_data, labels = load_and_preprocess_data(
    data_path='./data/sample_data.csv',
    sequence_length=60,
    num_features=10,
    graph_method='correlation',
    threshold=0.5
)
```

**What this does:**
- Loads the CSV data
- Creates time series sequences
- Builds a graph connecting related entities
- Normalizes all features

**Expected Output:**
```
Temporal data shape: (50, 60, 10)
  - 50 entities
  - 60 time steps
  - 10 features per step
Graph data:
  - 50 nodes (entities)
  - 20 node features
  - 156 edges (relationships)
```

### 6. Create the Model

Initialize the hybrid architecture:

```python
model = HybridRiskModel(
    num_temporal_features=10,
    sequence_length=60,
    transformer_dim=128,
    gnn_hidden_dim=128,
    num_classes=2
)
```

**What this does:**
- Creates a Transformer for time series analysis
- Creates a GNN for relationship modeling
- Combines both with a fusion layer
- Adds a classification head for risk prediction

**Expected Output:**
```
Model created successfully!
Total parameters: 1,234,567
```

### 7. Train the Model

This is the most important step:

```python
trainer = Trainer(model, train_loader, val_loader, test_loader)
history = trainer.train()
```

**What happens:**
- Model trains for multiple epochs (iterations)
- Each epoch shows a progress bar
- Loss decreases over time (good!)
- Accuracy increases over time (good!)
- Best model is saved automatically

**Expected Output (for each epoch):**
```
Epoch 1/20: 100%|████████| 12/12 [00:05<00:00, loss=0.65, acc=65.71%]

Epoch 1/20:
  Train Loss: 0.6543, Train Acc: 65.71%
  Val Loss: 0.5876, Val Acc: 70.00%
  Learning Rate: 0.001000
  Saved best model to ./checkpoints/best_model.pth
```

**Training Time:**
- With GPU: ~5-10 minutes
- With CPU: ~10-20 minutes

**Tips while training:**
- Watch the loss decrease
- Check if validation accuracy improves
- Don't worry if it fluctuates a bit
- You can stop early with Kernel > Interrupt

### 8. Evaluate the Model

After training, evaluate performance:

```python
test_metrics = trainer.evaluate(save_dir='./results')
```

**Expected Output:**
```
Test Set Results:
==================================================
Accuracy: 85.00%
PRECISION: 0.8421
RECALL: 0.8500
F1: 0.8460
AUC_ROC: 0.9123
```

**What these metrics mean:**
- **Accuracy**: 85% of predictions are correct
- **Precision**: Of predicted high-risk, 84% are actually high-risk
- **Recall**: Of actual high-risk, 85% are identified
- **F1**: Harmonic mean of precision and recall
- **AUC-ROC**: 0.91 is excellent (1.0 is perfect, 0.5 is random)

A confusion matrix will also be saved to `./results/confusion_matrix.png`.

### 9. Make Predictions

Use the trained model:

```python
model.eval()
with torch.no_grad():
    outputs = model(temporal_data, graph_data)
    predictions = outputs.argmax(dim=1)
    probabilities = torch.softmax(outputs, dim=1)
```

**Expected Output:**
```
Sample Predictions:
   True Label  Predicted Label  Prob Low Risk  Prob High Risk  Correct
0           0                0       0.923456        0.076544     True
1           1                1       0.234567        0.765433     True
2           0                0       0.876543        0.123457     True
...

Overall Accuracy: 85.00%
```

### 10. SHAP Explanations

Understand why the model makes predictions:

```python
explainer = SHAPExplainer(model, background_data, device)
shap_values, base_values = explainer.explain_instance(test_data, graph_data)
explainer.plot_summary(shap_values, features, save_path='./results/shap_summary.png')
```

**What this does:**
- Calculates feature importance for each prediction
- Shows which time steps and features matter most
- Creates visual explanations

**Expected Output:**
- SHAP summary plot showing feature importance
- SHAP waterfall plot for individual predictions

## 🎨 Understanding the Visualizations

### Training History Plot

Shows two graphs:
1. **Loss over time**: Should decrease (good!)
2. **Accuracy over time**: Should increase (good!)

**What to look for:**
- ✅ Decreasing training loss
- ✅ Increasing validation accuracy
- ⚠️ If validation loss increases while training loss decreases: overfitting
- ⚠️ If both losses stay flat: learning rate too low or data issues

### Confusion Matrix

A heatmap showing predictions vs. reality:

```
              Predicted
              Low  High
Actual Low    24    2
Actual High    1   11
```

**How to read:**
- **Top-left**: Correctly predicted low risk (24)
- **Bottom-right**: Correctly predicted high risk (11)
- **Top-right**: False positives - predicted high but actually low (2)
- **Bottom-left**: False negatives - predicted low but actually high (1)

### SHAP Summary Plot

Shows which features are most important:
- **Red**: High feature value
- **Blue**: Low feature value
- **Position**: How much it affects prediction

**How to interpret:**
- Features at the top are most important
- Red dots pushing right = high values increase risk
- Blue dots pushing left = low values decrease risk

## 🔧 Customization Options

### Change Training Duration

```python
# In the config setup cell, add:
Config.NUM_EPOCHS = 50  # More epochs for better training
Config.BATCH_SIZE = 16  # Smaller batch for less memory
```

### Try Different Graph Methods

```python
# Correlation-based (default)
temporal_data, graph_data, labels = load_and_preprocess_data(
    data_path='./data/sample_data.csv',
    graph_method='correlation',
    threshold=0.7  # Higher = fewer edges
)

# K-nearest neighbors
temporal_data, graph_data, labels = load_and_preprocess_data(
    data_path='./data/sample_data.csv',
    graph_method='knn',
    k=10  # Connect to 10 nearest neighbors
)
```

### Adjust Model Size

```python
# Smaller model (faster, less accurate)
model = HybridRiskModel(
    transformer_dim=64,
    transformer_layers=2,
    gnn_layers=2
)

# Larger model (slower, more accurate)
model = HybridRiskModel(
    transformer_dim=256,
    transformer_layers=6,
    gnn_layers=4
)
```

## 💾 Using Your Own Data

### Step 1: Prepare Your CSV

Your data should look like this:

```csv
entity_id,time_step,feature_1,feature_2,feature_3,price,label
0,0,0.5,1.2,-0.3,100.0,0
0,1,0.6,1.1,-0.2,101.5,0
0,2,0.4,1.3,-0.4,99.8,0
1,0,-0.2,0.8,0.5,50.0,1
1,1,-0.3,0.9,0.4,49.5,1
1,2,-0.1,0.7,0.6,51.2,1
```

**Required columns:**
- `entity_id`: Unique ID for each company/asset
- `time_step`: Sequential time index (0, 1, 2, ...)
- `feature_1`, `feature_2`, etc.: Your features
- `label`: Risk label (0=low risk, 1=high risk)

### Step 2: Update Configuration

```python
# Update these based on your data
Config.NUM_FEATURES = 15  # Number of feature columns
Config.SEQUENCE_LENGTH = 100  # Number of time steps
Config.NUM_CLASSES = 2  # Binary or multi-class
```

### Step 3: Load Your Data

```python
# Replace the data path
temporal_data, graph_data, labels = load_and_preprocess_data(
    data_path='./data/your_data.csv',  # Your file
    sequence_length=Config.SEQUENCE_LENGTH,
    num_features=Config.NUM_FEATURES
)
```

### Step 4: Train as Normal

Continue with the same training steps!

## 🐛 Common Issues and Solutions

### Issue 1: "ModuleNotFoundError"

**Error:** `ModuleNotFoundError: No module named 'torch'`

**Solution:**
```bash
# In terminal
pip install -r requirements.txt
```

### Issue 2: Kernel Crashes During Training

**Error:** Kernel dies or "out of memory"

**Solution:**
```python
# Reduce memory usage
Config.BATCH_SIZE = 4
Config.TRANSFORMER_LAYERS = 2
Config.GNN_LAYERS = 2
```

### Issue 3: Training is Very Slow

**Solution:**
```python
# Use smaller dataset for testing
df = generate_sample_data(
    num_entities=20,  # Smaller
    sequence_length=30  # Shorter
)
Config.NUM_EPOCHS = 10
```

### Issue 4: Poor Performance

**Solutions:**
1. **More training:** Increase `NUM_EPOCHS`
2. **More data:** Generate larger dataset
3. **Better features:** Add more informative features
4. **Hyperparameter tuning:** Adjust learning rate, model size

### Issue 5: Can't Save Files

**Solution:**
```python
# Make sure directories exist
import os
os.makedirs('./results', exist_ok=True)
os.makedirs('./checkpoints', exist_ok=True)
```

## 📱 Keyboard Shortcuts

Make your life easier:

| Action | Shortcut |
|--------|----------|
| Run cell and move to next | `Shift + Enter` |
| Run cell and stay | `Ctrl + Enter` |
| Insert cell below | `B` |
| Insert cell above | `A` |
| Delete cell | `D` + `D` (press twice) |
| Undo cell deletion | `Z` |
| Change to code | `Y` |
| Change to markdown | `M` |
| Save notebook | `Ctrl + S` |
| Show command palette | `Ctrl + Shift + C` |

## 💡 Pro Tips

1. **Run cells in order**: Don't skip cells!
2. **Read the comments**: Each cell has explanations
3. **Check outputs**: Make sure each step works
4. **Save often**: Use `Ctrl + S` frequently
5. **Experiment**: Try changing parameters
6. **Use GPU if available**: Much faster training
7. **Start small**: Test with small data first
8. **Check shapes**: Verify data dimensions match

## 📊 Expected Performance

With the default sample data (50 entities, 60 timesteps):

| Metric | Expected Range | Good Performance |
|--------|----------------|------------------|
| Accuracy | 70-90% | > 80% |
| F1 Score | 0.65-0.88 | > 0.75 |
| AUC-ROC | 0.75-0.95 | > 0.85 |
| Training Time (GPU) | 3-10 min | - |
| Training Time (CPU) | 10-25 min | - |

## 🎓 What's Happening Under the Hood

### The Hybrid Model

1. **Transformer Path:**
   - Takes time series data (60 steps × 10 features)
   - Uses attention to find important patterns
   - Outputs a 128-dimensional embedding

2. **GNN Path:**
   - Takes entity relationships (graph structure)
   - Learns how entities influence each other
   - Outputs a 128-dimensional embedding

3. **Fusion:**
   - Combines both embeddings
   - Creates a 256-dimensional representation

4. **Classification:**
   - Maps to risk prediction
   - Outputs probabilities for each class

### SHAP Explanations

SHAP uses game theory to explain predictions:
- Calculates contribution of each feature
- Shows positive/negative impact
- Provides local (per prediction) and global (overall) explanations

## 📚 Next Steps

After completing the tutorial:

1. **Experiment with hyperparameters** in `config.py`
2. **Try different architectures** (GCN vs GAT vs GraphSAGE)
3. **Use your own financial data**
4. **Fine-tune for your specific use case**
5. **Deploy the model** for production predictions

## 🆘 Need Help?

- **Read the code comments**: Detailed explanations
- **Check README.md**: Comprehensive documentation
- **Review JUPYTERLAB_GUIDE.md**: Additional tips
- **Open GitHub issue**: Report bugs or ask questions

## ✅ Success Checklist

- [ ] Installed all dependencies
- [ ] Ran first cell successfully
- [ ] Generated sample data
- [ ] Loaded and preprocessed data
- [ ] Created model
- [ ] Trained model (saw progress bars)
- [ ] Evaluated model (got metrics)
- [ ] Made predictions
- [ ] Generated SHAP plots
- [ ] Understand the workflow
- [ ] Ready to use custom data

---

**Congratulations! You're now ready to build financial risk models! 🎉📈**

If you have any questions, don't hesitate to ask. Happy modeling!
