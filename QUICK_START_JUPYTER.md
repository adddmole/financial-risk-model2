# Quick Start Guide - JupyterLab

Get started with the Financial Risk Model in 5 minutes!

## ⚡ Installation (2 minutes)

```bash
# 1. Navigate to project
cd financial-risk-model2

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install JupyterLab (if not already installed)
pip install jupyterlab

# 4. Verify installation
python verify_installation.py
```

## 🚀 Launch JupyterLab

```bash
jupyter lab
```

Your browser will open to `http://localhost:8888`

## 📓 Run the Tutorial

1. **Click** on `tutorial.ipynb` in the file browser (left side)
2. **Run** cells one by one with `Shift + Enter`
3. **Follow** the instructions in each cell

## 📋 Tutorial Steps

### Step 1: Setup (Run first cell)
```python
import torch
from config import Config
# Check if everything is installed
```

### Step 2: Generate Data (Run data generation cell)
```python
from data.generate_sample_data import generate_sample_data
df = generate_sample_data(num_entities=50, sequence_length=60)
```
Creates synthetic financial data in `./data/sample_data.csv`

### Step 3: Explore Data (Run visualization cells)
See price trajectories and risk patterns

### Step 4: Load Data (Run preprocessing cell)
```python
temporal_data, graph_data, labels = load_and_preprocess_data(
    data_path='./data/sample_data.csv',
    sequence_length=60,
    num_features=10
)
```

### Step 5: Create Model (Run model creation cell)
```python
model = HybridRiskModel(
    num_temporal_features=10,
    sequence_length=60,
    num_classes=2
)
```

### Step 6: Train Model (Run training cells - takes 5-15 minutes)
```python
trainer = Trainer(model, train_loader, val_loader, test_loader)
history = trainer.train()
```
Watch the progress bars and metrics!

### Step 7: Evaluate (Run evaluation cell)
```python
test_metrics = trainer.evaluate(save_dir='./results')
```
See accuracy, precision, recall, F1, and AUC-ROC scores

### Step 8: Make Predictions (Run prediction cells)
```python
predictions = model(temporal_data, graph_data)
```

### Step 9: SHAP Explanations (Run SHAP cells)
```python
explainer = SHAPExplainer(model, background_data, device)
shap_values = explainer.explain_instance(test_data, graph_data)
```
Understand feature importance!

## 💡 Quick Tips

- **Run cells in order**: Don't skip cells
- **Watch for errors**: Red output means something went wrong
- **Check shapes**: Make sure data dimensions match
- **Be patient**: Training takes time (5-15 minutes)
- **Save often**: Use `Ctrl + S` or `Cmd + S`

## 🎯 Expected Results

After running all cells, you should have:
- ✅ Generated sample data (3000 rows)
- ✅ Trained model saved in `./checkpoints/`
- ✅ Test accuracy around 80-90%
- ✅ Confusion matrix in `./results/`
- ✅ SHAP plots in `./results/`
- ✅ Training history plot

## 🔧 Common Issues

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### Kernel Crashes
```python
# Reduce memory usage
Config.BATCH_SIZE = 4
Config.NUM_EPOCHS = 10
```

### Training Too Slow
```python
# Use smaller dataset
df = generate_sample_data(num_entities=20, sequence_length=30)
```

## 📚 More Information

- **Detailed Guide**: See `HOW_TO_USE_IN_JUPYTERLAB.md`
- **Full Documentation**: See `README.md`
- **Setup Help**: See `JUPYTERLAB_GUIDE.md`

## ✅ Success Checklist

- [ ] Installed dependencies
- [ ] Launched JupyterLab
- [ ] Opened tutorial.ipynb
- [ ] Generated sample data
- [ ] Trained model successfully
- [ ] Saw evaluation metrics
- [ ] Generated SHAP plots

**You're done! Now experiment with your own data!** 🎉

## 🆘 Need Help?

1. Run `python verify_installation.py` to check setup
2. Read `HOW_TO_USE_IN_JUPYTERLAB.md` for detailed help
3. Check cell outputs for error messages
4. Open an issue on GitHub if stuck

---

**Happy modeling! 📊🚀**
