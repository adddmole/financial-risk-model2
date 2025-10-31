# Getting Started with Financial Risk Model

Welcome! This document will guide you to the right resources based on your needs.

## 🎯 Choose Your Path

### I'm New to This → Start Here!

**Read:** [QUICK_START_JUPYTER.md](QUICK_START_JUPYTER.md)

A 5-minute quick start guide to get you running in JupyterLab immediately.

**What you'll do:**
1. Install dependencies (2 min)
2. Launch JupyterLab (1 min)
3. Run the tutorial notebook (2 min to start)

Perfect for: First-time users who want to see it working quickly.

---

### I Want Step-by-Step Instructions

**Read:** [HOW_TO_USE_IN_JUPYTERLAB.md](HOW_TO_USE_IN_JUPYTERLAB.md)

A comprehensive guide with detailed explanations of every step.

**What you'll learn:**
- Detailed installation instructions
- How to use the tutorial notebook
- Understanding each step in the pipeline
- Troubleshooting common issues
- How to customize the system

Perfect for: Users who want to understand how everything works.

---

### I Need Setup Help

**Read:** [JUPYTERLAB_GUIDE.md](JUPYTERLAB_GUIDE.md)

A complete guide to setting up JupyterLab and understanding the interface.

**What you'll learn:**
- Installing Python and JupyterLab
- Understanding the JupyterLab interface
- Keyboard shortcuts
- Common operations
- Tips and tricks

Perfect for: Users new to JupyterLab or Jupyter notebooks.

---

### I Want Full Documentation

**Read:** [README.md](README.md)

Complete project documentation with architecture details, API reference, and examples.

**What you'll find:**
- System architecture diagrams
- Complete API documentation
- Command-line interface usage
- Configuration options
- Advanced customization
- Technical details

Perfect for: Developers and advanced users.

---

### I Want to Use Command Line

**Run these commands:**

```bash
# Generate sample data
python data/generate_sample_data.py

# Train model
python main.py --mode train --data-path ./data/sample_data.csv

# Evaluate model
python main.py --mode evaluate \
    --data-path ./data/sample_data.csv \
    --load-checkpoint ./checkpoints/best_model.pth

# Make predictions
python main.py --mode predict \
    --data-path ./data/new_data.csv \
    --load-checkpoint ./checkpoints/best_model.pth
```

Perfect for: Users who prefer terminal/command-line over notebooks.

---

## 📚 All Available Guides

| Document | Purpose | Reading Time |
|----------|---------|--------------|
| [QUICK_START_JUPYTER.md](QUICK_START_JUPYTER.md) | Fast start in JupyterLab | 5 min |
| [HOW_TO_USE_IN_JUPYTERLAB.md](HOW_TO_USE_IN_JUPYTERLAB.md) | Detailed JupyterLab guide | 20 min |
| [JUPYTERLAB_GUIDE.md](JUPYTERLAB_GUIDE.md) | JupyterLab setup and interface | 15 min |
| [README.md](README.md) | Full documentation | 30 min |
| [tutorial.ipynb](tutorial.ipynb) | Interactive tutorial | 30-60 min |
| [data/README.md](data/README.md) | Data format specs | 10 min |
| [results/README.md](results/README.md) | Understanding results | 5 min |

## 🛠️ Quick Commands

### Verify Installation
```bash
python verify_installation.py
```
Checks if all dependencies are installed correctly.

### Generate Sample Data
```bash
python data/generate_sample_data.py
```
Creates synthetic financial data for testing.

### Launch JupyterLab
```bash
jupyter lab
```
Opens JupyterLab in your browser.

### Train Model (CLI)
```bash
python main.py --mode train --data-path ./data/sample_data.csv
```

## 📊 What This System Does

The Financial Risk Model combines:

1. **Transformer** - Analyzes time series patterns (prices, returns, volatility)
2. **GNN** - Models relationships between entities (companies, assets)
3. **SHAP** - Explains predictions (which features matter most)

**Result:** Predict financial risk with interpretable AI

## 🎓 Learning Path

### Beginner Path (Recommended)
```
1. QUICK_START_JUPYTER.md (5 min)
   ↓
2. Open tutorial.ipynb in JupyterLab
   ↓
3. Run all cells, read explanations
   ↓
4. Experiment with your own data
```

### Detailed Path
```
1. JUPYTERLAB_GUIDE.md (setup)
   ↓
2. HOW_TO_USE_IN_JUPYTERLAB.md (detailed guide)
   ↓
3. tutorial.ipynb (hands-on practice)
   ↓
4. README.md (full documentation)
   ↓
5. Customize for your needs
```

### Advanced Path
```
1. README.md (full documentation)
   ↓
2. Review source code (models/, train.py, etc.)
   ↓
3. Modify config.py for your needs
   ↓
4. Use command-line interface
   ↓
5. Deploy to production
```

## 🔍 Find What You Need

### "I want to understand the model architecture"
→ Read [README.md](README.md) - "Model Architecture" section

### "I want to use my own data"
→ Read [data/README.md](data/README.md) and [HOW_TO_USE_IN_JUPYTERLAB.md](HOW_TO_USE_IN_JUPYTERLAB.md) - "Using Your Own Data" section

### "I'm getting errors during installation"
→ Read [JUPYTERLAB_GUIDE.md](JUPYTERLAB_GUIDE.md) - "Troubleshooting" section

### "How do I interpret the results?"
→ Read [results/README.md](results/README.md) and [HOW_TO_USE_IN_JUPYTERLAB.md](HOW_TO_USE_IN_JUPYTERLAB.md) - "Understanding the Visualizations"

### "I want to change hyperparameters"
→ Edit `config.py` or see [README.md](README.md) - "Configuration" section

### "How does SHAP work?"
→ Read [HOW_TO_USE_IN_JUPYTERLAB.md](HOW_TO_USE_IN_JUPYTERLAB.md) - "SHAP Explanations" section

## 🚀 Quick Start (3 Steps)

```bash
# 1. Install
pip install -r requirements.txt
pip install jupyterlab

# 2. Launch
jupyter lab

# 3. Open tutorial.ipynb and run cells!
```

## ✅ System Requirements

- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB+ recommended
- **Disk**: 2GB for dependencies + data
- **GPU**: Optional but recommended for faster training
- **OS**: Linux, macOS, or Windows

## 🎯 Quick Test

After installation, verify everything works:

```bash
# Check installation
python verify_installation.py

# Generate test data
python data/generate_sample_data.py

# Quick training test (5-10 min)
python main.py --mode train \
    --data-path ./data/sample_data.csv \
    --epochs 5 \
    --batch-size 8
```

## 💡 Tips for Success

1. **Start with the tutorial notebook** - It's interactive and well-explained
2. **Use small data first** - Test with 20-30 entities before scaling up
3. **Check each step** - Make sure outputs look correct before proceeding
4. **Save your work** - Use Ctrl+S frequently in JupyterLab
5. **Experiment** - Try different parameters and see what happens

## 🆘 Getting Help

1. **Check the guides** - Most questions are answered there
2. **Run verification** - `python verify_installation.py`
3. **Read error messages** - They usually tell you what's wrong
4. **Check common issues** - See troubleshooting sections in guides
5. **Open an issue** - On GitHub if you're still stuck

## 📧 Resources

- **Repository**: [github.com/adddmole/financial-risk-model2](https://github.com/adddmole/financial-risk-model2)
- **Tutorial**: `tutorial.ipynb` (open in JupyterLab)
- **Documentation**: All the `.md` files in this directory

---

**Ready to start? Choose your path above and begin your journey! 🚀**

**Recommended for most users:** [QUICK_START_JUPYTER.md](QUICK_START_JUPYTER.md) → `tutorial.ipynb`
