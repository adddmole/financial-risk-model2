"""
Verification script to check if all dependencies are installed correctly.
Run this script after installation to ensure everything is set up properly.
"""

import sys

def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(module_name)
        print(f"✓ {package_name} is installed")
        return True
    except ImportError:
        print(f"✗ {package_name} is NOT installed")
        return False

def main():
    """Run all verification checks."""
    print("=" * 60)
    print("Financial Risk Model - Installation Verification")
    print("=" * 60)
    print()
    
    all_ok = True
    
    # Check Python version
    print("1. Checking Python version...")
    python_version = sys.version_info
    print(f"   Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version.major >= 3 and python_version.minor >= 8:
        print("   ✓ Python version is compatible (3.8+)")
    else:
        print("   ✗ Python version should be 3.8 or higher")
        all_ok = False
    print()
    
    # Check core dependencies
    print("2. Checking core dependencies...")
    core_packages = [
        ('torch', 'PyTorch'),
        ('torch_geometric', 'PyTorch Geometric'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'Scikit-learn'),
    ]
    
    for module, name in core_packages:
        if not check_import(module, name):
            all_ok = False
    print()
    
    # Check visualization libraries
    print("3. Checking visualization libraries...")
    viz_packages = [
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
    ]
    
    for module, name in viz_packages:
        if not check_import(module, name):
            all_ok = False
    print()
    
    # Check SHAP
    print("4. Checking SHAP...")
    if not check_import('shap', 'SHAP'):
        all_ok = False
    print()
    
    # Check other dependencies
    print("5. Checking other dependencies...")
    other_packages = [
        ('tqdm', 'tqdm'),
        ('networkx', 'NetworkX'),
        ('scipy', 'SciPy'),
    ]
    
    for module, name in other_packages:
        if not check_import(module, name):
            all_ok = False
    print()
    
    # Check PyTorch details
    print("6. Checking PyTorch configuration...")
    try:
        import torch
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print("   ✓ GPU acceleration is available!")
        else:
            print("   ⚠ No GPU detected, will use CPU (slower but works)")
    except Exception as e:
        print(f"   ✗ Error checking PyTorch: {e}")
        all_ok = False
    print()
    
    # Check PyTorch Geometric
    print("7. Checking PyTorch Geometric modules...")
    try:
        from torch_geometric.nn import GCNConv, GATConv, SAGEConv
        print("   ✓ GCNConv is available")
        print("   ✓ GATConv is available")
        print("   ✓ SAGEConv is available")
    except ImportError as e:
        print(f"   ✗ PyTorch Geometric modules not fully available: {e}")
        all_ok = False
    print()
    
    # Check project files
    print("8. Checking project files...")
    import os
    
    required_files = [
        'config.py',
        'data_loader.py',
        'train.py',
        'main.py',
        'utils.py',
        'fix_financial.py',
        'models/__init__.py',
        'models/transformer.py',
        'models/gnn.py',
        'models/hybrid.py',
        'tutorial.ipynb',
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"   ✓ {file}")
        else:
            print(f"   ✗ {file} is missing")
            all_ok = False
    print()
    
    # Check directories
    print("9. Checking project directories...")
    required_dirs = ['models', 'data', 'results', 'checkpoints']
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"   ✓ {directory}/")
        else:
            print(f"   ⚠ {directory}/ will be created when needed")
    print()
    
    # Final result
    print("=" * 60)
    if all_ok:
        print("✓✓✓ All checks passed! You're ready to use the system!")
        print()
        print("Next steps:")
        print("1. Open JupyterLab: jupyter lab")
        print("2. Open tutorial.ipynb")
        print("3. Run the cells to train your first model")
        print()
        print("Or use command line:")
        print("  python data/generate_sample_data.py")
        print("  python main.py --mode train --data-path ./data/sample_data.csv")
    else:
        print("✗✗✗ Some checks failed. Please fix the issues above.")
        print()
        print("To install missing dependencies:")
        print("  pip install -r requirements.txt")
    print("=" * 60)

if __name__ == '__main__':
    main()
