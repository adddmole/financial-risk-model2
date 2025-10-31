"""
Utility functions for Financial Risk Model
"""
import os
import random
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from config import RESULTS_DIR, RANDOM_SEED


def set_seed(seed=RANDOM_SEED):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """Get available device (CUDA or CPU)"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_model(model, filename):
    """Save model checkpoint"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filepath = os.path.join(RESULTS_DIR, filename)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def load_model(model, filename):
    """Load model checkpoint"""
    filepath = os.path.join(RESULTS_DIR, filename)
    model.load_state_dict(torch.load(filepath, map_location='cpu'))
    print(f"Model loaded from {filepath}")
    return model


def plot_training_history(history, filename='training_history.png'):
    """Plot training and validation metrics"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot metrics
    if 'train_metric' in history and 'val_metric' in history:
        axes[1].plot(history['train_metric'], label='Train Metric')
        axes[1].plot(history['val_metric'], label='Val Metric')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Metric')
        axes[1].set_title('Training and Validation Metrics')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    filepath = os.path.join(RESULTS_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Training history plot saved to {filepath}")


def calculate_risk_score(prediction, thresholds):
    """Calculate risk category from prediction"""
    if prediction < thresholds['low']:
        return 'Low Risk'
    elif prediction < thresholds['medium']:
        return 'Medium Risk'
    else:
        return 'High Risk'


def save_predictions(predictions, true_labels, filename='predictions.csv'):
    """Save predictions to CSV file"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    df = pd.DataFrame({
        'true_label': true_labels,
        'prediction': predictions,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    
    filepath = os.path.join(RESULTS_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"Predictions saved to {filepath}")


def print_metrics(metrics, prefix=''):
    """Print evaluation metrics"""
    print(f"\n{prefix} Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def create_timestamped_filename(base_name, extension):
    """Create filename with timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{base_name}_{timestamp}.{extension}"
