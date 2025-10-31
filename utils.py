"""
Utility functions for data processing, visualization, and SHAP integration.
"""
import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import shap

from config import Config


def ensure_dir(directory: str) -> None:
    """Ensure directory exists, create if not."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_metrics(metrics: Dict[str, float], filepath: str) -> None:
    """Save metrics to JSON file."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {filepath}")


def load_metrics(filepath: str) -> Dict[str, float]:
    """Load metrics from JSON file."""
    with open(filepath, 'r') as f:
        metrics = json.load(f)
    return metrics


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional, for AUC-ROC)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    # Add AUC-ROC if probabilities provided
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) == 2:  # Binary classification
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
            else:  # Multi-class
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
        except Exception as e:
            print(f"Warning: Could not calculate AUC-ROC: {e}")
            metrics['auc_roc'] = 0.0
    
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         save_path: str, class_names: Optional[List[str]] = None) -> None:
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names or range(len(cm)),
                yticklabels=class_names or range(len(cm)))
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_training_history(history: Dict[str, List[float]], save_path: str) -> None:
    """Plot and save training history."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy
    if 'train_acc' in history:
        axes[1].plot(history['train_acc'], label='Train Accuracy')
    if 'val_acc' in history:
        axes[1].plot(history['val_acc'], label='Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {save_path}")


class SHAPExplainer:
    """SHAP explainer wrapper for the hybrid model."""
    
    def __init__(self, model, background_data: Tuple, device: torch.device):
        """
        Initialize SHAP explainer.
        
        Args:
            model: The trained model
            background_data: Tuple of (temporal_features, graph_data) for background
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.background_data = background_data
        
        # Prepare model for SHAP
        self.model.eval()
        
    def explain_instance(self, temporal_features: torch.Tensor, 
                        graph_data: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Explain a single instance using SHAP.
        
        Args:
            temporal_features: Temporal input features
            graph_data: Graph data object
            
        Returns:
            Tuple of (shap_values, base_values)
        """
        # Create a wrapper function for SHAP
        def model_wrapper(temporal_input):
            """Wrapper function for SHAP that takes numpy array."""
            temporal_tensor = torch.FloatTensor(temporal_input).to(self.device)
            with torch.no_grad():
                outputs = self.model(temporal_tensor, graph_data)
                probs = torch.softmax(outputs, dim=1)
            return probs.cpu().numpy()
        
        # Initialize DeepExplainer
        background_temporal, _ = self.background_data
        explainer = shap.DeepExplainer(
            model_wrapper,
            background_temporal.cpu().numpy()[:Config.SHAP_SAMPLE_SIZE]
        )
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(temporal_features.cpu().numpy())
        
        return shap_values, explainer.expected_value
    
    def plot_summary(self, shap_values: np.ndarray, features: np.ndarray,
                    feature_names: Optional[List[str]] = None,
                    save_path: str = None) -> None:
        """
        Plot SHAP summary plot.
        
        Args:
            shap_values: SHAP values from explain_instance
            features: Original features
            feature_names: Names of features
            save_path: Path to save plot
        """
        plt.figure(figsize=(12, 8))
        
        # Handle multi-class case
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class for binary
        
        # Flatten temporal features for visualization
        if len(features.shape) == 3:  # (batch, seq_len, features)
            features_flat = features.reshape(features.shape[0], -1)
            shap_flat = shap_values.reshape(shap_values.shape[0], -1)
        else:
            features_flat = features
            shap_flat = shap_values
        
        shap.summary_plot(
            shap_flat,
            features_flat,
            feature_names=feature_names,
            max_display=Config.SHAP_MAX_DISPLAY,
            show=False
        )
        
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP summary plot saved to {save_path}")
        plt.close()
    
    def plot_waterfall(self, shap_values: np.ndarray, features: np.ndarray,
                      feature_names: Optional[List[str]] = None,
                      instance_idx: int = 0,
                      save_path: str = None) -> None:
        """
        Plot SHAP waterfall plot for a single instance.
        
        Args:
            shap_values: SHAP values
            features: Original features
            feature_names: Names of features
            instance_idx: Index of instance to explain
            save_path: Path to save plot
        """
        # Handle multi-class case
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
        
        # Flatten features
        if len(features.shape) == 3:
            features_flat = features.reshape(features.shape[0], -1)
            shap_flat = shap_values.reshape(shap_values.shape[0], -1)
        else:
            features_flat = features
            shap_flat = shap_values
        
        plt.figure(figsize=(10, 8))
        
        # Create explanation object
        explanation = shap.Explanation(
            values=shap_flat[instance_idx],
            base_values=shap_flat.mean(axis=0).mean(),
            data=features_flat[instance_idx],
            feature_names=feature_names
        )
        
        shap.plots.waterfall(explanation, show=False, max_display=15)
        
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP waterfall plot saved to {save_path}")
        plt.close()


def save_checkpoint(model, optimizer, epoch: int, metrics: Dict[str, float],
                   filepath: str) -> None:
    """Save model checkpoint."""
    ensure_dir(os.path.dirname(filepath))
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, filepath: str) -> Tuple[int, Dict[str, float]]:
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']
    print(f"Checkpoint loaded from {filepath}")
    return epoch, metrics


def get_feature_names(num_features: int, sequence_length: int) -> List[str]:
    """Generate feature names for temporal features."""
    feature_names = []
    for t in range(sequence_length):
        for f in range(num_features):
            feature_names.append(f"t{t}_f{f}")
    return feature_names
