"""
Main entry point for the financial risk model.
Orchestrates training, evaluation, and SHAP explanation.
"""

import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
import config
from data_loader import get_data_loaders
from train import FinancialRiskModel, train_model, evaluate
from utils import set_seed, save_model, save_results, plot_training_history
import os


def explain_with_shap(model: FinancialRiskModel, test_loader, device: torch.device):
    """
    Generate SHAP explanations for model predictions.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run on
    """
    print("\nGenerating SHAP explanations...")
    
    # Prepare data for SHAP
    model.eval()
    
    # Get background and test samples
    background_data = []
    test_data = []
    
    for i, (time_series, graph_features, adj_matrix, targets) in enumerate(test_loader):
        if i == 0:
            # Use first batch as background
            background_data.append((time_series, graph_features, adj_matrix))
        if i <= 1:
            # Use first two batches for test samples
            test_data.append((time_series, graph_features, adj_matrix, targets))
        if i >= 1:
            break
    
    # Prepare background samples
    bg_ts, bg_gf, bg_adj = background_data[0]
    bg_ts = bg_ts[:config.SHAP_BACKGROUND_SIZE].to(device)
    bg_gf = bg_gf[:config.SHAP_BACKGROUND_SIZE].to(device)
    bg_adj = bg_adj.to(device)
    
    # Prepare test samples
    test_ts = test_data[0][0][:config.SHAP_TEST_SIZE].to(device)
    test_gf = test_data[0][1][:config.SHAP_TEST_SIZE].to(device)
    test_adj = test_data[0][2].to(device)
    
    # Create wrapper function for SHAP
    def model_wrapper(combined_input):
        """Wrapper function for SHAP that takes combined input."""
        batch_size = combined_input.shape[0]
        
        # Split input back into time series and graph features
        ts_size = config.SEQUENCE_LENGTH * config.N_FEATURES
        ts_data = combined_input[:, :ts_size].reshape(batch_size, config.SEQUENCE_LENGTH, config.N_FEATURES)
        gf_data = combined_input[:, ts_size:].reshape(batch_size, config.N_GRAPH_NODES, config.N_FEATURES)
        
        ts_tensor = torch.FloatTensor(ts_data).to(device)
        gf_tensor = torch.FloatTensor(gf_data).to(device)
        
        with torch.no_grad():
            outputs = model(ts_tensor, gf_tensor, test_adj)
        
        return outputs.cpu().numpy()
    
    # Flatten inputs for SHAP
    bg_combined = np.concatenate([
        bg_ts.cpu().numpy().reshape(bg_ts.shape[0], -1),
        bg_gf.cpu().numpy().reshape(bg_gf.shape[0], -1)
    ], axis=1)
    
    test_combined = np.concatenate([
        test_ts.cpu().numpy().reshape(test_ts.shape[0], -1),
        test_gf.cpu().numpy().reshape(test_gf.shape[0], -1)
    ], axis=1)
    
    # Create SHAP explainer
    explainer = shap.KernelExplainer(model_wrapper, bg_combined)
    
    # Calculate SHAP values
    print("Calculating SHAP values (this may take a while)...")
    shap_values = explainer.shap_values(test_combined[:10])  # Use subset for speed
    
    # Create SHAP summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, test_combined[:10], show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(config.SHAP_PLOT_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"SHAP explanation plot saved to {config.SHAP_PLOT_PATH}")
    
    # Calculate feature importance
    feature_importance = np.abs(shap_values).mean(axis=0)
    top_features_idx = np.argsort(feature_importance)[-10:][::-1]
    
    print("\nTop 10 most important features:")
    for i, idx in enumerate(top_features_idx):
        print(f"  {i+1}. Feature {idx}: {feature_importance[idx]:.4f}")
    
    return shap_values, feature_importance


def main():
    """
    Main function to run the financial risk model.
    """
    print("=" * 60)
    print("Financial Risk Model - Transformer + GNN + SHAP")
    print("=" * 60)
    
    # Create directories if they don't exist
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Set random seed for reproducibility
    set_seed(config.RANDOM_SEED)
    print(f"\nRandom seed set to {config.RANDOM_SEED}")
    
    # Set device
    device = config.DEVICE
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
        batch_size=config.BATCH_SIZE,
        n_samples=1000
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nInitializing model...")
    model = FinancialRiskModel()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        device=device,
        patience=config.PATIENCE
    )
    
    # Save model
    save_model(model, config.CHECKPOINT_PATH)
    
    # Plot training history
    plot_training_history(
        history['train_losses'],
        history['val_losses'],
        f"{config.RESULTS_DIR}training_history.png"
    )
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluation")
    print("=" * 60)
    criterion = torch.nn.MSELoss()
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  MSE: {test_metrics['mse']:.4f}")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  MAE: {test_metrics['mae']:.4f}")
    print(f"  R²: {test_metrics['r2']:.4f}")
    
    # SHAP explanations
    print("\n" + "=" * 60)
    print("SHAP Explanations")
    print("=" * 60)
    try:
        shap_values, feature_importance = explain_with_shap(model, test_loader, device)
    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        print("Continuing without SHAP explanations...")
        shap_values = None
        feature_importance = None
    
    # Save results
    results = {
        'test_loss': test_loss,
        'test_metrics': test_metrics,
        'best_val_loss': history['best_val_loss'],
        'config': {
            'n_features': config.N_FEATURES,
            'sequence_length': config.SEQUENCE_LENGTH,
            'n_graph_nodes': config.N_GRAPH_NODES,
            'transformer_d_model': config.TRANSFORMER_D_MODEL,
            'gnn_hidden_dim': config.GNN_HIDDEN_DIM,
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE
        }
    }
    
    if feature_importance is not None:
        results['top_features'] = {
            f'feature_{i}': float(importance) 
            for i, importance in enumerate(feature_importance[:20])
        }
    
    save_results(results, config.RESULTS_PATH)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {config.RESULTS_DIR}")
    print(f"Model saved to: {config.CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
