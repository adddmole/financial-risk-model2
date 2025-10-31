"""
Main entry point for Financial Risk Model
"""
import argparse
import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
import os

from config import (
    TRANSFORMER_CONFIG, GNN_CONFIG, HYBRID_CONFIG,
    TRAINING_CONFIG, SHAP_CONFIG, RESULTS_DIR, RISK_THRESHOLDS
)
from utils import (
    set_seed, get_device, load_model, print_metrics,
    save_predictions, calculate_risk_score
)
from data_loader import load_and_split_data, get_sample_for_shap
from models import HybridRiskModel
from train import Trainer, evaluate_model
from fix_financial import FinancialDataFixer, validate_financial_data


def setup_model(input_dim=10):
    """
    Setup the hybrid risk model
    
    Args:
        input_dim: Input feature dimension
    
    Returns:
        Initialized model
    """
    # Update configs with input dimension
    transformer_config = TRANSFORMER_CONFIG.copy()
    transformer_config['input_dim'] = input_dim
    
    gnn_config = GNN_CONFIG.copy()
    gnn_config['node_features'] = input_dim
    
    # Create model
    model = HybridRiskModel(
        transformer_config=transformer_config,
        gnn_config=gnn_config,
        fusion_dim=HYBRID_CONFIG['fusion_dim'],
        output_dim=HYBRID_CONFIG['output_dim']
    )
    
    return model


def train_model(args):
    """Train the financial risk model"""
    print("="*60)
    print("TRAINING FINANCIAL RISK MODEL")
    print("="*60)
    
    # Set seed
    set_seed()
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader, scaler, _ = load_and_split_data(
        batch_size=TRAINING_CONFIG['batch_size']
    )
    
    # Get input dimension from first batch
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch['features'].shape[-1]
    print(f"Input dimension: {input_dim}")
    
    # Setup model
    print("\nSetting up model...")
    model = setup_model(input_dim=input_dim)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Get device
    device = get_device()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay'],
        device=device
    )
    
    # Train
    print("\nStarting training...")
    history = trainer.train(
        epochs=TRAINING_CONFIG['epochs'],
        early_stopping_patience=TRAINING_CONFIG['early_stopping_patience']
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    model = load_model(model, 'best_model.pt')
    metrics, predictions, targets = evaluate_model(model, test_loader, device)
    
    print_metrics(metrics, prefix='Test')
    
    # Save predictions
    save_predictions(predictions, targets)
    
    print("\nTraining completed successfully!")
    
    return model, test_loader


def explain_model(model, test_loader, args):
    """Generate SHAP explanations for model predictions"""
    print("\n" + "="*60)
    print("GENERATING SHAP EXPLANATIONS")
    print("="*60)
    
    device = get_device()
    model = model.to(device)
    model.eval()
    
    # Get sample data
    print("\nPreparing data for SHAP analysis...")
    sample_data = []
    for i, batch in enumerate(test_loader):
        if i * test_loader.batch_size >= SHAP_CONFIG['num_samples']:
            break
        sample_data.append(batch)
    
    # Extract features from first batch for background
    background_batch = sample_data[0]
    background_features = background_batch['features'][:10].to(device)
    background_adj = background_batch['graph'][:10].to(device) if 'graph' in background_batch else None
    
    if background_adj is None:
        batch_size = background_features.size(0)
        num_nodes = background_features.size(-1)
        background_adj = torch.eye(num_nodes).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    
    # Create a wrapper function for SHAP
    def model_predict(features_np):
        """Wrapper for model prediction"""
        features = torch.FloatTensor(features_np).to(device)
        
        # Handle adjacency matrix
        batch_size = features.size(0)
        num_nodes = features.size(-1)
        adj = torch.eye(num_nodes).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        
        with torch.no_grad():
            predictions = model(features, adj)
        
        return predictions.cpu().numpy()
    
    print("\nCalculating SHAP values...")
    try:
        # Reshape for SHAP (SHAP expects 2D input)
        background_2d = background_features.cpu().numpy().reshape(
            background_features.size(0), -1
        )
        
        # Create explainer
        explainer = shap.KernelExplainer(
            lambda x: model_predict(x.reshape(-1, *background_features.shape[1:])),
            background_2d[:5]  # Use subset as background
        )
        
        # Get test samples
        test_batch = sample_data[1] if len(sample_data) > 1 else sample_data[0]
        test_features = test_batch['features'][:5].cpu().numpy()
        test_2d = test_features.reshape(test_features.shape[0], -1)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(test_2d)
        
        # Plot summary
        print("\nGenerating SHAP summary plot...")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values, 
            test_2d,
            show=False,
            max_display=min(20, test_2d.shape[1])
        )
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'shap_summary.png'))
        plt.close()
        
        print(f"SHAP summary plot saved to {RESULTS_DIR}/shap_summary.png")
        
    except Exception as e:
        print(f"Warning: Could not generate SHAP explanations: {e}")
        print("This is optional and does not affect model training.")
    
    print("\nSHAP analysis completed!")


def predict_risk(model, features, adj_matrix=None):
    """
    Predict risk for new data
    
    Args:
        model: Trained model
        features: Input features
        adj_matrix: Adjacency matrix (optional)
    
    Returns:
        Risk prediction and category
    """
    device = get_device()
    model = model.to(device)
    model.eval()
    
    # Prepare input
    if not isinstance(features, torch.Tensor):
        features = torch.FloatTensor(features)
    
    features = features.to(device)
    
    if adj_matrix is None:
        num_nodes = features.size(-1)
        adj_matrix = torch.eye(num_nodes).unsqueeze(0).to(device)
    elif not isinstance(adj_matrix, torch.Tensor):
        adj_matrix = torch.FloatTensor(adj_matrix).to(device)
    
    # Predict
    with torch.no_grad():
        prediction = model(features.unsqueeze(0), adj_matrix)
        prediction = prediction.item()
    
    # Calculate risk category
    risk_category = calculate_risk_score(prediction, RISK_THRESHOLDS)
    
    return prediction, risk_category


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Financial Risk Model')
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['train', 'evaluate', 'explain', 'all'],
        help='Mode: train, evaluate, explain, or all'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='best_model.pt',
        help='Path to saved model'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'train' or args.mode == 'all':
        # Train model
        model, test_loader = train_model(args)
        
        if args.mode == 'all':
            # Generate explanations
            explain_model(model, test_loader, args)
    
    elif args.mode == 'evaluate':
        # Load data
        _, _, test_loader, _, _ = load_and_split_data(
            batch_size=TRAINING_CONFIG['batch_size']
        )
        
        # Load model
        sample_batch = next(iter(test_loader))
        input_dim = sample_batch['features'].shape[-1]
        model = setup_model(input_dim=input_dim)
        model = load_model(model, args.model_path)
        
        # Evaluate
        device = get_device()
        metrics, predictions, targets = evaluate_model(model, test_loader, device)
        print_metrics(metrics, prefix='Test')
        
    elif args.mode == 'explain':
        # Load data
        _, _, test_loader, _, _ = load_and_split_data(
            batch_size=TRAINING_CONFIG['batch_size']
        )
        
        # Load model
        sample_batch = next(iter(test_loader))
        input_dim = sample_batch['features'].shape[-1]
        model = setup_model(input_dim=input_dim)
        model = load_model(model, args.model_path)
        
        # Generate explanations
        explain_model(model, test_loader, args)
    
    print("\n" + "="*60)
    print("PROCESS COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == '__main__':
    main()
