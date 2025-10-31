"""
Main execution script for Financial Risk Model.
Handles training, evaluation, and inference.
"""
import os
import argparse
import torch
import numpy as np
from typing import Optional

from config import Config, set_seed
from data_loader import load_and_preprocess_data, create_data_loaders
from models.hybrid import HybridRiskModel
from train import Trainer
from utils import (
    ensure_dir, save_metrics, plot_training_history,
    SHAPExplainer, load_checkpoint
)
from fix_financial import preprocess_financial_data


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Financial Risk Model')
    
    # Mode
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'evaluate', 'predict', 'preprocess'],
                       help='Execution mode')
    
    # Data arguments
    parser.add_argument('--data-path', type=str, default='./data/financial_data.csv',
                       help='Path to data file')
    parser.add_argument('--output-path', type=str, default='./data/processed_data.csv',
                       help='Path to save processed data')
    
    # Model arguments
    parser.add_argument('--load-checkpoint', type=str, default=None,
                       help='Path to checkpoint to load')
    parser.add_argument('--save-dir', type=str, default='./results',
                       help='Directory to save results')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    
    # Graph construction
    parser.add_argument('--graph-method', type=str, default='correlation',
                       choices=['correlation', 'knn', 'predefined'],
                       help='Graph construction method')
    parser.add_argument('--correlation-threshold', type=float, default=0.7,
                       help='Correlation threshold for graph edges')
    parser.add_argument('--knn-k', type=int, default=10,
                       help='K for KNN graph construction')
    
    # SHAP analysis
    parser.add_argument('--explain', action='store_true',
                       help='Generate SHAP explanations')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--no-cuda', action='store_true',
                       help='Disable CUDA')
    
    return parser.parse_args()


def setup_model(config: Config) -> HybridRiskModel:
    """
    Setup the hybrid risk model.
    
    Args:
        config: Configuration object
        
    Returns:
        Initialized model
    """
    model = HybridRiskModel(
        num_temporal_features=config.NUM_FEATURES,
        sequence_length=config.SEQUENCE_LENGTH,
        transformer_dim=config.TRANSFORMER_DIM,
        transformer_heads=config.TRANSFORMER_HEADS,
        transformer_layers=config.TRANSFORMER_LAYERS,
        transformer_ff_dim=config.TRANSFORMER_FF_DIM,
        transformer_dropout=config.TRANSFORMER_DROPOUT,
        num_node_features=config.NUM_NODE_FEATURES,
        gnn_hidden_dim=config.GNN_HIDDEN_DIM,
        gnn_layers=config.GNN_LAYERS,
        gnn_dropout=config.GNN_DROPOUT,
        gnn_type=config.GNN_TYPE,
        gnn_heads=config.GNN_HEADS,
        fusion_method=config.FUSION_METHOD,
        hybrid_hidden_dim=config.HYBRID_HIDDEN_DIM,
        hybrid_dropout=config.HYBRID_DROPOUT,
        num_classes=config.NUM_CLASSES
    )
    
    return model


def preprocess_mode(args):
    """Run preprocessing mode."""
    print("=" * 50)
    print("Preprocessing Financial Data")
    print("=" * 50)
    
    ensure_dir(os.path.dirname(args.output_path))
    
    preprocess_financial_data(
        data_path=args.data_path,
        output_path=args.output_path,
        price_col='price',
        entity_col='entity_id',
        time_col='time_step',
        label_method='volatility'
    )
    
    print(f"\nPreprocessed data saved to: {args.output_path}")


def train_mode(args):
    """Run training mode."""
    print("=" * 50)
    print("Training Financial Risk Model")
    print("=" * 50)
    
    # Set random seed
    set_seed(args.seed)
    
    # Update config with command line arguments
    if args.batch_size is not None:
        Config.BATCH_SIZE = args.batch_size
    if args.epochs is not None:
        Config.NUM_EPOCHS = args.epochs
    if args.lr is not None:
        Config.LEARNING_RATE = args.lr
    if args.no_cuda:
        Config.DEVICE = torch.device('cpu')
    
    # Print configuration
    Config.print_config()
    
    # Create directories
    ensure_dir(Config.CHECKPOINT_DIR)
    ensure_dir(Config.RESULTS_DIR)
    ensure_dir(Config.LOG_DIR)
    
    # Load and preprocess data
    print("\nLoading data...")
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at {args.data_path}")
        print("Please provide a valid data file or run in 'preprocess' mode first.")
        return
    
    temporal_data, graph_data, labels = load_and_preprocess_data(
        data_path=args.data_path,
        sequence_length=Config.SEQUENCE_LENGTH,
        num_features=Config.NUM_FEATURES,
        graph_method=args.graph_method,
        threshold=args.correlation_threshold,
        k=args.knn_k
    )
    
    print(f"Temporal data shape: {temporal_data.shape}")
    print(f"Graph data: {graph_data}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label distribution: {np.bincount(labels)}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        temporal_data,
        graph_data,
        labels,
        batch_size=Config.BATCH_SIZE,
        train_split=Config.TRAIN_SPLIT,
        val_split=Config.VAL_SPLIT,
        test_split=Config.TEST_SPLIT,
        random_seed=args.seed
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Setup model
    print("\nSetting up model...")
    model = setup_model(Config)
    
    # Load checkpoint if provided
    if args.load_checkpoint is not None and os.path.exists(args.load_checkpoint):
        print(f"Loading checkpoint from {args.load_checkpoint}")
        epoch, metrics = load_checkpoint(model, None, args.load_checkpoint)
        print(f"Loaded checkpoint from epoch {epoch}")
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=Config
    )
    
    # Train
    history = trainer.train()
    
    # Plot training history
    history_plot_path = os.path.join(Config.RESULTS_DIR, 'training_history.png')
    plot_training_history(history, history_plot_path)
    
    # Evaluate on test set
    test_metrics = trainer.evaluate(Config.RESULTS_DIR)
    
    # SHAP explanations
    if args.explain:
        print("\nGenerating SHAP explanations...")
        generate_shap_explanations(model, test_loader, Config)
    
    print("\nTraining complete!")


def evaluate_mode(args):
    """Run evaluation mode."""
    print("=" * 50)
    print("Evaluating Financial Risk Model")
    print("=" * 50)
    
    if args.load_checkpoint is None:
        print("Error: Please provide a checkpoint path with --load-checkpoint")
        return
    
    # Set random seed
    set_seed(args.seed)
    
    # Load data
    print("\nLoading data...")
    temporal_data, graph_data, labels = load_and_preprocess_data(
        data_path=args.data_path,
        sequence_length=Config.SEQUENCE_LENGTH,
        num_features=Config.NUM_FEATURES,
        graph_method=args.graph_method,
        threshold=args.correlation_threshold,
        k=args.knn_k
    )
    
    # Create data loaders
    _, _, test_loader = create_data_loaders(
        temporal_data,
        graph_data,
        labels,
        batch_size=Config.BATCH_SIZE,
        train_split=Config.TRAIN_SPLIT,
        val_split=Config.VAL_SPLIT,
        test_split=Config.TEST_SPLIT,
        random_seed=args.seed
    )
    
    # Setup model
    model = setup_model(Config)
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {args.load_checkpoint}")
    optimizer = torch.optim.Adam(model.parameters())  # Dummy optimizer
    epoch, metrics = load_checkpoint(model, optimizer, args.load_checkpoint)
    print(f"Loaded checkpoint from epoch {epoch}")
    
    # Evaluate
    trainer = Trainer(
        model=model,
        train_loader=test_loader,  # Dummy
        val_loader=test_loader,  # Dummy
        test_loader=test_loader,
        config=Config
    )
    
    ensure_dir(args.save_dir)
    test_metrics = trainer.evaluate(args.save_dir)
    
    # SHAP explanations
    if args.explain:
        print("\nGenerating SHAP explanations...")
        generate_shap_explanations(model, test_loader, Config)
    
    print("\nEvaluation complete!")


def predict_mode(args):
    """Run prediction mode."""
    print("=" * 50)
    print("Predicting with Financial Risk Model")
    print("=" * 50)
    
    if args.load_checkpoint is None:
        print("Error: Please provide a checkpoint path with --load-checkpoint")
        return
    
    # Load data
    print("\nLoading data...")
    temporal_data, graph_data, labels = load_and_preprocess_data(
        data_path=args.data_path,
        sequence_length=Config.SEQUENCE_LENGTH,
        num_features=Config.NUM_FEATURES,
        graph_method=args.graph_method
    )
    
    # Setup model
    model = setup_model(Config)
    model = model.to(Config.DEVICE)
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {args.load_checkpoint}")
    optimizer = torch.optim.Adam(model.parameters())  # Dummy optimizer
    epoch, metrics = load_checkpoint(model, optimizer, args.load_checkpoint)
    
    # Make predictions
    model.eval()
    temporal_tensor = torch.FloatTensor(temporal_data).to(Config.DEVICE)
    graph_data = graph_data.to(Config.DEVICE)
    
    with torch.no_grad():
        outputs = model(temporal_tensor, graph_data)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = outputs.argmax(dim=1)
    
    # Save predictions
    predictions_dict = {
        'predictions': predictions.cpu().numpy().tolist(),
        'probabilities': probabilities.cpu().numpy().tolist(),
        'labels': labels.tolist()
    }
    
    ensure_dir(args.save_dir)
    output_file = os.path.join(args.save_dir, 'predictions.json')
    save_metrics(predictions_dict, output_file)
    
    print(f"\nPredictions saved to: {output_file}")
    print(f"Prediction distribution: {np.bincount(predictions.cpu().numpy())}")


def generate_shap_explanations(model, data_loader, config):
    """Generate SHAP explanations."""
    # Get a sample of data for SHAP
    sample_temporal = []
    sample_graph = None
    
    for temporal_data, graph_data, _ in data_loader:
        sample_temporal.append(temporal_data)
        if sample_graph is None:
            sample_graph = graph_data
        if len(sample_temporal) * temporal_data.size(0) >= config.SHAP_SAMPLE_SIZE:
            break
    
    sample_temporal = torch.cat(sample_temporal, dim=0)[:config.SHAP_SAMPLE_SIZE]
    
    # Create SHAP explainer
    print("Initializing SHAP explainer...")
    explainer = SHAPExplainer(
        model,
        (sample_temporal, sample_graph),
        config.DEVICE
    )
    
    # Explain first few instances
    print("Generating SHAP values...")
    test_temporal = sample_temporal[:10]
    shap_values, base_values = explainer.explain_instance(test_temporal, sample_graph)
    
    # Plot SHAP summary
    shap_summary_path = os.path.join(config.RESULTS_DIR, 'shap_summary.png')
    explainer.plot_summary(
        shap_values,
        test_temporal.cpu().numpy(),
        save_path=shap_summary_path
    )
    
    # Plot SHAP waterfall for first instance
    shap_waterfall_path = os.path.join(config.RESULTS_DIR, 'shap_waterfall.png')
    explainer.plot_waterfall(
        shap_values,
        test_temporal.cpu().numpy(),
        instance_idx=0,
        save_path=shap_waterfall_path
    )
    
    print("SHAP explanations generated!")


def main():
    """Main function."""
    args = parse_args()
    
    if args.mode == 'preprocess':
        preprocess_mode(args)
    elif args.mode == 'train':
        train_mode(args)
    elif args.mode == 'evaluate':
        evaluate_mode(args)
    elif args.mode == 'predict':
        predict_mode(args)
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == '__main__':
    main()
