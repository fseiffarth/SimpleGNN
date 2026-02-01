"""
Basic example: Training a single GNN model on a dataset.

This example demonstrates how to:
1. Load a predefined dataset (Cora)
2. Create a predefined GNN model (GCN)
3. Train and evaluate the model
"""

from simplegnn import GCN, PlanetoidDataset, Benchmark, BenchmarkConfig


def main():
    # Create a configuration
    config = BenchmarkConfig(
        hidden_channels=64,
        num_layers=2,
        dropout=0.5,
        learning_rate=0.01,
        num_epochs=200,
        early_stopping_patience=20,
        verbose=True,
        seed=42
    )
    
    # Load dataset
    print("Loading Cora dataset...")
    dataset = PlanetoidDataset(name="Cora")
    print(dataset)
    print(f"\nDataset statistics: {dataset.get_stats()}\n")
    
    # Create model
    model = GCN(
        in_channels=dataset.num_features,
        hidden_channels=config.hidden_channels,
        out_channels=dataset.num_classes,
        num_layers=config.num_layers,
        dropout=config.dropout
    )
    print(f"Model: {model}\n")
    
    # Create benchmark and train
    benchmark = Benchmark(config)
    result = benchmark.train(model, dataset)
    
    # Print results
    print("\nDetailed Results:")
    print(f"Train Accuracy: {result['train_metrics']['accuracy']:.4f}")
    print(f"Val Accuracy:   {result['val_metrics']['accuracy']:.4f}")
    print(f"Test Accuracy:  {result['test_metrics']['accuracy']:.4f}")
    print(f"Test F1 Score:  {result['test_metrics']['f1_macro']:.4f}")
    print(f"Training Time:  {result['total_time']:.2f}s")


if __name__ == "__main__":
    main()
