"""
Comparison example: Comparing multiple GNN architectures.

This example demonstrates how to:
1. Compare multiple predefined GNN models (GCN, GAT, GraphSAGE)
2. Evaluate them on the same dataset
3. Generate a comparison table
"""

from simplegnn import GCN, GAT, GraphSAGE, PlanetoidDataset, Benchmark, BenchmarkConfig


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
    
    # Create models
    models = [
        GCN(
            in_channels=dataset.num_features,
            hidden_channels=config.hidden_channels,
            out_channels=dataset.num_classes,
            num_layers=config.num_layers,
            dropout=config.dropout
        ),
        GAT(
            in_channels=dataset.num_features,
            hidden_channels=8,  # 8 channels per head
            out_channels=dataset.num_classes,
            num_layers=config.num_layers,
            dropout=config.dropout,
            num_heads=8
        ),
        GraphSAGE(
            in_channels=dataset.num_features,
            hidden_channels=config.hidden_channels,
            out_channels=dataset.num_classes,
            num_layers=config.num_layers,
            dropout=config.dropout
        ),
    ]
    
    # Create benchmark and compare models
    benchmark = Benchmark(config)
    comparison_df = benchmark.compare_models(models, dataset)
    
    # Save results
    benchmark.save_results("results.csv")
    print("\nComparison results saved to results.csv")


if __name__ == "__main__":
    main()
