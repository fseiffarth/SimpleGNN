"""
Custom dataset example: Loading and benchmarking on a custom dataset.

This example demonstrates how to:
1. Create a custom dataset by extending BaseDataset
2. Use it with predefined models
"""

import torch
import numpy as np
from torch_geometric.data import Data
from simplegnn import BaseDataset, GCN, Benchmark, BenchmarkConfig


class SyntheticDataset(BaseDataset):
    """Synthetic dataset for demonstration."""

    def __init__(self, num_nodes=1000, num_features=16, num_classes=3, 
                 edge_prob=0.01, seed=42):
        super().__init__("Synthetic")
        self.num_nodes_val = num_nodes
        self.num_features_val = num_features
        self.num_classes_val = num_classes
        self.edge_prob = edge_prob
        self.seed = seed

    def load(self) -> Data:
        """Generate a synthetic graph dataset."""
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        
        # Generate random node features
        x = torch.randn(self.num_nodes_val, self.num_features_val)
        
        # Generate random labels
        y = torch.randint(0, self.num_classes_val, (self.num_nodes_val,))
        
        # Generate random edges (Erdős-Rényi)
        edge_list = []
        for i in range(self.num_nodes_val):
            for j in range(i + 1, self.num_nodes_val):
                if np.random.rand() < self.edge_prob:
                    edge_list.append([i, j])
                    edge_list.append([j, i])  # Undirected
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # Create train/val/test splits
        train_mask, val_mask, test_mask = self.split_data(
            train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=self.seed
        )
        
        return Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask
        )


def main():
    # Create configuration
    config = BenchmarkConfig(
        hidden_channels=32,
        num_layers=2,
        dropout=0.3,
        learning_rate=0.01,
        num_epochs=100,
        verbose=True,
        seed=42
    )
    
    # Create custom dataset
    print("Generating synthetic dataset...")
    dataset = SyntheticDataset(
        num_nodes=500,
        num_features=16,
        num_classes=3,
        edge_prob=0.02,
        seed=42
    )
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
    
    # Benchmark
    benchmark = Benchmark(config)
    result = benchmark.train(model, dataset)
    
    print("\nCustom dataset successfully benchmarked!")
    print(f"Test Accuracy: {result['test_metrics']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
