"""
Custom model example: Creating and benchmarking a custom GNN model.

This example demonstrates how to:
1. Create a custom GNN model by extending BaseGNN
2. Benchmark it alongside predefined models
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from simplegnn import BaseGNN, GCN, PlanetoidDataset, Benchmark, BenchmarkConfig


class CustomGNN(BaseGNN):
    """Custom GNN with residual connections."""

    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_layers=2, dropout=0.5, **kwargs):
        super().__init__(in_channels, hidden_channels, out_channels, 
                        num_layers, dropout)
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
        # Residual projection layers
        self.residual_projs = torch.nn.ModuleList()
        if in_channels != hidden_channels:
            self.residual_projs.append(torch.nn.Linear(in_channels, hidden_channels))
        else:
            self.residual_projs.append(torch.nn.Identity())
        
        for _ in range(num_layers - 2):
            self.residual_projs.append(torch.nn.Identity())

    def forward(self, x, edge_index, edge_weight=None):
        """Forward pass with residual connections."""
        # Process all layers except the last with residual connections
        for i, conv in enumerate(self.convs[:-1]):
            identity = self.residual_projs[i](x)
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = x + identity  # Residual connection
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer (no residual)
        x = self.convs[-1](x, edge_index, edge_weight)
        return x


def main():
    # Create configuration
    config = BenchmarkConfig(
        hidden_channels=64,
        num_layers=3,
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
    
    # Create models: custom vs. standard GCN
    models = [
        GCN(
            in_channels=dataset.num_features,
            hidden_channels=config.hidden_channels,
            out_channels=dataset.num_classes,
            num_layers=config.num_layers,
            dropout=config.dropout
        ),
        CustomGNN(
            in_channels=dataset.num_features,
            hidden_channels=config.hidden_channels,
            out_channels=dataset.num_classes,
            num_layers=config.num_layers,
            dropout=config.dropout
        ),
    ]
    
    # Benchmark and compare
    benchmark = Benchmark(config)
    comparison_df = benchmark.compare_models(models, dataset)
    
    print("\nCustom model successfully benchmarked!")


if __name__ == "__main__":
    main()
