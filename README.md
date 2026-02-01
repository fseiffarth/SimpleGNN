# SimpleGNN

A simple and extensible framework for benchmarking Graph Neural Networks (GNNs) on various datasets. SimpleGNN enables fair and reproducible comparisons between different GNN architectures with minimal code.

## Features

- **Predefined GNN Models**: Ready-to-use implementations of popular GNN architectures
  - GCN (Graph Convolutional Network)
  - GAT (Graph Attention Network)
  - GraphSAGE
- **Easy Custom Models**: Simple interface for implementing and benchmarking your own GNN architectures
- **Standard Benchmarks**: Built-in support for classic citation network datasets
  - Cora
  - CiteSeer
  - PubMed
- **Custom Datasets**: Easy integration of your own graph datasets
- **Standardized Evaluation**: Consistent metrics and evaluation protocols for fair comparison
- **Reproducibility**: Built-in support for seeding and configuration management

## Installation

```bash
# Clone the repository
git clone https://github.com/fseiffarth/SimpleGNN.git
cd SimpleGNN

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage: Training a Single Model

```python
from simplegnn import GCN, PlanetoidDataset, Benchmark, BenchmarkConfig

# Configure the benchmark
config = BenchmarkConfig(
    hidden_channels=64,
    num_layers=2,
    dropout=0.5,
    learning_rate=0.01,
    num_epochs=200,
    seed=42
)

# Load a dataset
dataset = PlanetoidDataset(name="Cora")

# Create a model
model = GCN(
    in_channels=dataset.num_features,
    hidden_channels=config.hidden_channels,
    out_channels=dataset.num_classes,
    num_layers=config.num_layers,
    dropout=config.dropout
)

# Train and evaluate
benchmark = Benchmark(config)
result = benchmark.train(model, dataset)

print(f"Test Accuracy: {result['test_metrics']['accuracy']:.4f}")
```

### Comparing Multiple Models

```python
from simplegnn import GCN, GAT, GraphSAGE, PlanetoidDataset, Benchmark, BenchmarkConfig

config = BenchmarkConfig(hidden_channels=64, num_epochs=200)
dataset = PlanetoidDataset(name="Cora")

# Create multiple models
models = [
    GCN(dataset.num_features, 64, dataset.num_classes),
    GAT(dataset.num_features, 8, dataset.num_classes, num_heads=8),
    GraphSAGE(dataset.num_features, 64, dataset.num_classes),
]

# Compare models
benchmark = Benchmark(config)
comparison_df = benchmark.compare_models(models, dataset)

# Save results
benchmark.save_results("comparison_results.csv")
```

### Creating Custom Models

```python
from simplegnn import BaseGNN, Benchmark, PlanetoidDataset
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class MyCustomGNN(BaseGNN):
    """Your custom GNN implementation."""
    
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_layers=2, dropout=0.5, **kwargs):
        super().__init__(in_channels, hidden_channels, out_channels, 
                        num_layers, dropout)
        
        # Define your layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index, edge_weight=None):
        # Implement your forward pass
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Use your custom model
dataset = PlanetoidDataset(name="Cora")
model = MyCustomGNN(dataset.num_features, 64, dataset.num_classes)

benchmark = Benchmark()
result = benchmark.train(model, dataset)
```

### Creating Custom Datasets

```python
from simplegnn import BaseDataset, GCN, Benchmark
import torch
from torch_geometric.data import Data

class MyCustomDataset(BaseDataset):
    """Your custom dataset."""
    
    def __init__(self, name="MyDataset"):
        super().__init__(name)
    
    def load(self) -> Data:
        # Load or generate your graph data
        x = torch.randn(100, 16)  # 100 nodes, 16 features
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        y = torch.randint(0, 3, (100,))  # 3 classes
        
        # Create splits
        train_mask, val_mask, test_mask = self.split_data()
        
        return Data(x=x, edge_index=edge_index, y=y,
                   train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

# Use your custom dataset
dataset = MyCustomDataset()
model = GCN(dataset.num_features, 32, dataset.num_classes)

benchmark = Benchmark()
result = benchmark.train(model, dataset)
```

## Examples

More detailed examples are available in the `examples/` directory:

- `basic_example.py`: Train a single GNN model
- `comparison_example.py`: Compare multiple GNN architectures
- `custom_model_example.py`: Implement and benchmark a custom GNN
- `custom_dataset_example.py`: Use a custom dataset

Run an example:
```bash
cd examples
python basic_example.py
```

## Configuration Options

The `BenchmarkConfig` class provides extensive configuration options:

```python
from simplegnn import BenchmarkConfig

config = BenchmarkConfig(
    # Model architecture
    hidden_channels=64,
    num_layers=2,
    dropout=0.5,
    
    # Training
    learning_rate=0.01,
    weight_decay=5e-4,
    num_epochs=200,
    early_stopping_patience=20,
    
    # Optimization
    optimizer="adam",  # 'adam', 'sgd', 'adamw'
    
    # Device
    device="auto",  # 'auto', 'cpu', 'cuda', 'mps'
    
    # Reproducibility
    seed=42,
    
    # Logging
    verbose=True,
    log_interval=10
)
```

## Evaluation Metrics

SimpleGNN provides comprehensive evaluation metrics:

- **Accuracy**: Overall classification accuracy
- **F1 Score**: Both macro and micro averaged F1 scores
- **Precision**: Macro averaged precision
- **Recall**: Macro averaged recall
- **Training Time**: Total time for training

All metrics are automatically computed and reported for train, validation, and test sets.

## Project Structure

```
SimpleGNN/
├── src/simplegnn/
│   ├── models/          # GNN model implementations
│   │   ├── base.py      # Base model interface
│   │   ├── gcn.py       # GCN implementation
│   │   ├── gat.py       # GAT implementation
│   │   └── graphsage.py # GraphSAGE implementation
│   ├── datasets/        # Dataset loaders
│   │   ├── base.py      # Base dataset interface
│   │   └── planetoid.py # Planetoid datasets (Cora, etc.)
│   ├── benchmark/       # Benchmarking framework
│   │   ├── config.py    # Configuration management
│   │   └── benchmark.py # Training and evaluation
│   └── metrics.py       # Evaluation metrics
├── examples/            # Example scripts
├── tests/              # Unit tests
└── README.md           # This file
```

## Contributing

Contributions are welcome! To add a new model or dataset:

1. **New Model**: Extend `BaseGNN` and implement the `forward` method
2. **New Dataset**: Extend `BaseDataset` and implement the `load` method
3. Submit a pull request with tests and documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use SimpleGNN in your research, please cite:

```bibtex
@software{simplegnn2026,
  title={SimpleGNN: A Simple Framework for Benchmarking Graph Neural Networks},
  author={SimpleGNN Contributors},
  year={2026},
  url={https://github.com/fseiffarth/SimpleGNN}
}
```

## Acknowledgments

SimpleGNN is built on top of [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/), an excellent library for deep learning on graphs.
