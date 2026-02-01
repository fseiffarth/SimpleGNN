"""SimpleGNN - A simple framework for benchmarking Graph Neural Networks."""

__version__ = "0.1.0"

from simplegnn.models import BaseGNN, GCN, GAT, GraphSAGE
from simplegnn.datasets import BaseDataset, PlanetoidDataset
from simplegnn.benchmark import Benchmark, BenchmarkConfig
from simplegnn.metrics import Metrics

__all__ = [
    "BaseGNN",
    "GCN",
    "GAT",
    "GraphSAGE",
    "BaseDataset",
    "PlanetoidDataset",
    "Benchmark",
    "BenchmarkConfig",
    "Metrics",
]
