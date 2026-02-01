"""Tests for benchmark."""

import pytest
import torch
from simplegnn.benchmark import Benchmark, BenchmarkConfig
from simplegnn.models import GCN
from simplegnn.datasets import BaseDataset
from torch_geometric.data import Data


class TinyDataset(BaseDataset):
    """Tiny dataset for fast testing."""
    
    def load(self) -> Data:
        x = torch.randn(30, 8)
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 0]
        ], dtype=torch.long)
        y = torch.randint(0, 2, (30,))
        
        train_mask = torch.zeros(30, dtype=torch.bool)
        val_mask = torch.zeros(30, dtype=torch.bool)
        test_mask = torch.zeros(30, dtype=torch.bool)
        
        train_mask[:15] = True
        val_mask[15:22] = True
        test_mask[22:] = True
        
        return Data(x=x, edge_index=edge_index, y=y,
                   train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)


def test_benchmark_config():
    """Test benchmark configuration."""
    config = BenchmarkConfig(
        hidden_channels=32,
        num_layers=2,
        dropout=0.3,
        learning_rate=0.01,
        num_epochs=10
    )
    
    assert config.hidden_channels == 32
    assert config.num_layers == 2
    assert config.dropout == 0.3
    assert config.learning_rate == 0.01
    assert config.num_epochs == 10


def test_benchmark_config_validation():
    """Test benchmark configuration validation."""
    with pytest.raises(AssertionError):
        BenchmarkConfig(hidden_channels=-1)
    
    with pytest.raises(AssertionError):
        BenchmarkConfig(num_layers=1)
    
    with pytest.raises(AssertionError):
        BenchmarkConfig(dropout=1.5)


def test_benchmark_train():
    """Test benchmark training."""
    config = BenchmarkConfig(
        hidden_channels=16,
        num_layers=2,
        num_epochs=5,
        verbose=False,
        seed=42
    )
    
    dataset = TinyDataset("tiny")
    model = GCN(
        in_channels=dataset.num_features,
        hidden_channels=config.hidden_channels,
        out_channels=dataset.num_classes,
        num_layers=config.num_layers
    )
    
    benchmark = Benchmark(config)
    result = benchmark.train(model, dataset)
    
    assert 'model_name' in result
    assert 'dataset_name' in result
    assert 'train_metrics' in result
    assert 'val_metrics' in result
    assert 'test_metrics' in result
    assert 'total_time' in result
    
    assert result['model_name'] == 'GCN'
    assert result['dataset_name'] == 'tiny'
    assert 'accuracy' in result['test_metrics']


def test_benchmark_compare_models():
    """Test model comparison."""
    config = BenchmarkConfig(
        hidden_channels=16,
        num_layers=2,
        num_epochs=3,
        verbose=False,
        seed=42
    )
    
    dataset = TinyDataset("tiny")
    
    models = [
        GCN(dataset.num_features, 16, dataset.num_classes, num_layers=2),
        GCN(dataset.num_features, 16, dataset.num_classes, num_layers=2),
    ]
    
    benchmark = Benchmark(config)
    comparison_df = benchmark.compare_models(models, dataset)
    
    assert len(comparison_df) == 2
    assert 'Model' in comparison_df.columns
    assert 'Test Accuracy' in comparison_df.columns
    assert 'Train Time (s)' in comparison_df.columns


def test_benchmark_device_setup():
    """Test device setup."""
    config_auto = BenchmarkConfig(device="auto")
    benchmark_auto = Benchmark(config_auto)
    assert benchmark_auto.device.type in ["cpu", "cuda", "mps"]
    
    config_cpu = BenchmarkConfig(device="cpu")
    benchmark_cpu = Benchmark(config_cpu)
    assert benchmark_cpu.device.type == "cpu"
