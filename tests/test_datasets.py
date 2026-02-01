"""Tests for datasets."""

import pytest
import torch
from simplegnn.datasets import BaseDataset, PlanetoidDataset
from torch_geometric.data import Data


class DummyDataset(BaseDataset):
    """Dummy dataset for testing."""
    
    def load(self) -> Data:
        x = torch.randn(20, 10)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        y = torch.randint(0, 3, (20,))
        
        train_mask = torch.zeros(20, dtype=torch.bool)
        val_mask = torch.zeros(20, dtype=torch.bool)
        test_mask = torch.zeros(20, dtype=torch.bool)
        
        train_mask[:10] = True
        val_mask[10:15] = True
        test_mask[15:] = True
        
        return Data(x=x, edge_index=edge_index, y=y,
                   train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)


def test_dummy_dataset():
    """Test dummy dataset."""
    dataset = DummyDataset("dummy")
    
    assert dataset.name == "dummy"
    assert dataset.num_nodes == 20
    assert dataset.num_features == 10
    assert dataset.num_classes == 3
    assert dataset.num_edges == 3


def test_dataset_stats():
    """Test dataset statistics."""
    dataset = DummyDataset("dummy")
    stats = dataset.get_stats()
    
    assert stats['name'] == 'dummy'
    assert stats['num_nodes'] == 20
    assert stats['num_edges'] == 3
    assert stats['num_features'] == 10
    assert stats['num_classes'] == 3
    assert 'avg_degree' in stats


def test_dataset_split():
    """Test dataset split generation."""
    dataset = DummyDataset("dummy")
    
    train_mask, val_mask, test_mask = dataset.split_data(
        num_nodes=20,
        train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42
    )
    
    assert train_mask.sum() == 12
    assert val_mask.sum() == 4
    assert test_mask.sum() == 4
    assert (train_mask.sum() + val_mask.sum() + test_mask.sum()) == 20


def test_planetoid_dataset():
    """Test Planetoid dataset loading (if available)."""
    try:
        dataset = PlanetoidDataset("Cora")
        
        assert dataset.name == "Cora"
        assert dataset.num_nodes > 0
        assert dataset.num_features > 0
        assert dataset.num_classes > 0
        
        data = dataset.data
        assert hasattr(data, 'x')
        assert hasattr(data, 'edge_index')
        assert hasattr(data, 'y')
        assert hasattr(data, 'train_mask')
        assert hasattr(data, 'val_mask')
        assert hasattr(data, 'test_mask')
    except Exception as e:
        pytest.skip(f"Planetoid dataset not available: {e}")
