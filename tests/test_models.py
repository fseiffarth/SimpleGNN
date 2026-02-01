"""Tests for GNN models."""

import pytest
import torch
from simplegnn.models import BaseGNN, GCN, GAT, GraphSAGE


@pytest.fixture
def graph_data():
    """Create simple graph data for testing."""
    x = torch.randn(10, 16)  # 10 nodes, 16 features
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    return x, edge_index


def test_gcn_forward(graph_data):
    """Test GCN forward pass."""
    x, edge_index = graph_data
    model = GCN(in_channels=16, hidden_channels=32, out_channels=5, num_layers=2)
    
    output = model(x, edge_index)
    
    assert output.shape == (10, 5)
    assert not torch.isnan(output).any()


def test_gat_forward(graph_data):
    """Test GAT forward pass."""
    x, edge_index = graph_data
    model = GAT(in_channels=16, hidden_channels=8, out_channels=5, 
                num_layers=2, num_heads=4)
    
    output = model(x, edge_index)
    
    assert output.shape == (10, 5)
    assert not torch.isnan(output).any()


def test_graphsage_forward(graph_data):
    """Test GraphSAGE forward pass."""
    x, edge_index = graph_data
    model = GraphSAGE(in_channels=16, hidden_channels=32, out_channels=5, num_layers=2)
    
    output = model(x, edge_index)
    
    assert output.shape == (10, 5)
    assert not torch.isnan(output).any()


def test_model_reset_parameters(graph_data):
    """Test model parameter reset."""
    x, edge_index = graph_data
    model = GCN(in_channels=16, hidden_channels=32, out_channels=5, num_layers=2)
    
    # Get initial output
    output1 = model(x, edge_index)
    
    # Reset and check output is different (with high probability)
    model.reset_parameters()
    output2 = model(x, edge_index)
    
    # Outputs should be different after reset
    assert not torch.allclose(output1, output2)


def test_model_config():
    """Test model configuration."""
    model = GCN(in_channels=16, hidden_channels=32, out_channels=5, 
                num_layers=3, dropout=0.3)
    
    config = model.get_config()
    
    assert config['model_class'] == 'GCN'
    assert config['in_channels'] == 16
    assert config['hidden_channels'] == 32
    assert config['out_channels'] == 5
    assert config['num_layers'] == 3
    assert config['dropout'] == 0.3


def test_model_eval_mode(graph_data):
    """Test model in eval mode."""
    x, edge_index = graph_data
    model = GCN(in_channels=16, hidden_channels=32, out_channels=5, num_layers=2)
    
    model.eval()
    output1 = model(x, edge_index)
    output2 = model(x, edge_index)
    
    # Outputs should be identical in eval mode
    assert torch.allclose(output1, output2)
