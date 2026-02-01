"""Tests for metrics."""

import pytest
import torch
from simplegnn.metrics import Metrics


def test_accuracy():
    """Test accuracy metric."""
    y_true = torch.tensor([0, 1, 2, 1, 0])
    y_pred = torch.tensor([0, 1, 2, 1, 0])
    
    acc = Metrics.accuracy(y_true, y_pred)
    assert acc == 1.0


def test_accuracy_with_logits():
    """Test accuracy with logits."""
    y_true = torch.tensor([0, 1, 2, 1, 0])
    y_pred = torch.tensor([
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
        [0.05, 0.9, 0.05],
        [0.9, 0.05, 0.05],
    ])
    
    acc = Metrics.accuracy(y_true, y_pred)
    assert acc == 1.0


def test_f1_score():
    """Test F1 score."""
    y_true = torch.tensor([0, 1, 2, 1, 0])
    y_pred = torch.tensor([0, 1, 2, 1, 0])
    
    f1_macro = Metrics.f1_score(y_true, y_pred, average='macro')
    assert f1_macro == 1.0
    
    f1_micro = Metrics.f1_score(y_true, y_pred, average='micro')
    assert f1_micro == 1.0


def test_precision():
    """Test precision metric."""
    y_true = torch.tensor([0, 1, 2, 1, 0])
    y_pred = torch.tensor([0, 1, 2, 1, 0])
    
    precision = Metrics.precision(y_true, y_pred)
    assert precision == 1.0


def test_recall():
    """Test recall metric."""
    y_true = torch.tensor([0, 1, 2, 1, 0])
    y_pred = torch.tensor([0, 1, 2, 1, 0])
    
    recall = Metrics.recall(y_true, y_pred)
    assert recall == 1.0


def test_cross_entropy_loss():
    """Test cross-entropy loss."""
    y_true = torch.tensor([0, 1, 2])
    y_pred = torch.tensor([
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
    ])
    
    loss = Metrics.cross_entropy_loss(y_true, y_pred)
    assert loss > 0
    assert loss < 1.0


def test_compute_all():
    """Test computing all metrics."""
    y_true = torch.tensor([0, 1, 2, 1, 0])
    y_pred = torch.tensor([
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
        [0.05, 0.9, 0.05],
        [0.9, 0.05, 0.05],
    ])
    
    metrics = Metrics.compute_all(y_true, y_pred, loss=True)
    
    assert 'accuracy' in metrics
    assert 'f1_macro' in metrics
    assert 'f1_micro' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'loss' in metrics
    
    assert metrics['accuracy'] == 1.0
    assert metrics['f1_macro'] == 1.0
