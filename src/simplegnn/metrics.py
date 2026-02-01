"""Evaluation metrics for GNN models."""

from typing import Dict, Any
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class Metrics:
    """Metrics for evaluating GNN model performance."""

    @staticmethod
    def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """Calculate accuracy.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels or logits
            
        Returns:
            Accuracy score
        """
        if y_pred.dim() > 1:
            y_pred = y_pred.argmax(dim=-1)
        return accuracy_score(y_true.cpu().numpy(), y_pred.cpu().numpy())

    @staticmethod
    def f1_score(y_true: torch.Tensor, y_pred: torch.Tensor, 
                 average: str = "macro") -> float:
        """Calculate F1 score.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels or logits
            average: Averaging method ('micro', 'macro', 'weighted')
            
        Returns:
            F1 score
        """
        if y_pred.dim() > 1:
            y_pred = y_pred.argmax(dim=-1)
        return f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), 
                       average=average, zero_division=0)

    @staticmethod
    def precision(y_true: torch.Tensor, y_pred: torch.Tensor,
                  average: str = "macro") -> float:
        """Calculate precision.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels or logits
            average: Averaging method ('micro', 'macro', 'weighted')
            
        Returns:
            Precision score
        """
        if y_pred.dim() > 1:
            y_pred = y_pred.argmax(dim=-1)
        return precision_score(y_true.cpu().numpy(), y_pred.cpu().numpy(),
                             average=average, zero_division=0)

    @staticmethod
    def recall(y_true: torch.Tensor, y_pred: torch.Tensor,
               average: str = "macro") -> float:
        """Calculate recall.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels or logits
            average: Averaging method ('micro', 'macro', 'weighted')
            
        Returns:
            Recall score
        """
        if y_pred.dim() > 1:
            y_pred = y_pred.argmax(dim=-1)
        return recall_score(y_true.cpu().numpy(), y_pred.cpu().numpy(),
                          average=average, zero_division=0)

    @staticmethod
    def cross_entropy_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """Calculate cross-entropy loss.
        
        Args:
            y_true: True labels
            y_pred: Predicted logits
            
        Returns:
            Cross-entropy loss
        """
        return F.cross_entropy(y_pred, y_true).item()

    @staticmethod
    def compute_all(y_true: torch.Tensor, y_pred: torch.Tensor,
                    loss: bool = False) -> Dict[str, float]:
        """Compute all metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels or logits
            loss: Whether to compute loss (requires logits)
            
        Returns:
            Dictionary of metric name to value
        """
        metrics = {
            'accuracy': Metrics.accuracy(y_true, y_pred),
            'f1_macro': Metrics.f1_score(y_true, y_pred, average='macro'),
            'f1_micro': Metrics.f1_score(y_true, y_pred, average='micro'),
            'precision': Metrics.precision(y_true, y_pred, average='macro'),
            'recall': Metrics.recall(y_true, y_pred, average='macro'),
        }
        
        if loss and y_pred.dim() > 1:
            metrics['loss'] = Metrics.cross_entropy_loss(y_true, y_pred)
        
        return metrics
