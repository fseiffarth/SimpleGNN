"""Base GNN model interface."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import torch
import torch.nn as nn


class BaseGNN(ABC, nn.Module):
    """Base class for all GNN models.
    
    This abstract class defines the interface that all GNN models must implement.
    It allows for easy integration of both predefined and custom GNN architectures.
    """

    def __init__(self, 
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 **kwargs):
        """Initialize the base GNN model.
        
        Args:
            in_channels: Number of input features per node
            hidden_channels: Number of hidden features per node
            out_channels: Number of output features/classes
            num_layers: Number of GNN layers
            dropout: Dropout probability
            **kwargs: Additional model-specific arguments
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

    @abstractmethod
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the GNN model.
        
        Args:
            x: Node feature matrix of shape [num_nodes, in_channels]
            edge_index: Edge indices of shape [2, num_edges]
            edge_weight: Optional edge weights of shape [num_edges]
            
        Returns:
            Output node embeddings/predictions of shape [num_nodes, out_channels]
        """
        pass

    def reset_parameters(self):
        """Reset model parameters."""
        for module in self.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization.
        
        Returns:
            Dictionary containing model configuration
        """
        return {
            'model_class': self.__class__.__name__,
            'in_channels': self.in_channels,
            'hidden_channels': self.hidden_channels,
            'out_channels': self.out_channels,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.get_config()})"
