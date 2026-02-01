"""Core functionality for SimpleGNN.

This module contains the core classes and functions for Graph Neural Networks.
"""


class GNNModel:
    """Base class for Graph Neural Network models."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        """Initialize GNN model.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x: Input features

        Returns:
            Model output
        """
        raise NotImplementedError("Subclasses must implement forward method")


def create_model(model_type, **kwargs):
    """Factory function to create GNN models.

    Args:
        model_type: Type of model to create
        **kwargs: Model-specific arguments

    Returns:
        Initialized GNN model
    """
    if model_type == "base":
        return GNNModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
