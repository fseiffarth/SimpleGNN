"""SimpleGNN: A PyTorch-based Graph Neural Network experimentation framework."""

__version__ = "0.1.0"
__author__ = "Florian Seiffarth"
__license__ = "Apache-2.0"

# Convenience imports
from simplegnn.framework.core import FrameworkMain
from simplegnn.models.model import GraphModel

__all__ = ["__version__", "FrameworkMain", "GraphModel"]
