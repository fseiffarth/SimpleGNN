"""Base dataset interface."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import torch
from torch_geometric.data import Data


class BaseDataset(ABC):
    """Base class for all datasets.
    
    This abstract class defines the interface that all datasets must implement.
    It allows for easy integration of both predefined and custom datasets.
    """

    def __init__(self, name: str, root: Optional[str] = None):
        """Initialize the dataset.
        
        Args:
            name: Name of the dataset
            root: Root directory for storing dataset files
        """
        self.name = name
        self.root = root or "./data"
        self._data = None
        self._num_classes = None
        self._num_features = None

    @abstractmethod
    def load(self) -> Data:
        """Load and return the dataset.
        
        Returns:
            PyTorch Geometric Data object containing:
                - x: Node features [num_nodes, num_features]
                - edge_index: Edge indices [2, num_edges]
                - y: Node labels [num_nodes]
                - train_mask: Training node mask [num_nodes]
                - val_mask: Validation node mask [num_nodes]
                - test_mask: Test node mask [num_nodes]
        """
        pass

    @property
    def data(self) -> Data:
        """Get the loaded data."""
        if self._data is None:
            self._data = self.load()
        return self._data

    @property
    def num_classes(self) -> int:
        """Get the number of classes in the dataset."""
        if self._num_classes is None:
            self._num_classes = int(self.data.y.max().item()) + 1
        return self._num_classes

    @property
    def num_features(self) -> int:
        """Get the number of node features in the dataset."""
        if self._num_features is None:
            self._num_features = self.data.x.size(1)
        return self._num_features

    @property
    def num_nodes(self) -> int:
        """Get the number of nodes in the dataset."""
        return self.data.x.size(0)

    @property
    def num_edges(self) -> int:
        """Get the number of edges in the dataset."""
        return self.data.edge_index.size(1)

    def split_data(self, 
                   num_nodes: Optional[int] = None,
                   train_ratio: float = 0.6,
                   val_ratio: float = 0.2,
                   test_ratio: float = 0.2,
                   seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create train/val/test splits.
        
        Args:
            num_nodes: Number of nodes. If None, uses self.num_nodes
            train_ratio: Proportion of nodes for training
            val_ratio: Proportion of nodes for validation
            test_ratio: Proportion of nodes for testing
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_mask, val_mask, test_mask)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"
        
        if num_nodes is None:
            num_nodes = self.num_nodes
        
        if seed is not None:
            # Use numpy for random permutation to avoid torch recursion issues
            import numpy as np
            np.random.seed(seed)
            indices = torch.from_numpy(np.random.permutation(num_nodes))
        else:
            indices = torch.randperm(num_nodes)
        
        train_size = int(num_nodes * train_ratio)
        val_size = int(num_nodes * val_ratio)
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True
        
        return train_mask, val_mask, test_mask

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics.
        
        Returns:
            Dictionary containing dataset statistics
        """
        return {
            'name': self.name,
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges,
            'num_features': self.num_features,
            'num_classes': self.num_classes,
            'avg_degree': self.num_edges / self.num_nodes,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', nodes={self.num_nodes}, " \
               f"edges={self.num_edges}, features={self.num_features}, classes={self.num_classes})"
