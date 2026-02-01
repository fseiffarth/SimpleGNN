"""Planetoid dataset loader (Cora, CiteSeer, PubMed)."""

from typing import Optional, Literal
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from simplegnn.datasets.base import BaseDataset


class PlanetoidDataset(BaseDataset):
    """Loader for Planetoid datasets (Cora, CiteSeer, PubMed).
    
    These are classic citation network datasets commonly used for benchmarking
    node classification methods.
    
    Reference:
        Revisiting Semi-Supervised Learning with Graph Embeddings
        https://arxiv.org/abs/1603.08861
    """

    def __init__(self, 
                 name: Literal["Cora", "CiteSeer", "PubMed"] = "Cora",
                 root: Optional[str] = None,
                 split: str = "public"):
        """Initialize Planetoid dataset.
        
        Args:
            name: Dataset name ('Cora', 'CiteSeer', or 'PubMed')
            root: Root directory for storing dataset files
            split: Split type ('public', 'full', or 'random')
        """
        super().__init__(name, root)
        self.split = split
        self._dataset = None

    def load(self) -> Data:
        """Load the Planetoid dataset.
        
        Returns:
            PyTorch Geometric Data object with train/val/test masks
        """
        if self._dataset is None:
            self._dataset = Planetoid(
                root=self.root,
                name=self.name,
                split=self.split
            )
        
        # Return the first (and only) graph
        return self._dataset[0]

    def __repr__(self) -> str:
        return f"PlanetoidDataset(name='{self.name}', split='{self.split}')"
