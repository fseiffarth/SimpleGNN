"""GNN models module."""

from simplegnn.models.base import BaseGNN
from simplegnn.models.gcn import GCN
from simplegnn.models.gat import GAT
from simplegnn.models.graphsage import GraphSAGE

__all__ = ["BaseGNN", "GCN", "GAT", "GraphSAGE"]
