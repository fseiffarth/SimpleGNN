"""GCN (Graph Convolutional Network) implementation."""

from typing import Optional
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from simplegnn.models.base import BaseGNN


class GCN(BaseGNN):
    """Graph Convolutional Network (Kipf & Welling, 2017).
    
    Reference:
        Semi-Supervised Classification with Graph Convolutional Networks
        https://arxiv.org/abs/1609.02907
    """

    def __init__(self, 
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 use_batch_norm: bool = False,
                 **kwargs):
        """Initialize GCN model.
        
        Args:
            in_channels: Number of input features per node
            hidden_channels: Number of hidden features per node
            out_channels: Number of output features/classes
            num_layers: Number of GCN layers (minimum 2)
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
            **kwargs: Additional arguments (ignored)
        """
        super().__init__(in_channels, hidden_channels, out_channels, num_layers, dropout)
        
        assert num_layers >= 2, "GCN requires at least 2 layers"
        
        self.use_batch_norm = use_batch_norm
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList() if use_batch_norm else None
        
        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        if use_batch_norm:
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            if use_batch_norm:
                self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels))
        
        # Output layer
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
        self.reset_parameters()

    def reset_parameters(self):
        """Reset model parameters."""
        for conv in self.convs:
            conv.reset_parameters()
        if self.use_batch_norm:
            for bn in self.batch_norms:
                bn.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of GCN.
        
        Args:
            x: Node feature matrix [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Optional edge weights [num_edges]
            
        Returns:
            Output node predictions [num_nodes, out_channels]
        """
        # Forward through all layers except the last
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer (no activation)
        x = self.convs[-1](x, edge_index, edge_weight)
        
        return x

    def get_config(self):
        """Get model configuration."""
        config = super().get_config()
        config['use_batch_norm'] = self.use_batch_norm
        return config
