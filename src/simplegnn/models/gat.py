"""GAT (Graph Attention Network) implementation."""

from typing import Optional
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from simplegnn.models.base import BaseGNN


class GAT(BaseGNN):
    """Graph Attention Network (Veličković et al., 2018).
    
    Reference:
        Graph Attention Networks
        https://arxiv.org/abs/1710.10903
    """

    def __init__(self, 
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 num_heads: int = 8,
                 concat_heads: bool = True,
                 **kwargs):
        """Initialize GAT model.
        
        Args:
            in_channels: Number of input features per node
            hidden_channels: Number of hidden features per node (per head)
            out_channels: Number of output features/classes
            num_layers: Number of GAT layers (minimum 2)
            dropout: Dropout probability
            num_heads: Number of attention heads
            concat_heads: Whether to concatenate attention heads (if False, average)
            **kwargs: Additional arguments (ignored)
        """
        super().__init__(in_channels, hidden_channels, out_channels, num_layers, dropout)
        
        assert num_layers >= 2, "GAT requires at least 2 layers"
        
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        self.convs = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(
            in_channels, 
            hidden_channels, 
            heads=num_heads,
            dropout=dropout,
            concat=concat_heads
        ))
        
        # Hidden layers
        in_dim = hidden_channels * num_heads if concat_heads else hidden_channels
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(
                in_dim,
                hidden_channels,
                heads=num_heads,
                dropout=dropout,
                concat=concat_heads
            ))
        
        # Output layer (average attention heads)
        self.convs.append(GATConv(
            in_dim,
            out_channels,
            heads=1,
            dropout=dropout,
            concat=False
        ))
        
        self.reset_parameters()

    def reset_parameters(self):
        """Reset model parameters."""
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of GAT.
        
        Args:
            x: Node feature matrix [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Optional edge weights (ignored for GAT)
            
        Returns:
            Output node predictions [num_nodes, out_channels]
        """
        # Forward through all layers except the last
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer (no activation)
        x = self.convs[-1](x, edge_index)
        
        return x

    def get_config(self):
        """Get model configuration."""
        config = super().get_config()
        config['num_heads'] = self.num_heads
        config['concat_heads'] = self.concat_heads
        return config
