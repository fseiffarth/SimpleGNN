"""GraphSAGE implementation."""

from typing import Optional
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from simplegnn.models.base import BaseGNN


class GraphSAGE(BaseGNN):
    """GraphSAGE: Inductive Representation Learning on Large Graphs (Hamilton et al., 2017).
    
    Reference:
        Inductive Representation Learning on Large Graphs
        https://arxiv.org/abs/1706.02216
    """

    def __init__(self, 
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 aggr: str = "mean",
                 normalize: bool = False,
                 **kwargs):
        """Initialize GraphSAGE model.
        
        Args:
            in_channels: Number of input features per node
            hidden_channels: Number of hidden features per node
            out_channels: Number of output features/classes
            num_layers: Number of SAGE layers (minimum 2)
            dropout: Dropout probability
            aggr: Aggregation method ('mean', 'max', 'sum')
            normalize: Whether to normalize embeddings
            **kwargs: Additional arguments (ignored)
        """
        super().__init__(in_channels, hidden_channels, out_channels, num_layers, dropout)
        
        assert num_layers >= 2, "GraphSAGE requires at least 2 layers"
        
        self.aggr = aggr
        self.normalize = normalize
        self.convs = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr, normalize=normalize))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr, normalize=normalize))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggr, normalize=normalize))
        
        self.reset_parameters()

    def reset_parameters(self):
        """Reset model parameters."""
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of GraphSAGE.
        
        Args:
            x: Node feature matrix [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Optional edge weights (ignored for GraphSAGE)
            
        Returns:
            Output node predictions [num_nodes, out_channels]
        """
        # Forward through all layers except the last
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer (no activation)
        x = self.convs[-1](x, edge_index)
        
        return x

    def get_config(self):
        """Get model configuration."""
        config = super().get_config()
        config['aggr'] = self.aggr
        config['normalize'] = self.normalize
        return config
