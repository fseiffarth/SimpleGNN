import torch
import torch_geometric

from datasets.graph_dataset import GraphDataset
from models.layers.mpnn_classical.gnn_conv import GNNConvLayer


class SAGEConv(GNNConvLayer):
    def __init__(self, layer_args):
        super(SAGEConv, self).__init__(layer_args)
        self.sage_args = {
            'in_channels': layer_args.get('in_channels'),
            'out_channels': layer_args.get('out_channels'),
            'aggr': layer_args.get('aggr', 'mean'),  # "mean", "max", "add"
            'normalize': layer_args.get('normalize', False),
            'root_weight': layer_args.get('root_weight', True),
            'project': layer_args.get('project', False),
            'bias': layer_args.get('bias', True),
        }
        self.layer = torch_geometric.nn.SAGEConv(**self.sage_args)


    def forward(self, node_representation:torch.Tensor, batch_data: GraphDataset, *args, **kwargs):
        x = node_representation
        node_representation = self.layer(node_representation, batch_data.edge_index)
        if self.batch_norm:
            node_representation = self.batch_norm_layer(node_representation)
        node_representation = self.activation(node_representation)
        if self.residual:
            node_representation = node_representation + x
        if self.dropout > 0 and self.training:
            node_representation = torch.nn.Dropout(self.dropout)(node_representation)
        return node_representation