import torch
import torch_geometric

from datasets.graph_dataset import GraphDataset
from models.layers.mpnn_classical.gnn_conv import GNNConvLayer


class GCNConv(GNNConvLayer):
    def __init__(self, layer_args):
        super(GCNConv, self).__init__(layer_args)
        gcn_args = {
            'in_channels': layer_args.get('in_channels'),
            'out_channels': layer_args.get('out_channels'),
            'improved': layer_args.get('improved', False),
            'cached': layer_args.get('cached', False),
            'add_self_loops': layer_args.get('add_self_loops', True),
            'normalize': layer_args.get('normalize', True),
            'bias': layer_args.get('bias', True)
        }
        self.layer = torch_geometric.nn.GCNConv(**gcn_args)

    def forward(self,node_representation:torch.Tensor, batch_data: GraphDataset, *args, **kwargs):
        x = node_representation
        node_representation = self.layer(node_representation, batch_data.edge_index)
        if self.batch_norm:
            node_representation = self.batch_norm_layer(node_representation)
        node_representation = self.activation(node_representation)
        if self.residual:
            node_representation = node_representation + x
        if self.dropout > 0 and self.training:
            node_representation = torch.nn.functional.dropout(node_representation, p=self.dropout, training=self.training)
        return node_representation