import torch
import torch_geometric

from simplegnn.datasets.graph_dataset import GraphDataset
from simplegnn.models.layers.mpnn_classical.gnn_conv import GNNConvLayer


class GATConv(GNNConvLayer):
    def __init__(self, layer_args):
        layer_args['name'] = 'GATConv'
        super(GATConv, self).__init__(layer_args)
        self.gat_args = {
            'in_channels': layer_args.get('in_features'),
            'out_channels': layer_args.get('out_features'),
            'heads': layer_args.get('num_heads', 1),
            'concat': layer_args.get('concat', False),
            'negative_slope': layer_args.get('negative_slope', 0.2),
            'add_self_loops': layer_args.get('add_self_loops', True),
            'edge_dim': layer_args.get('edge_dim', None),
            'fill_value': layer_args.get('fill_value', 'mean'),
            'bias': layer_args.get('bias', True),
        }
        self.merge_heads = layer_args.get('merge_heads', True)
        self.layer = torch_geometric.nn.GATConv(**self.gat_args)
        self.linear_merge_heads = torch.nn.Linear(self.gat_args['out_channels'] * self.gat_args['heads'], self.gat_args['out_channels']) if self.gat_args['concat'] else torch.nn.Linear(self.gat_args['out_channels'], self.gat_args['out_channels'])


    def forward(self, node_representation:torch.Tensor, batch_data: GraphDataset, *args, **kwargs):
        x = node_representation
        node_representation = self.layer(node_representation, batch_data.edge_index)
        if self.merge_heads and self.gat_args['concat'] and self.gat_args['heads'] > 1:
            node_representation = self.linear_merge_heads(node_representation)
        if self.batch_norm:
            if self.merge_heads:
                node_representation = self.batch_norm_layer(node_representation)
            else: # apply batch norm to each head separately
                node_representation = self.batch_norm_layer(node_representation.view(-1, self.gat_args['out_features'])).view(-1, self.gat_args['out_features'] * self.gat_args['heads'])
        node_representation = self.activation(node_representation)
        if self.residual:
            if self.merge_heads:
                node_representation = node_representation + x
            else: # add residual to each head separately
                node_representation = node_representation + x.repeat(1, self.gat_args['heads'])
        if self.dropout > 0 and self.training:
            node_representation = torch.nn.functional.dropout(node_representation, p=self.dropout, training=self.training)
        return node_representation