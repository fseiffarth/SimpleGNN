import torch
import torch_geometric

from simplegnn.datasets.graph_dataset import GraphDataset
from simplegnn.models.layers.mpnn_classical.gnn_conv import GNNConvLayer


class GATv2Conv(GNNConvLayer):
    def __init__(self, layer_args):
        super(GATv2Conv, self).__init__(layer_args)
        self.gatv2_args = {
            'in_channels': layer_args.get('in_channels'),
            'out_channels': layer_args.get('out_channels'),
            'heads': layer_args.get('heads', 1),
            'concat': layer_args.get('concat', False),
            'negative_slope': layer_args.get('negative_slope', 0.2),
            'add_self_loops': layer_args.get('add_self_loops', True),
            'edge_dim': layer_args.get('edge_dim', None),
            'fill_value': layer_args.get('fill_value', 'mean'),
            'bias': layer_args.get('bias', True),
            'share_weights': layer_args.get('share_weights', False),
        }
        self.use_edge_features = layer_args.get('edge_dim', None) is not None
        self.merge_heads = layer_args.get('merge_heads', True)
        self.layer = torch_geometric.nn.GATv2Conv(**self.gatv2_args)
        if self.merge_heads:
            self.linear_merge_heads = torch.nn.Linear(self.gatv2_args['out_channels'] * self.gatv2_args['heads'], self.gatv2_args['out_channels']) if self.gatv2_args['concat'] else torch.nn.Linear(self.gatv2_args['out_channels'], self.gatv2_args['out_channels'])

    def forward(self, node_representation:torch.Tensor, batch_data: GraphDataset, *args, **kwargs):
        x = node_representation
        if self.use_edge_features:
            node_representation = self.layer(node_representation, batch_data.edge_index, batch_data.edge_attributes)
        else:
            node_representation = self.layer(node_representation, batch_data.edge_index)
        if self.merge_heads and self.gatv2_args['concat'] and self.gatv2_args['heads'] > 1:
            node_representation = self.linear_merge_heads(node_representation)
        if self.batch_norm:
            if self.merge_heads:
                node_representation = self.batch_norm_layer(node_representation)
            else: # apply batch norm to each head separately
                node_representation = self.batch_norm_layer(node_representation.view(-1, self.gatv2_args['out_channels'])).view(-1, self.gatv2_args['out_channels'] * self.gatv2_args['heads'])
        node_representation = self.activation(node_representation)
        if self.residual:
            if self.merge_heads:
                node_representation = node_representation + x
            else: # add residual to each head separately
                node_representation = node_representation + x.repeat(1, self.gatv2_args['heads'])
        if self.dropout > 0 and self.training:
            node_representation = torch.nn.functional.dropout(node_representation, p=self.dropout, training=self.training)
        return node_representation