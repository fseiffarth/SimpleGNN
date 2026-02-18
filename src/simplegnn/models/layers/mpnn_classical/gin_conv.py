import torch
import torch_geometric
from torch.nn import Sequential, Linear, BatchNorm1d

from simplegnn.datasets.graph_dataset import GraphDataset
from simplegnn.models.layers.mpnn_classical.gnn_conv import GNNConvLayer


class GINConv(GNNConvLayer):
    def __init__(self, layer_args):
        layer_args['name'] = 'GINConv'
        super(GINConv, self).__init__(layer_args)
        self.use_edge_features = layer_args.get('edge_dim', None) is not None
        emb_dim = layer_args.get('out_features')
        neural_network = Sequential(Linear(emb_dim, 2 * emb_dim), BatchNorm1d(2 * emb_dim), self.activation,
                        Linear(2 * emb_dim, emb_dim))
        gin_args = {
            'nn': neural_network,
            'eps': layer_args.get('eps', 0.0),
            'train_eps': layer_args.get('train_eps', False),
            'edge_dim': layer_args.get('edge_dim', 0),
        }

        if gin_args['edge_dim'] == 0:
            gin_args.pop('edge_dim')
            self.layer = torch_geometric.nn.GINConv(**gin_args)
        else:
            self.layer = torch_geometric.nn.GINEConv(**gin_args)

    def forward(self, node_representation:torch.Tensor, batch_data: GraphDataset, *args, **kwargs):
        x = node_representation
        if self.use_edge_features:
            node_representation = self.layer(node_representation, batch_data.edge_index, batch_data.edge_attributes)
        else:
            node_representation = self.layer(node_representation, batch_data.edge_index)
        if self.batch_norm:
            node_representation = self.batch_norm_layer(node_representation)
        node_representation = self.activation(node_representation)
        if self.residual:
            node_representation = node_representation + x
        if self.dropout > 0 and self.training:
            node_representation = torch.nn.functional.dropout(node_representation, p=self.dropout, training=self.training)
        return node_representation