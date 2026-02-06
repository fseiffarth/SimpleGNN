from abc import abstractmethod, ABC

import torch
import torch_geometric
from torch._C.cpp import nn
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d

from datasets.graph_dataset import GraphDataset


class FrameworkLayer(torch.nn.Module, ABC):
    """
    A base class for the layers in our GNN framework.

    The input to the layer is a tensor of shape (C, N, F),
    where C is the number of channels (or heads), N is the number of nodes, and F is the number of features per node.
    For batch processing, the nodes from different graphs are concatenated along the N dimension.

    The output of the layer is a tensor of the shape (C', N, F'), where C' is the number of output channels (or heads),
    and F' is the number of output features per node.
    The number of nodes N remains the same as the input.
    """
    def __init__(self, layer_args=None):
        super(FrameworkLayer, self).__init__()
        # Mandatory layer arguments
        if layer_args is None:
            raise ValueError("layer_args must be provided")
        self.layer_args = layer_args

        # Get layer ID and name
        self.layer_id = self.layer_args.get('layer_id', None)
        if self.layer_id is None:
            raise ValueError("layer_id must be provided")
        self.name = self.layer_args.get('name', None)
        if self.name is None:
            raise ValueError("layer name must be provided")

        # Get layer seed for reproducibility
        self.seed = self.layer_args.get('seed', None)
        if self.seed is None:
            raise ValueError("seed must be provided")

        # Set layer precision
        self.precision = layer_args.get('dtype', None)
        if self.precision is None:
            raise ValueError("precision setting is not supported in this framework layer implementation")


        # Get input and output dimensions of the layer
        self.in_features = layer_args.get('in_features', None)
        if self.in_features is None:
            raise ValueError("in_features must be provided")
        self.out_features = layer_args.get('out_features', None)
        if self.out_features is None:
            raise ValueError("out_features must be provided")
        self.in_channels = layer_args.get('in_channels', None)
        if self.in_channels is None:
            raise ValueError("in_channels must be provided")
        self.out_channels = layer_args.get('out_channels', None)
        if self.out_channels is None:
            raise ValueError("out_channels must be provided")

        self.num_heads = layer_args.get('num_heads', 1)
        if 'heads' in layer_args:
            self.num_heads = 0
            for head in layer_args['heads']:
                self.num_heads += head['num']


        # Whether to use residual connections in this layer
        self.residual = layer_args.get('residual', False)
        # Whether to use batch normalization in this layer
        self.batch_norm = layer_args.get('batch_norm', False)
        if self.batch_norm:
            self.batch_norm_args = {
                'in_channels': self.in_channels,
                'eps': layer_args.get('batch_norm_eps', 1e-5),
                'momentum': layer_args.get('batch_norm_momentum', 0.1),
                'affine': layer_args.get('batch_norm_affine', True),
                'track_running_stats': layer_args.get('batch_norm_track_running_stats', True),
                'allow_single_element': layer_args.get('batch_norm_allow_single_element', False),
            }
            self.batch_norm_layer = torch_geometric.nn.BatchNorm(**self.batch_norm_args)
        # Dropout rate for this layer
        self.dropout = layer_args.get('dropout', 0.0)
        # Device to run the layer on
        self.device = layer_args.get('device', 'cpu')
        # bias for this layer
        self.bias = layer_args.get('bias', True) # whether to use bias in this layer, default is True

        # Activation function for this layer
        if 'activation' not in layer_args:
            self.activation = torch.nn.Identity()
        else:
            self.activation = eval(layer_args['activation'])

    @abstractmethod
    def forward(self, node_representation:torch.Tensor, batch_data: GraphDataset, *args, **kwargs):
        pass






















