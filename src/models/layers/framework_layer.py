from abc import abstractmethod, ABC

import torch
import torch_geometric
from torch._C.cpp import nn
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d

from datasets.graph_dataset import GraphDataset


class FrameworkLayer(torch.nn.Module, ABC):
    """
    A base class for the layers in our GNN framework.
    """
    def __init__(self, layer_args, device='cpu'):
        super(FrameworkLayer, self).__init__()
        self.layer_args = layer_args
        # Whether to use residual connections in this layer
        self.residual = layer_args.get('residual', False)
        # Whether to use batch normalization in this layer
        self.batch_norm = layer_args.get('batch_norm', False)
        if self.batch_norm:
            self.batch_norm_args = {
                'in_channels': layer_args.get('in_channels', None),
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
        self.device = device
        # Activation function for this layer
        self.activation = torch.nn.Identity()
        if 'activation' in layer_args:
            self.activation = eval(layer_args['activation'])

    @abstractmethod
    def forward(self, node_representation:torch.Tensor, batch_data: GraphDataset, *args, **kwargs):
        pass






















