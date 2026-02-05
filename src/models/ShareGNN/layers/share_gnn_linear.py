import math

import torch
from torch import nn

from datasets.graph_dataset import GraphDataset
from models.ShareGNN.utils import Layer
from models.layers.framework_layer import FrameworkLayer


class ShareGNNLinear(FrameworkLayer):
    """
    Wrapper class for a linear layer that ignores the pos argument
    """
    def __init__(self, seed:int, layer_id:int, layer:Layer, parameters, graph_data:GraphDataset, num_heads=None, in_features=None):
        """
        """
        layer_args = {
            'activation': layer.layer_dict['activation'],
            'residual': layer.layer_dict.get('residual', False),
            'batch_norm': layer.layer_dict.get('batch_norm', False),
            'dropout': layer.layer_dict.get('dropout', 0.0),
        }
        super(ShareGNNLinear, self).__init__(layer_id=layer_id, layer_args=layer_args, device=parameters.run_config.config.get('device', 'cpu'))
        torch.manual_seed(layer_id + seed)
        self.layer = layer
        # get the input features, i.e. the dimension of the input vector and out_features
        self.in_features = graph_data.num_node_features
        if in_features is not None:
            self.in_features = in_features
        self.in_features = layer.layer_dict.get('in_features', self.in_features)

        self.out_features = graph_data.num_node_features
        if out_features is not None:
            self.out_features = out_features
        self.out_features = layer.layer_dict.get('out_features', self.in_features)

        # determine the number of heads
        self.num_heads = 1
        if num_heads is not None:
            self.num_heads = num_heads
        self.num_heads = layer.layer_dict.get('num_heads', self.num_heads)


        self.bias = layer.layer_dict.get('bias', True)
        self.precision = torch.float
        if parameters.run_config.config.get('precision', 'float') == 'double':
            self.precision = torch.double

        self.mode = layer.layer_dict.get('mode', 'aggr_features') # mode can be headwise, aggr_heads, aggr_features. If headwise a linear transformation is applied to each head
        if self.mode == 'aggr_heads':
            k = math.sqrt(1.0 / (self.num_heads * self.in_features))
            self.Param_W = nn.Parameter(torch.nn.init.uniform_(torch.zeros(self.num_heads * self.in_features, self.out_features, dtype=self.precision), -k, k))
            self.Param_b = nn.Parameter(torch.nn.init.uniform_(torch.zeros(self.out_features, dtype=self.precision), -k, k))
        elif self.mode == 'aggr_features':
            k = math.sqrt(1.0/self.in_features)
            self.Param_W = nn.Parameter(torch.nn.init.uniform_(torch.zeros(self.in_features, self.out_features, dtype=self.precision), -k, k))
            self.Param_b = nn.Parameter(torch.nn.init.uniform_(torch.zeros(self.out_features, dtype=self.precision), -k, k))


        self.name = "Share GNN Linear Layer"


    def forward(self, node_representation:torch.Tensor, *args, **kwargs):
        """
        Forward pass of the layer
        param: x: torch.Tensor -> the input tensor
        param: pos: int -> the pos argument (ignored)
        """
        if self.mode == 'aggr_features':
            node_representation = node_representation @ self.Param_W
        elif self.mode == 'aggr_heads':
            # permute (C, N, F) to (N, C, F)
            node_representation = node_representation.permute(1,0,2)
            # convert to (N, CxF)
            node_representation = node_representation.reshape(node_representation.shape[0], -1)
            node_representation = node_representation @ self.Param_W
            #node_representation = node_representation.unsqueeze(0)

        if self.bias:
            node_representation = node_representation + self.Param_b
        node_representation = self.activation(node_representation)
        return node_representation
