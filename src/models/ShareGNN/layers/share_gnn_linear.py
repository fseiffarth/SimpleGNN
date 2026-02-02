import math

import torch
from torch import nn

from datasets.graph_dataset import GraphDataset
from models.ShareGNN.utils import activation_function, Layer


class ShareGNNLinear(nn.Module):
    """
    Wrapper class for a linear layer that ignores the pos argument
    """
    def __init__(self, seed:int, layer_id:int, layer:Layer, parameters, graph_data:GraphDataset, num_heads=None, input_features=None, output_features=None):
        """
        """
        super(ShareGNNLinear, self).__init__()
        torch.manual_seed(layer_id + seed)
        self.layer = layer
        self.layer_id = layer_id

        # get the input features, i.e. the dimension of the input vector and output_features
        self.input_features = graph_data.num_node_features
        if input_features is not None:
            self.input_features = input_features
        self.input_features = layer.layer_dict.get('input_features', self.input_features)

        self.output_features = graph_data.num_node_features
        if output_features is not None:
            self.output_features = output_features
        self.output_features = layer.layer_dict.get('output_features', self.input_features)

        # determine the number of heads
        self.num_heads = 1
        if num_heads is not None:
            self.num_heads = num_heads
        self.num_heads = layer.layer_dict.get('num_heads', self.num_heads)


        self.bias = layer.layer_dict.get('bias', True)
        self.activation = activation_function(layer.layer_dict.get('activation', 'None'), **layer.layer_dict.get('activation_kwargs', {}))
        self.precision = torch.float
        if parameters.run_config.config.get('precision', 'float') == 'double':
            self.precision = torch.double

        self.mode = layer.layer_dict.get('mode', 'aggr_features') # mode can be headwise, aggr_heads, aggr_features. If headwise a linear transformation is applied to each head
        if self.mode == 'aggr_heads':
            k = math.sqrt(1.0 / (self.num_heads * self.input_features))
            self.Param_W = nn.Parameter(torch.nn.init.uniform_(torch.zeros(self.num_heads * self.input_features, self.output_features, dtype=self.precision), -k, k))
            self.Param_b = nn.Parameter(torch.nn.init.uniform_(torch.zeros(self.output_features, dtype=self.precision), -k, k))
        elif self.mode == 'aggr_features':
            k = math.sqrt(1.0/self.input_features)
            self.Param_W = nn.Parameter(torch.nn.init.uniform_(torch.zeros(self.input_features, self.output_features, dtype=self.precision), -k, k))
            self.Param_b = nn.Parameter(torch.nn.init.uniform_(torch.zeros(self.output_features, dtype=self.precision), -k, k))


        self.name = "Linear Layer"


    def forward(self, x: torch.Tensor, pos:int=None):
        """
        Forward pass of the layer
        param: x: torch.Tensor -> the input tensor
        param: pos: int -> the pos argument (ignored)
        """
        if self.mode == 'aggr_features':
            x = x @ self.Param_W
        elif self.mode == 'aggr_heads':
            # permute (C, N, F) to (N, C, F)
            x = x.permute(1,0,2)
            # convert to (N, CxF)
            x = x.reshape(x.shape[0], -1)
            x = x @ self.Param_W
            #x = x.unsqueeze(0)

        if self.bias:
            x = x + self.Param_b
        x = self.activation(x)
        return x
