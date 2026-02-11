import math

import torch
from torch import nn

from simplegnn.datasets.graph_dataset import GraphDataset
from simplegnn.models.layers.framework_layer import FrameworkLayer


class LinearLayer(FrameworkLayer):
    def __init__(self, layer_args):
        """
        A standard linear layer that applies a linear transformation to the input tensor.
        The input tensor is expected to have the shape (C, N, F), where C is the number of channels (or heads),
        N is the number of nodes, and F is the number of features per node.
        The linear layer depends on the mode parameter:
            - 'channel_wise': applies a separate linear transformation to each head. The output tensor will have the shape (C, N, F'), where F' is the number of output features per node.
            - 'aggr_channels': aggregates the heads before applying a linear transformation. The output tensor will have the shape (N, F'), where F' is the number of output features per node.
            - 'aggr_features': aggregates the features before applying a linear transformation. The output tensor will have the shape (C, N, F'), where F' is the number of output features per node.
        The output tensor will have the shape (C, N, F'), where F' is the number of output features per node.
        If mode is set to 'channel_wise', a separate linear transformation is applied to each head.
        If mode is set to 'aggr_channels' or 'aggr_features', the heads are aggregated before applying the linear transformation.
        Default mode is 'aggr_features'.
        """
        layer_args['name'] = "Linear Layer"
        layer_args['mode'] = layer_args.get('mode', 'aggr_features')
        super(LinearLayer, self).__init__(layer_args=layer_args)
        self.mode = layer_args['mode']
        torch.manual_seed(self.layer_id + self.seed)
        if self.mode == 'aggr_channels':
            self.out_channels = 1
            k = math.sqrt(1.0 / (self.num_heads * self.in_features))
            self.Param_W = nn.Parameter(torch.nn.init.uniform_(torch.zeros(self.in_features, self.out_features, dtype=self.precision), -k, k))
            self.Param_b = nn.Parameter(torch.nn.init.uniform_(torch.zeros(self.out_features, dtype=self.precision), -k, k))
        elif self.mode == 'aggr_features':
            k = math.sqrt(1.0/self.in_features)
            self.Param_W = nn.Parameter(torch.nn.init.uniform_(torch.zeros(self.in_features, self.out_features, dtype=self.precision), -k, k))
            self.Param_b = nn.Parameter(torch.nn.init.uniform_(torch.zeros(self.out_features, dtype=self.precision), -k, k))
        elif self.mode == 'channel_wise':
            k = math.sqrt(1.0 / self.in_features)
            self.Param_W = nn.Parameter(torch.nn.init.uniform_(torch.zeros(self.num_heads, self.in_features, self.out_features, dtype=self.precision), -k, k))
            self.Param_b = nn.Parameter(torch.nn.init.uniform_(torch.zeros(self.out_channels, self.out_features, dtype=self.precision), -k, k))

        self.layer = torch.nn.Linear(self.in_features, self.out_features, self.bias, self.device, self.precision)

    def forward(self, node_representation:torch.Tensor, *args, **kwargs):
        """
        Forward pass of the layer
        param: x: torch.Tensor -> the input tensor
        param: pos: int -> the pos argument (ignored)
        """
        if self.mode == 'aggr_features':
            # apply linear transformation, input is (N, F) self.Param_W is (F, F') and output is (N, F')
            node_representation = node_representation @ self.Param_W
        elif self.mode == 'aggr_channels':
            # merge channels and features and apply a single linear transformation, i.e., input is (C, N, F) -> (N, CxF) self.Param_W is (CxF, F') and output is (1, N, F')
            # permute (C, N, F) to (N, C, F)
            node_representation = node_representation.permute(1,0,2)
            # convert to (N, CxF)
            node_representation = node_representation.reshape(node_representation.shape[0], -1)
            node_representation = node_representation @ self.Param_W
            #node_representation = node_representation.unsqueeze(0)
        elif self.mode == 'channel_wise':
            # apply a separate linear transformation to each head
            node_representation = node_representation @ self.Param_W

        if self.bias:
            node_representation = node_representation + self.Param_b
        node_representation = self.activation(node_representation)
        return node_representation