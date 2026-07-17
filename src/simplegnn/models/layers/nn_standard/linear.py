import math

import torch
from torch import nn

from simplegnn.datasets.graph_dataset import GraphDataset
from simplegnn.models.ShareGNN.utils import is_batched_pos
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
            - 'factorized': a CP-factorized map over the (channel, feature) plane, W[c, f, o] = sum_r A[c, r] * B[f, r] * C[r, o].
              Costs C*R + F*R + R*F' parameters instead of C*F*F' and outputs the shape (N, F').
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
            # The channels are flattened into the feature dimension before the
            # matmul, so the weight matrix needs num_heads * in_features rows.
            self.Param_W = nn.Parameter(torch.nn.init.uniform_(torch.zeros(self.num_heads * self.in_features, self.out_features, dtype=self.precision), -k, k))
            self.Param_b = nn.Parameter(torch.nn.init.uniform_(torch.zeros(self.out_features, dtype=self.precision), -k, k))
        elif self.mode == 'aggr_features':
            k = math.sqrt(1.0/self.in_features)
            self.Param_W = nn.Parameter(torch.nn.init.uniform_(torch.zeros(self.in_features, self.out_features, dtype=self.precision), -k, k))
            self.Param_b = nn.Parameter(torch.nn.init.uniform_(torch.zeros(self.out_features, dtype=self.precision), -k, k))
        elif self.mode == 'channel_wise':
            k = math.sqrt(1.0 / self.in_features)
            self.Param_W = nn.Parameter(torch.nn.init.uniform_(torch.zeros(self.num_heads, self.in_features, self.out_features, dtype=self.precision), -k, k))
            # One bias per channel; the singleton node dimension makes the add
            # broadcast over N instead of colliding with it.
            self.Param_b = nn.Parameter(torch.nn.init.uniform_(torch.zeros(self.num_heads, 1, self.out_features, dtype=self.precision), -k, k))
        elif self.mode == 'factorized':
            self.rank = layer_args.get('rank', 32)
            # CP decomposition of the (C, F, F') readout tensor. The dense map
            # would need C * F * F' weights; the factors need R * (C + F + F').
            fan_in = self.num_heads * self.in_features
            # Var(W) = R * Var(A) * Var(B) * Var(C) is matched to the uniform
            # nn.Linear initialization Var = k^2/3 with k = sqrt(1/fan_in).
            k = (9.0 / (self.rank * fan_in)) ** (1.0 / 6.0)
            self.Param_A = nn.Parameter(torch.nn.init.uniform_(torch.zeros(self.num_heads, self.rank, dtype=self.precision), -k, k))
            self.Param_B = nn.Parameter(torch.nn.init.uniform_(torch.zeros(self.in_features, self.rank, dtype=self.precision), -k, k))
            self.Param_C = nn.Parameter(torch.nn.init.uniform_(torch.zeros(self.rank, self.out_features, dtype=self.precision), -k, k))
            b = math.sqrt(1.0 / fan_in)
            self.Param_b = nn.Parameter(torch.nn.init.uniform_(torch.zeros(self.out_features, dtype=self.precision), -b, b))
            # the (C, F) plane is contracted away, so the output is channel-free
            self.out_channels = 1
        else:
            raise ValueError(f"Unsupported linear layer mode: '{self.mode}'. Supported modes: "
                             f"'aggr_features', 'aggr_channels', 'channel_wise', 'factorized'.")

    def forward(self, node_representation:torch.Tensor, *args, **kwargs):
        """
        Forward pass of the layer
        param: x: torch.Tensor -> the input tensor
        param: pos: int -> the pos argument (ignored)
        """
        # graph-level tensors carry a leading batch dimension in the batched
        # ShareGNN forward, (B, C, F), where the per-graph forward has (C, 1, F)
        batched = is_batched_pos(kwargs.get('pos', None))
        if self.mode == 'aggr_features':
            # apply linear transformation, input is (N, F) self.Param_W is (F, F') and output is (N, F')
            node_representation = node_representation @ self.Param_W
        elif self.mode == 'aggr_channels':
            # merge channels and features and apply a single linear transformation, i.e., input is (C, N, F) -> (N, CxF) self.Param_W is (CxF, F') and output is (N, F')
            # 2D input (N, CxF) arrives with the channels already flattened
            # (e.g. from the invariant message-passing layers) and is used as is
            if node_representation.dim() == 3 and not batched:
                # permute (C, N, F) to (N, C, F)
                node_representation = node_representation.permute(1,0,2)
                # convert to (N, CxF)
                node_representation = node_representation.reshape(node_representation.shape[0], -1)
            elif node_representation.dim() == 3:
                # batched graph-level (B, C, F) -> (B, CxF)
                node_representation = node_representation.reshape(node_representation.shape[0], -1)
            node_representation = node_representation @ self.Param_W
        elif self.mode == 'channel_wise':
            if batched and node_representation.dim() == 3:
                # batched graph-level (B, C, F) x (C, F, F') -> (B, C, F'); the
                # plain matmul would broadcast B against C instead
                node_representation = torch.einsum('bcf,cfo->bco', node_representation, self.Param_W)
                if self.bias:
                    node_representation = node_representation + self.Param_b.squeeze(1)
                return self.activation(node_representation)
            # apply a separate linear transformation to each head
            node_representation = node_representation @ self.Param_W
        elif self.mode == 'factorized':
            if not batched and node_representation.dim() == 3:
                # per-graph (C, N, F) -> (N, C, F) so that N stays the leading axis
                node_representation = node_representation.permute(1, 0, 2)
            # contract the (C, F) plane through the rank-R bottleneck
            node_representation = torch.einsum('...cf,fr->...cr', node_representation, self.Param_B)
            node_representation = torch.einsum('...cr,cr->...r', node_representation, self.Param_A)
            node_representation = node_representation @ self.Param_C

        if self.bias:
            node_representation = node_representation + self.Param_b
        node_representation = self.activation(node_representation)
        return node_representation
