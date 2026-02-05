import torch

from datasets.graph_dataset import GraphDataset
from models.layers.framework_layer import FrameworkLayer


class LinearLayer(FrameworkLayer):
    def __init__(self, layer_args):
        """
        A standard linear layer that applies a linear transformation to the input tensor.
        The input tensor is expected to have the shape (C, N, F), where C is the number of channels (or heads),
        N is the number of nodes, and F is the number of features per node.
        The linear layer depends on the mode parameter:
            mo
        The output tensor will have the shape (C, N, F'), where F' is the number of output features per node.
        If mode is set to 'headwise', a separate linear transformation is applied to each head.
        If mode is set to 'aggr_heads' or 'aggr_features', the heads are aggregated before applying the linear transformation.
        Default mode is 'aggr_features'.
        """
        layer_args['name'] = "Linear Layer"
        layer_args['mode'] = layer_args.get('mode', 'aggr_features')
        super(LinearLayer, self).__init__(layer_args=layer_args)
        self.layer = torch.nn.Linear(self.in_features, self.out_features, self.bias, self.device, self.precision)

    def forward(self, node_representation:torch.Tensor, batch_data: GraphDataset, *args, **kwargs):
        return self.activation(self.layer(node_representation))