import torch
from torch import nn

from simplegnn.models.layers.framework_layer import FrameworkLayer


class LayerNormalization(FrameworkLayer):
    def __init__(self, layer_args):
        layer_args['name'] = "Layer Normalization"
        super(LayerNormalization, self).__init__(layer_args=layer_args)

    def forward(self, node_representation:torch.Tensor, *args, **kwargs):
        """
        Forward pass of the layer normalization
        param: x: torch.Tensor -> the input tensor
        param: pos: int -> the pos argument (ignored)
        """
        # apply layer normalization
        if len(node_representation.shape) == 2:
            node_representation = nn.functional.layer_norm(node_representation, normalized_shape=[node_representation.shape[-1]])
        elif len(node_representation.shape) == 3:
            node_representation = nn.functional.layer_norm(node_representation, normalized_shape=[node_representation.shape[-2], node_representation.shape[-1]])
        return node_representation