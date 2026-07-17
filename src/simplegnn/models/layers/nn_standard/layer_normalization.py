import torch
from torch import nn

from simplegnn.models.layers.framework_layer import FrameworkLayer


class LayerNormalization(FrameworkLayer):
    """
    Standard layer normalization with learnable elementwise affine parameters.

    Normalizes over the feature (last) dimension only, so batched and
    per-graph forwards compute the same function for every input shape:
    (F,), (N, F), (C, N, F) node-level and (B, C, F) graph-level tensors
    all share F as the last dimension.

    Optional layer_args keys:
    - 'layer_norm_eps' : float (default 1e-5)
    - 'elementwise_affine' : bool (default True) -> learnable weight (and bias)
    - 'bias' : bool (default True) -> learnable bias, only used when
      elementwise_affine is True
    """
    def __init__(self, layer_args):
        layer_args['name'] = "Layer Normalization"
        super(LayerNormalization, self).__init__(layer_args=layer_args)
        self.elementwise_affine = layer_args.get('elementwise_affine', True)
        self.layer_norm = nn.LayerNorm(
            normalized_shape=self.out_features,
            eps=layer_args.get('layer_norm_eps', 1e-5),
            elementwise_affine=self.elementwise_affine,
            bias=self.bias,
            dtype=self.precision,
        )

    def forward(self, node_representation:torch.Tensor, *args, **kwargs):
        """
        Forward pass of the layer normalization
        param: node_representation: torch.Tensor -> the input tensor
        """
        return self.layer_norm(node_representation)
