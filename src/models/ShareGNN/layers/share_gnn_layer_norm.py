import torch
from torch import nn


class ShareGNNLayerNorm(nn.Module):
    def __init__(self, layer_id, num_heads, input_features=None, output_features=None):
        super(ShareGNNLayerNorm, self).__init__()
        self.name = "Layer Normalization"
        self.input_features = input_features
        self.output_features = output_features
        self.num_heads = num_heads
        self.layer_id = layer_id
    def forward(self, x: torch.Tensor, pos:int=None):
        """
        Forward pass of the layer normalization
        param: x: torch.Tensor -> the input tensor
        param: pos: int -> the pos argument (ignored)
        """
        # apply layer normalization
        if len(x.shape) == 2:
            x = nn.functional.layer_norm(x, normalized_shape=[x.shape[-1]])
        elif len(x.shape) == 3:
            x = nn.functional.layer_norm(x, normalized_shape=[x.shape[-2], x.shape[-1]])
        return x