import torch
from torch import nn


class ShareGNNActivation(nn.Module):
    def __init__(self, activation_function):
        super(ShareGNNActivation, self).__init__()
        self.activation_function = activation_function
        self.name = "Activation Function"

    def forward(self, x: torch.Tensor, pos:int=None):
        return self.activation_function(x)
