import torch

from datasets.graph_dataset import GraphDataset
from models.layers.framework_layer import FrameworkLayer


class ActivationLayer(FrameworkLayer):
    def __init__(self, layer_args):
        activation_function = layer_args.get('activation_function', torch.nn.Identity())
        super(ActivationLayer, self).__init__(layer_args)
        self.activation_function = activation_function
        self.name = "Activation Function"

    def forward(self, node_representation:torch.Tensor, batch_data: GraphDataset, *args, **kwargs):
        return self.activation_function(node_representation)