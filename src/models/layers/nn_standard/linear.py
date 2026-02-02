import torch

from datasets.graph_dataset import GraphDataset
from models.layers.framework_layer import FrameworkLayer


class LinearLayer(FrameworkLayer):
    def __init__(self, layer_args):
        super(LinearLayer, self).__init__(layer_args)
        self.layer = torch.nn.Linear(**layer_args)
        self.name = "Linear Layer"

    def forward(self, node_representation:torch.Tensor, batch_data: GraphDataset, *args, **kwargs):
        return self.activation(self.layer(node_representation))