import torch

from datasets.graph_dataset import GraphDataset
from models.layers.framework_layer import FrameworkLayer


class DropoutLayer(FrameworkLayer):
    def __init__(self, layer_args):
        p = layer_args.get('p', 0.5)
        super(DropoutLayer, self).__init__(layer_args)
        self.dropout = torch.nn.Dropout(p)
        self.name = "Dropout Layer"

    def forward(self, node_representation:torch.Tensor, batch_data: GraphDataset, *args, **kwargs):
        if self.training:
            return self.dropout(node_representation)
        else:
            return node_representation