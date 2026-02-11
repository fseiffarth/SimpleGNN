from abc import abstractmethod

import torch

from simplegnn.datasets.graph_dataset import GraphDataset
from simplegnn.models.layers.framework_layer import FrameworkLayer


class GNNConvLayer(FrameworkLayer):
    def __init__(self, layer_args):
        super(GNNConvLayer, self).__init__(layer_args)

    @abstractmethod
    def forward(self, node_representation:torch.Tensor, batch_data: GraphDataset, *args, **kwargs):
        return node_representation