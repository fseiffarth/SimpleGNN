import torch

from simplegnn.datasets.graph_dataset import GraphDataset
from simplegnn.models.layers.framework_layer import FrameworkLayer


class BatchNormLayer(FrameworkLayer):
    def __init__(self, layer_args):
        layer_args['batch_norm'] = True # Ensure batch_norm is set to True by default
        super(BatchNormLayer, self).__init__(layer_args)
        self.name = "Batch Normalization Layer"

    def forward(self, node_representation:torch.Tensor, batch_data: GraphDataset, *args, **kwargs):
        return self.batch_norm_layer(node_representation)