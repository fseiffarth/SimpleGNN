import torch
import torch_geometric

from simplegnn.datasets.graph_dataset import GraphDataset
from simplegnn.models.layers.framework_layer import FrameworkLayer


class GlobalPooling(FrameworkLayer):
    def __init__(self, layer_args):
        super(GlobalPooling, self).__init__(layer_args)
        self.mode = layer_args.get('mode', 'mean')
        self.pooling_function = None
        if self.mode == 'mean':
            self.pooling_function = torch_geometric.nn.global_mean_pool
        elif self.mode == 'max':
            self.pooling_function = torch_geometric.nn.global_max_pool
        elif self.mode == 'sum':
            self.pooling_function = torch_geometric.nn.global_add_pool
        else:
            raise ValueError(f"Unsupported pooling mode: {self.mode}")
        self.name = f"Global {self.mode.capitalize()} Pooling"


    def forward(self, node_representation:torch.Tensor, batch_data: GraphDataset, *args, **kwargs):
        return self.activation(self.pooling_function(node_representation, batch_data.batch))