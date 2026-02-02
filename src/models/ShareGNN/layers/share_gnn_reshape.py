import torch
from torch import nn

from datasets.graph_dataset import GraphDataset


class ShareGNNReshapeLayer(nn.Module):
    def __init__(self, layer_id, seed, layer, parameters, graph_data:GraphDataset, num_heads=None, input_features=None, output_features=None):
        super(ShareGNNReshapeLayer, self).__init__()
        self.name = "Reshape Layer"
        self.layer = layer
        self.shape = layer.layer_dict.get('shape', [-1,])
        if isinstance(self.shape, str):
            # if the shape is named flatten heads
            if self.shape == 'flatten_head':
                # flatten the heads, i.e. reshape the input tensor from (C, N, F) to (C*N, F)
                self.shape = [-1, self.layer.layer_dict.get('input_features', graph_data.num_node_features)]
        # get the input features, i.e. the dimension of the input vector and output_features
        self.input_features = graph_data.num_node_features
        if input_features is not None:
            self.input_features = input_features
        self.input_features = layer.layer_dict.get('input_features', self.input_features)
        self.output_features = graph_data.num_node_features
        if output_features is not None:
            self.output_features = output_features
        self.output_features = layer.layer_dict.get('output_features', self.output_features)

        # determine the number of heads
        self.num_heads = 1
        if num_heads is not None:
            self.num_heads = num_heads
        self.num_heads = layer.layer_dict.get('num_heads', self.num_heads)


    def forward(self, x:torch.Tensor, pos:int=None):
        x = x.reshape(shape=self.shape)
        return x
