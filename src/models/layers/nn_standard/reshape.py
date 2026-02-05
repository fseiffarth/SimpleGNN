import torch
from torch import nn

from datasets.graph_dataset import GraphDataset
from models.layers.framework_layer import FrameworkLayer


class Reshape(FrameworkLayer):
    def __init__(self,layer_args):
        """
        A layer that reshapes the input tensor to a specified shape.
        The input tensor is expected to have the shape (C, N, F),
        where C is the number of channels (or heads), N is the number of nodes, and F is the number of features per node.
        The shape argument can be a list of integers specifying the desired output shape,
        or the string 'flatten_head' to flatten the heads into the node dimension.
        Default shape is [-1, ], which flattens the entire tensor into a 1D vector.
        """
        layer_args['name'] = "Reshape Layer"
        super(Reshape, self).__init__(layer_args=layer_args)
        self.shape = layer_args.get('shape', [-1,])
        if isinstance(self.shape, str):
            # if the shape is named flatten heads
            if self.shape == 'flatten_head':
                # flatten the heads, i.e. reshape the input tensor from (C, N, F) to (C*N, F)
                self.shape = [-1, self.in_features]
                # update the out_features
                layer_args['out_features'] = self.in_features
                layer_args['out_channels'] = 1
                layer_args['num_heads'] = 1
        else:
            self.out_features = self.in_features*self.in_channels*self.num_heads
            self.out_channels = 1
            self.num_heads = 1



    def forward(self, node_representation:torch.Tensor, *args, **kwargs):
        node_representation = node_representation.reshape(shape=self.shape)
        return node_representation
