from abc import abstractmethod, ABC

import torch
from torch import nn

from simplegnn.datasets.graph_dataset import GraphDataset
from simplegnn.framework.utils.parameters import Parameters
from simplegnn.models.ShareGNN.utils import Layer
from simplegnn.models.layers.framework_layer import FrameworkLayer


class InvariantBasedLayer(FrameworkLayer, ABC):
    def __init__(self, parameters:Parameters, layer: Layer, graph_data: GraphDataset):
        super().__init__(layer_args=layer.layer_dict)
        # set seed for reproducibility
        torch.manual_seed(self.layer_id + self.seed)
        # layer information
        self.layer = layer
        self.para = parameters  # get the all the parameters of the experiment
        # get the underlying graph data
        self.graph_data = graph_data

        # Weights
        self.Param_W = None
        self.weight_distribution = None
        self.weight_distribution_slices = None
        self.weight_num = [] # number of weights per head
        self.current_W = torch.Tensor() # current weight matrix (for the graph considered in the forward pass)
        # Bias
        self.Param_b = None
        self.bias_distribution = None
        self.bias_distribution_slices = None
        self.bias_num = [] # number of biases per head
        self.current_B = torch.Tensor() # current bias matrix (for the graph considered in the forward pass)



        # number of node labels for message passing (per head)
        self.n_source_labels = []  # count of the different labels occuring for the first entry in the triple (each list entry stands for one head)
        self.source_label_descriptions = []  # graph invariant description (each list entry corresponds to one head)
        self.n_target_labels = []  # count of the different labels occuring for the second entry in the triple (each list entry stands for one head)
        self.target_label_descriptions = []  # graph invariant description (each list entry corresponds to one head)

        # pairwise properties for message passing (per head) e.g., the distance between two nodes
        self.n_properties = []  # counts of the different properties occuring in the third entry in the triple (each list entry corresponds to one head)
        self.property_descriptions = []

        # number of node labels for bias (per head)
        self.n_bias_labels = []  # count of the different labels occuring in the bias term (each list entry stands for one head)
        self.bias_label_descriptions = []  # graph invariant description (each list entry corresponds to one head)

    @abstractmethod
    def forward(self, node_representation:torch.Tensor, batch_data: GraphDataset, *args, **kwargs):
        pass
