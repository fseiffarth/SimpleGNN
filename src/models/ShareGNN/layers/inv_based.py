from abc import abstractmethod

import torch
from torch import nn

from datasets.graph_dataset import GraphDataset
from models.ShareGNN.utils import Layer
from models.layers.framework_layer import FrameworkLayer


class InvariantBasedLayer(FrameworkLayer):
    def __init__(self, layer_id, seed, parameters, layer: Layer, graph_data: GraphDataset, device='cpu', input_features=None, output_features=None):
        layer_args = {
            'activation': layer.layer_dict['activation'],
            'residual': layer.layer_dict.get('residual', False),
            'batch_norm': layer.layer_dict.get('batch_norm', False),
            'dropout': layer.layer_dict.get('dropout', 0.0),
        }
        super().__init__(layer_args=layer_args, device=device)
        # set seed for reproducibility
        torch.manual_seed(layer_id + seed)
        # id and name of the layer
        self.layer_id = layer_id
        # layer information
        self.layer = layer
        self.para = parameters  # get the all the parameters of the experiment
        self.name = f"Invariant Based Layer"
        # get the underlying graph data
        self.graph_data = graph_data
        self.precision = torch.float # set the precision of the weights
        if parameters.run_config.config.get('precision', 'float') == 'double':
            self.precision = torch.double

        # get the input features, i.e. the dimension of the input vector and output_features
        self.input_features = self.graph_data.num_node_features
        if input_features is not None:
            self.input_features = input_features
        self.output_features = self.graph_data.num_node_features
        if output_features is not None:
            self.output_features = output_features
        # number of heads
        self.num_heads = len(layer.layer_heads)

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
