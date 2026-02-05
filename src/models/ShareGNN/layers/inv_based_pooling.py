import time
from typing import Tuple

import matplotlib
import networkx as nx
import numpy as np
import torch
from torch import nn

from datasets.graph_dataset import GraphDataset
from datasets.utils.graph_drawing import GraphDrawing
from framework.utils.parameters import Parameters
from models.ShareGNN.layers.inv_based import InvariantBasedLayer
from models.ShareGNN.utils import Layer


class InvariantBasedAggregationLayer(InvariantBasedLayer):
    """
    This class represents an invariant based decoder layer of a ShareGNN
    :param
    layer_id: int -> the id of the layer in the network
    seed: int -> the seed for reproducibility
    :param Parameters -> the parameters of the network

    **forward(x: torch.Tensor, pos:int) -> out: torch.Tensor**
        - **x** is the input matrix of shape (N, F) where N is the number of nodes and F is the number of node features.
        - **pos** is the index of the graph in the graph_data
        - **out** is the output matrix of shape (H, N, F) where H is the number of heads and N is the number of nodes and F is the number of node features.
    """
    def __init__(self, parameters:Parameters, layer: Layer, graph_data: GraphDataset):
        layer.layer_dict['name'] = "Invariant Based Aggregation Layer"
        super(InvariantBasedAggregationLayer, self).__init__(parameters, layer, graph_data)
        torch.manual_seed(self.seed + self.layer_id)
        self.layer = layer
        # fixed output dimension of the layer
        self.output_dimension = layer.layer_dict.get('out_dim', None)
        if self.output_dimension is None:
            raise ValueError("out_dim must be provided for the InvariantBasedAggregationLayer")

        # Update the out_features in the layer dictionary
        layer.layer_dict['out_features'] = self.num_heads * self.out_features * self.output_dimension

        self.n_node_labels = [] # number of node labels per head
        self.node_label_descriptions = [] # node label descriptions per head
        # bias per head
        self.bias_list = [head.bias for head in layer.layer_heads]
        # is there any bias
        self.bias = any(self.bias_list)
        for i, head in enumerate(layer.layer_heads):
            self.node_label_descriptions.append(layer.get_source_string(i))
            self.n_node_labels.append(self.graph_data.node_labels[self.node_label_descriptions[i]].num_unique_node_labels)

        self.weight_num = np.sum(self.n_node_labels) * self.output_dimension
        self.weight_distribution = [None] * len(self.graph_data)

        for i, head in enumerate(layer.layer_heads):
            node_labels = self.graph_data.node_labels[self.node_label_descriptions[i]].node_labels
            # Set the bias weights
            _, indices, counts = torch.unique(node_labels, dim=0, return_inverse=True, return_counts=True, sorted=False)
            for idx in range(len(self.graph_data)):
                for out_dim_id in range(self.output_dimension):
                    new_weight_distribution = torch.zeros((self.graph_data.num_nodes[idx].item(), 4), dtype=torch.int64)
                    new_weight_distribution[:, 0] = i
                    new_weight_distribution[:, 1] = out_dim_id
                    new_weight_distribution[:, 2] = torch.arange(self.graph_data.num_nodes[idx].item()) # torch.arange(start=self.graph_data.slices['x'][idx], end=self.graph_data.slices['x'][idx+1], dtype=torch.int64)
                    new_weight_distribution[:, 3] = indices[self.graph_data.slices['x'][idx]:self.graph_data.slices['x'][idx+1]] + out_dim_id * self.n_node_labels[i]
                    if self.weight_distribution[idx] is None:
                        self.weight_distribution[idx] = new_weight_distribution.detach().clone()
                    else:
                        self.weight_distribution[idx] = torch.cat((self.weight_distribution[idx], new_weight_distribution), dim=0)


        # merge the bias distribution of all graphs (creating additionally slicing information)
        self.weight_distribution_slices = torch.tensor([0] + [len(w) for w in self.weight_distribution], dtype=torch.int64).cumsum(dim=0)
        self.weight_distribution = torch.cat([self.weight_distribution[i] for i in range(len(self.graph_data))], dim=0).to(self.device)

        self.Param_W = self.init_weights(self.weight_num, init_type='aggregation')


        if self.bias:
            self.Param_b = self.init_weights(shape=(self.num_heads, self.output_dimension, self.in_features), init_type='aggregation_bias').to(self.device)
        self.forward_step_time = 0



        # in case of pruning is turned on, save the original weights
        self.Param_W_original = None
        self.mask = None
        if 'prune' in self.para.run_config.config and self.para.run_config.config['prune']['enabled']:
            self.Param_W_original = self.Param_W.detach().clone()
            self.mask = torch.ones(self.Param_W.size(), requires_grad=False)

    def init_weights(self, shape, init_type=None):
        num_weights = np.prod(shape)
        weights = nn.Parameter(torch.zeros(shape, dtype=self.precision), requires_grad=True)
        weight_init = self.para.run_config.config.get('weight_initialization', None)
        if weight_init is not None:
            weight_initialization = weight_init.get(init_type, None)
            if weight_initialization is not None:
                if weight_initialization.get('type', None) == 'uniform':
                    torch.nn.init.uniform_(weights, a=weight_initialization.get('min', 0.0), b=weight_initialization.get('max', 1.0))
                elif weight_initialization.get('type', None) == 'normal':
                    torch.nn.init.normal_(weights, mean=weight_initialization.get('mean', 0.0), std=weight_initialization.get('std', 1.0))
                elif weight_initialization.get('type', None) == 'symmetric_normal':
                    # choose from two normal distributions one with positive and one with negative mean
                    # shuffle the indices
                    weight_arrange = torch.randperm(torch.arange(0, num_weights).size(0))
                    # initialize the weights with indeces in weight_arrange[0:num_weights//2] with positive mean and the rest with negative mean
                    new_weights = torch.zeros(num_weights, dtype=self.precision)
                    new_weights[weight_arrange[0:num_weights//2]] = torch.normal(mean=weight_initialization.get('mean', 0.0), std=weight_initialization.get('std', 1.0), size=(weight_arrange[0:num_weights//2].size(0),), dtype=self.precision)
                    new_weights[weight_arrange[num_weights//2:]] = -torch.normal(mean=weight_initialization.get('mean', 0.0), std=weight_initialization.get('std', 1.0), size=(weight_arrange[num_weights//2:].size(0),), dtype=self.precision)
                    # reshape new_weights to the shape of the weights
                    new_weights = new_weights.reshape(shape)
                    weights = nn.Parameter(new_weights, requires_grad=True)
                elif weight_initialization.get('type', None) == 'constant':
                    torch.nn.init.constant_(weights, weight_initialization.get('value', 0.01))
                elif weight_initialization.get('type', None) == 'lower_upper':
                    # calculate the range for the weights
                    lower, upper = -(1.0 / np.sqrt(num_weights)), (1.0 / np.sqrt(num_weights))
                    weights = nn.Parameter(lower + torch.randn(shape, dtype=self.precision) * (upper - lower))
                elif weight_initialization.get('type', None) == 'he':
                    std = np.sqrt(2.0 / num_weights)
                    weights = nn.Parameter(torch.randn(num_weights, dtype=self.precision) * std)
            else:
                torch.nn.init.constant_(weights, 0.01)
        else:
            torch.nn.init.constant_(weights, 0.01)
        return weights

    def set_weights(self, pos):
        input_size = self.graph_data.num_nodes[pos]
        self.current_W = torch.zeros((self.num_heads, self.output_dimension, input_size), dtype=self.precision).to(self.device)
        weight_distr = self.weight_distribution[self.weight_distribution_slices[pos]:self.weight_distribution_slices[pos+1]]
        param_indices = weight_distr[:, 3]
        matrix_indices = weight_distr[:, 0:3].T
        self.current_W[matrix_indices[0], matrix_indices[1], matrix_indices[2]] = torch.take(self.Param_W, param_indices)
        # divide the weights by the number of nodes in the graph
        #self.current_W = self.current_W / input_size
        pass

    def print_weights(self):
        print("Weights of the Resize layer")
        for x in self.Param_W:
            print("\t", x.data)

    def print_bias(self):
        print("Bias of the Resize layer")
        for x in self.Param_b:
            print("\t", x.data)


    def print_all(self):
        # print the layer name
        print("Layer: ", self.name)
        print("\tLearnable Weights:")
        # print non-zero/total parameters
        num_params = self.Param_W.numel()
        num_non_zero_params = torch.nonzero(self.Param_W).size(0)
        print(f"\t\tNon-zero parameters: {num_non_zero_params}/{num_params}")
        # print relative number of non-zero parameters
        print(f"\t\tRelative non-zero parameters: {num_non_zero_params / num_params * 100:.2f}%")
        # print the bias parameters
        print("\tLearnable Bias:")
        num_params = self.Param_b.numel()
        num_non_zero_params = torch.nonzero(self.Param_b).size(0)
        print(f"\t\tNon-zero parameters: {num_non_zero_params}/{num_params}")
        print(f"\t\tRelative non-zero parameters: {num_non_zero_params / num_params * 100:.2f}%")


    def forward(self, node_representation:torch.Tensor, batch_data: GraphDataset, *args, **kwargs):
        pos = kwargs.get('pos', 0)
        #x = x.view(-1)
        # remove first dim if x is of shape (1, N, F) (check if x is 3-dimensional)
        if node_representation.size(0) == 1 and node_representation.dim() == 3:
            node_representation = node_representation.squeeze(0)
        begin = time.time()
        self.set_weights(pos)
        node_representation = torch.einsum('cij,jk->cik', self.current_W, node_representation)
        if self.bias:
            node_representation = node_representation + self.Param_b
        self.forward_step_time += time.time() - begin
        return node_representation

    def get_weights(self):
        return [x.item() for x in self.Param_W]

    def get_bias(self):
        return [x.item() for x in self.Param_b[0]]

    def draw(self, ax, graph_id, graph_drawing: Tuple[GraphDrawing, GraphDrawing], head=0, out_dimension=0, with_graph=True, graph_only=False):
        # create graph
        graph = self.graph_data.create_nx_graph(graph_id, directed=False)
        if with_graph or graph_only:
            # draw the graph
            # root node is the one with label 0
            root_node = None
            for node in graph.nodes():
                if self.graph_data.node_labels['primary'].node_labels[graph_id][node] == 0:
                    root_node = node
                    break

            # if graph is circular use the circular layout
            pos = dict()
            if graph_drawing[0].draw_type == 'circle':
                # get circular positions around (0,0) starting with the root node at (-400,0)
                pos[root_node] = (400, 0)
                angle = 2 * np.pi / (graph.number_of_nodes())
                # iterate over the neighbors of the root node
                cur_node = root_node
                last_node = None
                counter = 0
                while len(pos) < graph.number_of_nodes():
                    neighbors = list(graph.neighbors(cur_node))
                    for next_node in neighbors:
                        if next_node != last_node:
                            counter += 1
                            pos[next_node] = (400 * np.cos(counter * angle), 400 * np.sin(counter * angle))
                            last_node = cur_node
                            cur_node = next_node
                            break
            elif graph_drawing[0].draw_type == 'kawai':
                pos = nx.kamada_kawai_layout(graph)
            elif graph_drawing[0].draw_type == 'shell':
                pos = nx.shell_layout(graph)
            elif graph_drawing[0].draw_type == 'bfs':
                pos = nx.bfs_layout(graph, 0)
            else:
                pos = nx.nx_pydot.graphviz_layout(graph)
            # keys to ints
            pos = {int(k): v for k, v in pos.items()}
            if graph_only:
                edge_labels = {}
                for (key1, key2, value) in graph.edges(data=True):
                    if "label" in value and len(value["label"]) > 1:
                        edge_labels[(key1, key2)] = int(value["label"][0])
                    else:
                        edge_labels[(key1, key2)] = ""
                nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=graph_drawing[0].edge_color,
                                       width=graph_drawing[0].edge_width)
                nx.draw_networkx_edge_labels(graph, pos=pos, edge_labels=edge_labels, ax=ax, font_size=8,
                                             font_color='black')
                # get node colors from the node labels using the plasma colormap
                cmap = graph_drawing[0].colormap
                norm = matplotlib.colors.Normalize(vmin=0,
                                                   vmax=self.graph_data.node_labels['primary'].num_unique_node_labels)
                node_colors = [cmap(norm(self.graph_data.node_labels['primary'].node_labels[graph_id][node])) for node
                               in graph.nodes()]
                nx.draw_networkx_nodes(graph, pos=pos, ax=ax, node_color=node_colors,
                                       node_size=graph_drawing[0].node_size)
                return
            nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=graph_drawing[1].edge_color, width=graph_drawing[1].edge_width, alpha=graph_drawing[1].edge_alpha*0.5)

        all_weights = np.array(self.get_weights())
        bias = self.get_bias()
        graph = self.graph_data.graphs[graph_id]
        weight_distribution = self.weight_distribution[graph_id]
        param_indices = np.array(weight_distribution[:, 3])
        matrix_indices = np.array(weight_distribution[:, 0:3])
        graph_weights = all_weights[param_indices]

        weight_min = np.min(graph_weights)
        weight_max = np.max(graph_weights)
        weight_max_abs = max(abs(weight_min), abs(weight_max))
        bias_min = np.min(bias)
        bias_max = np.max(bias)
        bias_max_abs = max(abs(bias_min), abs(bias_max))

        # use seismic colormap with maximum and minimum values from the weight matrix
        cmap = graph_drawing[1].colormap
        # normalize item number values to colormap
        normed_weight = (graph_weights + (-weight_min)) / (weight_max - weight_min)
        weight_colors = cmap(normed_weight)
        normed_bias = (bias + (-bias_min)) / (bias_max - bias_min)
        bias_colors = cmap(normed_bias)

        # draw the graph
        # if graph is circular use the circular layout
        pos = dict()
        if graph_drawing[0].draw_type == 'circle':
            # root node is the one with label 0
            root_node = None
            for i, node in enumerate(graph.nodes()):
                if i == 0:
                    print(f"First node: {self.graph_data.node_labels['primary'].node_labels[graph_id][node]}")
                if self.graph_data.node_labels['primary'].node_labels[graph_id][node] == 0:
                    root_node = node
                    break
            # get circular positions around (0,0) starting with the root node at (-400,0)
            pos[root_node] = (400, 0)
            angle = 2 * np.pi / (graph.number_of_nodes())
            # iterate over the neighbors of the root node
            cur_node = root_node
            last_node = None
            counter = 0
            while len(pos) < graph.number_of_nodes():
                neighbors = list(graph.neighbors(cur_node))
                for next_node in neighbors:
                    if next_node != last_node:
                        counter += 1
                        pos[next_node] = (400 * np.cos(counter * angle), 400 * np.sin(counter * angle))
                        last_node = cur_node
                        cur_node = next_node
                        break
        elif graph_drawing[0].draw_type == 'kawai':
            pos = nx.kamada_kawai_layout(graph)
        elif graph_drawing[0].draw_type == 'shell':
            pos = nx.shell_layout(graph)
        elif graph_drawing[0].draw_type == 'bfs':
            pos = nx.bfs_layout(graph,0)
        else:
            pos = nx.nx_pydot.graphviz_layout(graph)
        # keys to ints
        pos = {int(k): v for k, v in pos.items()}
        # graph to digraph with
        digraph = nx.DiGraph()
        for node in graph.nodes():
            digraph.add_node(node)



        node_colors = []
        node_sizes = []
        for i, index in enumerate(weight_distribution):
            c = index[0]
            o_dimension = index[1]
            node_idx = index[2]
            weight = index[3]
            if c == head and o_dimension == out_dimension:
                node_colors.append(weight_colors[i])
                node_sizes.append(graph_drawing[1].node_size * abs(graph_weights[i]) / weight_max_abs)

        nx.draw_networkx_nodes(digraph, pos=pos, ax=ax, node_color=node_colors, node_size=node_sizes)
