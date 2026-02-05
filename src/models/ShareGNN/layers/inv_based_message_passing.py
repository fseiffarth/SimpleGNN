import time
from pathlib import Path
from typing import Optional, Tuple

import networkx as nx
import numpy as np
import torch
from torch import nn
import sys

from datasets.graph_dataset import GraphDataset
from datasets.utils.graph_drawing import GraphDrawing
from framework.utils.parameters import Parameters
from models.ShareGNN.layers.inv_based import InvariantBasedLayer
from models.ShareGNN.utils import Layer


class InvariantBasedMessagePassingLayer(InvariantBasedLayer):
    """
    This class represents a message passing layer of the encoder of an ShareGNN.
    :param layer_id: the id of the layer
    :param seed: the seed for the random number generator
    :param parameters: the parameters of the experiment
    :param graph_data: the data of the graph dataset
    :param device: use 'cpu' or 'cuda' as device ('cpu' is recommended)
    :param input_feature_dimensions: the number of input features

    **forward(x: torch.Tensor, pos:int) -> out: torch.Tensor**
        - **x** is the input matrix of shape (N, F) where N is the number of nodes and F is the number of node features. F should be constant over all graphs (TODO allow different F for different graphs)
        - **pos** is the index of the graph in the graph_data
        - **out** is the output matrix of shape (H, N, F) where H is the number of heads and N is the number of nodes. F is the number of node features.
    """



    def __init__(self, parameters: Parameters, layer: Layer, graph_data: GraphDataset):
        """
        Constructor of the GraphConvLayer
        :param layer_id: the id of the layer
        :param seed: the seed for the random number generator
        :param parameters: the parameters of the experiment
        :param graph_data: the data of the graph dataset
        :param device: use 'cpu' or 'cuda' as device ('cpu' is recommended)
        :param input_feature_dimensions: the number of input features
        """
        layer.layer_dict['name'] = "Invariant Based Message Passing Layer"
        super(InvariantBasedMessagePassingLayer, self).__init__(parameters, layer, graph_data)

        for h_id, head in enumerate(layer.layer_heads):
            self.source_label_descriptions.append(layer.get_source_string(h_id))
            self.n_source_labels.append(graph_data.node_labels[self.source_label_descriptions[h_id]].num_unique_node_labels)
            self.target_label_descriptions.append(layer.get_target_string(h_id))
            self.n_target_labels.append(graph_data.node_labels[self.target_label_descriptions[h_id]].num_unique_node_labels)
            self.bias_label_descriptions.append(layer.get_bias_string(h_id))
            self.n_bias_labels.append(graph_data.node_labels[self.bias_label_descriptions[h_id]].num_unique_node_labels)
            self.property_descriptions.append(head.property_dict.get_property_string())
            self.n_properties.append(graph_data.properties[self.property_descriptions[h_id]].num_properties[(self.layer_id, h_id)])

        self.bias_list = [head.bias for head in layer.layer_heads]
        self.bias = any(self.bias_list)  # check if bias is used

        # Determine the number of weights and biases
        # There are two cases asymetric and symmetric, asymetric is the default, TODO add symmetric case
        self.skips = [0]
        self.skips_description = [None]
        self.skips_description_text = [None]
        self.weight_distribution = [None] * len(graph_data)
        self.bias_distribution = [None] * len(graph_data)

        # Iterate over all heads in the layer
        for i, head in enumerate(self.layer.layer_heads):
            # get all the valid property values for the head (e.g., the distances 0, 3, 6)
            valid_property_values = self.graph_data.properties[self.property_descriptions[i]].valid_values[(self.layer_id, i)]
            # apply the head and tail labels to the subdict
            source_labels = self.graph_data.node_labels[self.source_label_descriptions[i]].node_labels
            target_labels = self.graph_data.node_labels[self.target_label_descriptions[i]].node_labels
            bias_labels = self.graph_data.node_labels[self.bias_label_descriptions[i]].node_labels
            for key in valid_property_values:
                #print(f'Initialize head {i+1}/{len(self.layer.layer_heads)} with property {key}')
                property_subdict = self.graph_data.properties[self.property_descriptions[i]].properties[key]
                property_subdict_slices = self.graph_data.properties[self.property_descriptions[i]].properties_slices[key]
                labeled_subdict = property_subdict.detach().clone()
                labeled_subdict[:, 0] = source_labels[property_subdict[:, 0]]
                labeled_subdict[:, 1] = target_labels[property_subdict[:, 1]]
                # set all indices to -1 where the head or tail label is -1
                invalid_indices = torch.where(torch.logical_or(labeled_subdict[:, 0] == -1, labeled_subdict[:, 1] == -1))[0]
                do_invalid_indices_exist = len(invalid_indices) > 0
                if do_invalid_indices_exist:
                    max_first = torch.max(labeled_subdict[:, 0]) + 1
                    max_second = torch.max(labeled_subdict[:, 1]) + 1
                    labeled_subdict[invalid_indices] = torch.tensor([max_first, max_second])
                # get unique rows of the property subdict together with counts and indices
                _, indices, counts = torch.unique(labeled_subdict, dim=0, return_inverse=True, return_counts=True, sorted=False)
                if do_invalid_indices_exist:
                    counts[-1] = 0
                # set all indices to -1 where the count is smaller than the threshold TODO
                threshold = self.para.run_config.config.get('rule_occurrence_threshold', 1)
                upper_threshold = self.para.run_config.config.get('rule_occurrence_upper_threshold', None)
                num_weights = len(counts)
                if do_invalid_indices_exist:
                    num_weights -= 1
                if threshold > 1 or do_invalid_indices_exist or upper_threshold is not None:
                    # get a bool tensor from indices where the entry is true if the indices entry is in the unique_rows
                    if upper_threshold is not None:
                        valid_values = torch.where(torch.logical_and(counts >= threshold, counts <= upper_threshold))[0]
                    else:
                        valid_values = torch.where(counts >= threshold)[0]
                    valid_value_dict = {value.item(): idx for idx, value in enumerate(valid_values)}
                    valid_indices_bool = torch.isin(indices, valid_values)
                    valid_indices = torch.where(valid_indices_bool)[0]
                    # relabel indices
                    indices[valid_indices] = torch.tensor([valid_value_dict[idx.item()] for idx in indices[valid_indices]], dtype=torch.int64)
                    num_weights = len(valid_values)
                start_time = time.time()
                for idx in range(len(graph_data)):
                    # if number of graphs is larger than 10000 print progress
                    if len(graph_data) > 10000 and idx % 1000 == 0:
                        print(f'Head {i+1}/{len(self.layer.layer_heads)} with property {key}: {idx}/{len(graph_data)} graphs processed ({(idx/len(graph_data))*100:.2f}%) time so far (in s): {time.time()-start_time:.2f}',
                                end='\r', flush=True)
                    # get the valid indices for the current graph
                    if threshold > 1 or do_invalid_indices_exist or upper_threshold is not None:
                        valid_indices_graph = torch.where(valid_indices_bool[property_subdict_slices[idx]:property_subdict_slices[idx+1]])[0] + property_subdict_slices[idx]
                    else:
                        valid_indices_graph = torch.arange(property_subdict_slices[idx], property_subdict_slices[idx+1], dtype=torch.int64)
                    # create new tensor where each row is the concatenation of head_id, property_subdict_row, and indices
                    new_weight_distribution = torch.zeros((len(valid_indices_graph), 4), dtype=torch.int64)
                    new_weight_distribution[:, 0] = i
                    new_weight_distribution[:, 1:3] = property_subdict[valid_indices_graph] - self.graph_data.slices['x'][idx] # check if subtracting is necessary
                    new_weight_distribution[:, 3] = indices[valid_indices_graph] + self.skips[-1]
                    if self.weight_distribution[idx] is None:
                        self.weight_distribution[idx] = new_weight_distribution.detach().clone()
                    else:
                        self.weight_distribution[idx] = torch.cat((self.weight_distribution[idx], new_weight_distribution), dim=0)



                self.skips.append(self.skips[-1] + num_weights)
                self.skips_description.append({'head:': i, 'property': key, 'weights': num_weights})
                self.skips_description_text.append(f"Head {i} Property {key} has {num_weights} different weights")


            self.weight_num.append(self.skips[-1])
            # TODO symmetric case

            if self.bias:
                # Determine the number of different learnable parameters in the bias vector
                self.bias_num.append(self.in_features * self.n_bias_labels[i])
                # Set the bias weights
                _, indices, counts = torch.unique(bias_labels, dim=0, return_inverse=True, return_counts=True, sorted=False)
                for idx in range(len(graph_data)):
                    for feature_id in range(self.in_features):
                        new_bias_distribution = torch.zeros((graph_data.num_nodes[idx].item(), 4), dtype=torch.int64)
                        new_bias_distribution[:, 0] = i
                        new_bias_distribution[:, 1] = torch.arange(graph_data.num_nodes[idx].item(), dtype=torch.int64) # alternative torch.arange(start=graph_data.slices['x'][idx], end=graph_data.slices['x'][idx+1], dtype=torch.int64)
                        new_bias_distribution[:, 2] = feature_id
                        new_bias_distribution[:, 3] = indices[graph_data.slices['x'][idx]:graph_data.slices['x'][idx+1]] + feature_id * self.n_bias_labels[i]
                        if self.bias_distribution[idx] is None:
                            self.bias_distribution[idx] = new_bias_distribution.detach().clone()
                        else:
                            self.bias_distribution[idx] = torch.cat((self.bias_distribution[idx], new_bias_distribution), dim=0)

        # Merge the weight distribution of all graphs (creating additionally slicing information)
        self.weight_distribution_slices = torch.tensor([0] + [len(w) for w in self.weight_distribution], dtype=torch.int64).cumsum(dim=0)
        self.weight_distribution = torch.cat([self.weight_distribution[i] for i in range(len(graph_data))], dim=0).to(self.device)
        if self.bias:
            # Merge the bias distribution of all graphs (creating additionally slicing information)
            self.bias_distribution_slices = torch.tensor([0] + [len(b) for b in self.bias_distribution], dtype=torch.int64).cumsum(dim=0)
            self.bias_distribution = torch.cat([self.bias_distribution[i] for i in range(len(graph_data))], dim=0).to(self.device)


        if self.bias:
            #self.bias_map = np.arange(total_bias_num, dtype=np.int64).reshape((self.n_bias_labels, self.input_feature_dimension))
            self.Param_b = self.init_weights(np.sum(self.bias_num), init_type='convolution_bias').to(self.device)
        self.Param_W = self.init_weights(np.sum(self.weight_num), init_type='convolution').to(self.device)


        # TODO add pruning
        # in case of pruning is turned on, save the original weights
        self.Param_W_original = None
        self.mask = None
        if 'prune' in self.para.run_config.config and self.para.run_config.config['prune']['enabled']:
            self.Param_W_original = self.Param_W.detach().clone()
            self.mask = torch.ones(self.Param_W.size())

        self.forward_step_time = 0

    def init_weights(self, num_weights:np.float64, init_type:Optional[str]=None) -> nn.Parameter:
        """
        Initializes the weights, i.e., learnable parameters of the module
        :param num_weights: number of weights
        :param init_type: type of the weight initialization determined in the config file (convolution, or convolution bias)
        :return: the initialized weights
        """
        weights = nn.Parameter(torch.zeros(num_weights, dtype=self.precision), requires_grad=True)
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
                    weights = nn.Parameter(new_weights, requires_grad=True)

                elif weight_initialization.get('type', None) == 'constant':
                    torch.nn.init.constant_(weights, weight_initialization.get('value', 0.01))
                elif weight_initialization.get('type', None) == 'lower_upper':
                    # calculate the range for the weights
                    lower, upper = -(1.0 / np.sqrt(num_weights)), (1.0 / np.sqrt(num_weights))
                    weights = nn.Parameter(lower + torch.randn(num_weights, dtype=self.precision) * (upper - lower))
                elif weight_initialization.get('type', None) == 'he':
                    std = np.sqrt(2.0 / num_weights)
                    weights = nn.Parameter(torch.randn(num_weights, dtype=self.precision) * std)

            else:
                raise ValueError(f"Weight initialization type {init_type} is not supported")
        else:
            torch.nn.init.constant_(weights, 0.01)
        return weights

    def set_weights(self, pos:int) -> None:
        """
        Sets the precomputed weights for the graph at position pos in the graph dataset to the matrix
        :param pos:
        :return:
        """
        input_size = self.graph_data.num_nodes[pos].item()
        self.current_W = torch.zeros((self.num_heads, input_size, input_size), dtype=self.precision).to(self.device)
        graph_weight_distribution = self.weight_distribution[self.weight_distribution_slices[pos]:self.weight_distribution_slices[pos+1]]
        if len(graph_weight_distribution) != 0:
            # get third column of the weight_distribution: the index of self.Param_W
            weight_indices = graph_weight_distribution[:, 3]
            matrix_indices = graph_weight_distribution[:, 0:3].T
            # set current_W by using the matrix_indices with the values of the Param_W at the indices of param_indices
            self.current_W[matrix_indices[0], matrix_indices[1], matrix_indices[2]] = torch.take(self.Param_W, weight_indices)
        return

    def set_bias(self, pos) -> None:
        """
        Sets the precomputed bias term for the graph at position pos in the graph dataset
        :param pos:
        :return:
        """
        input_size = self.graph_data.num_nodes[pos].item()
        self.current_B = torch.zeros((self.num_heads, input_size, self.in_features), dtype=self.precision).to(self.device)
        graph_bias_distribution = self.bias_distribution[self.bias_distribution_slices[pos]:self.bias_distribution_slices[pos+1]]
        param_indices = graph_bias_distribution[:, 3]
        matrix_indices = graph_bias_distribution[:, 0:3].T
        self.current_B[matrix_indices[0], matrix_indices[1], matrix_indices[2]] = torch.take(self.Param_b, param_indices)
        return

    def print_layer_info(self)->None:
        """
        Print the layer information
        :return:
        """
        print("Layer" + self.__class__.__name__)

    def print_weights(self):
        print("Weights of the Convolution layer")
        string = ""
        for x in self.Param_W:
            string += str(x.data)
        print(string)

    def print_bias(self):
        print("Bias of the Convolution layer")
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
        # get pos from kwargs
        pos = kwargs.get('pos', 0)
        # automatically modifiy input if node_representation is 3-dimensional, i.e., (N, F) -> (1, N, F)
        if node_representation.dim() == 3:
            if node_representation.size(0) != 1:
                raise ValueError("Input tensor x must have size 1 in the first dimension for InvariantBasedMessagePassingLayer")
            node_representation = node_representation.squeeze(0)
        begin = time.time()
        # set the weights, i.e., sets self.current_W to (C, N, N) where C is the number of channels and N is the number of nodes in graph at position pos of the dataset
        self.set_weights(pos)

        self.forward_step_time += time.time() - begin
        if self.para.run_config.config.get('degree_matrix', False):
            node_representation = self.in_edges[pos]*torch.einsum('cij,jk->cik', torch.diag(self.D[pos]) @ self.current_W @ torch.diag(self.D[pos]), node_representation)
        elif self.para.run_config.config.get('use_in_degrees', False):
            node_representation = self.in_edges[pos]*torch.einsum('cij,jk->cik', self.current_W, node_representation)
        else:
            node_representation = torch.einsum('cij,jk->cik', self.current_W, node_representation)
        if self.bias:
            self.set_bias(pos)
            node_representation = node_representation + self.current_B
        node_representation = self.activation(node_representation)
        return node_representation


    def get_weights(self):
        # return the weights as a numpy array
        return np.array(self.Param_W.detach().cpu())

    def get_graph_weights(self, graph_id):
        return self.weight_distribution[self.weight_distribution_slices[graph_id]:self.weight_distribution_slices[graph_id + 1]]

    def get_bias(self):
        if self.bias:
            return np.array(self.Param_b.detach().cpu())
        else:
            return None

    def draw(self, ax, graph_id, graph_drawing: Tuple[GraphDrawing, GraphDrawing], head=0, filter_weights=None, with_graph=True, graph_only=False, draw_bias_labels=False,pos_path:str=''):
        # create graph
        graph = self.graph_data.create_nx_graph(graph_id, directed=False)
        pos = dict()
        # if pos_path is given and the file exists, load the positions from the file
        if pos_path != '' and Path(pos_path).is_file():
            pos = dict()
            with open(pos_path, 'r') as f:
                # iterate over the lines of the file
                for line in f:
                    # split the line by whitespaces
                    line = line.split()
                    # get the node id and the x and y position
                    pos[int(line[0])] = (float(line[1]), float(line[2]))
        if with_graph or graph_only:
            # draw the graph
            if not Path(pos_path).is_file():
                # if graph is circular use the circular layout
                if graph_drawing[0].draw_type == 'circle':
                    # root node is the one with label 0
                    root_node = None
                    for node in graph.nodes():
                        if self.graph_data.node_labels['primary'].node_labels[self.graph_data.slices['x'][graph_id] + node] == 0:
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
                    pos = nx.bfs_layout(graph, 0)
                else:
                    pos = nx.nx_pydot.graphviz_layout(graph)

                # keys to ints
                pos = {int(k): v for k, v in pos.items()}
                # if pos_path is given, save the positions to the file
                if pos_path != '':
                    with open(pos_path, 'w') as f:
                        for key, value in pos.items():
                            f.write(f"{key} {value[0]} {value[1]}\n")
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


                draw_node_labels = None
                num_unique_node_labels = 0
                if isinstance(self.graph_data.node_labels['primary'], NodeLabels):
                    draw_node_labels = self.graph_data.node_labels['primary'].node_labels
                    num_unique_node_labels = self.graph_data.node_labels['primary'].num_unique_node_labels
                elif isinstance(self.graph_data.node_labels['primary'], torch.Tensor):
                    draw_node_labels = self.graph_data.node_labels['primary']
                    num_unique_node_labels = torch.unique(draw_node_labels).size(0)
                else:
                    raise ValueError("Node labels are not of type NodeLabels or torch.Tensor")

                graph_node_labels = None
                if draw_bias_labels:
                    graph_node_labels = self.graph_data.node_labels[self.bias_label_descriptions[head]].node_labels[self.graph_data.slices['x'][graph_id]:self.graph_data.slices['x'][graph_id + 1]]
                    num_unique_node_labels = self.graph_data.node_labels[self.bias_label_descriptions[head]].num_unique_node_labels
                else:
                    if isinstance(self.graph_data.node_labels['primary'], NodeLabels):
                        graph_node_labels = self.graph_data.node_labels['primary'].node_labels[self.graph_data.slices['x'][graph_id]:self.graph_data.slices['x'][graph_id+1]]
                    elif isinstance(self.graph_data.node_labels['primary'], torch.Tensor):
                        graph_node_labels = self.graph_data.node_labels['primary'][self.graph_data.slices['x'][graph_id]:self.graph_data.slices['x'][graph_id+1]]
                    else:
                        raise ValueError("Node labels are not of type NodeLabels or torch.Tensor")

                cmap = graph_drawing[0].colormap
                norm = matplotlib.colors.Normalize(vmin=0, vmax=num_unique_node_labels)
                node_colors = [cmap(norm(graph_node_labels[node])) for node
                               in graph.nodes()]
                nx.draw_networkx_nodes(graph, pos=pos, ax=ax, node_color=node_colors,
                                       node_size=graph_drawing[0].node_size)
                #nx.draw_networkx_labels(graph, pos=pos, ax=ax, labels={node: node for node in graph.nodes()}, font_size=8)
                return
            nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=graph_drawing[1].edge_color, width=graph_drawing[1].edge_width, alpha=graph_drawing[1].edge_alpha*0.5)

        all_weights = np.array(self.get_weights())
        bias = self.get_bias()
        weight_distribution = self.get_graph_weights(graph_id)
        param_indices = np.array(weight_distribution[:, 3])
        matrix_indices = np.array(weight_distribution[:, 0:3])
        graph_weights = all_weights[param_indices]

        # sort weights
        if filter_weights is not None and len(graph_weights) != 0:
            sorted_weights = np.sort(np.array(list(set(graph_weights))))
            if filter_weights.get('percentage', None) is not None:
                percentage = filter_weights['percentage']
                lower_bound_weight = sorted_weights[int(len(sorted_weights) * percentage) - 1]
                upper_bound_weight = sorted_weights[int(len(sorted_weights) * (1 - percentage))]
            elif filter_weights.get('absolute', None) is not None:
                absolute = filter_weights['absolute']
                absolute = min(absolute, len(sorted_weights))
                lower_bound_weight = sorted_weights[absolute - 1]
                upper_bound_weight = sorted_weights[-absolute]
            # set all weights smaller than the lower bound and larger than the upper bound to zero
            upper_weights = np.where(graph_weights >= upper_bound_weight, graph_weights, 0)
            lower_weights = np.where(graph_weights <= lower_bound_weight, graph_weights, 0)

            weights = upper_weights + lower_weights
        else:
            weights = np.asarray(graph_weights)

        # if graph weights is empty, return
        if len(weights) == 0:
            weight_min = 0
            weight_max = 0
        else:
            weight_min = np.min(graph_weights)
            weight_max = np.max(graph_weights)
        weight_max_abs = max(abs(weight_min), abs(weight_max))
        # use seismic colormap with maximum and minimum values from the weight matrix
        cmap = graph_drawing[1].colormap
        # normalize item number values to colormap
        normed_weight = (graph_weights + (-weight_min)) / (weight_max - weight_min)
        weight_colors = cmap(normed_weight)

        if self.bias:
            bias_min = np.min(bias)
            bias_max = np.max(bias)
            bias_max_abs = max(abs(bias_min), abs(bias_max))
            normed_bias = (bias + (-bias_min)) / (bias_max - bias_min)
            bias_colors = cmap(normed_bias)




        # draw the graph
        # if graph is circular use the circular layout
        # draw the graph
        if pos == {}:
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
            # if pos_path is given, save the positions to the file
            if pos_path != '':
                with open(pos_path, 'w') as f:
                    for key, value in pos.items():
                        f.write(f"{key} {value[0]} {value[1]}\n")
        # graph to digraph with
        digraph = nx.DiGraph()
        for node in graph.nodes():
            digraph.add_node(node)


        if self.bias:
            node_colors = []
            node_sizes = []
            for node in digraph.nodes():
                node_label = self.graph_data.node_labels[self.bias_label_descriptions[head]].node_labels[self.graph_data.slices['x'][graph_id]:self.graph_data.slices['x'][graph_id+1]][node]
                node_colors.append(bias_colors[node_label])
                node_sizes.append(graph_drawing[1].node_size * abs(bias[node_label]) / bias_max_abs)

            nx.draw_networkx_nodes(digraph, pos=pos, ax=ax, node_color=node_colors, node_size=node_sizes)

        edge_widths = []
        for weight_id, entry in enumerate(weight_distribution):
            c = entry[0]
            if c == head:
                i = entry[1]
                j = entry[2]
                if weights[weight_id] != 0:
                    # add edge with weight as data
                    digraph.add_edge(i.item(), j.item(), weight=weight_id)
        curved_edges = [edge for edge in digraph.edges(data=True)]
        curved_edges_colors = []

        for edge in curved_edges:
            curved_edges_colors.append(weight_colors[edge[2]['weight']])
            edge_widths.append(graph_drawing[1].weight_edge_width * abs(weights[edge[2]['weight']]) / weight_max_abs)
        arc_rad = 0.25
        nx.draw_networkx_edges(digraph, pos, ax=ax, edgelist=curved_edges, edge_color=curved_edges_colors,
                               width=edge_widths,
                               connectionstyle=f'arc3, rad = {arc_rad}', arrows=True, arrowsize=graph_drawing[1].arrow_size, node_size=graph_drawing[1].node_size)
