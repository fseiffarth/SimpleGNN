import time
from typing import Tuple

import matplotlib
import networkx as nx
import numpy as np
import torch
from torch import nn

from simplegnn.datasets.graph_dataset import GraphDataset
from simplegnn.datasets.utils.graph_drawing import GraphDrawing, resolve_positions
from simplegnn.framework.utils.parameters import Parameters
from simplegnn.models.ShareGNN.layers.inv_based import InvariantBasedLayer
from simplegnn.models.ShareGNN.utils import Layer, is_batched_pos, range_gather


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
        - **out** is the graph embedding. By default (``flatten: True``) the H heads are folded into the
          feature dimension, giving shape (1, H*F) per graph — out_features = H*F, out_channels = 1.
          With ``flatten: False`` the head axis is kept, giving shape (H, 1, F) per graph
          (out_features = F, out_channels = H), which lets a channel_wise or factorized readout
          exploit the head/feature structure instead of flattening it away.
    """
    def __init__(self, parameters:Parameters, layer: Layer, graph_data: GraphDataset):
        layer.layer_dict['name'] = "Invariant Based Aggregation Layer"
        super(InvariantBasedAggregationLayer, self).__init__(parameters, layer, graph_data)

        self.flatten = layer.layer_dict.get('flatten', True)
        if self.flatten:
            self.out_features = self.in_features * self.num_heads
            # The output is a flat (H*F) vector per graph: all heads are folded
            # into out_features, so no separate channel dimension remains.
            self.out_channels = 1
        else:
            # Keep the (H, F) structure of the graph embedding so that a
            # downstream readout can exploit it (channel_wise / factorized
            # linear) instead of learning a free weight per (head, feature).
            self.out_features = self.in_features
            self.out_channels = self.num_heads

        self.n_node_labels = [] # number of node labels per head
        self.node_label_descriptions = [] # node label descriptions per head
        self.n_heads_per_label = [] # number of heads per node label description
        # bias per head
        self.bias_list = [head.bias for head in layer.layer_heads]
        # is there any bias
        self.bias = any(self.bias_list)
        for head_id, head in enumerate(layer.layer_heads):
            self.node_label_descriptions.append(layer.get_source_string(head_id))
            self.n_node_labels.append(self.graph_data.node_labels[self.node_label_descriptions[head_id]].num_unique_node_labels)
            self.n_heads_per_label.append(head.num)

        self.weight_num = np.sum([self.n_node_labels[i] * self.n_heads_per_label[i] for i in range(len(layer.layer_heads))])
        # FACTORED storage (see specs/10-invariant-layer-memory-factorization.md):
        # the old (total_nodes, num_heads) int64 matrix repeated each config's
        # unique-inverse across its replica columns, shifted by a constant
        # (6.5 GB on ZINC-full). Keep one int32 inverse per head-config plus a
        # per-column offset vector; the (N, H) index matrix is rebuilt on the
        # fly in _weight_index_matrix().
        col_offset = torch.zeros(self.num_heads, dtype=torch.int64)
        self._agg_col_ranges = []  # per head-config: (column_base, num_replicas)
        for head_id, head in enumerate(layer.layer_heads):
            node_labels = self.graph_data.node_labels[self.node_label_descriptions[head_id]].node_labels
            _, indices, counts = torch.unique(node_labels, dim=0, return_inverse=True, return_counts=True, sorted=False)
            column_base = int(np.sum(self.n_heads_per_label[:head_id], dtype=int))
            weight_base = int(np.sum([self.n_node_labels[i] * self.n_heads_per_label[i] for i in range(head_id)], dtype=int))
            # Non-persistent buffer: moved by net.to(device), kept out of state_dict
            self.register_buffer(f'_agg_idx_{head_id}', indices.to(torch.int32), persistent=False)
            self._agg_col_ranges.append((column_base, head.num))
            for h_num in range(head.num):
                col_offset[column_base + h_num] = h_num * self.n_node_labels[head_id] + weight_base
        self.register_buffer('_agg_col_offset', col_offset, persistent=False)
        self.register_buffer('_agg_x_slices', self.graph_data.slices['x'].clone(), persistent=False)
        # per-graph offsets as plain ints (Python slicing, no device syncs)
        self._agg_slices = [int(x) for x in self.graph_data.slices['x']]



        self.Param_W = self.init_weights(self.weight_num, init_type='aggregation')


        if self.bias:
            # No .to(self.device): that would demote the Parameter to a plain
            # tensor on CUDA and hide it from the optimizer (net.to moves it).
            self.Param_b = self.init_weights(shape=(self.num_heads, self.in_features), init_type='aggregation_bias')
        self.forward_step_time = 0
        self.profile_layers = self.para.run_config.config.get('profile_layers', False)



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

    def _weight_index_matrix(self, node_gather) -> torch.Tensor:
        """
        (N_sel, num_heads) Param_W indices for the selected nodes.

        node_gather is either a slice (contiguous per-graph node range) or an
        int64 index tensor (concatenated batch node ranges). Equivalent to
        slicing the old dense (total_nodes, num_heads) matrix.
        """
        first = getattr(self, '_agg_idx_0')[node_gather]
        idx = torch.empty((first.shape[0], self.num_heads), dtype=torch.int64, device=first.device)
        for head_id, (column_base, num) in enumerate(self._agg_col_ranges):
            cfg_idx = first if head_id == 0 else getattr(self, f'_agg_idx_{head_id}')[node_gather]
            cols = slice(column_base, column_base + num)
            idx[:, cols] = cfg_idx.long()[:, None] + self._agg_col_offset[cols][None, :]
        return idx

    def set_weights(self, pos):
        node_range = slice(self._agg_slices[pos], self._agg_slices[pos + 1])
        self.current_W = self.Param_W[self._weight_index_matrix(node_range)]
        # divide the weights by the number of nodes in the graph
        #self.current_W = self.current_W / input_size

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
        if is_batched_pos(pos):
            return self._forward_batched(node_representation, pos)
        begin = time.time() if self.profile_layers else None
        self.set_weights(pos)
        # (H, N) @ (N, F) -> (H, F); same as einsum('no,nf->of') with less dispatch overhead
        node_representation = torch.matmul(self.current_W.t(), node_representation)
        if self.bias:
            node_representation = node_representation + self.Param_b
        if self.flatten:
            node_representation = node_representation.flatten().unsqueeze(0)
        else:
            # (H, F) -> (H, 1, F): the framework's (C, N, F) convention with a
            # singleton node dimension, so channel-wise layers can follow
            node_representation = node_representation.unsqueeze(1)
        node_representation = self.activation(node_representation)
        if self.profile_layers:
            self.forward_step_time += time.time() - begin
        return node_representation

    def _forward_batched(self, node_representation: torch.Tensor, positions) -> torch.Tensor:
        """
        Batched aggregation over several graphs at once.

        ``node_representation`` is the row-wise concatenation of the batch
        graphs' node representations in the order given by ``positions``
        (duplicates allowed). Computes out[b, h, f] = sum_{i in graph b}
        W[i, h] * x[i, f] and returns shape (B, H * F) — row b equals the
        per-graph forward on positions[b]. With ``flatten: False`` the heads
        are kept as a separate axis and the shape is (B, H, F).

        Implemented as one padded batched matmul (B, H, N_max) @ (B, N_max, F):
        padded rows stay zero and contribute nothing to the sum. This avoids
        materializing the (total_nodes, H, F) contributions tensor of the
        segment-reduction formulation (~1.3 GB per 512-graph ZINC eval chunk
        and ~7x slower). The segment reduction is kept as a fallback for
        batches with very skewed graph sizes, where padding would dominate.
        """
        begin = time.time() if self.profile_layers else None
        device = node_representation.device
        dtype = node_representation.dtype
        positions = [int(p) for p in positions]
        counts_list = [self._agg_slices[p + 1] - self._agg_slices[p] for p in positions]
        batch_size = len(positions)
        total_nodes = node_representation.shape[0]
        n_max = max(counts_list, default=0)
        num_features = node_representation.shape[-1]
        counts = torch.as_tensor(counts_list, dtype=torch.int64, device=device)
        node_gather, _ = range_gather(self._agg_x_slices,
                                      torch.as_tensor(positions, dtype=torch.int64, device=self._agg_x_slices.device))
        node_weights = self.Param_W[self._weight_index_matrix(node_gather)]  # (total_nodes, H)
        if batch_size * n_max <= 4 * total_nodes:
            graph_idx = torch.repeat_interleave(
                torch.arange(batch_size, dtype=torch.int64, device=device), counts)
            offsets = torch.cumsum(counts, dim=0) - counts
            node_idx = torch.arange(total_nodes, dtype=torch.int64, device=device) - torch.repeat_interleave(offsets, counts)
            weights_padded = torch.zeros((batch_size, n_max, self.num_heads), dtype=dtype, device=device)
            features_padded = torch.zeros((batch_size, n_max, num_features), dtype=dtype, device=device)
            weights_padded[graph_idx, node_idx] = node_weights
            features_padded[graph_idx, node_idx] = node_representation
            out = torch.bmm(weights_padded.transpose(1, 2), features_padded)  # (B, H, F)
        else:
            # segment reduction: robust when one huge graph would force
            # excessive padding for the whole batch
            contributions = node_weights.unsqueeze(-1) * node_representation.unsqueeze(1)  # (total_nodes, H, F)
            graph_idx = torch.repeat_interleave(
                torch.arange(batch_size, dtype=torch.int64, device=device), counts)
            out = torch.zeros((batch_size, self.num_heads, num_features), dtype=dtype, device=device)
            out.index_add_(0, graph_idx, contributions)
        if self.bias:
            out = out + self.Param_b
        if self.flatten:
            out = out.flatten(start_dim=1)
        out = self.activation(out)
        if self.profile_layers:
            self.forward_step_time += time.time() - begin
        return out

    def get_weights(self):
        return self.Param_W.detach().cpu().numpy()

    def get_bias(self):
        if self.bias:
            return self.Param_b.detach().cpu().numpy()
        return None

    def draw(self, ax, graph_id, graph_drawing: Tuple[GraphDrawing, GraphDrawing], head=0, out_dimension=0, with_graph=True, graph_only=False, pos_path:str=''):
        """Draw one graph with its per-node pooling weights of the given head.

        out_dimension is kept for backwards compatibility but ignored: the
        factored weight storage shares one parameter per (node label, head),
        there is no output-dimension axis anymore.
        """
        graph = self.graph_data.create_nx_graph(graph_id, directed=False)
        labels = self.graph_data.node_labels['primary']
        node_offset = int(self._agg_slices[graph_id])
        node_end = int(self._agg_slices[graph_id + 1])

        # the circle layout starts its walk at the node with primary label 0
        root_node = None
        if graph_drawing[0].draw_type == 'circle':
            for node in graph.nodes():
                if labels.node_labels[node_offset + node] == 0:
                    root_node = node
                    break
        pos = resolve_positions(graph, graph_drawing[0].draw_type, pos_path=pos_path, root_node=root_node)

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
            cmap = graph_drawing[0].colormap
            norm = matplotlib.colors.Normalize(vmin=0, vmax=labels.num_unique_node_labels)
            node_colors = [cmap(norm(labels.node_labels[node_offset + node])) for node in graph.nodes()]
            nx.draw_networkx_nodes(graph, pos=pos, ax=ax, node_color=node_colors,
                                   node_size=graph_drawing[0].node_size)
            return
        if with_graph:
            nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=graph_drawing[1].edge_color,
                                   width=graph_drawing[1].edge_width, alpha=graph_drawing[1].edge_alpha*0.5)

        # per-node Param_W index of the drawn head
        all_weights = self.get_weights()
        node_range = slice(node_offset, node_end)
        param_indices = self._weight_index_matrix(node_range)[:, head].cpu().numpy()
        node_weights = all_weights[param_indices]

        weight_min = float(np.min(node_weights)) if node_weights.size else 0.0
        weight_max = float(np.max(node_weights)) if node_weights.size else 0.0
        weight_max_abs = max(abs(weight_min), abs(weight_max))
        weight_range = weight_max - weight_min
        if weight_range > 0:
            normed_weight = (node_weights - weight_min) / weight_range
        else:
            normed_weight = np.full_like(node_weights, 0.5)
        cmap = graph_drawing[1].colormap
        weight_colors = cmap(normed_weight)

        node_list = list(graph.nodes())
        node_colors = weight_colors[node_list]
        if weight_max_abs > 0:
            node_sizes = graph_drawing[1].node_size * np.abs(node_weights[node_list]) / weight_max_abs
        else:
            node_sizes = np.full(len(node_list), graph_drawing[1].node_size)
        nx.draw_networkx_nodes(graph, pos=pos, ax=ax, nodelist=node_list, node_color=node_colors,
                               node_size=list(node_sizes))
