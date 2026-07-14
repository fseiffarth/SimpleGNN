import time
from pathlib import Path
from typing import List, Optional, Tuple
import hashlib
import json

import matplotlib
import yaml
from datetime import datetime

import networkx as nx
import numpy as np
import torch
from torch import nn
import sys

from simplegnn.datasets.graph_dataset import GraphDataset
from simplegnn.datasets.utils.NodeLabels import NodeLabels
from simplegnn.datasets.utils.graph_drawing import GraphDrawing
from simplegnn.framework.utils.parameters import Parameters
from simplegnn.models.ShareGNN.layers.inv_based import InvariantBasedLayer
from simplegnn.models.ShareGNN.utils import Layer, is_batched_pos


class InvariantBasedMessagePassingLayer(InvariantBasedLayer):
    """
    ShareGNN message passing layer using invariant-based label and property aggregation.

    This layer implements the core ShareGNN architecture: message passing conditioned
    on node label pairs and pairwise properties (e.g., shortest path distance). It
    computes multi-head aggregations where each head focuses on specific label
    combinations and property values, enabling fine-grained structural feature extraction.

    Parameters
    ----------
    parameters : Parameters
        Experiment configuration containing paths, precision, and print settings.
    layer : Layer
        Layer configuration specifying:
        - layer_heads: List of head configurations (labels, properties, num heads)
        - layer_id: Position of this layer in the network
        - in_features, out_features: Dimensions
    graph_data : GraphDataset
        Graph dataset with precomputed:
        - node_labels: Dictionary of NodeLabels objects (source, target, bias labels)
        - properties: Dictionary of Properties objects (pairwise distances/features)

    Attributes
    ----------
    out_features : int
        Total output dimension (in_features × num_heads).
    num_heads : int
        Total number of heads across all head configurations.
    n_heads_per_label : list of int
        Number of heads for each label configuration.
    source_label_descriptions : list of str
        Label names used for source nodes in each head.
    target_label_descriptions : list of str
        Label names used for target nodes in each head.
    bias_label_descriptions : list of str
        Label names used for bias terms in each head.
    property_descriptions : list of str
        Property names (e.g., 'distance_0_3_6') for each head.
    n_source_labels : list of int
        Number of unique source labels per head.
    n_target_labels : list of int
        Number of unique target labels per head.
    n_bias_labels : list of int
        Number of unique bias labels per head.
    n_properties : list of int
        Number of property values per head.
    weight_num : list of int
        Number of weights allocated to each (source, target, property) combination.
    weight_offset : list of int
        Cumulative offsets into the weight parameter vector.
    bias : bool
        Whether any head uses bias terms.
    bias_list : list of bool
        Per-head bias flags.

    Notes
    -----
    **ShareGNN Message Passing Algorithm:**

    For each head h and each property value p (e.g., distance = 3):
    1. Identify node pairs (i, j) where:
        - Source label of node i matches head configuration
        - Target label of node j matches head configuration
        - Property(i, j) == p
    2. Look up weight w[source_label_i, target_label_j, p]
    3. Aggregate: out_i_h += w × input_j for all valid j
    4. Add bias b[bias_label_i] if enabled
    5. Stack outputs across heads: (H, N, F)

    **Weight Distribution:**

    Weights are distributed based on label-property combinations that actually
    occur in the dataset. The massive __init__() method:
    1. Scans all graphs to find valid (source, target, property) tuples
    2. Assigns weight indices to each unique combination
    3. Caches the index mappings for efficient forward passes
    4. Allocates bias parameters per unique bias label

    **Caching Strategy:**

    Index computations are expensive (O(|E| × |labels|²) worst case), so results
    are cached to disk with MD5-hashed keys. See get_cache_path(), _load_cached_indices(),
    and _save_cached_indices() for caching implementation.

    **Tensor Shapes:**
    - Input: (N, F) or (1, N, F) where N = nodes, F = features
    - Output: (H, N, F) where H = num_heads
    - Weights: (W,) where W = total weight count across all heads
    - Bias: (B,) where B = sum of unique bias labels across heads

    See Also
    --------
    InvariantBasedLayer : Base class
    models.ShareGNN.preprocessing.preprocessing : Generates labels and properties
    datasets.utils.node_labeling : Node labeling strategies
    datasets.utils.edge_labeling : Property computation

    Examples
    --------
    >>> # Typical configuration in YAML
    >>> layer_config = {
    ...     'layer_type': 'inv_based_message_passing',
    ...     'heads': [
    ...         {
    ...             'num': 4,
    ...             'source_labels': {'label_type': 'wl', 'depth': 3},
    ...             'target_labels': {'label_type': 'wl', 'depth': 3},
    ...             'property': {'type': 'distance', 'values': [0, 3, 6]}
    ...         }
    ...     ]
    ... }
    """



    def __init__(self, parameters: Parameters, layer: Layer, graph_data: GraphDataset):
        """
        Initialize the invariant-based message passing layer.

        Performs extensive preprocessing to determine weight distribution:
        1. Extracts label and property descriptions from layer configuration
        2. Scans all graphs to find valid (source_label, target_label, property) tuples
        3. Assigns weight indices to each unique combination (caching for efficiency)
        4. Allocates bias parameters per unique bias label
        5. Initializes weight and bias parameter tensors

        Parameters
        ----------
        parameters : Parameters
            Experiment parameters (paths, precision, device, printing options).
        layer : Layer
            Layer configuration with head specifications.
        graph_data : GraphDataset
            Dataset with precomputed node_labels and properties dictionaries.

        Notes
        -----
        **Initialization Phases:**

        **Phase 1: Extract Label and Property Metadata (lines 59-71)**
        - For each head: extract source/target/bias label names
        - Count unique labels (n_source_labels, n_target_labels, n_bias_labels)
        - Extract property description and count property values
        - Store number of heads per label configuration

        **Phase 2: Weight Distribution (lines 74-211)**
        - For each head and property value:
            a. Load or compute valid (source, target) label pairs
            b. Assign weight index to each pair
            c. Create index tensors for fast forward pass lookups
            d. Cache indices to disk for reuse
        - Accumulate weight counts and offsets
        - Store distribution chunks for each graph

        **Phase 3: Bias Setup (lines 213-231)**
        - If any head uses bias:
            a. Count unique bias labels across all graphs
            b. Assign bias index to each unique bias label
            c. Store bias offsets per graph
            d. Cache bias indices

        **Phase 4: Parameter Initialization (lines 233-260)**
        - Allocate weight tensor: size = total weight count
        - Allocate bias tensor: size = total unique bias labels (if bias enabled)
        - Initialize using configured strategy (xavier, kaiming, normal, uniform, zeros)
        - Store distribution and offset tensors for forward pass

        **Caching:**
        Index computations are expensive (can take minutes for large datasets).
        Results are cached with MD5-hashed keys based on:
        - Dataset, labels, properties, filter thresholds
        - Cache hit: instant loading
        - Cache miss: compute, save for next time

        Raises
        ------
        ValueError
            If label or property descriptions are missing from graph_data.
        FileNotFoundError
            If required label or property files are not found.

        See Also
        --------
        get_cache_path : Generates cache file paths
        _load_cached_indices : Loads cached index tensors
        _save_cached_indices : Saves computed indices
        init_weights : Weight initialization strategies
        """
        layer.layer_dict['name'] = "Invariant Based Message Passing Layer"
        super(InvariantBasedMessagePassingLayer, self).__init__(parameters, layer, graph_data)

        self.out_features = self.in_features * self.num_heads
        self.n_heads_per_label = [] # number of heads per node label description

        for h_id, head in enumerate(layer.layer_heads):
            self.source_label_descriptions.append(layer.get_source_string(h_id))
            self.n_source_labels.append(graph_data.node_labels[self.source_label_descriptions[h_id]].num_unique_node_labels)
            self.target_label_descriptions.append(layer.get_target_string(h_id))
            self.n_target_labels.append(graph_data.node_labels[self.target_label_descriptions[h_id]].num_unique_node_labels)
            self.bias_label_descriptions.append(layer.get_bias_string(h_id))
            self.n_bias_labels.append(graph_data.node_labels[self.bias_label_descriptions[h_id]].num_unique_node_labels)
            self.property_descriptions.append(head.property_dict.get_property_string())
            self.n_properties.append(graph_data.properties[self.property_descriptions[h_id]].num_properties[(self.layer_id, h_id)])
            self.n_heads_per_label.append(head.num)

        self.bias_list = [head.bias for head in layer.layer_heads]
        self.bias = any(self.bias_list)  # check if bias is used


        # Determine the number of weights and biases
        # There are two cases asymmetric and symmetric, asymmetric is the default, TODO add symmetric case
        self.weight_num = []
        self.weight_offset = [0]
        self.b_head_offset = 0
        self.weight_offset_description = [None]
        self.weight_offset_description_text = [None]

        # Coarse per-layer cache: the merged distributions are deterministic
        # given the layer configuration + dataset and are fold/run-invariant,
        # yet they used to be rebuilt for every fold x run x grid-config
        # combination. On a hit the whole head/property assembly is skipped.
        layer_cache_path = self._get_layer_cache_path()
        if not self._load_layer_distribution_cache(layer_cache_path):
            self._build_distributions(graph_data)
            self._save_layer_distribution_cache(layer_cache_path)

        self._finalize_initialization()
    def _build_distributions(self, graph_data: GraphDataset) -> None:
        """
        Build the merged weight/bias distribution tensors for all heads.

        Vectorized over graphs: the rows for all graphs are built in one pass
        per (head, property value), and the historical per-graph append order
        (head -> property value -> head replica) is reproduced bit for bit by a
        single stable sort over the row-wise graph ids at merge time.
        """
        num_graphs = len(graph_data)
        x_slices = self.graph_data.slices['x']
        num_nodes_per_graph = x_slices.diff()
        total_num_nodes = int(x_slices[-1])

        weight_chunks = []           # (K, 4) row blocks, all graphs at once
        weight_chunk_graph_ids = []  # (K,) graph id of every row in the block
        bias_chunks = []
        bias_chunk_graph_ids = []
        bias_unique_cache = {}       # bias label description -> unique inverse indices

        # Iterate over all heads in the layer
        for head_id, head in enumerate(self.layer.layer_heads):
            head_weight_num = []  # number of weights for the current head, used for debugging
            # get all the valid property values for the head (e.g., the distances 0, 3, 6)
            valid_property_values = self.graph_data.properties[self.property_descriptions[head_id]].valid_values[(self.layer_id, head_id)]
            # apply the head and tail labels to the subdict
            source_labels = self.graph_data.node_labels[self.source_label_descriptions[head_id]].node_labels
            target_labels = self.graph_data.node_labels[self.target_label_descriptions[head_id]].node_labels
            bias_labels = self.graph_data.node_labels[self.bias_label_descriptions[head_id]].node_labels
            current_head_id = 0
            for h_i in range(head_id):
                current_head_id += self.n_heads_per_label[h_i]

            for property_key in valid_property_values:
                #print(f'Initialize head {i+1}/{len(self.layer.layer_heads)} with property {property_key}')
                property_subdict = self.graph_data.properties[self.property_descriptions[head_id]].properties[property_key]
                property_subdict_slices = self.graph_data.properties[self.property_descriptions[head_id]].properties_slices[property_key]

                # OPTIMIZATION: Build labeled_subdict directly without clone (5-10% speedup)
                labeled_subdict = torch.stack([
                    source_labels[property_subdict[:, 0]],
                    target_labels[property_subdict[:, 1]]
                ], dim=1)

                # Initialize variables before try-except to avoid scoping issues
                do_invalid_indices_exist = False
                threshold = self.para.run_config.config.get('rule_occurrence_threshold', 1)
                upper_threshold = self.para.run_config.config.get('rule_occurrence_upper_threshold', None)

                cached_path = self.get_cache_path(head, property_key)

                try:
                    indices, counts, do_invalid_indices_exist = self._load_cached_indices(cached_path, head, property_key)
                    print(f"✓ Cache hit: head source label {self.source_label_descriptions[head_id]}, target label {self.target_label_descriptions[head_id]} with property {self.property_descriptions[head_id]} key {property_key} loaded from cache at {cached_path.name}")

                except (FileNotFoundError, Exception) as e:
                    print(f"⊗ Cache miss: head source label {self.source_label_descriptions[head_id]}, target label {self.target_label_descriptions[head_id]} with property {self.property_descriptions[head_id]} key {property_key} - computing indices and counts ({str(e)})")

                    # OPTIMIZATION: Handle invalid indices with masking (2-5% speedup)
                    invalid_mask = (labeled_subdict[:, 0] == -1) | (labeled_subdict[:, 1] == -1)
                    do_invalid_indices_exist = invalid_mask.any().item()

                    if do_invalid_indices_exist:
                        valid_mask = ~invalid_mask
                        max_first = labeled_subdict[valid_mask, 0].max().item() + 1 if valid_mask.any() else 0
                        max_second = labeled_subdict[valid_mask, 1].max().item() + 1 if valid_mask.any() else 0
                        labeled_subdict[invalid_mask, 0] = max_first
                        labeled_subdict[invalid_mask, 1] = max_second
                    else:
                        max_first = labeled_subdict[:, 0].max().item() + 1
                        max_second = labeled_subdict[:, 1].max().item() + 1

                    # OPTIMIZATION: Encode 2D rows as 1D scalars (10-50x speedup on torch.unique)
                    # For bounded integer labels, encode (a, b) as a*K + b where K > max(b)
                    # This converts 2D unique (slow, O(n²) row comparisons) to 1D unique (fast, O(n log n))
                    max_label = max(max_first, max_second) + 1
                    encoded_labels = labeled_subdict[:, 0] * max_label + labeled_subdict[:, 1]

                    # Fast 1D unique instead of slow 2D unique
                    _, indices, counts = torch.unique(encoded_labels, return_inverse=True, return_counts=True, sorted=False)
                    if do_invalid_indices_exist:
                        counts[-1] = 0

                    self._save_cached_indices(cached_path, head, property_key, indices, counts, do_invalid_indices_exist)

                # Threshold filtering (now do_invalid_indices_exist is always defined)
                num_weights = len(counts)
                if do_invalid_indices_exist:
                    num_weights -= 1
                if threshold > 1 or do_invalid_indices_exist or upper_threshold is not None:
                    # get a bool tensor from indices where the entry is true if the indices entry is in the unique_rows
                    if upper_threshold is not None:
                        valid_values = torch.where(torch.logical_and(counts >= threshold, counts <= upper_threshold))[0]
                    else:
                        valid_values = torch.where(counts >= threshold)[0]
                    valid_indices_bool = torch.isin(indices, valid_values)
                    # relabel indices using a vectorized mapping
                    mapping = torch.full((counts.size(0),), -1, dtype=torch.int64, device=indices.device)
                    mapping[valid_values] = torch.arange(valid_values.numel(), device=indices.device, dtype=torch.int64)
                    indices = mapping[indices]
                    num_weights = valid_values.numel()
                for n in range(self.n_heads_per_label[head_id]):
                    head_weight_num.append(num_weights)

                # VECTORIZED: build the rows for ALL graphs in one pass (the
                # old per-graph loop only partitioned rows that are already
                # stored grouped by graph)
                property_subdict_slices = torch.as_tensor(property_subdict_slices, dtype=torch.int64)
                if threshold > 1 or do_invalid_indices_exist or upper_threshold is not None:
                    kept_rows = torch.nonzero(valid_indices_bool, as_tuple=False).squeeze(1)
                else:
                    kept_rows = torch.arange(property_subdict.shape[0], dtype=torch.int64)
                # graph id of every kept row
                row_graph = torch.repeat_interleave(torch.arange(num_graphs, dtype=torch.int64),
                                                    property_subdict_slices.diff())
                kept_graph = row_graph[kept_rows]
                w_indices = indices[kept_rows]
                # global -> graph-local node ids
                p_indices = property_subdict[kept_rows] - x_slices[kept_graph].unsqueeze(1)
                base_offset = self.weight_offset[-1]
                for n in range(self.n_heads_per_label[head_id]):
                    block = torch.empty((kept_rows.numel(), 4), dtype=torch.int64)
                    block[:, 0] = current_head_id + n
                    block[:, 1:3] = p_indices
                    block[:, 3] = w_indices + base_offset + n * num_weights
                    weight_chunks.append(block)
                    weight_chunk_graph_ids.append(kept_graph)

                for n in range(self.n_heads_per_label[head_id]):
                    self.weight_offset.append(self.weight_offset[-1] + num_weights)
                    self.weight_offset_description.append({'head:': head_id, 'property': property_key, 'weights': num_weights})
                    self.weight_offset_description_text.append(f"Head {head_id} Property {property_key} has {num_weights} different weights")

            # TODO symmetric case

            if self.bias:
                # Set the bias weights; the unique inverse only depends on the
                # bias label description, so it is computed once per description
                bias_label_key = self.bias_label_descriptions[head_id]
                if bias_label_key not in bias_unique_cache:
                    _, bias_indices, _ = torch.unique(bias_labels, dim=0, return_inverse=True, return_counts=True, sorted=False)
                    bias_unique_cache[bias_label_key] = bias_indices
                bias_indices = bias_unique_cache[bias_label_key]
                # VECTORIZED over graphs: graph id and graph-local position per node
                node_graph = torch.repeat_interleave(torch.arange(num_graphs, dtype=torch.int64), num_nodes_per_graph)
                arranged_tensor = torch.arange(total_num_nodes, dtype=torch.int64) - x_slices[node_graph]
                for n in range(self.n_heads_per_label[head_id]):
                    for feature_id in range(self.in_features):
                        w_index_offset = n*self.in_features*self.n_bias_labels[head_id] + feature_id*self.n_bias_labels[head_id]
                        block = torch.empty((total_num_nodes, 4), dtype=torch.int64)
                        block[:, 0] = current_head_id + n
                        block[:, 1] = arranged_tensor
                        block[:, 2] = feature_id
                        block[:, 3] = bias_indices + self.b_head_offset + w_index_offset
                        bias_chunks.append(block)
                        bias_chunk_graph_ids.append(node_graph)
                # Determine the number of different learnable parameters in the bias vector
                for n in range(self.n_heads_per_label[head_id]):
                    self.bias_num.append(self.in_features * self.n_bias_labels[head_id])
                    self.b_head_offset += self.bias_num[-1]

            self.weight_num += head_weight_num
        # All weight distributions are computed, now merge them and create the final weight distribution tensor and bias distribution tensor.
        # A single stable sort by graph id restores the historical per-graph
        # chunk order (chunks were appended head -> property -> head replica).
        weight_distribution, self.weight_distribution_slices = self._merge_distribution_chunks(
            weight_chunks, weight_chunk_graph_ids, num_graphs)
        # Non-persistent buffer: moved by net.to(device), kept out of state_dict
        self.register_buffer('weight_distribution', weight_distribution, persistent=False)
        if self.bias:
            bias_distribution, self.bias_distribution_slices = self._merge_distribution_chunks(
                bias_chunks, bias_chunk_graph_ids, num_graphs)
            self.register_buffer('bias_distribution', bias_distribution, persistent=False)

    @staticmethod
    def _merge_distribution_chunks(chunks, chunk_graph_ids, num_graphs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Merge row blocks into one (rows, slices) pair ordered by graph id."""
        if chunks:
            rows = torch.cat(chunks, dim=0)
            graph_ids = torch.cat(chunk_graph_ids)
            order = torch.argsort(graph_ids, stable=True)
            rows = rows[order]
            rows_per_graph = torch.bincount(graph_ids, minlength=num_graphs)
        else:
            rows = torch.zeros((0, 4), dtype=torch.int64)
            rows_per_graph = torch.zeros(num_graphs, dtype=torch.int64)
        slices = torch.cat([torch.zeros(1, dtype=torch.int64), rows_per_graph.cumsum(dim=0)])
        return rows, slices

    def _finalize_initialization(self):
        """
        Allocate the learnable parameters and forward-pass structures.

        Runs after the distributions are available (built or cache-loaded).
        """
        if self.bias:
            # Set learnable parameters for the bias.
            # NOTE: no .to(self.device) here — on CUDA that would return a plain
            # (non-Parameter) tensor, silently removing the weights from
            # net.parameters() so the optimizer never updates them. Devices are
            # handled by the whole-model net.to(device) call.
            self.Param_b = self.init_weights(np.sum(self.bias_num), init_type='convolution_bias')

        # Set learnable parameters for the weights (see device note above)
        self.Param_W = self.init_weights(np.sum(self.weight_num), init_type='convolution')


        # TODO add pruning
        # in case of pruning is turned on, save the original weights
        #self.Param_W_original = None
        #self.mask = None
        #if 'prune' in self.para.run_config.config and self.para.run_config.config['prune']['enabled']:
        #    self.Param_W_original = self.Param_W.detach().clone()
        #    self.mask = torch.ones(self.Param_W.size())

        self.forward_step_time = 0

        # Per-graph node counts as plain Python ints: avoids a per-forward
        # .item() call (a host-device sync point once the data lives on GPU)
        self._num_nodes_list = [int(n) for n in self.graph_data.num_nodes]

        # The degree-matrix normalization branches relied on attributes
        # (self.D, self.in_edges) that are never assigned anywhere; enabling
        # them crashed in forward. Fail early instead of at forward time.
        if self.para.run_config.config.get('degree_matrix', False) or self.para.run_config.config.get('use_in_degrees', False):
            raise ValueError("The 'degree_matrix' and 'use_in_degrees' options are not supported: "
                             "their implementation was incomplete (self.D / self.in_edges were never initialized).")

        # Forward mode: 'sparse' builds a per-graph block COO matrix and uses
        # torch.sparse.mm; 'dense' keeps the original scatter into a dense
        # (H, N, N) tensor (equivalence oracle / fallback). 'auto' (default)
        # picks by graph size: measured on CPU, dense wins for small graphs
        # (sparse construction overhead dominates) while sparse wins for large
        # ones (no (H, N, N) allocation; ~6x faster at N=1024).
        forward_config = self.para.run_config.config.get('share_gnn_forward', None) or {}
        self.forward_mode_config = forward_config.get('mode', 'auto')
        if self.forward_mode_config not in ('auto', 'sparse', 'dense'):
            raise ValueError(f"share_gnn_forward.mode must be 'auto', 'sparse' or 'dense', got '{self.forward_mode_config}'")
        self.forward_mode = self.forward_mode_config
        if self.forward_mode == 'auto':
            self.forward_mode = 'sparse' if max(self._num_nodes_list, default=0) >= 256 else 'dense'
        # The *batched* forward has its own mode (see _use_dense_batch): the two
        # implementations trade off differently than in the per-graph case, and
        # the winner depends on the device, so 'auto' is resolved per forward.
        self._dense_batch_max_nodes = forward_config.get('dense_batch_max_nodes', 256)
        self._dense_batch_max_bytes = forward_config.get('dense_batch_max_bytes', 128 * 1024 ** 2)
        self._precision_itemsize = torch.empty(0, dtype=self.precision).element_size()
        # Per-forward wall-clock timing is pure overhead (and misleading on
        # CUDA without a synchronize) — gate it behind a config flag.
        self.profile_layers = self.para.run_config.config.get('profile_layers', False)

        self._build_forward_index_structures()

    def _build_forward_index_structures(self) -> None:
        """
        Precompute per-graph sparse index structures for the forward pass.

        Folds the head dimension into the row index (row = head * N + i) so the
        per-graph weight matrix becomes a single 2-D block COO matrix of shape
        (num_heads * N, N). The entries are sorted per graph block so the COO
        tensor can be constructed with ``is_coalesced=True`` (skipping the
        runtime coalesce). Only the *index* structure is precomputed here — the
        weight values are gathered from ``Param_W`` inside every forward pass,
        which is the only differentiable link back to the parameters.
        Precomputing the values instead would silently freeze training.
        """
        weight_distribution = self.weight_distribution
        row_counts = self.weight_distribution_slices.diff()
        nodes_per_graph = torch.as_tensor(self._num_nodes_list, dtype=torch.int64)
        n_per_row = torch.repeat_interleave(nodes_per_graph, row_counts)
        graph_per_row = torch.repeat_interleave(torch.arange(len(self._num_nodes_list), dtype=torch.int64), row_counts)

        rows = weight_distribution[:, 0] * n_per_row + weight_distribution[:, 1]
        cols = weight_distribution[:, 2]
        # sort within each graph block by (row, col); the (head, i, j) cells are
        # unique per graph, so sorted order makes the block a valid coalesced COO
        linear_index = rows * n_per_row + cols
        max_linear = int(linear_index.max().item()) + 1 if linear_index.numel() > 0 else 1
        order = torch.argsort(graph_per_row * max_linear + linear_index)

        self.register_buffer('_fwd_indices', torch.stack([rows, cols])[:, order].contiguous(), persistent=False)
        self.register_buffer('_fwd_param_idx', weight_distribution[order, 3].contiguous(), persistent=False)
        # per-graph offsets as plain ints (Python slicing, no device syncs)
        self._fwd_slices = [int(x) for x in self.weight_distribution_slices]

        # Second, NODE-MAJOR ordering for the batched forward: row = i * H + h.
        # With node-major rows the global row of graph g becomes
        # o_g * H + local_row (o_g = node offset in the batch), so concatenated
        # per-graph blocks occupy strictly increasing row ranges and the batch
        # block-diagonal COO can be built with is_coalesced=True as well.
        rows_b = weight_distribution[:, 1] * self.num_heads + weight_distribution[:, 0]
        linear_b = rows_b * n_per_row + cols
        max_linear_b = int(linear_b.max().item()) + 1 if linear_b.numel() > 0 else 1
        order_b = torch.argsort(graph_per_row * max_linear_b + linear_b)
        self.register_buffer('_fwd_b_rows', rows_b[order_b].contiguous(), persistent=False)
        self.register_buffer('_fwd_b_cols', cols[order_b].contiguous(), persistent=False)
        self.register_buffer('_fwd_b_param_idx', weight_distribution[order_b, 3].contiguous(), persistent=False)
        if self.bias:
            self._bias_slices = [int(x) for x in self.bias_distribution_slices]

    def _get_layer_cache_path(self) -> Path:
        """
        Cache path for the merged per-layer weight/bias distributions.

        Unlike the fine-grained (indices, counts) cache, the stored artifact is
        post-threshold-filtering, so the thresholds and every head's full
        configuration (including the resolved valid property values) must be
        part of the cache key.
        """
        heads_metadata = []
        for head_id in range(len(self.layer.layer_heads)):
            valid_values = self.graph_data.properties[self.property_descriptions[head_id]].valid_values[(self.layer_id, head_id)]
            heads_metadata.append({
                'source_label': self.source_label_descriptions[head_id],
                'target_label': self.target_label_descriptions[head_id],
                'bias_label': self.bias_label_descriptions[head_id],
                'property': self.property_descriptions[head_id],
                'valid_property_values': [str(v) for v in valid_values],
                'num': int(self.n_heads_per_label[head_id]),
                'bias': bool(self.bias_list[head_id]),
            })
        cache_metadata = {
            'format_version': 1,
            'dataset': self.para.db,
            'dataset_size': len(self.graph_data),
            'layer_id': self.layer_id,
            'in_features': int(self.in_features),
            'heads': heads_metadata,
            'threshold': self.para.run_config.config.get('rule_occurrence_threshold', 1),
            'upper_threshold': self.para.run_config.config.get('rule_occurrence_upper_threshold', None),
        }
        metadata_str = json.dumps(cache_metadata, sort_keys=True)
        cache_hash = hashlib.md5(metadata_str.encode()).hexdigest()[:12]
        data_path = Path(self.para.run_config.config['paths']['data'])
        cache_dir = data_path / 'caches'
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f'layerdist_{cache_hash}.pt'

    _LAYER_CACHE_FORMAT_VERSION = 1

    def _load_layer_distribution_cache(self, cached_path: Path) -> bool:
        """
        Load the merged per-layer distributions from disk. Returns True on a
        usable cache hit; on any mismatch or error falls back to a rebuild.
        """
        if not cached_path.exists():
            return False
        try:
            cached = torch.load(str(cached_path), weights_only=False)
            if not isinstance(cached, dict) or cached.get('format_version') != self._LAYER_CACHE_FORMAT_VERSION:
                return False
            required = ['weight_num', 'weight_offset', 'weight_offset_description',
                        'weight_offset_description_text', 'weight_distribution',
                        'weight_distribution_slices', 'bias_num', 'b_head_offset']
            if self.bias:
                required += ['bias_distribution', 'bias_distribution_slices']
            if any(key not in cached for key in required):
                return False
            # validate everything BEFORE registering buffers (register twice raises)
            self.weight_num = list(cached['weight_num'])
            self.weight_offset = list(cached['weight_offset'])
            self.weight_offset_description = list(cached['weight_offset_description'])
            self.weight_offset_description_text = list(cached['weight_offset_description_text'])
            self.bias_num = list(cached['bias_num'])
            self.b_head_offset = int(cached['b_head_offset'])
            self.weight_distribution_slices = cached['weight_distribution_slices']
            self.register_buffer('weight_distribution', cached['weight_distribution'], persistent=False)
            if self.bias:
                self.bias_distribution_slices = cached['bias_distribution_slices']
                self.register_buffer('bias_distribution', cached['bias_distribution'], persistent=False)
            print(f"✓ Layer cache hit: merged weight/bias distributions loaded from {cached_path.name}")
            return True
        except Exception as e:
            print(f"⚠ Warning: failed to load layer distribution cache {cached_path}: {e}")
            return False

    def _save_layer_distribution_cache(self, cached_path: Path) -> None:
        """Save the merged per-layer distributions. Non-fatal on failure."""
        try:
            cache_data = {
                'format_version': self._LAYER_CACHE_FORMAT_VERSION,
                'weight_num': list(self.weight_num),
                'weight_offset': list(self.weight_offset),
                'weight_offset_description': list(self.weight_offset_description),
                'weight_offset_description_text': list(self.weight_offset_description_text),
                'weight_distribution': self.weight_distribution,
                'weight_distribution_slices': self.weight_distribution_slices,
                'bias_num': list(self.bias_num),
                'b_head_offset': int(self.b_head_offset),
            }
            if self.bias:
                cache_data['bias_distribution'] = self.bias_distribution
                cache_data['bias_distribution_slices'] = self.bias_distribution_slices
            torch.save(cache_data, str(cached_path))
            file_size_mb = cached_path.stat().st_size / (1024 * 1024)
            meta_path = cached_path.with_suffix('.json')
            with open(meta_path, 'w') as f:
                json.dump({
                    'created': datetime.now().isoformat(),
                    'dataset': self.para.db,
                    'layer_id': self.layer_id,
                    'num_weight_rows': int(self.weight_distribution.shape[0]),
                    'num_weights': int(np.sum(self.weight_num)),
                }, f, indent=2)
            print(f"  Cached layer distributions ({file_size_mb:.2f} MB): {cached_path.name}")
        except Exception as e:
            print(f"⚠ Warning: Failed to save layer distribution cache to {cached_path}: {e}")

    def get_cache_path(self, head, property_key) -> Path:
        """
        Generate cache path for indices/counts of a specific head and property.

        Cache key includes all configuration that affects the computed indices:
        - Dataset, layer, head identifiers
        - Property configuration
        - Label configurations (source, target)
        - Filtering thresholds
        """
        # Build metadata dictionary for cache key
        cache_metadata = {
            'dataset': self.para.db,
            'dataset_size': len(self.graph_data),
            'property': self.property_descriptions[self.layer.layer_heads.index(head)],
            'property_key': str(property_key),
            'source_label': self.source_label_descriptions[self.layer.layer_heads.index(head)],
            'target_label': self.target_label_descriptions[self.layer.layer_heads.index(head)],
        }

        # Generate hash from metadata
        metadata_str = json.dumps(cache_metadata, sort_keys=True)
        cache_hash = hashlib.md5(metadata_str.encode()).hexdigest()[:12]

        # Construct cache directory path
        data_path = Path(self.para.run_config.config['paths']['data'])
        cache_dir = data_path / 'caches'
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Return cache file path
        return cache_dir / f'{cache_hash}.pt'

    def _load_cached_indices(self, cached_path: Path, head, property_key) -> tuple:
        """
        Load cached indices, counts and the invalid-pair flag from disk.

        Returns:
            (indices, counts, do_invalid_indices_exist)

        Raises:
            FileNotFoundError: If cache file doesn't exist
            Exception: If cache is corrupted, incompatible, or from an older
                format that did not store `do_invalid_indices_exist` (treated
                as a cache miss so the file self-heals on re-save).
        """
        if not cached_path.exists():
            raise FileNotFoundError(f"Cache file not found: {cached_path}")

        try:
            # Load cache file
            cached_data = torch.load(str(cached_path), weights_only=False)

            # Validate structure
            if not isinstance(cached_data, dict):
                raise ValueError("Invalid cache format: expected dict")

            if 'indices' not in cached_data or 'counts' not in cached_data:
                raise ValueError("Invalid cache format: missing indices or counts")

            if 'do_invalid_indices_exist' not in cached_data:
                # Old cache format: without this flag a cache hit would treat the
                # invalid (-1 label) bucket as a real weight. Recompute and re-save.
                raise ValueError("Invalid cache format: missing do_invalid_indices_exist")

            indices = cached_data['indices']
            counts = cached_data['counts']
            do_invalid_indices_exist = bool(cached_data['do_invalid_indices_exist'])

            # Validate tensor types
            if not isinstance(indices, torch.Tensor) or not isinstance(counts, torch.Tensor):
                raise ValueError("Invalid cache format: indices/counts must be tensors")

            return indices, counts, do_invalid_indices_exist

        except Exception as e:
            # If any error, treat as cache miss
            raise Exception(f"Failed to load cache: {e}")

    def _save_cached_indices(self, cached_path: Path, head, property_key, indices: torch.Tensor, counts: torch.Tensor, do_invalid_indices_exist: bool = False) -> None:
        """
        Save computed indices, counts and the invalid-pair flag to disk cache.

        Saves both the tensor data (.pt) and human-readable metadata (.json).
        Non-fatal: logs warning if save fails but doesn't raise exception.
        """
        try:
            # Prepare cache data
            cache_data = {
                'indices': indices,
                'counts': counts,
                'do_invalid_indices_exist': bool(do_invalid_indices_exist),
                'metadata': {
                    'created': datetime.now().isoformat(),
                    'source_label': self.source_label_descriptions[self.layer.layer_heads.index(head)],
                    'target_label': self.target_label_descriptions[self.layer.layer_heads.index(head)],
                    'property': self.property_descriptions[self.layer.layer_heads.index(head)],
                    'property_key': str(property_key),
                    'indices_shape': list(indices.shape),
                    'counts_shape': list(counts.shape),
                    'num_unique_pairs': len(counts),
                    'do_invalid_indices_exist': bool(do_invalid_indices_exist),
                }
            }

            # Save tensor data
            torch.save(cache_data, str(cached_path))

            # Calculate file size
            file_size_mb = cached_path.stat().st_size / (1024 * 1024)

            # Save human-readable metadata alongside
            meta_path = cached_path.with_suffix('.json')
            with open(meta_path, 'w') as f:
                json.dump(cache_data['metadata'], f, indent=2)

            print(f"  Cached {file_size_mb:.2f} MB: {cached_path.name}")

        except Exception as e:
            print(f"⚠ Warning: Failed to save cache to {cached_path}: {e}")
            # Non-fatal: continue without caching

    def init_weights(self, num_weights:np.float64, init_type:Optional[str]=None) -> nn.Parameter:
        """
        Initialize learnable weight parameters with configured strategy.

        Supports multiple initialization schemes: uniform, normal, symmetric normal,
        constant, lower/upper bound, and He initialization. Configuration is read
        from para.run_config.config['weight_initialization'][init_type].

        Parameters
        ----------
        num_weights : np.float64 or int
            Number of weight parameters to initialize.
        init_type : str, optional
            Weight category key in the configuration file (e.g., 'convolution',
            'convolution_bias'). If None, uses default constant initialization.

        Returns
        -------
        nn.Parameter
            Initialized weight parameter tensor of shape (num_weights,) with
            requires_grad=True.

        Notes
        -----
        **Supported Initialization Types:**

        - 'uniform': Uniform distribution U(min, max)
        - 'normal': Normal distribution N(mean, std)
        - 'symmetric_normal': Half weights from N(mean, std), half from N(-mean, std)
        - 'constant': All weights set to constant value
        - 'lower_upper': Uniform in [-1/√n, 1/√n] where n = num_weights
        - 'he': He initialization with std = √(2/n)

        If init_type is not in config or no config exists, defaults to constant 0.01.

        **Configuration Format (YAML):**
        ```yaml
        weight_initialization:
          convolution:
            type: normal
            mean: 0.0
            std: 0.1
          convolution_bias:
            type: constant
            value: 0.0
        ```

        Raises
        ------
        ValueError
            If init_type is specified but not supported by the configuration.

        See Also
        --------
        set_weights : Uses these parameters during forward pass
        __init__ : Calls this method to initialize Param_W and Param_b
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
        input_size = self._num_nodes_list[pos]
        self.current_W = torch.zeros((self.num_heads, input_size, input_size), dtype=self.precision, device=self.Param_W.device)
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
        input_size = self._num_nodes_list[pos]
        self.current_B = torch.zeros((self.num_heads, input_size, self.in_features), dtype=self.precision, device=self.Param_b.device)
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
        """
        Forward pass: apply invariant-based message passing to node representations.

        Computes multi-head aggregation conditioned on node label pairs and pairwise
        properties. For each head, aggregates neighbor features weighted by learned
        parameters specific to (source_label, target_label, property) combinations.

        Parameters
        ----------
        node_representation : torch.Tensor
            Input node features. Shape: (N, F) where:
            - N: number of nodes in the current graph
            - F: in_features (feature dimension)
        batch_data : GraphDataset
            Graph dataset (required by FrameworkLayer interface, not used here).
        *args
            Additional positional arguments (unused).
        **kwargs
            Keyword arguments:
            - 'pos' : int
                Index of the graph in the dataset (default: 0). Used to select
                graph-specific weight and bias configurations.

        Returns
        -------
        torch.Tensor
            Updated node representations. Shape: (N, H×F) where:
            - N: number of nodes (unchanged)
            - H: num_heads (number of aggregation heads)
            - F: in_features
            Output is flattened: (H, N, F) → (N, H×F)

        Notes
        -----
        **Algorithm:**
        1. Set graph-specific weights via set_weights(pos):
            - Constructs current_W: (H, N, N) sparse weight matrix
            - Each entry current_W[h, i, j] = weight for head h, node i, neighbor j
        2. Perform message passing via einsum:
            - Standard: current_W @ node_representation
            - With degree normalization: D @ current_W @ D @ x (if use_degree_matrix)
            - With in-degree norm: current_W @ x scaled by in_edges (if use_in_degrees)
        3. Add bias terms if enabled via set_bias(pos):
            - current_B: (H, N, F) bias for each head and node
        4. Permute and flatten: (H, N, F) → (N, F, H) → (N, H×F)
        5. Apply activation function

        **Einsum Operations:**
        - 'hij,jf->hif': (H, N, N) @ (N, F) → (H, N, F)
            Aggregates features from neighbors j to node i for each head h
        - 'cij,jk->cik': With degree matrix multiplication (variant)

        **Timing:**
        Forward pass time is accumulated in self.forward_step_time for profiling.

        **Graph-Specific Computation:**
        The weight matrix current_W is reconstructed for each graph based on:
        - Graph size (N varies across graphs)
        - Label distributions (different graphs have different label patterns)
        - Precomputed weight_distribution (indices into Param_W)

        See Also
        --------
        set_weights : Constructs current_W from weight parameters
        set_bias : Constructs current_B from bias parameters
        __init__ : Precomputes weight and bias distributions
        """
        # get pos from kwargs
        pos = kwargs.get('pos', 0)
        if is_batched_pos(pos):
            return self._forward_batched(node_representation, pos)
        begin = time.time() if self.profile_layers else None
        num_nodes = self._num_nodes_list[pos]
        if self.forward_mode == 'sparse':
            # Gather the weight values from Param_W (differentiable — this must
            # happen every forward, see _build_forward_index_structures) and
            # multiply as one block COO matrix of shape (H * N, N).
            start, end = self._fwd_slices[pos], self._fwd_slices[pos + 1]
            values = self.Param_W[self._fwd_param_idx[start:end]]
            current_W = torch.sparse_coo_tensor(self._fwd_indices[:, start:end], values,
                                                (self.num_heads * num_nodes, num_nodes),
                                                is_coalesced=True)
            node_representation = torch.sparse.mm(current_W, node_representation).view(self.num_heads, num_nodes, -1)
        else:
            # dense fallback: scatter into (H, N, N) and use a batched matmul
            self.set_weights(pos)
            node_representation = torch.matmul(self.current_W, node_representation)
        if self.bias:
            self.set_bias(pos)
            node_representation = node_representation + self.current_B
        node_representation = node_representation.permute(1, 2, 0)
        # merge dimensions 1 and 2
        node_representation = node_representation.flatten(start_dim=1)
        node_representation = self.activation(node_representation)
        if self.profile_layers:
            self.forward_step_time += time.time() - begin
        return node_representation

    def _use_dense_batch(self, device: torch.device, batch_size: int, max_nodes: int) -> bool:
        """
        Pick the batched implementation: padded dense or block-diagonal sparse.

        Unlike the per-graph case, the winner depends on the *device*, so
        ``mode: auto`` is resolved here instead of in __init__ (measured on
        NCI1, 64 graphs of <= 93 nodes, one conv layer, forward+backward):

            implementation            CPU        GPU (Radeon 890M)
            per-graph dense         11.3 ms      15.8 ms
            batched sparse          40.8 ms       6.2 ms
            batched padded-dense     3.1 ms       7.6 ms

        On CPU the padded (B, H, N_max, N_max) matmul goes through BLAS and
        beats both the sparse mm and the per-graph loop; on GPU the sparse
        block-diagonal mm wins because it does not pay for the padding, and
        kernel-launch overhead (the reason batching helps there) is already
        gone. An explicit ``mode`` overrides this choice.

        Dense padding costs B * H * N_max^2 elements, which grows quadratically
        with the largest graph in the batch, so ``auto`` also falls back to
        sparse beyond ``dense_batch_max_nodes`` nodes or
        ``dense_batch_max_bytes`` of padded weights.
        """
        if self.forward_mode_config == 'sparse':
            return False
        if self.forward_mode_config == 'dense':
            return True
        if device.type != 'cpu':
            return False
        if max_nodes >= self._dense_batch_max_nodes:
            return False
        dense_bytes = batch_size * self.num_heads * max_nodes * max_nodes * self._precision_itemsize
        return dense_bytes <= self._dense_batch_max_bytes

    def _forward_batched(self, node_representation: torch.Tensor, positions) -> torch.Tensor:
        """
        Batched forward over several graphs at once.

        ``node_representation`` must be the row-wise concatenation of the batch
        graphs' node features, in the order given by ``positions`` (duplicate
        graph ids are allowed and get independent node blocks). Returns the
        concatenated per-node output, shape (sum(N_g), F * H) — row-identical to
        running the per-graph forward on each graph. Both implementations
        (see _use_dense_batch) produce bit-identical results.
        """
        begin = time.time() if self.profile_layers else None
        positions = [int(p) for p in positions]
        sizes = [self._num_nodes_list[p] for p in positions]
        node_offsets = [0]
        for size in sizes:
            node_offsets.append(node_offsets[-1] + size)

        if self._use_dense_batch(node_representation.device, len(positions), max(sizes, default=0)):
            out = self._batched_dense_messages(node_representation, positions, sizes, node_offsets)
        else:
            out = self._batched_sparse_messages(node_representation, positions, node_offsets)

        if self.bias:
            out = out + self._batched_bias(positions, node_offsets, out.device)

        # (N, H, F) -> (N, F, H) -> (N, F*H): same output layout as the
        # per-graph path's (H, N, F).permute(1, 2, 0).flatten(1)
        out = self.activation(out.permute(0, 2, 1).flatten(start_dim=1))
        if self.profile_layers:
            self.forward_step_time += time.time() - begin
        return out

    def _batched_sparse_messages(self, node_representation: torch.Tensor, positions: List[int],
                                 node_offsets: List[int]) -> torch.Tensor:
        """One block-diagonal sparse mm over the concatenated batch nodes -> (N_total, H, F)."""
        device = node_representation.device
        num_heads = self.num_heads
        total_nodes = node_offsets[-1]
        offsets = torch.as_tensor(node_offsets[:-1], dtype=torch.int64, device=device)

        # concatenate the node-major per-graph index blocks and shift them by
        # the batch node offsets (rows are node-major, see
        # _build_forward_index_structures, so the result stays sorted/coalesced)
        nnz = torch.as_tensor([self._fwd_slices[p + 1] - self._fwd_slices[p] for p in positions],
                              dtype=torch.int64, device=device)
        rows = torch.cat([self._fwd_b_rows[self._fwd_slices[p]:self._fwd_slices[p + 1]] for p in positions])
        cols = torch.cat([self._fwd_b_cols[self._fwd_slices[p]:self._fwd_slices[p + 1]] for p in positions])
        param_idx = torch.cat([self._fwd_b_param_idx[self._fwd_slices[p]:self._fwd_slices[p + 1]] for p in positions])
        rows = rows + torch.repeat_interleave(offsets * num_heads, nnz)
        cols = cols + torch.repeat_interleave(offsets, nnz)

        # gathering from Param_W every forward keeps the graph differentiable
        values = self.Param_W[param_idx]
        current_W = torch.sparse_coo_tensor(torch.stack([rows, cols]), values,
                                            (total_nodes * num_heads, total_nodes),
                                            is_coalesced=True)
        return torch.sparse.mm(current_W, node_representation).view(total_nodes, num_heads, -1)

    def _batched_dense_messages(self, node_representation: torch.Tensor, positions: List[int],
                                sizes: List[int], node_offsets: List[int]) -> torch.Tensor:
        """
        Pad the batch to (B, H, N_max, N_max) @ (B, 1, N_max, F) -> (N_total, H, F).

        Uses the raw (head, i, j, param_idx) rows of weight_distribution, so no
        extra index buffers are needed. The padded rows/columns stay zero and
        are dropped again when the per-node outputs are gathered back into the
        concatenated (N_total, ...) layout, so the padding cannot leak between
        graphs.
        """
        device = node_representation.device
        num_heads = self.num_heads
        batch_size = len(positions)
        max_nodes = max(sizes, default=0)
        total_nodes = node_offsets[-1]

        counts = torch.as_tensor([self._fwd_slices[p + 1] - self._fwd_slices[p] for p in positions],
                                 dtype=torch.int64, device=device)
        blocks = torch.cat([self.weight_distribution[self._fwd_slices[p]:self._fwd_slices[p + 1]]
                            for p in positions])
        graph_slot = torch.repeat_interleave(
            torch.arange(batch_size, dtype=torch.int64, device=device), counts)

        # gathering from Param_W every forward keeps the graph differentiable
        current_W = torch.zeros((batch_size, num_heads, max_nodes, max_nodes),
                                dtype=self.precision, device=device)
        current_W[graph_slot, blocks[:, 0], blocks[:, 1], blocks[:, 2]] = self.Param_W[blocks[:, 3]]

        # scatter the concatenated node features into the padded (B, N_max, F) layout
        node_slot, local_index = self._batched_node_index(sizes, node_offsets, device)
        padded_x = torch.zeros((batch_size, max_nodes, node_representation.shape[1]),
                               dtype=self.precision, device=device)
        padded_x[node_slot, local_index] = node_representation

        out = torch.matmul(current_W, padded_x.unsqueeze(1))  # (B, H, N_max, F)
        # gather the real nodes back, dropping the padding: -> (N_total, H, F)
        return out[node_slot, :, local_index].view(total_nodes, num_heads, -1)

    def _batched_node_index(self, sizes: List[int], node_offsets: List[int],
                            device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """(batch slot, node index within its graph) for every row of the batch."""
        sizes_tensor = torch.as_tensor(sizes, dtype=torch.int64, device=device)
        offsets = torch.as_tensor(node_offsets[:-1], dtype=torch.int64, device=device)
        node_slot = torch.repeat_interleave(
            torch.arange(len(sizes), dtype=torch.int64, device=device), sizes_tensor)
        local_index = (torch.arange(node_offsets[-1], dtype=torch.int64, device=device)
                       - torch.repeat_interleave(offsets, sizes_tensor))
        return node_slot, local_index

    def _batched_bias(self, positions: List[int], node_offsets: List[int],
                      device: torch.device) -> torch.Tensor:
        """Scatter Param_b into the concatenated (N_total, H, F) bias tensor."""
        bias_rows = torch.cat([self.bias_distribution[self._bias_slices[p]:self._bias_slices[p + 1]]
                               for p in positions])
        b_counts = torch.as_tensor([self._bias_slices[p + 1] - self._bias_slices[p] for p in positions],
                                   dtype=torch.int64, device=device)
        offsets = torch.as_tensor(node_offsets[:-1], dtype=torch.int64, device=device)
        node_idx = bias_rows[:, 1] + torch.repeat_interleave(offsets, b_counts)
        current_B = torch.zeros((node_offsets[-1], self.num_heads, self.in_features),
                                dtype=self.precision, device=device)
        current_B[node_idx, bias_rows[:, 0], bias_rows[:, 2]] = torch.take(self.Param_b, bias_rows[:, 3])
        return current_B


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
