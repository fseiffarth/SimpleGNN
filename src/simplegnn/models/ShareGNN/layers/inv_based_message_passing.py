import os
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
from simplegnn.datasets.utils.graph_drawing import GraphDrawing, filter_weight_bounds, resolve_positions
from simplegnn.framework.utils.parameters import Parameters
from simplegnn.models.ShareGNN.layers.inv_based import InvariantBasedLayer
from simplegnn.models.ShareGNN.utils import Layer, is_batched_pos, range_gather
from simplegnn.utils.utils import available_memory_bytes


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

    Index computations are expensive (O(|E| × |labels|²) worst case), so the
    (indices, counts) results are cached to disk with MD5-hashed keys. See
    get_cache_path(), _load_cached_indices(), and _save_cached_indices().
    A second, coarse per-layer cache of the fully merged distributions can be
    enabled with `cache: {layer_distributions: True}` in the hyperparameter
    config; it is off by default because the files are large (GBs on
    ZINC-scale datasets) and are orphaned by any layer-config change.

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
        - Dataset, labels, properties (thresholds are applied after loading)
        - Cache hit: instant loading
        - Cache miss: compute, save for next time
        The merged per-layer distributions can additionally be cached with
        `cache: {layer_distributions: True}` (off by default; large files).

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
        # All heads are folded into out_features (downstream layers see a flat
        # (N, F*H) feature dimension), so no separate channel dimension remains.
        self.out_channels = 1
        # Transformer-style pre-norm block: out = x + Conv(LayerNorm(x)). Both
        # keys default to off, so existing configs are bit-identical. The
        # residual is needed even with self-loop-like configs: distance 0 is
        # usually excluded from the property values, so a node's own feature
        # never reaches its own output otherwise (see specs/15).
        self.pre_layer_norm = layer.layer_dict.get('pre_layer_norm', False)
        if self.pre_layer_norm:
            self.pre_norm = nn.LayerNorm(self.in_features)
        # Optional degree normalization of the aggregation (off by default).
        # The "degree" of a node is the number of neighbors a head actually
        # aggregates — the nonzero pattern of its weight matrix, so distance-k
        # pairs count and a distance-0 self loop counts as well.
        #   'row':       w[h,i,j] /= deg_h(i)            (mean aggregation)
        #   'symmetric': w[h,i,j] /= sqrt(deg_h(i) * deg_h(j))   (GCN-style)
        # The factors are weight-independent, so they are computed from the
        # index structure and cached along with it.
        self.degree_normalization = layer.layer_dict.get('degree_normalization', None)
        if self.degree_normalization in (False, 'none'):
            self.degree_normalization = None
        if self.degree_normalization not in (None, 'row', 'symmetric'):
            raise ValueError(f"degree_normalization must be 'row', 'symmetric' or omitted, "
                             f"got '{self.degree_normalization}'")
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

        # Coarse per-layer cache (opt-in via `cache: {layer_distributions: True}`
        # in the hyperparameter config): the merged distributions are
        # deterministic given the layer configuration + dataset and are
        # fold/run-invariant, so on a hit the whole head/property assembly is
        # skipped. Off by default: the files are large (GBs on ZINC-scale
        # datasets) and any config edit orphans them, while the fine-grained
        # (indices, counts) cache below already keeps the expensive
        # torch.unique results, making the rebuild cheap.
        cache_config = self.para.run_config.config.get('cache') or {}
        if cache_config.get('layer_distributions', False):
            layer_cache_path = self._get_layer_cache_path()
            if not self._load_layer_distribution_cache(layer_cache_path):
                self._build_distributions(graph_data)
                self._save_layer_distribution_cache(layer_cache_path)
        else:
            self._build_distributions(graph_data)

        self._finalize_initialization()

    def _build_distributions(self, graph_data: GraphDataset) -> None:
        """
        Build the factored per-head parameter-index structures.

        Historically this materialized one (head, i, j, param_idx) int64 row
        per matching node pair per head for the WHOLE dataset (plus an
        equivalent per-(head, node, feature) bias enumeration), which needed
        ~190 GB on ZINC-full and got the process OOM-killed (see
        specs/10-invariant-layer-memory-factorization.md). The (i, j) pairs are
        identical across heads and already stored once in the shared Properties
        object, so per head only a single int32 parameter-index vector aligned
        to the shared graph-major pair rows is kept (`_pv_<head_id>` buffers,
        -1 = pair not used by this head). Head-replica shifts are pure
        arithmetic (`_nw_<head_id>` tables). The bias reduces to one int32
        per-node label-index vector per unique bias description
        (`_bias_idx_<slot>`) plus a (num_heads, in_features) offset table.
        The explicit rows are re-assembled per batch by _assemble_rows().
        """
        num_graphs = len(graph_data)
        x_slices = self.graph_data.slices['x']
        total_num_nodes = int(x_slices[-1])

        # fail fast with a clear error instead of an OOM kill (the estimate is
        # computed from slice metadata only — no large allocation happens yet)
        self._check_memory_budget(graph_data)

        bias_unique_cache = {}       # bias label description -> unique inverse indices
        indices_cache_hits = 0       # fine-grained (indices, counts) cache stats
        indices_cache_misses = 0

        # shared consolidated pair views, one per property description
        self._desc_slot = {}         # property description -> buffer slot
        self._cfg_meta = []          # per head-config: desc slot, head base, replicas
        # bias: one per-node index vector per unique bias description
        self._bias_slot = {}         # bias label description -> buffer slot
        self._bias_cols_by_slot = {} # slot -> list of head columns using it
        b_off = torch.zeros((self.num_heads, self.in_features), dtype=torch.int64)

        # Iterate over all heads in the layer
        for head_id, head in enumerate(self.layer.layer_heads):
            head_weight_num = []  # number of weights for the current head, used for debugging
            # get all the valid property values for the head (e.g., the distances 0, 3, 6)
            prop = self.graph_data.properties[self.property_descriptions[head_id]]
            valid_property_values = prop.valid_values[(self.layer_id, head_id)]
            cons = prop.consolidated(x_slices)
            # apply the head and tail labels to the subdict
            source_labels = self.graph_data.node_labels[self.source_label_descriptions[head_id]].node_labels
            target_labels = self.graph_data.node_labels[self.target_label_descriptions[head_id]].node_labels
            bias_labels = self.graph_data.node_labels[self.bias_label_descriptions[head_id]].node_labels
            current_head_id = 0
            for h_i in range(head_id):
                current_head_id += self.n_heads_per_label[h_i]

            desc = self.property_descriptions[head_id]
            if desc not in self._desc_slot:
                slot = len(self._desc_slot)
                self._desc_slot[desc] = slot
                # shared CPU tensors; registered per layer so net.to(device)
                # moves them (a second invariant layer using the same
                # description gets its own device copy after .to)
                self.register_buffer(f'_lp_{slot}', cons.lp, persistent=False)
                self.register_buffer(f'_kid_{slot}', cons.key_id, persistent=False)
                self.register_buffer(f'_pslices_{slot}', cons.slices, persistent=False)
            desc_slot = self._desc_slot[desc]
            self._cfg_meta.append({
                'desc_slot': desc_slot,
                'head_base': current_head_id,
                'num': self.n_heads_per_label[head_id],
            })
            # parameter index of every consolidated pair row (replica 0);
            # -1 = row not used by this head (value not selected, combo below
            # the occurrence threshold, or invalid -1 label)
            pv = torch.full((cons.total_rows,), -1, dtype=torch.int32)
            # num_weights per property value, for the replica shift n * nw[key]
            nw_key = torch.zeros(len(cons.keys), dtype=torch.int64)

            for property_key in valid_property_values:
                #print(f'Initialize head {i+1}/{len(self.layer.layer_heads)} with property {property_key}')
                property_subdict = self.graph_data.properties[self.property_descriptions[head_id]].properties[property_key]
                property_subdict_slices = self.graph_data.properties[self.property_descriptions[head_id]].properties_slices[property_key]

                # Initialize variables before try-except to avoid scoping issues
                do_invalid_indices_exist = False
                threshold = self.para.run_config.config.get('rule_occurrence_threshold', 1)
                upper_threshold = self.para.run_config.config.get('rule_occurrence_upper_threshold', None)

                cached_path = self.get_cache_path(head, property_key)

                try:
                    indices, counts, do_invalid_indices_exist = self._load_cached_indices(cached_path, head, property_key)
                    indices_cache_hits += 1

                except Exception as e:
                    indices_cache_misses += 1
                    if not isinstance(e, FileNotFoundError):
                        print(f"⊗ Cache miss: head source label {self.source_label_descriptions[head_id]}, target label {self.target_label_descriptions[head_id]} with property {self.property_descriptions[head_id]} key {property_key} - computing indices and counts ({str(e)})")

                    # OPTIMIZATION: Build labeled_subdict directly without clone
                    # (5-10% speedup). Only needed on a miss — on a hit this
                    # would be a wasted multi-million-row gather per key.
                    labeled_subdict = torch.stack([
                        source_labels[property_subdict[:, 0]],
                        target_labels[property_subdict[:, 1]]
                    ], dim=1)

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

                    # Fast 1D unique instead of slow 2D unique. sorted=True is
                    # required: the invalid bucket was encoded as the largest
                    # value, so only in sorted order is counts[-1] guaranteed
                    # to be the invalid bucket.
                    _, indices, counts = torch.unique(encoded_labels, return_inverse=True, return_counts=True, sorted=True)
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
                    # relabel indices using a vectorized mapping (-1 = dropped,
                    # which is exactly the pv sentinel for "row unused")
                    mapping = torch.full((counts.size(0),), -1, dtype=torch.int64, device=indices.device)
                    mapping[valid_values] = torch.arange(valid_values.numel(), device=indices.device, dtype=torch.int64)
                    indices = mapping[indices]
                    num_weights = valid_values.numel()
                for n in range(self.n_heads_per_label[head_id]):
                    head_weight_num.append(num_weights)

                # write the (absolute, replica-0) parameter index of every pair
                # of this property value into the head's pv vector — dropped
                # rows keep the -1 sentinel
                base_offset = self.weight_offset[-1]
                target = cons.target_rows(property_key)
                pv[target] = torch.where(indices >= 0, indices + base_offset, indices).to(torch.int32)
                nw_key[cons.key_index[property_key]] = num_weights
                total_params = base_offset + num_weights * self.n_heads_per_label[head_id]
                if total_params > torch.iinfo(torch.int32).max:
                    raise ValueError(
                        f"Layer {self.layer_id}: {total_params:,} weight parameters exceed the "
                        f"int32 range of the factored pv storage — reduce heads/labels/properties.")

                for n in range(self.n_heads_per_label[head_id]):
                    self.weight_offset.append(self.weight_offset[-1] + num_weights)
                    self.weight_offset_description.append({'head:': head_id, 'property': property_key, 'weights': num_weights})
                    self.weight_offset_description_text.append(f"Head {head_id} Property {property_key} has {num_weights} different weights")

            # TODO symmetric case

            # Non-persistent buffers: moved by net.to(device), out of state_dict
            self.register_buffer(f'_pv_{head_id}', pv, persistent=False)
            self.register_buffer(f'_nw_{head_id}', nw_key, persistent=False)

            if self.bias:
                # The old enumeration stored one (head, node, feature, param)
                # row per node x feature x head replica — a pure broadcast of
                # the per-node unique inverse (37 GB on ZINC-full). Keep only
                # the (total_nodes,) inverse per unique bias description and
                # the (num_heads, in_features) offset table b_off; the bias is
                # gathered as Param_b[b_off[h, f] + bias_idx[node]] in forward.
                bias_label_key = self.bias_label_descriptions[head_id]
                if bias_label_key not in bias_unique_cache:
                    _, bias_indices, _ = torch.unique(bias_labels, dim=0, return_inverse=True, return_counts=True, sorted=False)
                    bias_unique_cache[bias_label_key] = bias_indices
                    slot = len(self._bias_slot)
                    self._bias_slot[bias_label_key] = slot
                    self.register_buffer(f'_bias_idx_{slot}', bias_indices.to(torch.int32), persistent=False)
                bias_slot = self._bias_slot[bias_label_key]
                for n in range(self.n_heads_per_label[head_id]):
                    self._bias_cols_by_slot.setdefault(bias_slot, []).append(current_head_id + n)
                    for feature_id in range(self.in_features):
                        w_index_offset = n*self.in_features*self.n_bias_labels[head_id] + feature_id*self.n_bias_labels[head_id]
                        b_off[current_head_id + n, feature_id] = self.b_head_offset + w_index_offset
                # Determine the number of different learnable parameters in the bias vector
                for n in range(self.n_heads_per_label[head_id]):
                    self.bias_num.append(self.in_features * self.n_bias_labels[head_id])
                    self.b_head_offset += self.bias_num[-1]

            self.weight_num += head_weight_num
        if indices_cache_hits or indices_cache_misses:
            print(f"Layer {self.layer_id} indices cache: {indices_cache_hits} hits, "
                  f"{indices_cache_misses} misses (computed and cached)")
        if self.bias:
            self.register_buffer('_b_off', b_off, persistent=False)
        # per-graph node offsets for single-graph/bias assembly
        self._x_slice_list = [int(v) for v in x_slices]
        self.register_buffer('_x_slices', x_slices.clone(), persistent=False)

    def _check_memory_budget(self, graph_data: GraphDataset) -> None:
        """
        Estimate the factored index-structure footprint from slice metadata and
        fail fast (MemoryError) if it clearly exceeds the available RAM.

        The estimate covers the shared consolidated pair views (int32 pair ids
        + int16 value ids + per-value offsets) and one int32 pv vector per
        head-config. Computed before anything large is allocated, so a
        misconfigured run on a huge dataset dies with an actionable message
        instead of an OOM kill.
        """
        desc_rows = {}
        pv_rows = 0
        for head_id in range(len(self.layer.layer_heads)):
            desc = self.property_descriptions[head_id]
            prop = self.graph_data.properties[desc]
            if desc not in desc_rows:
                desc_rows[desc] = sum(int(prop.properties_slices[k][-1]) for k in prop.properties)
            pv_rows += desc_rows[desc]
        shared_rows = sum(desc_rows.values())
        estimate = shared_rows * (8 + 2) + pv_rows * 4  # lp int32x2 + key_id int16 + pv int32
        estimate += len(desc_rows) * len(graph_data) * 8 * 4  # slices/per-key offsets (approx)
        if self.bias:
            estimate += int(graph_data.slices['x'][-1]) * 4  # per-node bias indices
        if estimate > (1 << 30):
            print(f"Layer {self.layer_id}: estimated invariant index memory "
                  f"{estimate / 2**30:.1f} GiB "
                  f"({shared_rows:,} shared pair rows, {pv_rows:,} pv entries)")
        available = available_memory_bytes()
        if available is not None and estimate > 0.8 * available:
            raise MemoryError(
                f"Invariant layer {self.layer_id} would need ~{estimate / 2**30:.1f} GiB of index "
                f"structures but only {available / 2**30:.1f} GiB of RAM are available. "
                f"Reduce the property value list (e.g. fewer distances), the number of heads, "
                f"or raise rule_occurrence_threshold before running this configuration.")

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
                             "their implementation was incomplete (self.D / self.in_edges were never initialized). "
                             "Use the per-layer option degree_normalization: 'row' | 'symmetric' on the "
                             "invariant_based_convolution layer instead.")

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
        # memoized sparse index structure of the last per-graph forward:
        # (pos, coalesced indices, sorted Param_W gather indices). Weight-
        # independent, so it stays valid across optimizer steps; hit on every
        # forward for single-graph (node-level) tasks.
        self._sparse_row_cache = None

        # Optional materialized-row cache: precompute the assembled rows for
        # the whole dataset once (the old representation, minus its build
        # transients) and serve batches by slicing. 'auto' (default)
        # materializes only when the rows fit comfortably in memory, so small
        # datasets get the old per-step speed while ZINC-full-scale datasets
        # stay factored. 'share_gnn_forward: {precompute_rows: True/False}'
        # forces either behavior.
        self._rows_materialized = False
        precompute = forward_config.get('precompute_rows', 'auto')
        if precompute not in ('auto', True, False):
            raise ValueError(f"share_gnn_forward.precompute_rows must be 'auto', True or False, got '{precompute}'")
        if precompute is not False:
            max_bytes = forward_config.get('precompute_rows_max_bytes', 4 * 1024 ** 3)
            total_rows = sum(
                int((getattr(self, f'_pv_{h}') >= 0).sum()) * meta['num']
                for h, meta in enumerate(self._cfg_meta))
            rows_bytes = total_rows * 16  # 4x int32 columns
            available = available_memory_bytes()
            fits = rows_bytes <= max_bytes and (available is None or rows_bytes <= 0.25 * available)
            if precompute is True and not fits:
                raise MemoryError(
                    f"Layer {self.layer_id}: precompute_rows=True would materialize "
                    f"{rows_bytes / 2**30:.1f} GiB of assembled rows "
                    f"(limit {max_bytes / 2**30:.1f} GiB / 25% of available RAM). "
                    f"Raise precompute_rows_max_bytes or use precompute_rows: 'auto'/False.")
            if fits:
                self._materialize_rows(total_rows)

    def _materialize_rows(self, total_rows: int) -> None:
        """
        Assemble the rows of every graph once (in graph chunks, graph-major)
        and register them as int32 buffers. _assemble_rows then serves batches
        with a single range-gather per column instead of the per-config
        assembly — the old representation's per-step speed at a
        deliberately-bounded memory cost.
        """
        num_graphs = len(self._num_nodes_list)
        rc_heads = torch.empty(total_rows, dtype=torch.int32)
        rc_i = torch.empty(total_rows, dtype=torch.int32)
        rc_j = torch.empty(total_rows, dtype=torch.int32)
        rc_params = torch.empty(total_rows, dtype=torch.int32)
        rows_per_graph = torch.zeros(num_graphs, dtype=torch.int64)
        write = 0
        chunk = 4096
        for start in range(0, num_graphs, chunk):
            positions = list(range(start, min(start + chunk, num_graphs)))
            heads, i_local, j_local, params, graph_slot = self._assemble_rows(positions)
            # group the chunk's rows by graph (stable: keeps assembly order within a graph)
            order = torch.argsort(graph_slot, stable=True)
            n = heads.shape[0]
            rc_heads[write:write + n] = heads[order].to(torch.int32)
            rc_i[write:write + n] = i_local[order].to(torch.int32)
            rc_j[write:write + n] = j_local[order].to(torch.int32)
            rc_params[write:write + n] = params[order].to(torch.int32)
            rows_per_graph[start:start + len(positions)] = torch.bincount(graph_slot, minlength=len(positions))
            write += n
        slices = torch.cat([torch.zeros(1, dtype=torch.int64), rows_per_graph.cumsum(dim=0)])
        # Non-persistent buffers: moved by net.to(device), kept out of state_dict
        self.register_buffer('_rc_heads', rc_heads, persistent=False)
        self.register_buffer('_rc_i', rc_i, persistent=False)
        self.register_buffer('_rc_j', rc_j, persistent=False)
        self.register_buffer('_rc_params', rc_params, persistent=False)
        self.register_buffer('_rc_slices', slices, persistent=False)
        self._rows_materialized = True

    def _assemble_rows(self, positions):
        """
        Re-assemble the explicit (head, i, j, param_idx) rows for a list of
        graphs from the factored structures.

        Returns (heads, i_local, j_local, param_idx, graph_slot), all int64
        1-D tensors of equal length; graph_slot indexes into `positions`.
        This replaces slicing the old dataset-wide `weight_distribution`: the
        per-batch cost is a handful of vectorized gathers, while the dataset-
        scale storage stays factored (see _build_distributions). The weight
        VALUES are still gathered from Param_W by the callers on every forward
        — that gather is the differentiable link back to the parameters.
        """
        device = getattr(self, '_pv_0').device
        positions_t = torch.as_tensor(positions, dtype=torch.int64, device=device)
        if self._rows_materialized:
            gather_idx, slot_of_row = range_gather(self._rc_slices, positions_t)
            return (self._rc_heads[gather_idx].long(), self._rc_i[gather_idx].long(),
                    self._rc_j[gather_idx].long(), self._rc_params[gather_idx].long(),
                    slot_of_row)
        views = {}
        parts_h, parts_ij, parts_p, parts_g = [], [], [], []
        for head_id, meta in enumerate(self._cfg_meta):
            slot = meta['desc_slot']
            if slot not in views:
                views[slot] = range_gather(getattr(self, f'_pslices_{slot}'), positions_t)
            gather_idx, slot_of_row = views[slot]
            pv = getattr(self, f'_pv_{head_id}')[gather_idx]
            mask = pv >= 0
            params0 = pv[mask].long()
            sel = gather_idx[mask]
            lp_m = getattr(self, f'_lp_{slot}')[sel].long()
            slots_m = slot_of_row[mask]
            kid_m = None
            if meta['num'] > 1:
                kid_m = getattr(self, f'_kid_{slot}')[sel].long()
                nw = getattr(self, f'_nw_{head_id}')
            for n in range(meta['num']):
                parts_h.append(torch.full_like(params0, meta['head_base'] + n))
                parts_ij.append(lp_m)
                parts_p.append(params0 if n == 0 else params0 + n * nw[kid_m])
                parts_g.append(slots_m)
        heads = torch.cat(parts_h)
        ij = torch.cat(parts_ij)
        params = torch.cat(parts_p)
        graph_slot = torch.cat(parts_g)
        return heads, ij[:, 0], ij[:, 1], params, graph_slot

    def _degree_norm_scale(self, heads, node_i, node_j, total_nodes):
        """
        Per-nonzero degree normalization factors for assembled rows.

        ``node_i``/``node_j`` are batch-global node ids (per-graph ids plus the
        graph's node offset), so counts never mix across graphs or heads. The
        degree of (head h, node n) is the number of nonzeros of h's weight
        matrix in row (out-degree) resp. column (in-degree) n. Returns a
        tensor aligned with the input rows; multiply it into the gathered
        Param_W values. Weight-independent, so cacheable with the indices.
        """
        num_keys = total_nodes * self.num_heads
        keys_i = node_i * self.num_heads + heads
        counts_i = torch.bincount(keys_i, minlength=num_keys).clamp(min=1)
        if self.degree_normalization == 'row':
            return (1.0 / counts_i[keys_i].to(self.precision))
        keys_j = node_j * self.num_heads + heads
        counts_j = torch.bincount(keys_j, minlength=num_keys).clamp(min=1)
        return 1.0 / torch.sqrt((counts_i[keys_i] * counts_j[keys_j]).to(self.precision))

    @staticmethod
    def _atomic_torch_save(data, path: Path) -> None:
        """torch.save via a temp file + os.replace.

        Parallel joblib workers share the cache paths; writing in place would
        let a reader see a partially written file. os.replace is atomic on
        POSIX, so readers only ever see complete files (last writer wins).
        """
        tmp_path = path.with_name(f'{path.name}.tmp.{os.getpid()}')
        try:
            torch.save(data, str(tmp_path))
            os.replace(tmp_path, path)
        finally:
            tmp_path.unlink(missing_ok=True)

    @staticmethod
    def _atomic_json_dump(obj, path: Path) -> None:
        """json.dump via a temp file + os.replace (see _atomic_torch_save)."""
        tmp_path = path.with_name(f'{path.name}.tmp.{os.getpid()}')
        try:
            with open(tmp_path, 'w') as f:
                json.dump(obj, f, indent=2)
            os.replace(tmp_path, path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def _cache_dir(self) -> Path:
        cache_dir = Path(self.para.run_config.config['paths']['data']) / 'caches'
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _layer_cache_key_dict(self) -> dict:
        """
        Cache key for the merged per-layer weight/bias distributions.

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
        return {
            'format_version': 1,
            'dataset': self.para.db,
            'dataset_size': len(self.graph_data),
            'layer_id': self.layer_id,
            'in_features': int(self.in_features),
            'heads': heads_metadata,
            'threshold': self.para.run_config.config.get('rule_occurrence_threshold', 1),
            'upper_threshold': self.para.run_config.config.get('rule_occurrence_upper_threshold', None),
        }

    def _get_layer_cache_path(self) -> Path:
        """Cache path for the merged per-layer weight/bias distributions."""
        metadata_str = json.dumps(self._layer_cache_key_dict(), sort_keys=True)
        cache_hash = hashlib.md5(metadata_str.encode()).hexdigest()[:12]
        return self._cache_dir() / f'layerdist_{cache_hash}.pt'

    # Bumped 2 -> 3: the cached artifact changed from the merged dataset-wide
    # (head, i, j, param) row tensors to the factored structures (per-head pv
    # vectors, replica tables, bias index vectors). Old caches rebuild.
    _LAYER_CACHE_FORMAT_VERSION = 3

    def _load_layer_distribution_cache(self, cached_path: Path) -> bool:
        """
        Load the factored per-layer structures from disk. Returns True on a
        usable cache hit; on any mismatch or error falls back to a rebuild.

        The shared consolidated pair views (_lp_*/_kid_*/_pslices_*) are not
        cached — they are deterministic per Properties object and rebuilt via
        prop.consolidated() (and typically already exist, shared, in memory).
        """
        if not cached_path.exists():
            return False
        try:
            cached = torch.load(str(cached_path), weights_only=False)
            if not isinstance(cached, dict) or cached.get('format_version') != self._LAYER_CACHE_FORMAT_VERSION:
                return False
            required = ['weight_num', 'weight_offset', 'weight_offset_description',
                        'weight_offset_description_text', 'bias_num', 'b_head_offset',
                        'cfg_meta', 'desc_names', 'pv', 'nw']
            if self.bias:
                required += ['bias_slot', 'bias_cols_by_slot', 'bias_idx', 'b_off']
            if any(key not in cached for key in required):
                return False
            n_cfgs = len(self.layer.layer_heads)
            # validate everything BEFORE registering buffers: a failure after
            # register_buffer would make the rebuild fallback register the same
            # name twice and crash
            if len(cached['pv']) != n_cfgs or len(cached['nw']) != n_cfgs:
                return False
            if any(not isinstance(t, torch.Tensor) for t in cached['pv'] + cached['nw']):
                return False
            if self.bias and any(not isinstance(t, torch.Tensor) for t in cached['bias_idx']):
                return False

            self.weight_num = list(cached['weight_num'])
            self.weight_offset = list(cached['weight_offset'])
            self.weight_offset_description = list(cached['weight_offset_description'])
            self.weight_offset_description_text = list(cached['weight_offset_description_text'])
            self.bias_num = list(cached['bias_num'])
            self.b_head_offset = int(cached['b_head_offset'])
            self._cfg_meta = list(cached['cfg_meta'])

            x_slices = self.graph_data.slices['x']
            self._desc_slot = {}
            for slot, desc in enumerate(cached['desc_names']):
                cons = self.graph_data.properties[desc].consolidated(x_slices)
                self._desc_slot[desc] = slot
                self.register_buffer(f'_lp_{slot}', cons.lp, persistent=False)
                self.register_buffer(f'_kid_{slot}', cons.key_id, persistent=False)
                self.register_buffer(f'_pslices_{slot}', cons.slices, persistent=False)
            for head_id in range(n_cfgs):
                self.register_buffer(f'_pv_{head_id}', cached['pv'][head_id], persistent=False)
                self.register_buffer(f'_nw_{head_id}', cached['nw'][head_id], persistent=False)
            if self.bias:
                self._bias_slot = dict(cached['bias_slot'])
                self._bias_cols_by_slot = dict(cached['bias_cols_by_slot'])
                for slot, idx in enumerate(cached['bias_idx']):
                    self.register_buffer(f'_bias_idx_{slot}', idx, persistent=False)
                self.register_buffer('_b_off', cached['b_off'], persistent=False)
            self._x_slice_list = [int(v) for v in x_slices]
            self.register_buffer('_x_slices', x_slices.clone(), persistent=False)
            print(f"✓ Layer {self.layer_id} distribution cache hit: {cached_path.name}")
            return True
        except Exception as e:
            print(f"⚠ Warning: failed to load layer distribution cache {cached_path}: {e}")
            # drop any buffers registered before the failure: the rebuild
            # fallback re-registers the same names and register_buffer raises
            # on duplicates
            for name in [n for n in list(self._buffers) if n.startswith(
                    ('_pv_', '_nw_', '_lp_', '_kid_', '_pslices_', '_bias_idx_', '_b_off', '_x_slices'))]:
                del self._buffers[name]
            return False

    def _save_layer_distribution_cache(self, cached_path: Path) -> None:
        """Save the factored per-layer structures. Non-fatal on failure."""
        try:
            n_cfgs = len(self.layer.layer_heads)
            desc_names = [None] * len(self._desc_slot)
            for desc, slot in self._desc_slot.items():
                desc_names[slot] = desc
            cache_data = {
                'format_version': self._LAYER_CACHE_FORMAT_VERSION,
                'weight_num': list(self.weight_num),
                'weight_offset': list(self.weight_offset),
                'weight_offset_description': list(self.weight_offset_description),
                'weight_offset_description_text': list(self.weight_offset_description_text),
                'bias_num': list(self.bias_num),
                'b_head_offset': int(self.b_head_offset),
                'cfg_meta': list(self._cfg_meta),
                'desc_names': desc_names,
                'pv': [getattr(self, f'_pv_{h}') for h in range(n_cfgs)],
                'nw': [getattr(self, f'_nw_{h}') for h in range(n_cfgs)],
            }
            if self.bias:
                cache_data['bias_slot'] = dict(self._bias_slot)
                cache_data['bias_cols_by_slot'] = dict(self._bias_cols_by_slot)
                cache_data['bias_idx'] = [getattr(self, f'_bias_idx_{s}') for s in range(len(self._bias_slot))]
                cache_data['b_off'] = self._b_off
            self._atomic_torch_save(cache_data, cached_path)
            file_size_mb = cached_path.stat().st_size / (1024 * 1024)
            self._atomic_json_dump({
                'created': datetime.now().isoformat(),
                'dataset': self.para.db,
                'layer_id': self.layer_id,
                'num_pv_entries': int(sum(getattr(self, f'_pv_{h}').numel() for h in range(n_cfgs))),
                'num_weights': int(np.sum(self.weight_num)),
                # the exact key dict behind the filename hash, so unexpected
                # cache misses can be diagnosed by diffing two sidecars
                'cache_key': self._layer_cache_key_dict(),
            }, cached_path.with_suffix('.json'))
            print(f"  Cached layer distributions ({file_size_mb:.2f} MB): {cached_path.name}")
        except Exception as e:
            print(f"⚠ Warning: Failed to save layer distribution cache to {cached_path}: {e}")

    def _indices_cache_key_dict(self, head, property_key) -> dict:
        """
        Cache key for the (indices, counts) of a specific head and property.

        Deliberately excludes thresholds and layer_id: the raw unique-pair
        indices only depend on the dataset, the label descriptions and the
        property value, so they are shared across layers, thresholds and
        network variants (threshold filtering happens after loading).
        """
        head_id = self.layer.layer_heads.index(head)
        return {
            'dataset': self.para.db,
            'dataset_size': len(self.graph_data),
            'property': self.property_descriptions[head_id],
            'property_key': str(property_key),
            'source_label': self.source_label_descriptions[head_id],
            'target_label': self.target_label_descriptions[head_id],
        }

    def get_cache_path(self, head, property_key) -> Path:
        """Generate cache path for indices/counts of a specific head and property."""
        metadata_str = json.dumps(self._indices_cache_key_dict(head, property_key), sort_keys=True)
        cache_hash = hashlib.md5(metadata_str.encode()).hexdigest()[:12]
        return self._cache_dir() / f'{cache_hash}.pt'

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
                    'indices_shape': list(indices.shape),
                    'counts_shape': list(counts.shape),
                    'num_unique_pairs': len(counts),
                    'do_invalid_indices_exist': bool(do_invalid_indices_exist),
                    # the exact key dict behind the filename hash, so unexpected
                    # cache misses can be diagnosed by diffing two sidecars
                    'cache_key': self._indices_cache_key_dict(head, property_key),
                }
            }

            # Save tensor data (atomically: paths are shared across joblib workers)
            self._atomic_torch_save(cache_data, cached_path)

            # Calculate file size
            file_size_mb = cached_path.stat().st_size / (1024 * 1024)

            # Save human-readable metadata alongside
            self._atomic_json_dump(cache_data['metadata'], cached_path.with_suffix('.json'))

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
                elif weight_initialization.get('type', None) == 'mean_aggregation':
                    # Fan-in-scaled normal WITH a nonzero (mean-aggregation) prior.
                    # A nonzero weight mean makes the invariant conv start as
                    # coherent (scaled) mean-aggregation, which is a much stronger
                    # ZINC init than zero-mean random projections. Defaults
                    # (gain=1.0, mean_ratio=-0.5) reproduce the old 'lower_upper'
                    # init exactly: mean=-1/sqrt(n), std=2/sqrt(n). See
                    # specs/14-zinc-weight-initialization.md.
                    gain = weight_initialization.get('gain', 1.0)
                    mean_ratio = weight_initialization.get('mean_ratio', -0.5)
                    std = gain * 2.0 / np.sqrt(num_weights)
                    torch.nn.init.normal_(weights, mean=mean_ratio * std, std=std)

            else:
                raise ValueError(f"Weight initialization type {init_type} is not supported")
        else:
            torch.nn.init.constant_(weights, 0.01)
        return weights

    def set_weights(self, pos:int) -> None:
        """
        Sets the weights for the graph at position pos in the graph dataset to the matrix
        :param pos:
        :return:
        """
        input_size = self._num_nodes_list[pos]
        self.current_W = torch.zeros((self.num_heads, input_size, input_size), dtype=self.precision, device=self.Param_W.device)
        heads, i_local, j_local, params, _ = self._assemble_rows([pos])
        if heads.numel() != 0:
            values = self.Param_W[params]
            if self.degree_normalization is not None:
                values = values * self._degree_norm_scale(heads, i_local, j_local, input_size)
            self.current_W[heads, i_local, j_local] = values
        return

    def set_bias(self, pos) -> None:
        """
        Sets the bias term for the graph at position pos in the graph dataset
        :param pos:
        :return:
        """
        input_size = self._num_nodes_list[pos]
        start, end = self._x_slice_list[pos], self._x_slice_list[pos + 1]
        # every head column is covered (see _build_distributions), so empty is safe
        self.current_B = torch.empty((self.num_heads, input_size, self.in_features), dtype=self.precision, device=self.Param_b.device)
        for slot, cols in self._bias_cols_by_slot.items():
            idx = getattr(self, f'_bias_idx_{slot}')[start:end].long()      # (N,)
            off = self._b_off[cols]                                          # (C, F)
            # current_B[c, node, f] = Param_b[b_off[c, f] + bias_idx[node]]
            self.current_B[cols] = self.Param_b[off[:, None, :] + idx[None, :, None]]
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
            - With degree_normalization 'row': each weight divided by the
              receiving node's aggregation count (mean aggregation)
            - With degree_normalization 'symmetric': divided by
              sqrt(out-count(i) * in-count(j)) (GCN-style)
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
        - Rows assembled on demand from the factored index structures
          (see _assemble_rows), gathered from Param_W

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
        x_in = node_representation
        if self.pre_layer_norm:
            node_representation = self.pre_norm(node_representation)
        num_nodes = self._num_nodes_list[pos]
        if self.forward_mode == 'sparse':
            # Assemble the graph's rows, sort them into coalesced order and
            # multiply as one block COO matrix of shape (H * N, N). The weight
            # values are gathered from Param_W every forward (differentiable —
            # precomputing them would silently freeze training).
            # The index structure (assembly + argsort) only depends on the
            # graph, not on the weights, so it is memoized for the last pos:
            # single-graph node tasks (and repeated evaluations of the same
            # graph) then pay only for the value gather and the sparse mm.
            cache = self._sparse_row_cache
            if cache is not None and cache[0] == pos and cache[1].device == node_representation.device:
                indices, params_sorted, scale_sorted = cache[1], cache[2], cache[3]
            else:
                heads, i_local, j_local, params, _ = self._assemble_rows([pos])
                rows = heads * num_nodes + i_local
                # (head, i, j) cells are unique per graph, so sorting by the linear
                # cell id yields a valid coalesced COO ordering
                order = torch.argsort(rows * num_nodes + j_local)
                indices = torch.stack([rows, j_local])[:, order]
                params_sorted = params[order]
                scale_sorted = None
                if self.degree_normalization is not None:
                    scale_sorted = self._degree_norm_scale(heads, i_local, j_local, num_nodes)[order]
                self._sparse_row_cache = (pos, indices, params_sorted, scale_sorted)
            values = self.Param_W[params_sorted]
            if scale_sorted is not None:
                values = values * scale_sorted
            current_W = torch.sparse_coo_tensor(indices, values,
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
        if self.residual:
            # feature-aligned with the (N, F*H) output layout (column f*H + h),
            # NOT x.repeat(1, H) which would match the PyG h*F + f layout
            node_representation = node_representation + x_in.repeat_interleave(self.num_heads, dim=1)
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
        x_in = node_representation
        if self.pre_layer_norm:
            node_representation = self.pre_norm(node_representation)
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
        if self.residual:
            out = out + x_in.repeat_interleave(self.num_heads, dim=1)
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

        heads, i_local, j_local, params, graph_slot = self._assemble_rows(positions)
        # node-major rows: the block-diagonal COO row of graph g's cell is
        # (i + node_offset_g) * H + head, its column j + node_offset_g
        node_i = i_local + offsets[graph_slot]
        node_j = j_local + offsets[graph_slot]
        rows = node_i * num_heads + heads
        cols = node_j
        # cells are unique per graph, so sorting by the linear id gives a
        # valid coalesced ordering
        order = torch.argsort(rows * total_nodes + cols)

        # gathering from Param_W every forward keeps the graph differentiable
        values = self.Param_W[params[order]]
        if self.degree_normalization is not None:
            values = values * self._degree_norm_scale(heads, node_i, node_j, total_nodes)[order]
        current_W = torch.sparse_coo_tensor(torch.stack([rows, cols])[:, order], values,
                                            (total_nodes * num_heads, total_nodes),
                                            is_coalesced=True)
        return torch.sparse.mm(current_W, node_representation).view(total_nodes, num_heads, -1)

    def _batched_dense_messages(self, node_representation: torch.Tensor, positions: List[int],
                                sizes: List[int], node_offsets: List[int]) -> torch.Tensor:
        """
        Pad the batch to (B, H, N_max, N_max) @ (B, 1, N_max, F) -> (N_total, H, F).

        Uses the assembled (head, i, j, param_idx) rows of the batch graphs.
        The padded rows/columns stay zero and are dropped again when the
        per-node outputs are gathered back into the concatenated (N_total, ...)
        layout, so the padding cannot leak between graphs.
        """
        device = node_representation.device
        num_heads = self.num_heads
        batch_size = len(positions)
        max_nodes = max(sizes, default=0)
        total_nodes = node_offsets[-1]

        heads, i_local, j_local, params, graph_slot = self._assemble_rows(positions)

        # gathering from Param_W every forward keeps the graph differentiable
        current_W = torch.zeros((batch_size, num_heads, max_nodes, max_nodes),
                                dtype=self.precision, device=device)
        values = self.Param_W[params]
        if self.degree_normalization is not None:
            offsets = torch.as_tensor(node_offsets[:-1], dtype=torch.int64, device=device)
            values = values * self._degree_norm_scale(heads, i_local + offsets[graph_slot],
                                                      j_local + offsets[graph_slot], total_nodes)
        current_W[graph_slot, heads, i_local, j_local] = values

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
        """Gather Param_b into the concatenated (N_total, H, F) bias tensor."""
        positions_t = torch.as_tensor(positions, dtype=torch.int64, device=device)
        node_gather, _ = range_gather(self._x_slices, positions_t)
        # every head column is covered (see _build_distributions)
        current_B = torch.empty((node_offsets[-1], self.num_heads, self.in_features),
                                dtype=self.precision, device=device)
        for slot, cols in self._bias_cols_by_slot.items():
            idx = getattr(self, f'_bias_idx_{slot}')[node_gather].long()   # (N_total,)
            off = self._b_off[cols]                                         # (C, F)
            # current_B[node, c, f] = Param_b[b_off[c, f] + bias_idx[node]]
            current_B[:, cols, :] = self.Param_b[idx[:, None, None] + off[None, :, :]]
        return current_B


    def get_weights(self):
        # return the weights as a numpy array
        return self.Param_W.detach().cpu().numpy()

    def get_graph_weights(self, graph_id):
        """(K, 4) rows [head, i_local, j_local, param_idx] for one graph (assembled on demand)."""
        heads, i_local, j_local, params, _ = self._assemble_rows([graph_id])
        return torch.stack([heads, i_local, j_local, params], dim=1)

    def get_bias(self):
        if self.bias:
            return self.Param_b.detach().cpu().numpy()
        else:
            return None

    def _primary_node_labels(self):
        """Flat per-node primary label tensor and its number of unique labels."""
        labels = self.graph_data.node_labels['primary']
        if isinstance(labels, NodeLabels):
            return labels.node_labels, labels.num_unique_node_labels
        if isinstance(labels, torch.Tensor):
            return labels, torch.unique(labels).size(0)
        raise ValueError("Node labels are not of type NodeLabels or torch.Tensor")

    def draw(self, ax, graph_id, graph_drawing: Tuple[GraphDrawing, GraphDrawing], head=0, filter_weights=None, with_graph=True, graph_only=False, draw_bias_labels=False, pos_path:str=''):
        graph = self.graph_data.create_nx_graph(graph_id, directed=False)
        node_offset = int(self.graph_data.slices['x'][graph_id])
        node_end = int(self.graph_data.slices['x'][graph_id + 1])
        primary_labels, num_unique_node_labels = self._primary_node_labels()

        # the circle layout starts its walk at the node with primary label 0
        root_node = None
        if graph_drawing[0].draw_type == 'circle':
            for node in graph.nodes():
                if primary_labels[node_offset + node] == 0:
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
            if draw_bias_labels:
                bias_labels = self.graph_data.node_labels[self.bias_label_descriptions[head]]
                graph_node_labels = bias_labels.node_labels[node_offset:node_end]
                num_unique_node_labels = bias_labels.num_unique_node_labels
            else:
                graph_node_labels = primary_labels[node_offset:node_end]
            cmap = graph_drawing[0].colormap
            norm = matplotlib.colors.Normalize(vmin=0, vmax=num_unique_node_labels)
            node_colors = [cmap(norm(graph_node_labels[node])) for node in graph.nodes()]
            nx.draw_networkx_nodes(graph, pos=pos, ax=ax, node_color=node_colors,
                                   node_size=graph_drawing[0].node_size)
            return
        if with_graph:
            nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=graph_drawing[1].edge_color,
                                   width=graph_drawing[1].edge_width, alpha=graph_drawing[1].edge_alpha*0.5)

        cmap = graph_drawing[1].colormap
        all_weights = self.get_weights()
        # (K, 4) rows [head, i_local, j_local, param_idx] of this graph
        rows = self.get_graph_weights(graph_id).cpu().numpy()
        graph_weights = all_weights[rows[:, 3]]
        weights = filter_weight_bounds(graph_weights, filter_weights)

        weight_min = float(np.min(graph_weights)) if graph_weights.size else 0.0
        weight_max = float(np.max(graph_weights)) if graph_weights.size else 0.0
        weight_max_abs = max(abs(weight_min), abs(weight_max))
        weight_range = weight_max - weight_min
        if weight_range > 0:
            normed_weight = (graph_weights - weight_min) / weight_range
        else:
            normed_weight = np.full_like(graph_weights, 0.5)
        weight_colors = cmap(normed_weight)

        digraph = nx.DiGraph()
        digraph.add_nodes_from(graph.nodes())

        if self.bias:
            bias = self.get_bias()
            bias_min = float(np.min(bias))
            bias_max = float(np.max(bias))
            bias_max_abs = max(abs(bias_min), abs(bias_max))
            bias_range = bias_max - bias_min
            normed_bias = (bias - bias_min) / bias_range if bias_range > 0 else np.full_like(bias, 0.5)
            bias_colors = cmap(normed_bias)
            bias_labels = self.graph_data.node_labels[self.bias_label_descriptions[head]].node_labels[node_offset:node_end]
            # per-node bias parameter of the drawn head (feature 0):
            # Param_b[b_off[head, 0] + bias_label] (see _build_distributions)
            bias_param_idx = int(self._b_off[head, 0]) + bias_labels.cpu().numpy().astype(np.int64)
            node_param_idx = bias_param_idx[list(digraph.nodes())]
            node_colors = bias_colors[node_param_idx]
            if bias_max_abs > 0:
                node_sizes = graph_drawing[1].node_size * np.abs(bias[node_param_idx]) / bias_max_abs
            else:
                node_sizes = np.full(len(node_param_idx), graph_drawing[1].node_size)
            nx.draw_networkx_nodes(digraph, pos=pos, ax=ax, node_color=node_colors, node_size=list(node_sizes))

        # one drawn edge per (i, j) pair of the selected head; if several
        # parameters share a pair the last row wins (matches the previous
        # digraph.add_edge overwrite behavior)
        selected = np.flatnonzero((rows[:, 0] == head) & (weights != 0))
        edge_rows = {(int(rows[row_id, 1]), int(rows[row_id, 2])): row_id for row_id in selected}
        if edge_rows and weight_max_abs > 0:
            edge_list = list(edge_rows.keys())
            row_ids = np.fromiter(edge_rows.values(), dtype=np.int64)
            edge_colors = weight_colors[row_ids]
            edge_widths = graph_drawing[1].weight_edge_width * np.abs(weights[row_ids]) / weight_max_abs
            digraph.add_edges_from(edge_list)
            nx.draw_networkx_edges(digraph, pos, ax=ax, edgelist=edge_list, edge_color=edge_colors,
                                   width=list(edge_widths),
                                   connectionstyle='arc3, rad = 0.25', arrows=True,
                                   arrowsize=graph_drawing[1].arrow_size, node_size=graph_drawing[1].node_size)
