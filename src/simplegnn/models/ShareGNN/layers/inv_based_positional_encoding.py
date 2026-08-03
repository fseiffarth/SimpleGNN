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


class InvariantBasedPositionalEncodingLayer(InvariantBasedLayer):
    """
    Invariant-based positional encoding layer of a ShareGNN.

    Each head maps every node to an ID derived from one node invariant (e.g.
    induced cycles, WL labels) and looks up ``num`` learned entries for that
    ID — the same semantics as ``num`` on an aggregation head: independent
    weights tied to the node's label value. A head therefore contributes
    ``num`` features per node, and the layer outputs one vector per node
    (the concatenation over heads). By default the embeddings are
    concatenated onto the incoming node features (``concatenate_input:
    True``), so the atom-type/input signal stays intact and a following
    linear layer mixes them; with ``concatenate_input: False`` the
    embeddings alone are returned (a pure node encoder).

    YAML interface::

        - { layer_type: invariant_based_positional_encoding,
            heads: [
              { num: 8, labels: { label_type: induced_cycles, min_cycle_length: 5, max_cycle_length: 10 } },
              { num: 8, labels: { label_type: wl_labeled, depth: 0 } },
            ],
          }

    **forward(x: torch.Tensor, pos) -> out: torch.Tensor**
        - **x** is the input matrix of shape (N, F).
        - **pos** is the index of the graph in the graph_data (or a list of
          indices for the batched forward; x is then the row-wise
          concatenation of those graphs' node features).
        - **out** has shape (N, F + sum_h num_h) with
          ``concatenate_input: True`` (the default), or (N, sum_h num_h)
          without the input features.
    """
    def __init__(self, parameters: Parameters, layer: Layer, graph_data: GraphDataset):
        layer.layer_dict['name'] = "Invariant Based Positional Encoding Layer"
        super(InvariantBasedPositionalEncodingLayer, self).__init__(parameters, layer, graph_data)

        self.concatenate_input = layer.layer_dict.get('concatenate_input', True)

        self.n_node_labels = []  # number of unique node labels per head
        self.node_label_descriptions = []  # node label descriptions per head
        self.n_heads_per_label = []  # learned entries per label value (= features per node) per head

        # Row layout of the flat Param_W for head h, node-ID i, entry k:
        # offset_h + i * num_h + k. Per head we keep the int32 unique-inverse
        # of the node labels (_pe_idx_{h}) plus the i-independent flat entry
        # offsets (_pe_off_{h}, shape (num_h,)), so the forward gather is a
        # single broadcasted index into Param_W.
        offset = 0
        for head_id, head in enumerate(layer.layer_heads):
            desc = layer.get_source_string(head_id)
            self.node_label_descriptions.append(desc)
            n_labels = graph_data.node_labels[desc].num_unique_node_labels
            self.n_node_labels.append(n_labels)
            self.n_heads_per_label.append(head.num)

            node_labels = graph_data.node_labels[desc].node_labels
            _, indices = torch.unique(node_labels, dim=0, return_inverse=True, sorted=False)
            # Non-persistent buffers: moved by net.to(device), kept out of state_dict
            self.register_buffer(f'_pe_idx_{head_id}', indices.to(torch.int32), persistent=False)
            self.register_buffer(f'_pe_off_{head_id}',
                                 offset + torch.arange(head.num, dtype=torch.int64),
                                 persistent=False)
            offset += head.num * n_labels

        self.weight_num = offset
        self.pe_dim = int(np.sum(self.n_heads_per_label))
        # get_model_layer sets the generic out_features/out_channels from the
        # heads key; override them here (same pattern as the conv layer)
        self.out_features = (self.in_features if self.concatenate_input else 0) + self.pe_dim
        self.out_channels = 1

        self.register_buffer('_pe_x_slices', self.graph_data.slices['x'].clone(), persistent=False)
        # per-graph offsets as plain ints (Python slicing, no device syncs)
        self._pe_slices = [int(x) for x in self.graph_data.slices['x']]

        self.Param_W = self.init_weights(self.weight_num, init_type='positional_encoding')
        # no bias: an embedding table is already a free parameter per ID
        self.bias = False

        self.forward_step_time = 0
        self.profile_layers = self.para.run_config.config.get('profile_layers', False)

    def init_weights(self, shape, init_type=None):
        num_weights = np.prod(shape)
        weights = nn.Parameter(torch.zeros(shape, dtype=self.precision), requires_grad=True)
        weight_init = self.para.run_config.config.get('weight_initialization', None)
        weight_initialization = None
        if weight_init is not None:
            weight_initialization = weight_init.get(init_type, None)
        if weight_initialization is not None:
            if weight_initialization.get('type', None) == 'uniform':
                torch.nn.init.uniform_(weights, a=weight_initialization.get('min', 0.0), b=weight_initialization.get('max', 1.0))
            elif weight_initialization.get('type', None) == 'normal':
                torch.nn.init.normal_(weights, mean=weight_initialization.get('mean', 0.0), std=weight_initialization.get('std', 1.0))
            elif weight_initialization.get('type', None) == 'constant':
                torch.nn.init.constant_(weights, weight_initialization.get('value', 0.01))
            elif weight_initialization.get('type', None) == 'lower_upper':
                lower, upper = -(1.0 / np.sqrt(num_weights)), (1.0 / np.sqrt(num_weights))
                weights = nn.Parameter(lower + torch.randn(shape, dtype=self.precision) * (upper - lower))
            else:
                raise ValueError(f"Weight initialization type {weight_initialization.get('type', None)} "
                                 f"is not supported for positional encoding")
        else:
            # Unlike the conv/pooling layers, a constant fallback would make all
            # embeddings identical (uninformative at init), so default to a
            # small zero-mean normal instead.
            torch.nn.init.normal_(weights, mean=0.0, std=0.1)
        return weights

    def export_weight_keys(self) -> dict:
        """
        Name every Param_W slot with dataset-independent keys (spec 18 B1).

        Layout (see __init__): the slot of (head h, label slot i, entry k) is
        offset_h + i * num_h + k, where i is the torch.unique inverse of the
        head's node labels. The unique VALUES are reconstructed from the
        stored inverse buffer, so the export is exact regardless of
        torch.unique's internal ordering.
        """
        from simplegnn.framework.utils.transfer import label_ids_to_hashes

        heads = []
        for head_id, num in enumerate(self.n_heads_per_label):
            description = self.node_label_descriptions[head_id]
            node_labels_obj = self.graph_data.node_labels[description]
            inverse = getattr(self, f'_pe_idx_{head_id}').detach().cpu().long()
            labels = node_labels_obj.node_labels.detach().cpu().long()
            n_labels = int(self.n_node_labels[head_id])
            values = torch.full((n_labels,), -1, dtype=torch.int64)
            values[inverse] = labels
            heads.append({
                'head_id': head_id,
                'label': description,
                'has_hashes': node_labels_obj.label_hashes is not None,
                'canonical': bool(node_labels_obj.has_canonical_hashes),
                'offset': int(getattr(self, f'_pe_off_{head_id}')[0]),
                'n_labels': n_labels,
                'num_entries': int(num),
                'label_hash': label_ids_to_hashes(values, node_labels_obj),
                'counts': torch.bincount(inverse, minlength=n_labels),
            })
        return {
            'layer_type': 'invariant_based_positional_encoding',
            'param_w_size': int(self.Param_W.numel()),
            'param_b_size': 0,
            'heads': heads,
        }

    def _gather_embeddings(self, node_gather) -> torch.Tensor:
        """
        (N_sel, pe_dim) embedding block for the selected nodes.

        node_gather is either a slice (contiguous per-graph node range) or an
        int64 index tensor (concatenated batch node ranges) — a per-node op,
        so the single-graph and batched paths share this gather.
        """
        head_embeddings = []
        for head_id, num in enumerate(self.n_heads_per_label):
            idx = getattr(self, f'_pe_idx_{head_id}')[node_gather].long()
            flat = idx[:, None] * num + getattr(self, f'_pe_off_{head_id}')[None, :]
            head_embeddings.append(self.Param_W[flat])
        return torch.cat(head_embeddings, dim=1)

    def forward(self, node_representation: torch.Tensor, batch_data: GraphDataset, *args, **kwargs):
        pos = kwargs.get('pos', 0)
        begin = time.time() if self.profile_layers else None
        if node_representation.dim() != 2:
            raise ValueError(f"Invariant based positional encoding expects 2D input (N, F), "
                             f"got shape {tuple(node_representation.shape)}")
        if is_batched_pos(pos):
            positions = torch.as_tensor([int(p) for p in pos], dtype=torch.int64,
                                        device=self._pe_x_slices.device)
            node_gather, _ = range_gather(self._pe_x_slices, positions)
        else:
            node_gather = slice(self._pe_slices[pos], self._pe_slices[pos + 1])
        embeddings = self._gather_embeddings(node_gather)
        if self.concatenate_input:
            out = torch.cat([node_representation, embeddings], dim=1)
        else:
            out = embeddings
        out = self.activation(out)
        if self.profile_layers:
            self.forward_step_time += time.time() - begin
        return out

    def get_weights(self):
        return self.Param_W.detach().cpu().numpy()

    def get_bias(self):
        return None

    def get_graph_embedding_ids(self, graph_id, head=0) -> np.ndarray:
        """Per-node embedding ID (the head's invariant class) of one graph."""
        node_range = slice(self._pe_slices[graph_id], self._pe_slices[graph_id + 1])
        return getattr(self, f'_pe_idx_{head}')[node_range].long().cpu().numpy()

    def get_graph_weights(self, graph_id, head=0, entry=0) -> np.ndarray:
        """Per-node embedding value of one graph for entry ``entry`` of ``head``.

        Nodes that share the head's invariant class share one parameter, so
        the returned vector repeats a value wherever the encoder cannot tell
        two nodes apart.
        """
        num = self.n_heads_per_label[head]
        ids = self.get_graph_embedding_ids(graph_id, head=head)
        flat = ids * num + int(getattr(self, f'_pe_off_{head}')[entry])
        return self.get_weights()[flat]

    def draw(self, ax, graph_id, graph_drawing: Tuple[GraphDrawing, GraphDrawing], head=0, out_dimension=0,
             with_graph=True, graph_only=False, color_by='weight', pos_path: str = ''):
        """Draw one graph with the node encoding of the given head.

        ``out_dimension`` selects which of the head's ``num`` learned entries
        is drawn (the head contributes ``num`` features per node).

        Modes:
        - ``graph_only=True``: only the graph, nodes colored by their primary
          node label (same reference drawing as the other invariant layers).
        - ``color_by='weight'`` (default): node color and size encode the
          learned embedding value of the selected entry.
        - ``color_by='label'``: node color encodes the head's invariant class
          ID, i.e. which nodes share an embedding, at constant node size.
        """
        if not 0 <= head < len(self.n_heads_per_label):
            raise ValueError(f"head {head} is out of range for {len(self.n_heads_per_label)} heads")
        num = self.n_heads_per_label[head]
        if not 0 <= out_dimension < num:
            raise ValueError(f"out_dimension {out_dimension} is out of range for head {head} with num={num}")
        if color_by not in ('weight', 'label'):
            raise ValueError(f"color_by must be 'weight' or 'label', got {color_by}")

        graph = self.graph_data.create_nx_graph(graph_id, directed=False)
        labels = self.graph_data.node_labels['primary']
        node_offset = self._pe_slices[graph_id]

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
                                   width=graph_drawing[1].edge_width, alpha=graph_drawing[1].edge_alpha * 0.5)

        node_list = list(graph.nodes())
        cmap = graph_drawing[1].colormap
        if color_by == 'label':
            embedding_ids = self.get_graph_embedding_ids(graph_id, head=head)
            norm = matplotlib.colors.Normalize(vmin=0, vmax=max(self.n_node_labels[head] - 1, 1))
            node_colors = [cmap(norm(embedding_ids[node])) for node in node_list]
            node_sizes = np.full(len(node_list), graph_drawing[1].node_size)
        else:
            node_weights = self.get_graph_weights(graph_id, head=head, entry=out_dimension)
            weight_min = float(np.min(node_weights)) if node_weights.size else 0.0
            weight_max = float(np.max(node_weights)) if node_weights.size else 0.0
            weight_max_abs = max(abs(weight_min), abs(weight_max))
            weight_range = weight_max - weight_min
            if weight_range > 0:
                normed_weight = (node_weights - weight_min) / weight_range
            else:
                normed_weight = np.full_like(node_weights, 0.5)
            node_colors = cmap(normed_weight)[node_list]
            if weight_max_abs > 0:
                node_sizes = graph_drawing[1].node_size * np.abs(node_weights[node_list]) / weight_max_abs
            else:
                node_sizes = np.full(len(node_list), graph_drawing[1].node_size)
        nx.draw_networkx_nodes(graph, pos=pos, ax=ax, nodelist=node_list, node_color=node_colors,
                               node_size=list(node_sizes))
