from typing import List, Optional

import networkx as nx
import numpy as np

def standard_node_labeling(graphs: List[nx.Graph]):
    """
    Standard node labeling method. It gets the primary_node_labels from the graphs in graphs
    :param graphs: a list of networkx graphs
    :return: None
    """
    node_labels = []
    unique_node_labels = []
    db_unique_node_labels = {}
    for graph in graphs:
        node_labels.append([0] * len(graph.nodes))
        unique_node_labels.append({})
        for node in graph.nodes(data=True):
            # check if the node has a label
            if 'primary_node_labels' in node[1]:
                if type(node[1]['primary_node_labels']) == int or type(node[1]['primary_node_labels']) == float:
                    node_label = node[1]['primary_node_labels']
                elif len(node[1]['primary_node_labels']) > 0:
                    node_label = node[1]['primary_node_labels'][0]
                else:
                    node_label = 0
            else:
                node_label = 0
            node_labels[-1][node[0]] = node_label
            if node_label not in unique_node_labels[-1]:
                unique_node_labels[-1][node_label] = 1
            else:
                unique_node_labels[-1][node_label] += 1
            if node_label not in db_unique_node_labels:
                db_unique_node_labels[node_label] = 1
            else:
                db_unique_node_labels[node_label] += 1
    # sort the db_unique_node_labels by the key
    db_unique_node_labels = dict(sorted(db_unique_node_labels.items()))
    return node_labels, unique_node_labels, db_unique_node_labels

def degree_node_labeling(graphs: List[nx.Graph]):
    node_labels = []
    unique_node_labels = []
    db_unique_node_labels = {}
    for graph in graphs:
        node_labels.append([0]*len(graph.nodes))
        unique_node_labels.append({})
        for node in graph.nodes(data=True):
            # get degree of node
            degree = graph.degree(node[0])
            node_labels[-1][node[0]] = degree
            if degree not in unique_node_labels[-1]:
                unique_node_labels[-1][degree] = 1
            else:
                unique_node_labels[-1][degree] += 1
            if degree not in db_unique_node_labels:
                db_unique_node_labels[degree] = 1
            else:
                db_unique_node_labels[degree] += 1
    # sort the db_unique_node_labels by the key
    db_unique_node_labels = dict(sorted(db_unique_node_labels.items()))
    return node_labels, unique_node_labels, db_unique_node_labels

def _wl_color_refinement(edge_src, edge_dst, num_nodes, colors, rounds):
    """
    Exact vectorized WL color refinement on flat disjoint-union edge arrays.

    edge_src/edge_dst must contain both directions of every undirected edge.
    Returns the color partition after `rounds` refinement rounds (or earlier if
    the partition stabilizes). Produces the same partition as
    nx.weisfeiler_lehman_subgraph_hashes with iterations=rounds (nx >= 3.5
    semantics), but without building a union graph or hashing strings.
    """
    N = int(num_nodes)
    colors = np.unique(np.asarray(colors, dtype=np.int64), return_inverse=True)[1]
    if N == 0 or rounds <= 0:
        return colors
    src = np.asarray(edge_src, dtype=np.int64)
    dst = np.asarray(edge_dst, dtype=np.int64)
    deg = np.bincount(dst, minlength=N)
    order = np.argsort(dst, kind="stable")
    dst_s, src_s = dst[order], src[order]
    start = np.concatenate([[0], np.cumsum(deg)[:-1]])
    node_order = np.argsort(deg, kind="stable")
    deg_sorted = deg[node_order]
    dmax = int(deg.max()) if deg.size else 0
    C = int(colors.max()) + 1
    for _ in range(rounds):
        # per-node sorted neighbor colors via one sort of the packed key
        key = dst_s * np.int64(C) + colors[src_s]
        key.sort()
        neigh_sorted = key - dst_s * np.int64(C)
        # iterated pairwise folding in a strictly growing id namespace
        acc = colors.astype(np.int64, copy=True)
        A = C
        for k in range(dmax):
            i0 = np.searchsorted(deg_sorted, k + 1)
            sel = node_order[i0:]
            if sel.size == 0:
                break
            code = acc[sel] * np.int64(C) + neigh_sorted[start[sel] + k]
            uq, inv = np.unique(code, return_inverse=True)
            acc[sel] = A + inv
            A += uq.size
        uq, colors = np.unique(acc, return_inverse=True)
        if uq.size == C:          # partition stable -> stays stable
            break
        C = uq.size
    return colors


def _wl_labels_to_output(colors, graphs):
    """Compact colors to first-appearance ids and split them per graph."""
    label_dict = {}
    string_labels = [label_dict.setdefault(int(c), len(label_dict)) for c in colors]
    graph_node_labels = []
    unique_node_labels = []
    db_unique_node_labels = {}
    counter = 0
    for graph in graphs:
        node_number = len(graph.nodes)
        graph_node_labels.append(string_labels[counter:counter + node_number])
        unique_node_labels.append({})
        for node_label in graph_node_labels[-1]:
            unique_node_labels[-1][node_label] = unique_node_labels[-1].get(node_label, 0) + 1
            db_unique_node_labels[node_label] = db_unique_node_labels.get(node_label, 0) + 1
        counter += node_number
    return graph_node_labels, unique_node_labels, db_unique_node_labels


def weisfeiler_lehman_node_labeling(graphs: List[nx.Graph], depth: int = 3, labeled: bool = False, base_labels: Optional[dict] = None, with_edge_labels: bool = False):
    if with_edge_labels:
        # edge-labeled WL is not covered by the vectorized refinement
        return _weisfeiler_lehman_node_labeling_nx(graphs, depth=depth, labeled=labeled, base_labels=base_labels, with_edge_labels=with_edge_labels)

    # build disjoint-union numbered edge arrays straight from the nx graphs
    num_nodes = sum(len(g.nodes) for g in graphs)
    edge_src, edge_dst = [], []
    offset = 0
    for graph in graphs:
        index = {node: i + offset for i, node in enumerate(graph.nodes())}
        for u, v in graph.edges():
            ui, vi = index[u], index[v]
            edge_src.append(ui)
            edge_dst.append(vi)
            edge_src.append(vi)
            edge_dst.append(ui)
        offset += len(graph.nodes)

    if labeled:
        if base_labels is not None:
            colors = base_labels['labels'].node_labels.cpu().numpy().astype(np.int64)
        else:
            value_to_id = {}
            colors = np.empty(num_nodes, dtype=np.int64)
            i = 0
            for graph in graphs:
                for _, data in graph.nodes(data=True):
                    value = data['primary_node_labels']
                    if isinstance(value, list):
                        value = tuple(value)
                    colors[i] = value_to_id.setdefault(value, len(value_to_id))
                    i += 1
        rounds = depth + 1                       # mirrors iterations=depth+1 of the nx variant
    else:
        colors = np.zeros(num_nodes, dtype=np.int64)   # nx >= 3.5 trivial init
        rounds = depth

    final_colors = _wl_color_refinement(edge_src, edge_dst, num_nodes, colors, rounds)
    return _wl_labels_to_output(final_colors, graphs)


def _weisfeiler_lehman_node_labeling_nx(graphs: List[nx.Graph], depth: int = 3, labeled: bool = False, base_labels: Optional[dict] = None, with_edge_labels: bool = False):
    unique_node_labels = []
    db_unique_node_labels = {}
    union_graph = nx.disjoint_union_all(graphs)
    # check if the base_labels is not None
    if base_labels is not None:
        for i, node in enumerate(union_graph.nodes(data=True)):
            # add base label as node attribute
            node[1]['base_label'] = base_labels['labels'].node_labels[node[0]].item()
    hash_dict = []
    if labeled:
        if base_labels is not None:
            # use the base label as the node attribute
            if with_edge_labels:
                # use the base label as the node attribute and edge attribute
                hashes = nx.weisfeiler_lehman_subgraph_hashes(union_graph, iterations=depth+1, node_attr='base_label', edge_attr='primary_edge_labels')
            else:
                hashes = nx.weisfeiler_lehman_subgraph_hashes(union_graph, iterations=depth+1, node_attr='base_label')
        else:
            if with_edge_labels:
                # use the primary label as the node attribute and edge attribute
                hashes = nx.weisfeiler_lehman_subgraph_hashes(union_graph, iterations=depth+1, node_attr='primary_node_labels', edge_attr='primary_edge_labels')
            else:
                hashes = nx.weisfeiler_lehman_subgraph_hashes(union_graph, iterations=depth+1, node_attr='primary_node_labels')
    else:
        hashes = nx.weisfeiler_lehman_subgraph_hashes(union_graph, iterations=depth)
    largest_int = 0

    # iterate over the keys of the hashes dictionary
    for node in hashes:
        # iterate over the subgraph hashes
        for i, subgraph_hash in enumerate(hashes[node], 0):
            if len(hash_dict) <= i:
                hash_dict.append({})
            if subgraph_hash not in hash_dict[i]:
                hash_dict[i][subgraph_hash] = len(hash_dict[i])
                if len(hash_dict[i]) > largest_int:
                    largest_int = len(hash_dict[i])
    # convert hashes to dict of int list of ints
    for node in hashes:
        for i, subgraph_hash in enumerate(hashes[node], 0):
            hashes[node][i] = hash_dict[i][subgraph_hash]
    # get the digits of the largest int
    digits = len(str(largest_int))
    # int hashes to ints
    string_labels = []
    for node in hashes:
        string_labels.append(''.join([str(x).zfill(digits) for x in hashes[node]]))
    label_dict = {}
    for label in string_labels:
        if label not in label_dict:
            label_dict[label] = len(label_dict)
    for i, label in enumerate(string_labels):
        string_labels[i] = label_dict[label]
    # make graph labels from labels
    graph_node_labels = []
    counter = 0
    for graph in graphs:
        node_number = len(graph.nodes)
        graph_node_labels.append(string_labels[counter:counter + node_number])
        unique_node_labels.append({})
        for node_label in graph_node_labels[-1]:
            if node_label not in unique_node_labels[-1]:
                unique_node_labels[-1][node_label] = 1
            else:
                unique_node_labels[-1][node_label] += 1
            if node_label not in db_unique_node_labels:
                db_unique_node_labels[node_label] = 1
            else:
                db_unique_node_labels[node_label] += 1
        counter += node_number
    return graph_node_labels, unique_node_labels, db_unique_node_labels