import gzip
import os
import pickle
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import yaml
from scipy.sparse.csgraph import shortest_path

from simplegnn.datasets.graph_dataset import GraphDataset
from simplegnn.datasets.utils.node_labeling import load_labels
from simplegnn.utils.utils import convert_to_list


def _finalize_property_dicts(pair_chunks: dict, counts: dict):
    """
    Convert per-graph pair chunks and per-graph counts into the stored format:
    key -> (num_pairs, 2) long tensor and key -> cumulative-count tensor of
    length num_graphs + 1.
    """
    valid_properties = set(pair_chunks.keys())
    final_dict = {key: torch.from_numpy(np.concatenate(chunks).astype(np.int64))
                  for key, chunks in pair_chunks.items()}
    slices_dict = {key: torch.from_numpy(np.concatenate(([0], np.cumsum(graph_counts))).astype(np.int64))
                   for key, graph_counts in counts.items()}
    return valid_properties, final_dict, slices_dict


def write_distance_properties(graph_data:GraphDataset, cutoff=None, out_path: Path = Path(), save_times=None) -> None:
    l = 'distances'
    if cutoff is not None:
        l += f"_cutoff_{cutoff}"
    out = out_path.joinpath(f"{graph_data.name}_properties_{l}.pt")
    out_yml = out_path.joinpath(f"{graph_data.name}_properties_{l}.yml")
    # check if the files already exists and if not create it
    if not os.path.exists(out) or not os.path.exists(out_yml):
        if graph_data.nx_graphs is None:
            graph_data.create_nx_graphs(directed=False)
        start_time = time.time()
        num_graphs = len(graph_data.nx_graphs)
        pair_chunks = {}   # distance -> list of (num_pairs, 2) arrays, one per graph
        counts = {}        # distance -> per-graph pair counts
        for graph_id, graph in enumerate(graph_data.nx_graphs):
            if graph_id % 100 == 0:
                print(f"Processing graph {graph_id} of {num_graphs}")
            if graph.number_of_nodes() == 0:
                continue
            offset = int(graph_data.slices['x'][graph_id])
            adjacency = nx.to_scipy_sparse_array(graph, format='csr')
            dist = shortest_path(adjacency, directed=False, unweighted=True)
            reachable = np.isfinite(dist)
            if cutoff is not None:
                reachable &= dist <= cutoff
            src, dst = np.nonzero(reachable)
            dist_values = dist[src, dst].astype(np.int64)
            for distance in np.unique(dist_values):
                sel = dist_values == distance
                pairs = np.stack([src[sel] + offset, dst[sel] + offset], axis=1)
                key = int(distance)
                pair_chunks.setdefault(key, []).append(pairs)
                counts.setdefault(key, np.zeros(num_graphs, dtype=np.int64))[graph_id] = len(pairs)

        valid_properties, final_dict, slices_dict = _finalize_property_dicts(pair_chunks, counts)

        # save list of dictionaries to a pickle file
        pickle_data = pickle.dumps((valid_properties, final_dict, slices_dict))
        # compress with gzip
        with open(out, 'wb') as f:
            f.write(gzip.compress(pickle_data))

        #fs.torch_save(
        #    (valid_properties, properties_dict), str(out)
        #)
        # save an additional .info file that stores the set of valid_properties as a yml file
        valid_properties_dict = {"valid_values": list(valid_properties), 'description': 'Distance',
                                 'list_of_values': f'{list(valid_properties)}'}
        with open(out_yml, 'w') as f:
            yaml.dump(valid_properties_dict, f)
        if save_times is not None:
            with open(save_times, 'a') as f:
                f.write(f"{graph_data.name}, distance, {time.time() - start_time}\n")
    else:
        print(f"File {out} already exists. Skipping.")


def write_distance_circle_properties(graph_data:GraphDataset, label_path, db_name, cutoff, out_path:Path = Path(), save_times=None) -> None:
    out = out_path.joinpath(f"{db_name}_circle_distances.prop")
    out_yml = out_path.joinpath(f"{db_name}_circle_distances.yml")
    # check if the file already exists and if not create it
    if not os.path.exists(out) or not os.path.exists(out_yml):
        distances = []
        circle_labels = load_labels(f"{label_path}{db_name}_cycles_20_labels.txt")
        label_combinations = circle_labels.num_unique_node_labels ** 2
        valid_properties = set()
        start_time = time.time()
        for graph_id, graph in enumerate(graph_data.graphs):
            if graph_id % 100 == 0:
                print(f"Processing graph {graph_id} of {len(graph_data.nx_graphs)}")
            d = dict(nx.all_pairs_shortest_path_length(graph, cutoff=cutoff))
            # use d to make a dictionary of pairs for each distance
            new_d = {}
            for key, value in d.items():
                for key2, value2 in value.items():
                    if value2 in new_d:
                        new_d[value2].append((key, key2))
                    else:
                        new_d[value2] = [(key, key2)]
                pass
            distances.append(new_d)
            for key in new_d.keys():
                valid_properties.add(key)

        final_properties = []
        valid_properties.clear()

        for graph_id, graph in enumerate(graph_data.graphs):
            final_dict = {}
            for key, value in distances[graph_id].items():
                for (i, j) in value:
                    label_i = circle_labels.node_labels[graph_id][i]
                    label_j = circle_labels.node_labels[graph_id][j]
                    # determine the final label
                    final_label = (key, label_i, label_j)
                    if final_label in final_dict:
                        final_dict[final_label].append((i, j))
                    else:
                        final_dict[final_label] = [(i, j)]
                    valid_properties.add(final_label)
            final_properties.append(final_dict)

        # sort valid properties by tuple 1,2,3 entries
        valid_properties = sorted(valid_properties, key=lambda x: (x[0], x[1], x[2]))
        # save list of dictionaries to a pickle file
        pickle_data = pickle.dumps(final_properties)

        # compress with gzip
        with open(out, 'wb') as f:
            f.write(gzip.compress(pickle_data))
        v_properties = [convert_to_list(x) for x in valid_properties]
        circle_properties = [convert_to_list(x) for x in valid_properties if x[1] == 1 and x[2] == 1]
        no_circle_properties = [convert_to_list(x) for x in valid_properties if x[1] == 0 and x[2] == 0]
        in_circle_properties = [convert_to_list(x) for x in valid_properties if x[1] == 0 and x[2] == 1]
        out_circle_properties = [convert_to_list(x) for x in valid_properties if x[1] == 1 and x[2] == 0]
        # save an additional .info file that stores the set of valid_properties as a yml file
        valid_properties_dict = {"valid_values": list(v_properties), 'description': 'Distance, In cycle -> In cycle',
                                 'list_of_values': f'{valid_properties}', 'list_of_values_circle': f'{circle_properties}', 'list_of_values_no_circle': f'{no_circle_properties}', 'list_of_values_in_circle': f'{in_circle_properties}', 'list_of_values_out_circle': f'{out_circle_properties}'}
        with open(out_path.joinpath(f"{db_name}_circle_distances.yml"), 'w') as f:
            yaml.dump(valid_properties_dict, f)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{db_name}, circle_distance, {time.time() - start_time}\n")
            except:
                print("Could not write to file")
                pass
    else:
        print(f"File {out} already exists. Skipping.")



def _edge_label_distance_keys(graph: nx.Graph, cutoff=None):
    """
    For every ordered node pair (s, t), compute
    (distance, #shortest paths, per-label edge occurrence counts summed over
    all shortest s-t paths) via one BFS per source with a DP over the
    shortest-path DAG — no path enumeration.

    Yields (s, t, key) with key = (distance, number_of_paths, label_tuple)
    where label_tuple is positional by label value, trimmed of trailing zeros
    (the same key format the path-enumerating implementation produced).
    """
    nodes = list(graph.nodes())
    n = len(nodes)
    index = {node: i for i, node in enumerate(nodes)}
    neighbors = [[] for _ in range(n)]
    max_label = 0
    for u, v, data in graph.edges(data=True):
        label = data['primary_edge_labels']
        if int(label) != label:
            raise ValueError("Edge label is not an integer.")
        label = int(label)
        if label < 0:
            raise ValueError("Edge labels must be non-negative integers.")
        max_label = max(max_label, label)
        neighbors[index[u]].append((index[v], label))
        neighbors[index[v]].append((index[u], label))
    width = max_label + 1

    for s in range(n):
        dist = np.full(n, -1, dtype=np.int64)
        sigma = np.zeros(n, dtype=np.int64)              # number of shortest paths s -> v
        label_counts = np.zeros((n, width), dtype=np.int64)  # summed label occurrences over those paths
        dist[s] = 0
        sigma[s] = 1
        queue = [s]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            if cutoff is not None and dist[u] >= cutoff:
                continue
            for v, label in neighbors[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                if dist[v] == dist[u] + 1:
                    sigma[v] += sigma[u]
                    label_counts[v] += label_counts[u]
                    label_counts[v, label] += sigma[u]
        for t in range(n):
            if t == s or dist[t] == -1:
                continue
            occurrences = label_counts[t]
            last = np.nonzero(occurrences)[0][-1]
            key = (int(dist[t]), int(sigma[t]), tuple(int(x) for x in occurrences[:last + 1]))
            yield nodes[s], nodes[t], key


def write_distance_edge_properties(graph_data:GraphDataset, out_path:Path = Path(), cutoff=None, save_times=None) -> None:
    l = 'edge_label_distances'
    if cutoff is not None:
        l += f"_cutoff_{cutoff}"
    out = out_path.joinpath(f"{graph_data.name}_properties_{l}.pt")
    out_yml = out_path.joinpath(f"{graph_data.name}_properties_{l}.yml")
    # check if the file already exists and if not create it
    if not os.path.exists(out) or not os.path.exists(out_yml):
        if graph_data.nx_graphs is None:
            graph_data.create_nx_graphs(directed=False)
        start_time = time.time()
        num_graphs = len(graph_data.nx_graphs)
        pair_chunks = {}   # key -> list of (num_pairs, 2) arrays
        counts = {}        # key -> per-graph pair counts
        for graph_id, graph in enumerate(graph_data.nx_graphs):
            if graph_id % 100 == 0:
                print(f"Processing graph {graph_id} of {num_graphs}")
            offset = int(graph_data.slices['x'][graph_id])
            graph_pairs = {}
            for start_node, end_node, key in _edge_label_distance_keys(graph, cutoff=cutoff):
                graph_pairs.setdefault(key, []).append([start_node + offset, end_node + offset])
            for key, pairs in graph_pairs.items():
                pair_chunks.setdefault(key, []).append(np.asarray(pairs, dtype=np.int64))
                counts.setdefault(key, np.zeros(num_graphs, dtype=np.int64))[graph_id] = len(pairs)

        valid_properties, final_dict, slices_dict = _finalize_property_dicts(pair_chunks, counts)

        # save list of dictionaries to a pickle file
        pickle_data = pickle.dumps((valid_properties, final_dict, slices_dict))
        # compress with gzip
        with open(out, 'wb') as f:
            f.write(gzip.compress(pickle_data))

        #fs.torch_save(
        #    (valid_properties, properties_dict), str(out)
        #)
        # save an additional .info file that stores the set of valid_properties as a yml file
        valid_properties_dict = {"valid_values": list(valid_properties), 'description': 'Distance',
                                 'list_of_values': f'{list(valid_properties)}'}
        with open(out_yml, 'w') as f:
            yaml.dump(valid_properties_dict, f)
        if save_times is not None:
            with open(save_times, 'a') as f:
                f.write(f"{graph_data.name}, edge_label_distance, {time.time() - start_time}\n")
    else:
        print(f"File {out} already exists. Skipping.")
