import gzip
import os
import pickle
import time
from pathlib import Path

import networkx as nx
import torch
import yaml
import copy

from simplegnn.datasets.graph_dataset import GraphDataset
from simplegnn.datasets.utils.node_labeling import load_labels
from simplegnn.utils.utils import convert_to_list


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
        distances = {}
        slices_dict = {}
        for graph_id, graph in enumerate(graph_data.nx_graphs):
            if graph_id % 100 == 0:
                print(f"Processing graph {graph_id} of {len(graph_data.nx_graphs)}")
            for key in slices_dict:
                slices_dict[key].append(slices_dict[key][-1])
            d = dict(nx.all_pairs_shortest_path_length(graph, cutoff=cutoff))
            # use d to make a dictionary of pairs for each distance
            for node_1, other_nodes in d.items():
                for node_2, distance in other_nodes.items():
                    if distance in distances:
                        distances[distance].append([node_1+graph_data.slices['x'][graph_id].item(), node_2+graph_data.slices['x'][graph_id].item()])
                        if distance in slices_dict:
                            slices_dict[distance][-1] += 1
                    else:
                        distances[distance] = [[node_1+graph_data.slices['x'][graph_id].item(), node_2+graph_data.slices['x'][graph_id].item()]]
                        slices_dict[distance] = [0] * (graph_id + 2)
                        slices_dict[distance][-1] = 1


        valid_properties = set(distances.keys())
        final_dict = {}
        for key in valid_properties:
            final_dict[key] = torch.tensor(distances[key], dtype=torch.long)
        for key in slices_dict:
            slices_dict[key] = torch.tensor(slices_dict[key], dtype=torch.long)

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
        property_keys = {}
        slices_dict = {}
        for graph_id, graph in enumerate(graph_data.nx_graphs):
            if graph_id % 100 == 0:
                print(f"Processing graph {graph_id} of {len(graph_data.nx_graphs)}")
            for key in slices_dict:
                slices_dict[key].append(slices_dict[key][-1])
            graph_map = {}
            d = dict(nx.all_pairs_all_shortest_paths(graph))
            # replace the end nodes with the label of the edge between them
            # copy d
            d_edges = copy.deepcopy(d)
            for key, value in d.items():
                for key2, value2 in value.items():
                    for path_id, shortest_path in enumerate(value2):
                        edge_label_sequence = []
                        if len(shortest_path) == 1 or (cutoff is not None and len(shortest_path) > cutoff + 1):
                            d_edges[key].pop(key2, None)
                        else:
                            for i in range(0, len(shortest_path) - 1):
                                edge_start = shortest_path[i]
                                edge_end = shortest_path[i + 1]
                                # get the label of the edge
                                edge_label = graph[edge_start][edge_end]['primary_edge_labels']
                                edge_label_sequence.append(edge_label)
                                try:
                                    isinstance(edge_label, int)
                                except:
                                    raise ValueError("Edge label is not an integer.")

                            d_edges[key][key2][path_id] = edge_label_sequence
            for start_node in graph.nodes:
                for end_node in graph.nodes:
                    if start_node in d_edges and end_node in d_edges[start_node]:
                        paths = d[start_node][end_node]
                        paths_labels = d_edges[start_node][end_node]
                        distance = len(paths[0]) - 1
                        number_of_paths = len(paths)
                        label_occurrences = []
                        for path in paths_labels:
                            for label in path:
                                label = int(label)
                                while len(label_occurrences) <= label:
                                    label_occurrences.append(0)
                                label_occurrences[label] += 1
                        label_tuple = (distance, number_of_paths, tuple(label_occurrences))
                        if label_tuple in property_keys:
                            property_keys[label_tuple].append([start_node + graph_data.slices['x'][graph_id].item(),
                                                        end_node + graph_data.slices['x'][graph_id].item()])
                            if label_tuple in slices_dict:
                                slices_dict[label_tuple][-1] += 1
                        else:
                            property_keys[label_tuple] = [[start_node + graph_data.slices['x'][graph_id].item(),
                                                    end_node + graph_data.slices['x'][graph_id].item()]]
                            slices_dict[label_tuple] = [0] * (graph_id + 2)
                            slices_dict[label_tuple][-1] = 1

        valid_properties = set(property_keys.keys())
        final_dict = {}
        for key in valid_properties:
            final_dict[key] = torch.tensor(property_keys[key], dtype=torch.long)
        for key in slices_dict:
            slices_dict[key] = torch.tensor(slices_dict[key], dtype=torch.long)

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
