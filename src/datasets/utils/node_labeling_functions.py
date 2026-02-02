from typing import List, Optional

import networkx as nx

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

def weisfeiler_lehman_node_labeling(graphs: List[nx.Graph], depth: int = 3, labeled: bool = False, base_labels: Optional[dict] = None, with_edge_labels: bool = False):
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