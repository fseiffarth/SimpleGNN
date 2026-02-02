from typing import List
import networkx as nx
import numpy as np


def RingDiagonals( data_size=1200, ring_size=100,*args, **kwargs) -> (List[nx.Graph], List[int]):
    """
    Create a dataset of ring graphs with diagonals.
    :param data_size: number of graphs to create
    :param ring_size: number of nodes in each ring
    :return: a list of graphs and a list of labels
    """
    graphs = []
    labels = []
    seed = 16
    np.random.seed(seed)
    class_counter = [0, 0]
    while sum(class_counter) < data_size:
        G = nx.cycle_graph(ring_size)
        # add random 1-dim labels and 3-dim features to nodes and edges
        for node in G.nodes():
            G.nodes[node]['label'] = np.random.randint(0, 2)
            G.nodes[node]['feature'] = [np.random.rand(), np.random.rand(), np.random.rand()]
        for edge in G.edges():
            G[edge[0]][edge[1]]['primary_node_labels'] = np.random.randint(0, 2)
            G[edge[0]][edge[1]]['feature'] = [np.random.rand(), np.random.rand(), np.random.rand()]

        # get two random nodes in the ring and connect them with an edge
        diag_start = np.random.randint(ring_size)
        while True:
            diag_end = np.random.randint(ring_size)
            if diag_end != diag_start:
                break
        # get the distance in the ring between the two nodes
        dist = nx.shortest_path_length(G, diag_start, diag_end)
        G.add_edge(diag_start, diag_end)
        G[diag_start][diag_end]['primary_edge_labels'] = np.random.randint(0, 2)
        G[diag_start][diag_end]['feature'] = [np.random.rand(), np.random.rand(), np.random.rand()]
        # determine the label of the graph G
        # Case 1: Edge Label of the diagonal is 1
        # Case 2: Labels of the two end nodes of the diagonal are the same
        # Case 3: Distance between the two end nodes of the diagonal greater than 25
        # => Then the graph label is 1, else 0
        graph_label = 0
        edge = G.edges[diag_start, diag_end]
        if 'primary_edge_labels' in edge and edge['primary_edge_labels'] == 1:
            graph_label = 1
        elif G.nodes[diag_start]['primary_node_labels'] == G.nodes[diag_end]['primary_node_labels']:
            graph_label = 1
        elif dist > 13:
            graph_label = 1
        if class_counter[graph_label] >= data_size / 2:
            continue
        else:
            class_counter[graph_label] += 1
            labels.append(graph_label)
            graphs.append(G)

    return graphs, labels
