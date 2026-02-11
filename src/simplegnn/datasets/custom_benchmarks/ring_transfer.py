from typing import List

import networkx as nx
import numpy as np


def RingTransfer(data_size, node_dimension, ring_size, seed)-> tuple[List[nx.Graph], List[np.ndarray[float]]]:
    """
    # generate data_size number of ring graphs
    """
    graphs = []
    labels = []
    # seed numpy
    np.random.seed(seed)
    while len(graphs) < data_size:
        G = nx.Graph()
        for j in range(0, ring_size):
            G.add_node(j, label=0)
        for j in range(0, ring_size):
            G.add_edge(j % ring_size, (j + 1) % ring_size)
        # permute the Ids of the nodes
        random_permutation = np.random.permutation(ring_size)
        G = nx.relabel_nodes(G, {i: random_permutation[i] for i in range(ring_size)})
        # set all graph node labels to 0
        for node in G.nodes():
            G.nodes[node]['primary_node_labels'] = 0
        # get a random node and the one on the opposite and the one on 90 degree and 270 and assign random labels from the list {1,2,3,4}
        pos = np.random.randint(0, ring_size)
        node_0 = random_permutation[pos]
        node_1 = random_permutation[(pos + ring_size // 4) % ring_size]
        # feature matrix
        feature_matrix = np.ones((ring_size, node_dimension))
        feature_matrix[node_0] = np.random.rand(node_dimension)
        feature_matrix[node_1] = np.random.rand(node_dimension)
        # for each node add flattened feature matrix as 'attribute' to the node
        for node in G.nodes():
            G.nodes[node]['attribute'] = feature_matrix[node].flatten()
        # add the flattened feature matrix as graph label
        labels.append(feature_matrix[node_0] + feature_matrix[node_1])
        graphs.append(G)
    return graphs, labels
