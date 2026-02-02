from typing import List

import networkx as nx
import numpy as np


def NodeClassificationTest(data_size=1, max_size=1000, num_node_features=1,seed=764,*args, **kwargs) -> (List[nx.Graph], List[int]):
    graphs = []
    labels = []
    np.random.seed(seed)
    for i in range(data_size):
        size = np.random.randint(1, max_size + 1)
        G = nx.Graph()
        for j in range(size):
            G.add_node(j, primary_node_labels=0)
        for j in range(size - 1):
            G.add_edge(j, j + 1)
        # create random seqeunce of size size of 0s and 1s
        rand_sequence = np.random.randint(0, 2, size)
        # assign the labels to the nodes
        for j in range(size):
            G.nodes[j]['primary_node_labels'] = rand_sequence[j]
        graphs.append(G)
        # check wheter first and last node have the same label 0
        valid = rand_sequence[0]
        labels.append(valid)
    return graphs, labels
