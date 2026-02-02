from typing import List

import networkx as nx
import numpy as np

def EvenPairs(data_size=1500, max_size=40, seed=764,*args, **kwargs) -> tuple[List[nx.Graph], List[int]]:
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
        valid = rand_sequence[0] == rand_sequence[-1]
        labels.append(valid)
    return graphs, labels

def ParityCheck(data_size=1500, max_size=40, seed=764,*args, **kwargs) -> tuple[List[nx.Graph], List[int]]:
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
        # count number of 1s in the sequence
        even = np.count_nonzero(rand_sequence) % 2
        labels.append(even)
    return graphs, labels


def FirstChar(data_size=1500, max_size=40, seed=764,*args, **kwargs) -> tuple[List[nx.Graph], List[int]]:
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
        # check whether first and last node have the same label 0
        valid = rand_sequence[0]
        labels.append(valid)
    return graphs, labels


def LastChar(data_size=1500, max_size=40, seed=764,*args, **kwargs) -> tuple[List[nx.Graph], List[int]]:
    graphs = []
    labels = []
    import numpy as np
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
        # check whether last node is 1
        valid = rand_sequence[-1]
        labels.append(valid)
    return graphs, labels