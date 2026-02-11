from typing import List

import networkx as nx
import numpy as np


def EvenOddRings(data_size=1200, ring_size=100, difficulty=1, count=False, seed=764,*args, **kwargs) -> (List[nx.Graph], List[int]):
    """
    Create a benchmark dataset consisting of labeled rings with ring_size nodes and labels.
    The label of the graph is determined by the following:
    - Select the node with label and the node with distance ring_size//2 say x and the ones with distances ring_size//4, ring_size//8, say y_1, y_2 and z_1, z_2
    Now consider the numbers:
    a = 1 + x
    b = y_1 + y_2
    c = z_1 + z_2
    and distinct the cases odd and even. This defines the 8 possible labels of the graphs.
    """
    graphs = []
    labels = []
    # seed numpy
    np.random.seed(seed)
    class_number = 0
    permutation_storage = []
    while len(graphs) < data_size:
        G = nx.Graph()
        label_permutation = np.random.permutation(ring_size)
        for j in range(0, ring_size):
            G.add_node(j, primary_node_labels=label_permutation[j])
        for j in range(0, ring_size):
            G.add_edge(j % ring_size, (j + 1) % ring_size)
        # permute the Ids of the nodes
        random_permutation = np.random.permutation(ring_size)

        # make random permutation start with 0
        r_perm = np.roll(random_permutation, -np.where(random_permutation == 0)[0][0])
        # to list
        r_perm = r_perm.tolist()
        if r_perm not in permutation_storage:
            # add permutation to storage
            permutation_storage.append(r_perm)

            G = nx.relabel_nodes(G, {i: random_permutation[i] for i in range(ring_size)})
            if count:
                class_number = 2
                opposite_nodes = []
                for node in G.nodes(data=True):
                    node_label = node[1]['primary_node_labels']
                    node_id = node[0]
                    pos = np.where(random_permutation == node_id)[0][0]
                    # get opposite node in the ring
                    opposite_node = random_permutation[(pos + ring_size // 2) % ring_size]
                    # get opposite node label in the ring
                    opposite_node_label = G.nodes[opposite_node]['primary_node_labels']
                    # add node_label + opposite_node_label to opposite_nodes
                    opposite_nodes.append(node_label + opposite_node_label)
                # count odd and even entries in opposite_nodes
                odd_count = np.count_nonzero(np.array(opposite_nodes) % 2)
                even_count = len(opposite_nodes) - odd_count
                if odd_count > even_count:
                    label = 1
                else:
                    label = 0
            else:
                # find graph node with label 0
                for node in G.nodes(data=True):
                    if node[1]['primary_node_labels'] == 0:
                        node_0 = node[0]
                        break
                # get index of node_0 in random_permutation
                pos = np.where(random_permutation == node_0)[0][0]
                node_1 = random_permutation[(pos + ring_size // 4) % ring_size]
                node_2 = random_permutation[(pos + ring_size // 2) % ring_size]
                node_3 = random_permutation[(pos - ring_size // 4) % ring_size]
                # get the neighbors of node_0
                node_4 = random_permutation[(pos + 1) % ring_size]
                node_5 = random_permutation[(pos - 1 + ring_size) % ring_size]

                label_node_1 = G.nodes[node_1]['primary_node_labels']
                label_node_2 = G.nodes[node_2]['primary_node_labels']
                label_node_3 = G.nodes[node_3]['primary_node_labels']
                label_node_4 = G.nodes[node_4]['primary_node_labels']
                label_node_5 = G.nodes[node_5]['primary_node_labels']

                # add the labels of the nodes
                a = 0 + label_node_2
                b = label_node_1 + label_node_3
                c = label_node_4 + label_node_5

                if difficulty == 1:
                    label = a % 2
                    class_number = 2
                elif difficulty == 2:
                    label = 2 * (a % 2) + b % 2
                    class_number = 4
                elif difficulty == 3:
                    label = 4 * (a % 2) + 2 * (b % 2) + c % 2
                    class_number = 8

            # get unique label count and append if count for label is smaller than data_size//6
            unique_labels, counts = np.unique(labels, return_counts=True)
            if label not in labels or counts[unique_labels == label] < data_size // class_number:
                graphs.append(G)
                labels.append(label)
    # shuffle the graphs and labels
    perm = np.random.permutation(len(graphs))
    graphs = [graphs[i] for i in perm]
    labels = [labels[i] for i in perm]
    return graphs, labels