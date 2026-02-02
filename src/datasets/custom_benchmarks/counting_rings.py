from typing import List

import networkx as nx


def CountingRings(data_size=1000, ring_size=3, min_rings=0, max_rings=9, seed=42, *args, **kwargs) -> (List[nx.Graph], List[int]):
    ring_counts = list(range(min_rings, max_rings+1))
    mean_graph_size = max_rings*ring_size
    graphs = []
    labels = []
    for num_rings in ring_counts:
        for graph_id in range(data_size//len(ring_counts)):
            # add num_rings rings to the graph
            rings = [nx.cycle_graph(ring_size) for _ in range(num_rings)]
            # if len(rings) == 0: create a random tree with mean size of mean_graph_size
            if num_rings == 0:
                tree_size = np.random.randint(mean_graph_size -mean_graph_size//2, mean_graph_size + mean_graph_size//2)
                graphs.append(nx.random_tree(tree_size))
                labels.append(0)
            else:
                current_graph = rings[0]
                # iteratively take randomly a node or an edge and glue a new ring to it
                for glueing in range(1, num_rings):
                    rand_int = np.random.randint(0, 2)
                    if rand_int == 0:
                        # get a random node from the current graph
                        rand_node_current = np.random.randint(0, len(current_graph.nodes))
                        rand_node_ring = np.random.randint(0, len(rings[glueing].nodes))
                        current_graph = glue_graphs(current_graph, rings[glueing], rand_node_current, rand_node_ring, plot=False)
                    else:
                        # get a random edge from the current graph
                        rand_edge_current = np.random.randint(0, len(current_graph.edges))
                        rand_edge_ring = np.random.randint(0, len(rings[glueing].edges))
                        current_graph = glue_graphs_edge(current_graph, rings[glueing], rand_edge_current, rand_edge_ring, plot=False)

                # get number of nodes in the current graph
                num_nodes = len(current_graph.nodes)
                # get difference to mean_graph_size
                diff = mean_graph_size - num_nodes
                # if diff > 0 add random nodes to the graph
                nodes_to_add = np.random.randint(num_nodes, mean_graph_size + diff)
                mean_tree_size = 5
                num_trees = nodes_to_add // mean_tree_size
                trees = [nx.random_tree(np.random.randint(1, 2*mean_tree_size - 1)) for _ in range(num_trees)]
                for tree in trees:
                    # get random node from the current graph
                    rand_node_current = np.random.randint(0, len(current_graph.nodes))
                    rand_node_tree = np.random.randint(0, len(tree.nodes))
                    current_graph = glue_graphs(current_graph, tree, rand_node_current, rand_node_tree, plot=False)

                graphs.append(current_graph)
                labels.append(num_rings)
    return graphs, labels