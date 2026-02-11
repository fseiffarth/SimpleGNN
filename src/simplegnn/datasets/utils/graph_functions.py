import networkx as nx
from matplotlib import pyplot as plt


def glue_graphs(G1, G2, node1, node2, plot=False):
    '''
    Glue together two graphs G1 and G2 and replace node2 in G2 with node1 in G1
    '''
    G = nx.Graph()
    # add nodes from G1
    for i, node in enumerate(G1.nodes()):
        # check if node is labeled
        if 'primary_node_labels' in G1.nodes[node]:
            G.add_node(i, label=G1.nodes[node]['primary_node_labels'])
        else:
            G.add_node(i)
    # add edges from G1
    for edge in G1.edges():
        G.add_edge(edge[0], edge[1])
    # create a node map for G2
    node_map = {}
    for i, node in enumerate(G2.nodes()):
        if node == node2:
            node_map[node] = node1
        else:
            if node < node2:
                node_map[node] = i + G.number_of_nodes()
            else:
                node_map[node] = i + G.number_of_nodes() - 1
    for edge in G2.edges():
        G.add_edge(node_map[edge[0]], node_map[edge[1]])
    if plot:
        # draw the graph G with pydot draw G1 in blue and G2 in red
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, nodelist=range(0, G1.number_of_nodes()), pos=pos, node_color='b')
        nx.draw_networkx_nodes(G, nodelist=range(G1.number_of_nodes(), G.number_of_nodes()), pos=pos, node_color='r')
        nx.draw_networkx_labels(G, pos, labels={node: node for node in G.nodes()})
        nx.draw_networkx_edges(G, pos)
        plt.show()
    return G

def glue_graphs_edge(G1, G2, edge1, edge2, plot=False):
    '''
    Glue together two graphs G1 and G2 and replace edge2 in G2 with edge1 in G1
    '''
    G = nx.Graph()
    # add nodes from G1
    for i, node in enumerate(G1.nodes()):
        # check if node is labeled
        if 'primary_node_labels' in G1.nodes[node]:
            G.add_node(i, label=G1.nodes[node]['primary_node_labels'])
        else:
            G.add_node(i)
    # add edges from G1
    end_nodes_edge_1 = [0,0]
    end_nodes_edge_2 = [0,0]
    for i, edge in enumerate(G1.edges()):
        G.add_edge(edge[0], edge[1])
        if i == edge1:
            end_nodes_edge_1 = [edge[0], edge[1]]

    end_nodes_edge_2 = [list(G2.edges)[edge2][0], list(G2.edges)[edge2][1]]
    new_node_id_map = {}
    new_node_id_map[end_nodes_edge_2[0]] = end_nodes_edge_1[0]
    new_node_id_map[end_nodes_edge_2[1]] = end_nodes_edge_1[1]
    new_node_id_map_inverse = {}
    for node in G2.nodes():
        if node not in end_nodes_edge_2:
            G.add_node(G.number_of_nodes())
            new_node_id_map[node] = G.number_of_nodes() - 1
            new_node_id_map_inverse[new_node_id_map[node]] = node
    for i, edge in enumerate(G2.edges()):
        G.add_edge(new_node_id_map[edge[0]], new_node_id_map[edge[1]])
    if plot:
        # draw the graph G with pydot draw G1 in blue and G2 in red
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, nodelist=range(0, G1.number_of_nodes()), pos=pos, node_color='b')
        nx.draw_networkx_nodes(G, nodelist=range(G1.number_of_nodes(), G.number_of_nodes()), pos=pos, node_color='r')
        nx.draw_networkx_labels(G, pos, labels={node: node for node in G.nodes()})
        nx.draw_networkx_edges(G, pos)
        plt.show()
    return G