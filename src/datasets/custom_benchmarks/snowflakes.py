'''
Created on 26.09.2019

@author:
'''

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from networkx.drawing import nx_pydot

from datasets.utils.graph_functions import glue_graphs


def PartA(plot=False):
    G = nx.Graph()
    # create 8 nodes
    for i in range(0, 8):
        G.add_node(i, label=0)
    # create edges from 0 to 1,2,3,4 from 1, 3 to 5 and from 2, 4 to 6 and from 5,6 to 7 from 2 to 3 and from 4 to 5
    G.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (3, 5), (2, 6), (4, 6), (5, 7), (6, 7), (1, 2), (3, 4)])
    if plot:
        # draw the graph G with pydot
        pos = nx_pydot.pydot_layout(G, prog='neato', root=0)
        nx.draw_networkx_nodes(G, pos)
        nx.draw_networkx_edges(G, pos)
        plt.show()
    return G


def PartB(plot=False):
    G = nx.Graph()
    # create 8 nodes
    for i in range(0, 8):
        G.add_node(i, label=0)
    # create edges from 0 to 1,2,3,4 from 1, 3 to 5 and from 2, 4 to 6 and from 5,6 to 7
    G.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (2, 5), (3, 6), (4, 6), (5, 7), (6, 7), (1, 2), (3, 4)])
    if plot:
        # draw the graph G with pydot
        pos = nx_pydot.pydot_layout(G, prog='neato', root=0)
        nx.draw_networkx_nodes(G, pos)
        nx.draw_networkx_edges(G, pos)
        plt.show()
    return G


def M0(plot=False):
    G = glue_graphs(PartA(), PartA(), 0, 0)
    if plot:
        # draw the graph G with pydot
        pos = nx_pydot.pydot_layout(G, prog='neato', root=0)
        nx.draw_networkx_nodes(G, pos)
        nx.draw_networkx_edges(G, pos)
        plt.show()
    return G


def M1(plot=False):
    # glue together snowflakeA and snowflakeB at node 0
    G = glue_graphs(PartA(), PartB(), 0, 0)
    if plot:
        # draw the graph G with pydot
        pos = nx_pydot.pydot_layout(G, prog='neato', root=0)
        nx.draw_networkx_nodes(G, pos)
        nx.draw_networkx_edges(G, pos)
        plt.show()
    return G


def M3(plot=False):
    # glue together two copies of snowflakeB at node 0
    G = glue_graphs(PartB(), PartB(), 0, 0)
    if plot:
        # draw the graph G with pydot
        pos = nx_pydot.pydot_layout(G, prog='neato', root=0)
        nx.draw_networkx_nodes(G, pos)
        nx.draw_networkx_edges(G, pos)
        plt.show()
    return G


def Snowflake(part_list=[0, 1, 2, 3], size=4, plot=False):
    '''
    Glue the parts from partlist, i.e., M0, M1, M2, M3 together to a snowflake graph
    '''
    # create ring graph
    if size == 1:
        G = nx.Graph()
        G.add_node(0)
    else:
        G = nx.circulant_graph(size, [1])
    part_dict = {0: (M0(), 7), 1: (M1(), 7), 2: (M1(), 14), 3: (M3(), 7)}
    # create snowflake graph
    for i, part in enumerate(part_list):
        G = glue_graphs(G, part_dict[part][0], i, part_dict[part][1], plot=False)
    if plot:
        # draw the graph G with pydot
        # for each graph in part list create a shell
        shell_list = []
        shells = 7
        start_idx = size
        # shells 0,1,2,3 and 4,5,21,22,38,39,55,56, ... and
        for i in range(0, shells):
            shell = []
            if i == 0:
                shell_list.append(list(range(0, size)))
            elif i == 1:
                for j in range(0, size):
                    shell.append(size + j * 14)
                    shell.append(size + j * 14 + 1)
            elif i == 2:
                for j in range(0, size):
                    shell.append(size + j * 14 + 2)
                    shell.append(size + j * 14 + 3)
                    shell.append(size + j * 14 + 4)
                    shell.append(size + j * 14 + 5)
            elif i == 3:
                for j in range(0, size):
                    shell.append(size + j * 14 + 6)
            elif i == 4:
                for j in range(0, size):
                    shell.append(size + j * 14 + 7)
                    shell.append(size + j * 14 + 8)
                    shell.append(size + j * 14 + 9)
                    shell.append(size + j * 14 + 10)
            elif i == 5:
                for j in range(0, size):
                    shell.append(size + j * 14 + 11)
                    shell.append(size + j * 14 + 12)
            elif i == 6:
                for j in range(0, size):
                    shell.append(size + j * 14 + 13)
            if i > 0:
                shell_list.append(shell)
        pos = nx.kamada_kawai_layout(G)
        nx.draw_networkx_nodes(G, pos, node_size=50)
        nx.draw_networkx_edges(G, pos)
        plt.show()
    return G


def Snowflakes(smallest_snowflake=1, largest_snowflake=20, flakes_per_size=10, plot=False, seed=4837257, generation_type='binary'):
    '''
    Create a list of snowflake graphs with sizes from smallest_snowflake to largest_snowflake
    '''
    np.random.seed(seed)
    snowflakes = []
    labels = []
    if type(flakes_per_size) == int:
        # create a list of snowflakes with sizes from smallest_snowflake to largest_snowflake
        for i in range(smallest_snowflake, largest_snowflake + 1):
            for j in range(0, flakes_per_size):
                # create random part list of size i
                part_list = np.random.randint(0, 4, i)
                snowflakes.append(Snowflake(part_list=part_list, size=i))
                if generation_type == 'count':
                    # count the number of parts in the snowflake
                    part_count = np.zeros(4)
                    for part in part_list:
                        part_count[part] += 1
                    # divide part count by largest snowflake size
                    part_count = part_count / largest_snowflake
                    # label the snowflake with the number of parts
                    labels.append(part_count)
                if generation_type == 'binary':
                    # create random list with one 1 and the rest 0 of length i
                    rand_index1 = np.random.randint(0, i)
                    rand_index2 = np.random.randint(0, i)
                    rand_index3 = np.random.randint(0, i)

                    label = (rand_index1*part_list[rand_index1] + 2 * rand_index2*part_list[rand_index2] + 3 * rand_index3*part_list[rand_index3]) % 2

                    rand_index2 = i + 14 * rand_index2
                    rand_index3 = i + 7 + 14 * rand_index3
                    # add node labels to the last graph in snowflakes
                    for node in snowflakes[-1].nodes():
                        if node == rand_index1:
                            snowflakes[-1].nodes[node]['primary_node_labels'] = 1
                        elif node >= i:
                            # set node to random label between 0 and 4
                            rand_node_label = np.random.randint(0, 1)
                            snowflakes[-1].nodes[node]['primary_node_labels'] = rand_node_label
                        #elif node == rand_index2:
                        #    snowflakes[-1].nodes[node]['primary_node_labels'] = 1
                        #elif node == rand_index3:
                        #    snowflakes[-1].nodes[node]['primary_node_labels'] = 1
                        else:
                            rand_node_label = np.random.randint(0, 1)
                            snowflakes[-1].nodes[node]['primary_node_labels'] = rand_node_label
                    label = part_list[rand_index1]
                    labels.append(label)

    if type(flakes_per_size) == list:
        # create a list of snowflakes with sizes from smallest_snowflake to largest_snowflake
        for i in range(smallest_snowflake, largest_snowflake + 1):
            for j in range(0, flakes_per_size[i - smallest_snowflake]):
                # create random part list of size i
                part_list = np.random.randint(0, 4, i)
                snowflakes.append(Snowflake(part_list=part_list, size=i))
                # count the number of parts in the snowflake
                part_count = np.zeros(4)
                for part in part_list:
                    part_count[part] += 1
                # divide part count by largest snowflake size
                part_count = part_count / largest_snowflake
                # label the snowflake with the number of parts
                labels.append(part_count)
    if flakes_per_size == 'all':
        for i in range(smallest_snowflake, largest_snowflake + 1):
            # create 1 million part counts and get the unique ones
            part_counts = set()
            for j in range(0, 1000000):
                # create random part list of size i
                part_list = np.random.randint(0, 4, i)
                # count the number of parts in the snowflake
                part_count = np.zeros(4)
                for part in part_list:
                    part_count[part] += 1
                part_counts.add(tuple(part_count))

            # part_counts to list of numpy arrays
            part_counts = [np.array(part_count) for part_count in part_counts]
            # divide part count by largest snowflake size
            part_count = part_count / largest_snowflake
            for part_count in part_counts:
                # make part list from part count
                part_list = []


            # get all
            for j in range(0, 4 ** i):
                # create random part list of size i
                part_list = np.random.randint(0, 4, i)
                snowflakes.append(Snowflake(part_list=part_list, size=i))
                # count the number of parts in the snowflake
                part_count = np.zeros(4)
                for part in part_list:
                    part_count[part] += 1
                # label the snowflake with the number of parts
                labels.append(part_count)
    if plot:
        # plot all snowflakes
        for i, snowflake in enumerate(snowflakes):
            # label as title
            label = labels[i]
            plt.title(str(label))
            pos = nx.kamada_kawai_layout(snowflake)
            nx.draw_networkx_nodes(snowflake, pos, node_size=50)
            nx.draw_networkx_edges(snowflake, pos)
            nx.draw_networkx_labels(snowflake, pos, font_size=8, labels={node: snowflake.nodes[node]['primary_node_labels'] for node in snowflake.nodes()})
            plt.show()
    return snowflakes, labels


def example_graph1():
    G = nx.Graph()
    G.add_node(0, label=np.array([0]))
    G.add_node(1, label=np.array([0]))
    G.add_node(2, label=np.array([1]))
    G.add_node(3, label=np.array([1]))
    G.add_node(4, label=np.array([0]))
    G.add_node(5, label=np.array([0]))

    G.add_edges_from([(0, 2), (1, 2), (2, 3), (3, 4), (3, 5)])
    return G


def example_graph2():
    G = nx.Graph()
    G.add_node(0, label=np.array([0]))
    G.add_node(1, label=np.array([0]))
    G.add_node(2, label=np.array([1]))
    G.add_node(3, label=np.array([0]))
    G.add_node(4, label=np.array([0]))
    G.add_node(5, label=np.array([0]))

    G.add_edges_from([(0, 2), (1, 2), (2, 3), (3, 4), (3, 5)])
    return G


def circle_graph(n=100):
    G = nx.Graph()

    for i in range(0, n):
        G.add_node(i, label=np.array([i % 2]), abc=i % 2)

    for i in range(0, n):
        G.add_edge(i % n, (i + 1) % n)
    return G


def double_circle(n=50, m=50):
    G = circle_graph(n)
    for i in range(n, n + m):
        G.add_node(i, label=np.array([i % 2]), abc=i % 2)

    for i in range(n, n + m):
        G.add_edge(i % m, (i + 1) % m)
    return G


def main():
    #Snowflake(part_list=[0, 1, 2, 3], size=4, plot=True)
    Snowflakes(smallest_snowflake=3, largest_snowflake=10, flakes_per_size=10, plot=True, seed=764, generation_type='binary')


if __name__ == '__main__':
    main()
