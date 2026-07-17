from pathlib import Path

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors


def load_positions(pos_path):
    """Load node positions from a whitespace-separated file, or None if absent.

    File format (one node per line): ``<node_id> <x> <y>``.
    """
    if pos_path is None or str(pos_path) == '' or not Path(pos_path).is_file():
        return None
    pos = dict()
    with open(pos_path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) == 3:
                pos[int(parts[0])] = (float(parts[1]), float(parts[2]))
    return pos


def save_positions(pos, pos_path):
    if pos_path is None or str(pos_path) == '':
        return
    with open(pos_path, 'w') as f:
        for key, value in pos.items():
            f.write(f"{key} {value[0]} {value[1]}\n")


def compute_positions(graph, draw_type, root_node=None):
    """Node positions for a networkx graph using the configured layout.

    ``draw_type='circle'`` walks the graph from ``root_node`` and places the
    nodes on a circle of radius 400 in visit order (intended for ring-shaped
    graphs); any node the walk cannot reach is appended in iteration order so
    the layout terminates on arbitrary graphs.
    """
    if draw_type == 'circle':
        if root_node is None or root_node not in graph:
            root_node = next(iter(graph.nodes()))
        pos = {root_node: (400.0, 0.0)}
        angle = 2 * np.pi / max(graph.number_of_nodes(), 1)
        cur_node = root_node
        last_node = None
        counter = 0
        while len(pos) < graph.number_of_nodes():
            for next_node in graph.neighbors(cur_node):
                if next_node != last_node and next_node not in pos:
                    counter += 1
                    pos[next_node] = (400 * np.cos(counter * angle), 400 * np.sin(counter * angle))
                    last_node = cur_node
                    cur_node = next_node
                    break
            else:
                for node in graph.nodes():
                    if node not in pos:
                        counter += 1
                        pos[node] = (400 * np.cos(counter * angle), 400 * np.sin(counter * angle))
                break
        return pos
    if draw_type == 'kawai':
        pos = nx.kamada_kawai_layout(graph)
    elif draw_type == 'shell':
        pos = nx.shell_layout(graph)
    elif draw_type == 'bfs':
        pos = nx.bfs_layout(graph, 0)
    else:
        pos = nx.nx_pydot.graphviz_layout(graph)
    return {int(k): v for k, v in pos.items()}


def resolve_positions(graph, draw_type, pos_path='', root_node=None):
    """Positions from the cache file if present, else computed and cached."""
    pos = load_positions(pos_path)
    if pos is not None:
        return pos
    pos = compute_positions(graph, draw_type, root_node=root_node)
    save_positions(pos, pos_path)
    return pos


def filter_weight_bounds(graph_weights, filter_weights):
    """Zero out all weights except the largest and smallest ones.

    Keeps a weight if it is among the ``absolute`` largest or ``absolute``
    smallest unique values, or within the top/bottom ``percentage`` fraction
    of unique values; everything in between is set to 0 (i.e. not drawn).
    """
    graph_weights = np.asarray(graph_weights)
    if filter_weights is None or graph_weights.size == 0:
        return graph_weights
    unique_weights = np.unique(graph_weights)  # sorted ascending
    n = len(unique_weights)
    if filter_weights.get('percentage', None) is not None:
        keep = int(n * filter_weights['percentage'])
    elif filter_weights.get('absolute', None) is not None:
        keep = int(filter_weights['absolute'])
    else:
        raise ValueError("filter_weights needs a 'percentage' or 'absolute' key")
    keep = min(max(keep, 1), n)
    lower_bound = unique_weights[keep - 1]
    upper_bound = unique_weights[n - keep]
    return np.where((graph_weights <= lower_bound) | (graph_weights >= upper_bound), graph_weights, 0)

class CustomColorMap:
    def __init__(self):
        aqua = (0.0, 0.6196, 0.8902)
        # 89,189,247
        skyblue = (0.3490, 0.7412, 0.9686)
        fuchsia = (232 / 255.0, 46 / 255.0, 130 / 255.0)
        violet = (152 / 255.0, 48 / 255.0, 130 / 255.0)
        white = (1.0, 1.0, 1.0)
        # darknavy 12,18,43
        darknavy = (12 / 255.0, 18 / 255.0, 43 / 255.0)

        # Define the three colors and their positions
        lamarr_colors = [aqua, white, fuchsia]  # Color 3 (RGB values)

        positions = [0.0, 0.5, 1.0]  # Positions of the colors (range: 0.0 to 1.0)

        # Create a colormap using LinearSegmentedColormap
        self.cmap = mcolors.LinearSegmentedColormap.from_list('custom_colormap', list(zip(positions, lamarr_colors)))

class TabColorMap:
    def __init__(self):
        cmap1 = plt.get_cmap('tab20')
        cmap2 = plt.get_cmap('tab20b')
        cmap3 = plt.get_cmap('tab20c')
        cmap4 = plt.get_cmap('Dark2')
        cmap5 = plt.get_cmap('Set2')
        # merge cmap1, cmap2, cmap3
        colors = []
        for i in range(20):
            colors.append(cmap1(i))
            colors.append(cmap2(i))
            colors.append(cmap3(i))
        for i in range(8):
            colors.append(cmap4(i))
        for i in range(12):
            colors.append(cmap5(i))
        # randomly shuffle the colors
        import random
        # set seed
        random.seed(42)
        random.shuffle(colors)
        self.cmap = mcolors.ListedColormap(colors)

class RandomColorMap:
    def __init__(self, cmap_name:str, number_intervalls:int=1000, seed:int=42):
        # split the colormap into number_intervalls
        cmap = plt.get_cmap(cmap_name)
        colors = []
        for i in range(number_intervalls):
            colors.append(cmap(i/number_intervalls))
        # randomly shuffle the colors
        import random
        # set seed
        random.seed(seed)
        random.shuffle(colors)
        self.cmap = mcolors.ListedColormap(colors)


class GraphDrawing:
    def __init__(self, node_size=10.0, edge_width=1.0, weight_edge_width=1.0, weight_arrow_size=5.0, edge_color='black', edge_alpha=1, node_color='black', draw_type=None, colormap=plt.get_cmap('tab20')):
        self.node_size = node_size
        self.edge_width = edge_width
        self.weight_edge_width = weight_edge_width
        self.edge_color = edge_color
        self.edge_alpha = edge_alpha
        self.node_color = node_color
        self.arrow_size = weight_arrow_size
        self.draw_type = draw_type
        self.colormap = colormap