from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

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