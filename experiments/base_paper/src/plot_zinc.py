from pathlib import Path

import matplotlib.colors as mcolors

import matplotlib.pyplot as plt
import numpy as np

from src.Experiment.ExperimentMain import ExperimentMain
from src.utils.GraphDrawing import GraphDrawing
from src.utils.load_splits import Load_Splits


def parameter_update():
    plt.rcParams.update({
        "font.family": "serif",  # use serif/main font for text elements
        "font.size": 30,
        "text.usetex": True,  # use inline math for ticks
        "pgf.rcfonts": False,  # don't setup fonts from rc parameters
        "pgf.texsystem": "lualatex",
        "pgf.preamble": "\n".join([
            r"\usepackage{url}",  # load additional packages
            r"\usepackage{unicode-math}",  # unicode math setup
            r"\setmainfont{DejaVu Serif}",  # serif font via preamble
        ])
    })


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


def main():
    parameter_update()
    experiment = ExperimentMain(Path('paper_experiments/regression/ZINC/configs/main_config_ZINC.yml'))
    experiment.ExperimentPreprocessing()
    graph_ids = [500]
    db_name = 'ZINC'
    net = experiment.load_model(db_name=db_name, config_id=0, run_id=0, validation_id=0)
    n = len(graph_ids)
    m = 4

    fig, axs = plt.subplots(nrows=n, ncols=m, figsize=(5 * m, 5 * n))
    plt.subplots_adjust(wspace=0, hspace=0)
    graph_drawing = (
        GraphDrawing(node_size=40,
                     edge_width=1,
                     draw_type='kawai'),
        GraphDrawing(node_size=40, edge_width=1,
                     weight_edge_width=2.5,
                     weight_arrow_size=10,
                     draw_type='kawai',
                     colormap=CustomColorMap().cmap)
    )
    # use plasma colormap for the bias
    graph_bias_drawing = (
        GraphDrawing(node_size=40, edge_width=1, colormap=plt.cm.plasma, draw_type='kawai'),
        GraphDrawing(node_size=40, edge_width=1, weight_edge_width=2.5, weight_arrow_size=10, draw_type='kawai'),
    )

    Path('paper_experiments/Results/Latex/Plots/Positions/').mkdir(exist_ok=True, parents=True)
    save_pos_path = Path('paper_experiments/Results/Latex/Plots/Positions/')
    pos_path = save_pos_path.joinpath(f'{db_name}_{graph_ids[0]}_pos.txt')

    for idx, graph_id in enumerate(graph_ids):
        axs_id = axs
        if len(graph_ids) > 1:
            axs_id = axs[idx]

        # get convolution layer
        convolution_layer = net.net_layers[2]
        # draw all the five heads
        # convolution_layer.draw(ax=axs_id[0], graph_id=graph_id, graph_drawing=graph_drawing, graph_only=True)
        convolution_layer.draw(ax=axs[0], graph_id=graph_ids[0], graph_drawing=graph_drawing, graph_only=True,
                               pos_path=pos_path)
        convolution_layer.draw(ax=axs_id[1], graph_id=graph_id, graph_drawing=graph_drawing, filter_weights=None,
                               head=0)
        convolution_layer.draw(ax=axs_id[2], graph_id=graph_id, graph_drawing=graph_drawing, filter_weights=None,
                               head=10)
        convolution_layer.draw(ax=axs_id[3], graph_id=graph_id, graph_drawing=graph_drawing, filter_weights=None,
                               head=18)

    # use latex backend for matplotlib
    graph_ids_string = '_'.join([str(x) for x in graph_ids])
    plt.savefig(f'paper_experiments/Results/Latex/{db_name}_{graph_ids_string}_message_passing.pdf', bbox_inches='tight', backend='pgf')
    plt.show()

    return


if __name__ == '__main__':
    main()