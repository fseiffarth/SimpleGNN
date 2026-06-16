from pathlib import Path

import matplotlib.colors as mcolors

import matplotlib.pyplot as plt
import numpy as np

from simplegnn.framework.core import FrameworkMain
from simplegnn.datasets.utils.graph_drawing import GraphDrawing
from simplegnn.framework.utils.preprocessing import load_splits

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
        lamarr_colors = [aqua, white, fuchsia] # Color 3 (RGB values)

        positions = [0.0, 0.5, 1.0]  # Positions of the colors (range: 0.0 to 1.0)

        # Create a colormap using LinearSegmentedColormap
        self.cmap = mcolors.LinearSegmentedColormap.from_list('custom_colormap', list(zip(positions, lamarr_colors)))



def main():
    parameter_update()
    experiment = FrameworkMain(Path('experiments/base_paper/regression/substructure_counting/configs/main_config_substructure_counting.yml'))
    experiment.preprocessing(num_threads=1)
    graph_ids = [3947]
    for counter, db_name in enumerate(['triangle', 'tri_tail', 'cycle5', 'cycle4', 'cycle6', 'star', 'substructure_counting']):
        validation_id = 0
        configuration = experiment.network_configurations[db_name][0]
        split_data = load_splits(configuration['paths']['splits'])
        test_data = np.asarray(split_data['test'][validation_id], dtype=int)
        # get five random graphs from the test data
        #np.random.seed(42)  # for reproducibility
        #graph_ids = np.random.choice(test_data, size=1, replace=False)
        net = experiment.load_model(db_name=db_name, config_id=0, run_id=0, validation_id=validation_id, best=False)
        outputs, labels, accuracy = experiment.evaluate_model_on_graphs(db_name=db_name, graph_ids=graph_ids, config_id=0, run_id=0, validation_id=validation_id, best=False)
        arg_max_outputs = np.argmax(outputs, axis=1)
        correct_outputs = np.equal(arg_max_outputs, labels)
        n = len(graph_ids)
        m = 5

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
            GraphDrawing(node_size=40, edge_width=1, colormap=plt.cm.plasma,draw_type='kawai'),
            GraphDrawing(node_size=40, edge_width=1, weight_edge_width=2.5, weight_arrow_size=10,draw_type='kawai'),
        )

        
        for idx, graph_id in enumerate(graph_ids):
            axs_id = axs
            if len(graph_ids) > 1:
                axs_id = axs[idx]
                
            # get convolution layer
            convolution_layer = net.net_layers[0]
            aggregation_layer = net.net_layers[-1]
            # draw all the five heads
            #convolution_layer.draw(ax=axs_id[0], graph_id=graph_id, graph_drawing=graph_drawing, graph_only=True)
            convolution_layer.draw(ax=axs_id[0], graph_id=graph_id, graph_drawing=graph_drawing, filter_weights=None, head=0)
            convolution_layer.draw(ax=axs_id[1], graph_id=graph_id, graph_drawing=graph_drawing, filter_weights=None, head=1)
            convolution_layer.draw(ax=axs_id[2], graph_id=graph_id, graph_drawing=graph_drawing, filter_weights=None,
                                   head=2)
            convolution_layer.draw(ax=axs_id[3], graph_id=graph_id, graph_drawing=graph_drawing, filter_weights=None,
                                   head=3)
            convolution_layer.draw(ax=axs_id[4], graph_id=graph_id, graph_drawing=graph_drawing, filter_weights=None,
                                      head=4)


        # add subplots column and row titles
        axs_title = axs
        if len(graph_ids) > 1:
            axs_title = axs_title[0]
        #axs_title[0].set_title(f'Graph')
        if counter == 0:
            axs_title[0].set_title(f'Head: $3$-Cycle')
            axs_title[1].set_title(f'Head: $4$-Cycle')
            axs_title[2].set_title(f'Head: $5$-Cycle')
            axs_title[3].set_title(f'Head: $6$-Cycle')
            axs_title[4].set_title(f'Head: Degree')

        # add
        for idx, graph_id in enumerate(graph_ids):
            axs_title[idx].set_ylabel(f'Task: {db_name}')
            #if db_name != 'substructure_counting':
            #    axs_title[idx].set_ylabel(f'Task: {db_name} - target: {labels[0].item():.4f} - pred: {outputs[0].item():4f}')
            #else:
            #    axs_title[idx].set_ylabel(f'Task: {db_name} - target: {labels} - pred: {outputs}')


        # use latex backend for matplotlib
        plt.savefig(f'experiments/base_paper/regression/substructure_counting/Plots/{db_name}_substructure_counting.pdf', bbox_inches='tight', backend='pgf')
        plt.show()


    return


if __name__ == '__main__':
    main()