import sys
from pathlib import Path

import matplotlib.pyplot as plt

from simplegnn.datasets.utils.graph_drawing import GraphDrawing, CustomColorMap

# base_paper is not a package; add its src/ dir so plot_common is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from plot_common import (setup_pgf, save_latex_figure, get_experiment, get_model,
                         figure_path, position_path)


def main():
    config_path = 'experiments/base_paper/regression/substructure_counting/configs/main_config_substructure_counting.yml'
    graph_ids = [3947]
    db_names = ['triangle', 'tri_tail', 'cycle5', 'cycle4', 'cycle6', 'star', 'substructure_counting']
    if all(figure_path(f'{db_name}_substructure_counting.pdf').exists() for db_name in db_names):
        return

    setup_pgf()
    experiment = get_experiment(config_path)
    experiment.preprocessing(num_threads=1)
    for counter, db_name in enumerate(db_names):
        output_path = figure_path(f'{db_name}_substructure_counting.pdf')
        if output_path.exists():
            continue
        validation_id = 0
        net = get_model(config_path, db_name, config_id=0, run_id=0, validation_id=validation_id, best=False)
        # the commented-out target/pred ylabels below need model outputs; if
        # re-enabled, evaluate via experiment.evaluate_model_on_graphs(...)
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

            pos_path = position_path(f'{db_name}_{graph_id}_pos.txt')

            # get convolution layer
            convolution_layer = net.net_layers[0]
            aggregation_layer = net.net_layers[-1]
            # draw all the five heads
            convolution_layer.draw(ax=axs_id[0], graph_id=graph_id, graph_drawing=graph_drawing, filter_weights=None,
                                   head=0, pos_path=pos_path)
            convolution_layer.draw(ax=axs_id[1], graph_id=graph_id, graph_drawing=graph_drawing, filter_weights=None,
                                   head=1, pos_path=pos_path)
            convolution_layer.draw(ax=axs_id[2], graph_id=graph_id, graph_drawing=graph_drawing, filter_weights=None,
                                   head=2, pos_path=pos_path)
            convolution_layer.draw(ax=axs_id[3], graph_id=graph_id, graph_drawing=graph_drawing, filter_weights=None,
                                   head=3, pos_path=pos_path)
            convolution_layer.draw(ax=axs_id[4], graph_id=graph_id, graph_drawing=graph_drawing, filter_weights=None,
                                   head=4, pos_path=pos_path)

        # add subplots column and row titles
        axs_title = axs
        if len(graph_ids) > 1:
            axs_title = axs_title[0]
        if counter == 0:
            axs_title[0].set_title(f'Head: $3$-Cycle')
            axs_title[1].set_title(f'Head: $4$-Cycle')
            axs_title[2].set_title(f'Head: $5$-Cycle')
            axs_title[3].set_title(f'Head: $6$-Cycle')
            axs_title[4].set_title(f'Head: Degree')

        for idx, graph_id in enumerate(graph_ids):
            axs_title[idx].set_ylabel(f'Task: {db_name}')
            #if db_name != 'substructure_counting':
            #    axs_title[idx].set_ylabel(f'Task: {db_name} - target: {labels[0].item():.4f} - pred: {outputs[0].item():4f}')
            #else:
            #    axs_title[idx].set_ylabel(f'Task: {db_name} - target: {labels} - pred: {outputs}')

        save_latex_figure(fig, output_path)


if __name__ == '__main__':
    main()
