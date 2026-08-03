import matplotlib.pyplot as plt

from simplegnn.datasets.utils.graph_drawing import GraphDrawing, CustomColorMap

from plot_common import (setup_pgf, save_latex_figure, get_experiment, get_model,
                         figure_path, position_path)


def main():
    graph_ids = [500]
    db_name = 'ZINC'
    graph_ids_string = '_'.join([str(x) for x in graph_ids])
    output_path = figure_path(f'{db_name}_{graph_ids_string}_message_passing.pdf')
    if output_path.exists():
        return

    setup_pgf()
    config_path = 'experiments/base_paper/regression/ZINC/configs/main_config_ZINC.yml'
    experiment = get_experiment(config_path)
    experiment.preprocessing(num_threads=1)
    net = get_model(config_path, db_name, config_id=0, run_id=0, validation_id=0)
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

    for idx, graph_id in enumerate(graph_ids):
        axs_id = axs
        if len(graph_ids) > 1:
            axs_id = axs[idx]

        pos_path = position_path(f'{db_name}_{graph_id}_pos.txt')

        # get convolution layer
        convolution_layer = net.net_layers[2]
        # draw all the five heads
        convolution_layer.draw(ax=axs_id[0], graph_id=graph_id, graph_drawing=graph_drawing, graph_only=True,
                               pos_path=pos_path)
        convolution_layer.draw(ax=axs_id[1], graph_id=graph_id, graph_drawing=graph_drawing, filter_weights=None,
                               head=0, pos_path=pos_path)
        convolution_layer.draw(ax=axs_id[2], graph_id=graph_id, graph_drawing=graph_drawing, filter_weights=None,
                               head=10, pos_path=pos_path)
        convolution_layer.draw(ax=axs_id[3], graph_id=graph_id, graph_drawing=graph_drawing, filter_weights=None,
                               head=18, pos_path=pos_path)

    save_latex_figure(fig, output_path)


if __name__ == '__main__':
    main()
