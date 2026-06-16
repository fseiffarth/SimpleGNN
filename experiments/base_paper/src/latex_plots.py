from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simplegnn.framework.core import FrameworkMain
from simplegnn.models.ShareGNN.layers.inv_based_message_passing import InvariantBasedMessagePassingLayer
from simplegnn.datasets.utils.graph_drawing import GraphDrawing, CustomColorMap, RandomColorMap


def ablation_threshold(dataset, threshold_type):
    if not Path(f'results/base_paper/classification/Latex/Plots/ablation_threshold_{dataset}_{threshold_type}.pdf').exists():
        plt.rcParams.update({
            "font.family": "serif",  # use serif/main font for text elements
            "font.size": 12,
            "text.usetex": True,  # use inline math for ticks
            "pgf.rcfonts": False,  # don't setup fonts from rc parameters
            "pgf.texsystem": "lualatex",
            "pgf.preamble": "\n".join([
                r"\usepackage{url}",  # load additional packages
                r"\usepackage{unicode-math}",  # unicode math setup
                r"\setmainfont{DejaVu Serif}",  # serif font via preamble
            ])
        })
        ablation_results_path = Path(f'results/base_paper/classification/Ablation/{threshold_type}/')

        ablation_results = dict()
        for i in list(range(1, 21)) + [30, 40, 50]:
            path = ablation_results_path.joinpath(f'{i}/')
            # check if the path exists
            if Path(path).exists():
                results_str, results = share_gnn_results('ShareGNN', [dataset], path, False)
                if i in ablation_results and isinstance(ablation_results[i], dict):
                    ablation_results[i]['accuracy'] = results[0][0]
                    ablation_results[i]['std'] = results[0][1]
                else:
                    ablation_results[i] = {'accuracy': results[0][0], 'std': results[0][1]}
        # get number of parameters
        for i in list(range(1, 21)) + [30, 40, 50]:
            path = ablation_results_path.joinpath(f'{i}/')
            if Path(path).exists():
                # get the file from results folder that contains Best and Network
                for file in Path(f'{path}/{dataset}/Results/classification_old').iterdir():
                    if 'Best_Configuration' in file.name and 'Network' in file.name:
                        with open(file, 'r') as f:
                            data = f.read()
                            data = data.split('\n')
                            for line in data:
                                if 'Total trainable parameters' in line:
                                    num_parameters = int(line.split(':')[-1].strip())
                                    ablation_results[i]['parameters'] = num_parameters
                                    break
                        break
        # get avg best epoch
        for i in list(range(1, 21)) + [30, 40, 50]:
            path = ablation_results_path.joinpath(f'{i}/')
            if Path(path).exists():
                # get the file from results folder that contains Best and Network
                df = pd.read_csv(f'{path}/{dataset}/summary_best_mean.csv', delimiter=",")
                ablation_results[i]['mean_epoch'] = df['Epoch Mean'].values[0]
        # get avg best epoch and avg epoch runtime
        for i in list(range(1, 21)) + [30, 40, 50]:
            path = ablation_results_path.joinpath(f'{i}/')
            if Path(path).exists():
                # get the file from results folder that contains Best and Network
                df_all = None
                for file in Path(f'{path}/{dataset}/Results/classification_old').iterdir():
                    if f'{dataset}_Configuration' in file.name and '.csv' in file.suffix:
                        df = pd.read_csv(file, delimiter=";")
                        # concatenate the dataframes
                        if df_all is None:
                            df_all = df
                        else:
                            df_all = pd.concat([df_all, df], ignore_index=True)
                # get the mean of all epoch times
                mean_epoch_time = df_all['EpochTime'].mean()
                ablation_results[i]['mean_epoch_time'] = mean_epoch_time

        fig, ax1 = plt.subplots()

        # ticks inside
        plt.tick_params(axis='both', direction='in')
        # set title to dataset
        # plt.title(f'{dataset}')
        #
        # create a bar plot with x-axis as keys of ablation_results and y-axis as accuracy
        ax1.errorbar(ablation_results.keys(), [ablation_results[i]['accuracy'] for i in ablation_results],
                     yerr=[ablation_results[i]['std'] for i in ablation_results], fmt='o', capsize=5)
        ax1.set_ylabel('Accuracy in \\%')
        # set range to 80 - 90
        #ax1.set_ylim([80, 90])


        # add number of parameters to the right y-axis in thousand
        ax2 = plt.gca().twinx()
        # ticks at the inside
        ax2.tick_params(axis='y', direction='in')
        ax2.plot(ablation_results.keys(), [ablation_results[i]['parameters']/1000 for i in ablation_results], 'r', marker='s')
        ax2.set_ylabel('Weights (in thousands)')
        # set range to 0 - 400
        #ax2.set_ylim([0, 400])

        #  add one legend for both axes
        if threshold_type == 'Lower':
            plt.figlegend(['Accuracy in \\%', 'Weights in thousands)'], loc=(0.42, 0.79))
        elif threshold_type == 'Upper':
            plt.figlegend(['Accuracy in \\%', 'Weights in thousands)'], loc=(0.2, 0.79))
        elif threshold_type == 'LowerUpper':
            plt.figlegend(['Accuracy in \\%', 'Weights in thousands)'], loc=(0.42, 0.79))
        # set ticks to list(range(1, 21)) + [30, 40, 50]
        plt.xticks(list(range(1, 21, 2)))
        # set x-axis label to the figure
        if threshold_type == 'Lower':
            ax1.set_xlabel('Minimum \\# of Occurrences per Shared Weight (Encoder)')
        elif threshold_type == 'Upper':
            ax1.set_xlabel('Maximum \\# of Occurrences per Shared Weight (Encoder)')
        elif threshold_type == 'LowerUpper':
            ax1.set_xlabel('Minimum \\# of Occurrences per Shared Weight (Encoder)')
        plt.savefig(f'results/base_paper/classification/Latex/Plots/ablation_threshold_{dataset}_{threshold_type}.pdf', bbox_inches='tight', backend='pgf')
        pass

def ablation_distance(dataset='NCI1', max_distance=12, fontsize=9):
    if not Path(f'results/base_paper/classification/Latex/Plots/ablation_distance_{dataset}_{max_distance}.pdf').exists():
        plt.rcParams.update({
            "font.family": "serif",  # use serif/main font for text elements
            "font.size": 12,
            "text.usetex": True,  # use inline math for ticks
            "pgf.rcfonts": False,  # don't setup fonts from rc parameters
            "pgf.texsystem": "lualatex",
            "pgf.preamble": "\n".join([
                r"\usepackage{url}",  # load additional packages
                r"\usepackage{unicode-math}",  # unicode math setup
                r"\setmainfont{DejaVu Serif}",  # serif font via preamble
            ])
        })

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        path = Path(f'results/base_paper/classification/Distance/{dataset}/')
        # get the summary.csv file
        df = pd.read_csv(path.joinpath('summary.csv'), delimiter=",")
        model_layers_depths = []
        for i in range(1, 21):
            model_layers_depths.append((1, i))
        for i in range(1, 11):
            model_layers_depths.append((2, i))
        for i in range(1, 7):
            model_layers_depths.append((3, i))
        for i in range(1, 6):
            model_layers_depths.append((4, i))
        for i in range(1, 5):
            model_layers_depths.append((5, i))
        for i in range(1, 4):
            model_layers_depths.append((6, i))
        for i in range(1, 3):
            model_layers_depths.append((7, i))
        for i in range(1, 3):
            model_layers_depths.append((8, i))
        for i in range(1, 3):
            model_layers_depths.append((9, i))
        for i in range(1, 3):
            model_layers_depths.append((10, i))

        # add layer depth information to df
        df['Layers'] = [model_layers_depths[i][0] for i in range(len(model_layers_depths))]
        df['Depth'] = [model_layers_depths[i][1] for i in range(len(model_layers_depths))]

        max_depth = max_distance + 1
        np_array = np.zeros((11, max_depth))
        # iterate over rows of the dataframe and fill the np_array
        for i, row in df.iterrows():
            if row['Depth'] < max_depth:
                np_array[int(row['Layers']), int(row['Depth'])] = row['Test Accuracy Mean']

        # plot the np_array in a coordinate system
        # normalize viridis to min, max of np_array where min is not zero
        cmap_custom = plt.colormaps['Blues']
        norm = plt.Normalize(vmin=np_array[np_array != 0].min() - 10.0, vmax=np_array.max())
        # set 0 to white
        cmap_custom.set_under('white')

        plt.figure()
        ax = plt.gca()
        im = plt.imshow(np_array, cmap=cmap_custom, norm=norm)
        # add the values to the plot
        for i in range(np_array.shape[0]):
            for j in range(np_array.shape[1]):
                array_color = cmap_custom(norm(np_array[i, j]))
                if np_array[i, j] != 0:
                    # if color is dark, add white text, else add black text
                    if np_array[i, j] > 63:
                        plt.text(j, i, '$\\mathbf{' + f'{np_array[i, j]:.1f}' + '}$', ha='center', va='center', color='white', fontsize=fontsize)
                    else:
                        plt.text(j, i, '$\\mathbf{' + f'{np_array[i, j]:.1f}' + '}$', ha='center', va='center', color='black', fontsize=fontsize)
                else:
                    # add -
                    #plt.text(j, i, '-', ha='center', va='center', color='black')
                    pass
        # invert y-axis
        plt.gca().invert_yaxis()
        # set x and y min to 1
        plt.xlim(0.5, max_depth - 1 + 0.5)
        plt.ylim(0.5, 10.5)
        # add tick at 1
        plt.xticks(range(1, max_depth))
        plt.yticks(range(1, 11))
        # set x-axis to Layers
        plt.ylabel('Encoder Layers')
        # set y-axis to Depth
        plt.xlabel('Maximum Message-Passing Distance $D$')
        # plt.title(f'{dataset}')

        # add colorbar and set height to axes height
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax).set_label('Accuracy in $\\%$')
        plt.savefig(f'results/base_paper/classification/Latex/Plots/ablation_distance_{dataset}_{max_distance}.pdf', bbox_inches='tight', backend='pgf')
        pass

def plot_network(path, db_name, graph_ids, filtering, draw_type=None, with_labels_from_invariant=True, with_aggregation=False, molecule=False, channel=0, headers=True):
    graph_id_string = '_'.join([str(graph_id) for graph_id in graph_ids])
    # check if file exists
    if not Path(f'results/base_paper/classification/Latex/Plots/visualization_{db_name}_{graph_id_string}.pdf').exists():
        #mpl.use("pgf")
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            "font.family": "serif",  # use serif/main font for text elements
            "font.size": 18,
            "text.usetex": True,  # use inline math for ticks
            "pgf.rcfonts": False,  # don't setup fonts from rc parameters
            "pgf.texsystem": "lualatex",
            "pgf.preamble": "\n".join([
                r"\usepackage{url}",  # load additional packages
                r"\usepackage{unicode-math}",  # unicode math setup
                r"\setmainfont{DejaVu Serif}",  # serif font via preamble
            ])
        })

        # remove matplotlib frame
        # remove frame from each side of plot
        plt.rcParams['axes.spines.left'] = False
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.bottom'] = False
        #experiment = FrameworkMain(Path('experiments/base_paper/classification/configs/main_config_fair_real_world.yml'))
        #experiment = FrameworkMain(Path('Examples/TUExample/classification/configs/config_main.yml'))
        experiment = FrameworkMain(Path(path))

        net = experiment.load_model(db_name=db_name, run_id=0, validation_id=0, best=True)
        num_convolution_layers = 0
        for layers in net.net_layers:
            if isinstance(layers, InvariantBasedMessagePassingLayer):
                num_convolution_layers += 1
        #sort_indices, steps = rules_vs_occurences(convolution_layer, db_name, channel)
        #rules_vs_occurences_properties(convolution_layer)
        #rules_vs_weights(convolution_layer, sort_indices, steps, db_name, channel)

        n = len(graph_ids)
        column_for_invariants = 1 if with_labels_from_invariant else 0
        m = 1 + len(filtering)*num_convolution_layers + column_for_invariants

        fig, axs = plt.subplots(nrows=n, ncols=m, figsize=(5*m, 5*n))
        plt.subplots_adjust(wspace=0, hspace=0)
        graph_drawing = (
            GraphDrawing(node_size=160, edge_width=1, draw_type=draw_type),
            GraphDrawing(node_size=160, edge_width=1, weight_edge_width=2.5, weight_arrow_size=10,
                         colormap=CustomColorMap().cmap, draw_type=draw_type)
        )
        # use plasma colormap for the bias
        graph_bias_drawing = (
            GraphDrawing(node_size=160, edge_width=1, colormap=RandomColorMap('nipy_spectral', 99999).cmap, draw_type=draw_type),
            GraphDrawing(node_size=160, edge_width=1, weight_edge_width=2.5, weight_arrow_size=10, draw_type=draw_type)
        )

        Path('results/base_paper/classification/Latex/Plots/Positions/').mkdir(exist_ok=True, parents=True)
        save_pos_path = Path('results/base_paper/classification/Latex/Plots/Positions/')

        if len(graph_ids) == 1:
            pos_path = save_pos_path.joinpath(f'{db_name}_{graph_ids[0]}_pos.txt')
            convolution_layer = net.net_layers[0]
            convolution_layer.draw(ax=axs[0], graph_id=graph_ids[0], graph_drawing=graph_drawing, graph_only=True, pos_path=pos_path)
            if with_labels_from_invariant:
                convolution_layer.draw(ax=axs[1], graph_id=graph_ids[0], graph_drawing=graph_bias_drawing, graph_only=True, draw_bias_labels=True, pos_path=pos_path)

            for i in range(0, num_convolution_layers):
                # get convolution layer
                convolution_layer = net.net_layers[i]
                for filter_weights in filtering:
                    convolution_layer.draw(ax=axs[2+i], graph_id=graph_ids[0], graph_drawing=graph_drawing, filter_weights=filter_weights, pos_path=pos_path)

            # add subplots column and row titles
            if molecule:
                axs[0].set_title(f'Atomic Numbers')
            else:
                axs[0].set_title(f'Node Labels')
            if with_labels_from_invariant:
                axs[1].set_title(f'Labels from Invariant')
            for i in range(0, num_convolution_layers):
                if num_convolution_layers > 1:
                    for j, filter_weights in enumerate(filtering):
                        if filter_weights is None:
                            axs[1+column_for_invariants+ i*len(filtering) + j].set_title(f'Layer {i+1}')
                        elif 'absolute' in filter_weights:
                            axs[1+column_for_invariants+ i*len(filtering) + j].set_title(f'Layer {i+1} Top ${filter_weights["absolute"]}$ Weights')
                else:
                    for j, filter_weights in enumerate(filtering):
                        if filter_weights is None:
                            axs[1+column_for_invariants+ i*len(filtering) + j].set_title(f'All Weights')
                        elif 'absolute' in filter_weights:
                            axs[1+column_for_invariants+ i*len(filtering) + j].set_title(f'Top ${filter_weights["absolute"]}$ Weights')

            axs[0].set_ylabel(f'Graph Label: {net.graph_data.y[graph_ids[0]].item()}')
        else:
            for idx, graph_id in enumerate(graph_ids):
                pos_path = save_pos_path.joinpath(f'{db_name}_{graph_id}_pos.txt')

                # get convolution layer
                convolution_layer = net.net_layers[0]
                convolution_layer.draw(ax=axs[idx][0], graph_id=graph_id, graph_drawing=graph_drawing, graph_only=True, pos_path=pos_path)
                if with_labels_from_invariant:
                    convolution_layer.draw(ax=axs[idx][1], graph_id=graph_id, graph_drawing=graph_bias_drawing, graph_only=True, draw_bias_labels=True, pos_path=pos_path)
                for i in range(0, num_convolution_layers):
                    # get convolution layer
                    convolution_layer = net.net_layers[i]
                    for j, filter_weights in enumerate(filtering):
                        convolution_layer.draw(ax=axs[idx][1+column_for_invariants+ i*len(filtering) + j], graph_id=graph_id, graph_drawing=graph_drawing, filter_weights=filter_weights, pos_path=pos_path)

            # add subplots column and row titles
            # add subplots column and row titles
            if molecule:
                axs[0][0].set_title(f'Atomic Numbers')
            else:
                axs[0][0].set_title(f'Node Labels')
            if with_labels_from_invariant:
                axs[0][1].set_title(f'Labels from Invariant')
            for i in range(0, num_convolution_layers):
                if num_convolution_layers > 1:
                    for j, filter_weights in enumerate(filtering):
                        if filter_weights is None:
                            axs[0][1 + column_for_invariants + i*len(filtering) + j].set_title(f'Layer {i+1}')
                        elif 'absolute' in filter_weights:
                            axs[0][1 + column_for_invariants + i*len(filtering) + j].set_title(f'Layer {i+1} Top ${filter_weights["absolute"]}$ Weights')
                else:
                    for j, filter_weights in enumerate(filtering):
                        if filter_weights is None:
                            axs[0][1 + column_for_invariants + i*len(filtering) + j].set_title(f'All Weights')
                        elif 'absolute' in filter_weights:
                            axs[0][1 + column_for_invariants + i*len(filtering) + j].set_title(f'Top ${filter_weights["absolute"]}$ Weights')



            for idx, graph_id in enumerate(graph_ids):
                axs.axis('off')
                axs[idx][0].set_ylabel(f'Graph Label: ${net.graph_data.y[graph_id].item()}$')


        plt.savefig(f'results/base_paper/classification/Latex/Plots/visualization_{db_name}_{graph_id_string}.pdf', bbox_inches='tight', backend='pgf')
        # remove matplotlib frame
        # remove frame from each side of plot
        plt.rcParams['axes.spines.left'] = True
        plt.rcParams['axes.spines.right'] = True
        plt.rcParams['axes.spines.top'] = True
        plt.rcParams['axes.spines.bottom'] = True


def plot_specific_graphs_from_db(path, db_name, graph_ids, draw_type=None, node_size=200, output_path=None):
    if output_path is None:
        output_path = Path(f'results/base_paper/classification/Latex/Plots/')
    else:
        output_path = Path(output_path)
    if not output_path.joinpath(f'{db_name}_{"_".join(map(str, graph_ids))}.pdf').exists():
        # make dir f'scripts/Evaluation/Drawing/Graphs/{db_name}/' if it does not exist
        output_path.mkdir(exist_ok=True, parents=True)

        #mpl.use("pgf")
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            "font.family": "serif",  # use serif/main font for text elements
            "font.size": 12,
            "text.usetex": True,  # use inline math for ticks
            "pgf.rcfonts": False,  # don't setup fonts from rc parameters
            "pgf.texsystem": "lualatex",
            "pgf.preamble": "\n".join([
                r"\usepackage{url}",  # load additional packages
                r"\usepackage{unicode-math}",  # unicode math setup
                r"\setmainfont{DejaVu Serif}",  # serif font via preamble
            ])
        })
        # remove matplotlib frame
        # remove frame from each side of plot
        plt.rcParams['axes.spines.left'] = False
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.bottom'] = False

        experiment = FrameworkMain(Path(path))
        net = experiment.load_model(db_name=db_name, run_id=0, validation_id=0, best=True)

        fig, ax = plt.subplots(1, len(graph_ids), figsize=(5*len(graph_ids), 5*1))
        for i, graph_id in enumerate(graph_ids):
            plt.subplots_adjust(wspace=0, hspace=0)
            graph_drawing = (
                GraphDrawing(node_size=node_size, edge_width=1, draw_type=draw_type),
                GraphDrawing(node_size=node_size, edge_width=1, weight_edge_width=2.5, weight_arrow_size=10,
                             colormap=CustomColorMap().cmap, draw_type=draw_type)
            )
            # get convolution layer
            convolution_layer = net.net_layers[0]
            convolution_layer.draw(ax=ax[i], graph_id=graph_id, graph_drawing=graph_drawing, graph_only=True)

            # add subplots column and row titles
            #axs.set_title(f'Graphs with Atomic Numbers')
            ax[i].set_xlabel(f'Graph Label: ${net.graph_data.y[graph_id]}$')



        plt.savefig(output_path.joinpath(f'{db_name}_{"_".join(map(str, graph_ids))}.pdf', bbox_inches='tight', backend='pgf'))

        # remove matplotlib frame
        # remove frame from each side of plot
        plt.rcParams['axes.spines.left'] = True
        plt.rcParams['axes.spines.right'] = True
        plt.rcParams['axes.spines.top'] = True
        plt.rcParams['axes.spines.bottom'] = True

def rules_vs_occurences(layer: InvariantBasedMessagePassingLayer, db_name, channel=0, appendix='') -> np.ndarray:
    if not Path(f'results/base_paper/classification/Latex/Plots/occurrences_per_rule_{db_name}{appendix}.png').exists():

        plt.rcParams.update({
            "font.family": "serif",  # use serif/main font for text elements
            "font.size": 14,
            "text.usetex": True,  # use inline math for ticks
            "pgf.rcfonts": False,  # don't setup fonts from rc parameters
            "pgf.texsystem": "lualatex",
            "pgf.preamble": "\n".join([
                r"\usepackage{url}",  # load additional packages
                r"\usepackage{unicode-math}",  # unicode math setup
                r"\setmainfont{DejaVu Serif}",  # serif font via preamble
            ])
        })

        weight_distribution = layer.weight_distribution
        num_weights = layer.Param_W.shape[0]
        weight_array = np.zeros(num_weights)
        weights = weight_distribution[:, 3]
        # get unique counts of entries in weights
        weight_array = np.bincount(weights)
        # sort the weight_array (largest occurence first) and save the sorted indices
        sort_indices = np.argsort(weight_array)[::-1]

        weight_array = weight_array[sort_indices]


        # get array such that in entry i is the index of weight_array where value is the first time larger than i
        # iterate over weight_array in reverse order
        steps = np.zeros(11)
        current_step = 1
        for i in range(num_weights-1, 0, -1):
            if weight_array[i] != current_step:
                steps[current_step] = i - 1
                current_step += 1
                if current_step == 11:
                    break

        property_colors = plt.get_cmap('tab20').colors
        if layer.property_descriptions[channel] == 'distances':
            property_legend = [f'Distance {i}' for i in range(layer.n_properties[channel])]
        else:
            property_legend = [f'{layer.property_descriptions[channel]} {i}' for i in range(layer.n_properties[channel])]
        f = lambda x : np.max(np.where(x >= np.array(layer.weight_offset)))
        f_vectorized = np.vectorize(f)
        # get property id from sort indices using the skips
        property_indices = f_vectorized(sort_indices)

        node_colors = np.array(property_colors)[property_indices]

        # plot the distribution of the rules with legend
        fig, ax = plt.subplots()
        for i, p in enumerate(range(layer.n_properties[channel])):
            ax.scatter([], [], color=property_colors[i], label=property_legend[i])
        ax.scatter(np.arange(num_weights), weight_array, s=1.0, alpha=1, c=node_colors)
        # add legend title

        # add vertical lines for the steps
        for i in range(1, 11):
            ax.axvline(steps[i], color='black', linestyle='--', linewidth=0.5)

        ax.legend(loc='upper right')
        plt.xlabel('Weights of Encoder (Sorted by Occurrences)')
        plt.ylabel('\\# Occurrences in Dataset')
        # plt.title(f'{dataset}')

        #plt.title('Number of occurrences per rule')
        # use pgf backend for latex
        plt.savefig(f'results/base_paper/classification/Latex/Plots/occurrences_per_rule_{db_name}{appendix}.png', bbox_inches='tight')
        return sort_indices, steps
    return None

def rules_vs_weights(layer:InvariantBasedMessagePassingLayer, sort_indices:np.ndarray, steps,db_name, channel=0, appendix=''):
    if not Path(f'results/base_paper/classification/Latex/Plots/weights_per_rule_{db_name}{appendix}.png').exists():
        weights = layer.Param_W.detach().cpu().numpy()
        weights = weights[sort_indices]
        # colors from tab20
        property_colors = plt.get_cmap('tab20').colors
        if layer.property_descriptions[channel] == 'distances':
            property_legend = [f'Distance {i}' for i in range(layer.n_properties[channel])]
        else:
            property_legend = [f'{layer.property_descriptions[channel]} {i}' for i in range(layer.n_properties[channel])]
        f = lambda x : np.max(np.where(x >= np.array(layer.weight_offset)))
        f_vectorized = np.vectorize(f)
        # get property id from sort indices using the skips
        property_indices = f_vectorized(sort_indices)

        node_colors = np.array(property_colors)[property_indices]

        # plot the distribution of the rules with legend
        fig, ax = plt.subplots()
        for i, p in enumerate(range(layer.n_properties[channel])):
            ax.scatter([], [], color=property_colors[i], label=property_legend[i])

        # add vertical lines for the steps
        for i in range(1, 11):
            ax.axvline(steps[i], color='black', linestyle='--', linewidth=0.5)

        ax.scatter(np.arange(len(weights)), weights, s=1, alpha=1, c=node_colors)
        ax.legend(loc='upper right')
        plt.xlabel('Weights of Encoder (Sorted by Occurrences)')
        plt.ylabel('Weight Value')
        # plt.title(f'{dataset}')

        #plt.title('Distribution of rules')
        plt.savefig(f'results/base_paper/classification/Latex/Plots/weights_per_rule_{db_name}{appendix}.png', bbox_inches='tight')


def plot_shared_weights(path, db_name, appendix=''):
    if appendix != '':
        appendix = f'_{appendix}'
    paths = [f'results/base_paper/classification/Latex/Plots/occurrences_per_rule_{db_name}{appendix}.png',
             f'results/base_paper/classification/Latex/Plots/weights_per_rule_{db_name}{appendix}.png']
    if not all([Path(p).exists() for p in paths]):
        experiment = FrameworkMain(Path(path))
        net = experiment.load_model(db_name=db_name, best=True)
        convolution_layer = net.net_layers[0]
        channel = 0
        sort_indices, steps = rules_vs_occurences(convolution_layer, db_name, channel, appendix)
        #rules_vs_occurences_properties(convolution_layer)
        rules_vs_weights(convolution_layer, sort_indices, steps, db_name, channel, appendix)


def main():
    # create Latex dir under Results
    Path('results/base_paper/classification/Latex').mkdir(parents=True, exist_ok=True)
    Path('results/base_paper/classification/Latex/Plots').mkdir(parents=True, exist_ok=True)
    plot_network_path = 'experiments/base_paper/classification/configs/main_config_fair_real_world.yml'
    plot_network_path_random = 'experiments/base_paper/classification/configs/main_config_fair_real_world_random_variation.yml'
    plot_network_path_synthetic = 'experiments/base_paper/classification/configs/main_config_fair_synthetic.yml'
    plot_regression_path = 'experiments/base_paper/regression/ZINC/configs/main_config_ZINC.yml'

    plot_network_path_ablation_threshold = lambda x : f'experiments/base_paper/classification/configs/ablation/threshold/lower/main_config_ablation_threshold_{x}.yml'
    plot_network_path_ablation_distance = 'experiments/base_paper/classification/configs/ablation/distances/main_config_ablation_distances.yml'


    plot_network(plot_regression_path, 'ZINC', [500], draw_type='kawai', filtering=[None, {'absolute' : 3}], molecule=True, headers=False)
    plot_network(plot_network_path_random, 'DHFR', [272, 273], draw_type='kawai', filtering=[None, {'absolute' : 3}], molecule=True)


    plot_shared_weights(plot_network_path_random, 'DHFR')
    plot_shared_weights(plot_network_path_random, 'IMDB-BINARY')
    plot_shared_weights(plot_network_path_random, 'IMDB-MULTI')
    plot_shared_weights(plot_network_path, 'NCI1')
    plot_shared_weights(plot_network_path, 'NCI109')
    plot_shared_weights(plot_network_path, 'Mutagenicity')

    plot_shared_weights(plot_network_path_ablation_threshold(10), 'NCI1', appendix='lower_10')
    plot_shared_weights(plot_network_path_ablation_threshold(10), 'NCI109', appendix='lower_10')
    plot_shared_weights(plot_network_path_ablation_threshold(10), 'Mutagenicity', appendix='lower_10')
    plot_shared_weights(plot_network_path_ablation_threshold(10), 'DHFR', appendix='lower_10')
    plot_shared_weights(plot_network_path_ablation_threshold(10), 'IMDB-BINARY', appendix='lower_10')
    plot_shared_weights(plot_network_path_ablation_threshold(10), 'IMDB-MULTI', appendix='lower_10')

    plot_shared_weights(plot_network_path_ablation_distance, 'NCI1', appendix='distance')
    plot_shared_weights(plot_network_path_ablation_distance, 'NCI109', appendix='distance')
    plot_shared_weights(plot_network_path_ablation_distance, 'Mutagenicity', appendix='distance')
    plot_shared_weights(plot_network_path_ablation_distance, 'DHFR', appendix='distance')



    plot_specific_graphs_from_db(plot_network_path_synthetic,db_name='EvenOddRings2_16', graph_ids=[3,2,1,4], draw_type='circle')
    plot_specific_graphs_from_db(plot_network_path_synthetic,db_name='EvenOddRingsCount16', graph_ids=[0,5], draw_type='circle')
    plot_specific_graphs_from_db(plot_network_path_synthetic,db_name='LongRings100',  graph_ids=[1,3,5], draw_type='circle', node_size=100)
    plot_specific_graphs_from_db(plot_network_path_synthetic,db_name='Snowflakes', graph_ids=[9,120,372,501], draw_type='kawai', node_size=50)
    plot_specific_graphs_from_db(plot_network_path_synthetic,db_name='CSL',  graph_ids=[0,16,31], draw_type='kawai')

    plot_network(plot_network_path_synthetic, 'EvenOddRings2_16', [2, 1, 4], with_labels_from_invariant=False, draw_type='circle', filtering=[None, {'absolute' : 5}])
    plot_network(plot_network_path_synthetic, 'EvenOddRingsCount16', [0,5,6], with_labels_from_invariant=False, draw_type='circle', filtering=[None])
    plot_network(plot_network_path_synthetic, 'Snowflakes', [500,120,476], draw_type='kawai', filtering=[None, {'absolute' : 3}])
    plot_network(plot_network_path_synthetic, 'CSL', [0,16,31], draw_type='kawai', filtering=[None])
    plot_network(plot_network_path_random, 'IMDB-MULTI', [25,805,1265], draw_type='kawai', filtering=[None, {'absolute' : 3}])
    plot_network(plot_network_path_random, 'IMDB-BINARY', [101,68,612], draw_type='kawai', filtering=[None, {'absolute' : 3}])

    plot_network(plot_network_path, 'NCI1', [216, 320, 655], draw_type='kawai', filtering=[None, {'absolute' : 3}])
    plot_network(plot_network_path, 'NCI109', [56, 18, 3165], draw_type='kawai', filtering=[None, {'absolute' : 3}])
    plot_network(plot_network_path, 'Mutagenicity', [1654, 257, 360], draw_type='kawai', filtering=[None, {'absolute' : 3}])

    ablation_distance('NCI1', 20, fontsize=6)
    ablation_distance('NCI109', 20, fontsize=6)
    ablation_distance('Mutagenicity', 20, fontsize=6)
    ablation_distance('DHFR', 20, fontsize=6)

    ablation_distance('NCI1', 10)
    ablation_distance('NCI109', 10)
    ablation_distance('Mutagenicity', 10)
    ablation_distance('DHFR', 10)

    ablation_distance('NCI1')
    ablation_distance('NCI109')
    ablation_distance('Mutagenicity')
    ablation_distance('DHFR')



    ablation_threshold('NCI1', 'Lower')
    ablation_threshold('NCI1', 'Upper')
    ablation_threshold('NCI1', 'LowerUpper')
    ablation_threshold('NCI109', 'Lower')
    ablation_threshold('NCI109', 'Upper')
    ablation_threshold('NCI109', 'LowerUpper')
    ablation_threshold('Mutagenicity', 'Lower')
    ablation_threshold('Mutagenicity', 'Upper')
    ablation_threshold('Mutagenicity', 'LowerUpper')
    ablation_threshold('DHFR', 'Lower')
    ablation_threshold('DHFR', 'Upper')
    ablation_threshold('DHFR', 'LowerUpper')
    ablation_threshold('IMDB-BINARY', 'Lower')
    ablation_threshold('IMDB-BINARY', 'Upper')
    ablation_threshold('IMDB-BINARY', 'LowerUpper')
    ablation_threshold('IMDB-MULTI', 'Lower')
    ablation_threshold('IMDB-MULTI', 'Upper')
    ablation_threshold('IMDB-MULTI', 'LowerUpper')
    ablation_distance('Mutagenicity')
    ablation_distance('DHFR')







if __name__ == '__main__':
    main()