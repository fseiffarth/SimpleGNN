import os
from pathlib import Path

from src.Experiment.ExperimentMain import ExperimentMain
from src.Experiment.RunConfiguration import get_run_configs
from src.Preprocessing.GraphData.GraphData import get_graph_data
from src.utils.EvaluationFinal import model_selection_evaluation
from src.Preprocessing.load_labels import load_labels
from src.utils.utils import save_graphs


def create_dataset(dataset_name, paths:dict[str, Path], output_path:Path, layers=None, with_degree=False):
    # load the graphs
    data_path = paths['data']
    label_path = paths['labels']
    splits_path = paths['splits']
    graph_data = get_graph_data(dataset_name, data_path, use_labels=True, use_attributes=False, graph_format='NEL')


    for l in layers:
        layer_label_path = label_path.joinpath(f"{dataset_name}_{l.get_layer_string()}_labels.txt")
        if os.path.exists(layer_label_path):
            g_labels = load_labels(path=layer_label_path)
            # add g_labels as node attributes to the graph_data
            for i, g in enumerate(graph_data.graphs):
                node_labels = g_labels.node_labels[i]
                for node in g.nodes:
                    if 'attr' in g.nodes[node]:
                        g.nodes[node]['attr'].append(node_labels[node])
                    else:
                        g.nodes[node]['attr'] = [node_labels[node]]
                    # delete the attribute key and value from the node data dict
                    g.nodes[node].pop('attribute', None)
    # graph_labels to 0,1,2 ...
    # if there exist graph labels -1, 1 shift to 0,1
    if min(graph_data.graph_labels) == -1 and max(graph_data.graph_labels) == 1:
        for i, label in enumerate(graph_data.graph_labels):
            graph_data.graph_labels[i] += 1
            graph_data.graph_labels[i] //= 2
    # if the graph labels start from 1, shift to 0,1,2 ...
    if min(graph_data.graph_labels) == 1:
        for i, label in enumerate(graph_data.graph_labels):
            graph_data.graph_labels[i] -= 1


    save_graphs(path=output_path, db_name=f'{dataset_name}Features', graphs=graph_data.graphs, labels=graph_data.graph_labels, with_degree=with_degree, graph_format='NEL')
    # copy the split data in the processed folder and rename it to dataset_nameFeatures_splits.json
    source_path = splits_path.joinpath(f"{dataset_name}_splits.json")
    target_path = output_path.joinpath(f"{dataset_name}Features/processed/{dataset_name}Features_splits.json")
    os.system(f"cp {source_path} {target_path}")



def get_gnn_comparison_data(main_config_path:Path, output_path:Path, db_name:str, with_degree=False, with_features=True):
    ### Real World Data
    experiment = ExperimentMain(Path(main_config_path))
    experiment_configuration = experiment.network_configurations[db_name][0]
    run_configs = get_run_configs(experiment_configuration)
    experiment_configuration['best_model'] = True
    # get the best configuration and run it
    best_config_id = model_selection_evaluation(db_name=db_name,
                                                get_best_model=True,
                                                experiment_config=experiment_configuration)

    best_config = run_configs[best_config_id]
    graph_data = get_graph_data(db_name, experiment_configuration['paths']['data'], graph_format='RuleGNNDataset')
    graph_data.create_nx_graphs()

    layer_label_strings = set()
    layer_label_strings.add('primary')
    for l in best_config.layers:
        layer_label_strings.update(l.get_layer_label_strings())

    for layer_label_string in layer_label_strings:
            layer_label_path = experiment_configuration['paths']['labels'].joinpath(f'{db_name}').joinpath(f"{db_name}_labels_{layer_label_string}.pt")
            if os.path.exists(layer_label_path):
                g_labels = load_labels(path=layer_label_path)
                # add g_labels as node attributes to the graph_data
                for i, g in enumerate(graph_data.nx_graphs):
                    node_labels = g_labels.node_labels[graph_data.slices['x'][i]:graph_data.slices['x'][i+1]]
                    for node in g.nodes:
                        # remove additional attributes
                        if layer_label_string == 'primary':
                            g.nodes[node]['primary_node_labels'] = node_labels[node].item()
                        else:
                            if with_features:
                                if 'attr' in g.nodes[node]:
                                    g.nodes[node]['attr'].append(node_labels[node].item() + 1)
                                else:
                                    g.nodes[node]['attr'] = [node_labels[node].item() + 1]
                                # delete the attribute key and value from the node data dict
                            g.nodes[node].pop('attribute', None)
    # graph_labels to 0,1,2 ...
    # if there exist graph labels -1, 1 shift to 0,1
    if min(graph_data.y) == -1 and max(graph_data.y) == 1:
        for i, label in enumerate(graph_data.graph_labels):
            graph_data.graph_labels[i] += 1
            graph_data.graph_labels[i] //= 2
    # if the graph labels start from 1, shift to 0,1,2 ...
    if min(graph_data.y) == 1:
        for i, label in enumerate(graph_data.graph_labels):
            graph_data.graph_labels[i] -= 1

    if with_features:
        save_graphs(path=output_path, db_name=f'{db_name}Features', graphs=graph_data.nx_graphs, labels=graph_data.y.tolist(), with_degree=with_degree, graph_format='NEL')
    else:
        save_graphs(path=output_path, db_name=f'{db_name}', graphs=graph_data.nx_graphs, labels=graph_data.y.tolist(), with_degree=with_degree, graph_format='NEL')
    # copy the split data in the processed folder and rename it to db_nameFeatures_splits.json
    source_path = experiment_configuration['paths']['splits'].joinpath(f"{db_name}_splits.json")
    if with_features:
        target_path = output_path.joinpath(f"{db_name}Features/processed/{db_name}Features_splits.json")
    else:
        target_path = output_path.joinpath(f"{db_name}/processed/{db_name}_splits.json")
    os.system(f"cp {source_path} {target_path}")


def main():
    for db_name in ['IMDB-BINARY', 'IMDB-MULTI']:
        get_gnn_comparison_data(main_config_path=Path('paper_experiments/Configs/main_config_fair_real_world_random_variation.yml'),
                                output_path=Path(f'paper_experiments/DataGNNComparison/'),
                                db_name=db_name)
        get_gnn_comparison_data(main_config_path=Path('paper_experiments/Configs/main_config_fair_real_world_random_variation.yml'),
                                output_path=Path(f'paper_experiments/DataGNNComparison/'),
                                db_name=db_name, with_features=False)
    for db_name in ['DHFR', 'NCI1', 'NCI109', 'Mutagenicity']:
        get_gnn_comparison_data(main_config_path=Path('paper_experiments/Configs/main_config_fair_real_world.yml'),
                                output_path=Path(f'paper_experiments/DataGNNComparison/'),
                                db_name=db_name)
        get_gnn_comparison_data(main_config_path=Path('paper_experiments/Configs/main_config_fair_real_world.yml'),
                                output_path=Path(f'paper_experiments/DataGNNComparison/'),
                                db_name=db_name, with_features=False)
    for db_name in ['CSL', 'EvenOddRings2_16', 'EvenOddRingsCount16', 'LongRings100', 'Snowflakes']:
        get_gnn_comparison_data(main_config_path=Path('paper_experiments/Configs/main_config_fair_synthetic.yml'),
                                output_path=Path(f'paper_experiments/DataGNNComparison/'),
                                db_name=db_name)
        get_gnn_comparison_data(main_config_path=Path('paper_experiments/Configs/main_config_fair_synthetic.yml'),
                                output_path=Path(f'paper_experiments/DataGNNComparison/'),
                                db_name=db_name, with_features=False)







if __name__ == "__main__":
    main()
