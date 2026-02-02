import json
import os
from pathlib import Path

from datasets.graph_dataset import GraphDataset
from datasets.utils.edge_labeling import Properties
from datasets.utils.node_labeling import load_labels, get_label_string
from framework.utils.parameters import Parameters
from models.ShareGNN.preprocessing.preprocessing import preprocessing_from_config
from utils.utils import save_graphs


class Preprocessing:
    """
    Preprocessing class to load the data, generate the splits, labels and properties and save them in the correct folders.
    params:
    dataset_configuration: dict: configuration for the dataset
    """
    def __init__(self, dataset_configurations):
        for configuration in dataset_configurations:
            self.db_name = configuration['name']
            self.graph_data : GraphDataset = None
            # load the config file
            self.experiment_configuration = configuration
            self.generation_times_labels_path = None
            self.generation_times_properties_path = None

            # create the folders and files for the results and preprocessing data
            self.create_folders()
            # generate the data only if it does not exist (i.e. the processed folder is empty)
            if not Path(self.experiment_configuration['paths']['data']).joinpath(f'{self.db_name}').joinpath('processed').joinpath(f'data.pt').is_file():
                dataset = self.db_name
                data_generation = self.experiment_configuration['data_generation']
                data_generation_args = self.experiment_configuration.get('generate_function_args', None)
                self.generate_data(dataset, data_generation, data_generation_args)
            else:
                # load the graph data
                self.load_data()
            # generate the split files
            self.load_configuration_splits()

            # do model-dependent preprocessing
            if configuration.get('with_invariant_layers', False):
                preprocessing_from_config(configuration, self.graph_data, self.generation_times_labels_path, self.generation_times_properties_path)


    def create_folders(self):
        # create config folders if they do not exist
        self.experiment_configuration['paths']['data'].mkdir(exist_ok=True, parents=True)
        # create results folder if it does not exist
        self.experiment_configuration['paths']['results'].mkdir(exist_ok=True, parents=True)
        # create folders plots, weights, models and results in the results folder under the db_name
        self.experiment_configuration['paths']['results'].joinpath(self.db_name).joinpath('Plots').mkdir(exist_ok=True, parents=True)
        self.experiment_configuration['paths']['results'].joinpath(self.db_name).joinpath('Weights').mkdir(exist_ok=True, parents=True)
        self.experiment_configuration['paths']['results'].joinpath(self.db_name).joinpath('Models').mkdir(exist_ok=True, parents=True)
        self.experiment_configuration['paths']['results'].joinpath(self.db_name).joinpath('Results').mkdir(exist_ok=True, parents=True)
        # if splits path ends with a file, get the parent folder
        if self.experiment_configuration['paths']['splits'].suffix == '.json':
            self.experiment_configuration['paths']['splits'].parent.mkdir(exist_ok=True, parents=True)
        else:
            self.experiment_configuration['paths']['splits'].mkdir(exist_ok=True, parents=True)
        # create folders labels, properties, splits only if they are not None
        if self.experiment_configuration['paths'].get('labels', None) is not None:
            self.experiment_configuration['paths']['labels'].mkdir(exist_ok=True, parents=True)
        if self.experiment_configuration['paths'].get('properties', None) is not None:
            self.experiment_configuration['paths']['properties'].mkdir(exist_ok=True, parents=True)




        # if not exists create the generation_times_labels.txt and generation_times_properties.txt in the Results folder
        if not Path(self.experiment_configuration['paths']['results']).joinpath('generation_times_labels.txt').exists():
            with open(Path(self.experiment_configuration['paths']['results']).joinpath('generation_times_labels.txt'), 'w') as f:
                f.write('Generation times for labels\n')
        if not Path(self.experiment_configuration['paths']['results']).joinpath('generation_times_properties.txt').exists():
            with open(Path(self.experiment_configuration['paths']['results']).joinpath('generation_times_properties.txt'), 'w') as f:
                f.write('Generation times for properties\n')
        self.generation_times_labels_path = self.experiment_configuration['paths']['results'].joinpath('generation_times_labels.txt')
        self.generation_times_properties_path = self.experiment_configuration['paths']['results'].joinpath('generation_times_properties.txt')

    def generate_data(self, dataset, data_generation_type, data_generation_args):
        # generate the graph data

        if isinstance(data_generation_type, list):
            # Create union of the graphs for transfer learning
            if not isinstance(data_generation_args, list):
                data_generation_args = [data_generation_args] * len(data_generation_type)
            zip_list = list(zip(data_generation_type, data_generation_args, self.experiment_configuration['single_datasets']))
            # generate the graph data for each list entry
            for data_gen, data_gen_args, d in zip_list:
                self.generate_data(d, data_gen, data_gen_args)
            # merge the generated datasets
            graphs = []
            for data_generation_type, data_generation_args, dataset in zip_list:
                graphs.append(GraphDataset(root=str(self.experiment_configuration['paths']['data']),
                                              name=dataset,
                                              from_existing_data=data_generation_type,
                                              task=self.experiment_configuration.get('task', None)
                                              ))
            # merge the graphs
            GraphDataset(root=str(self.experiment_configuration['paths']['data']),
                            name=self.experiment_configuration['name'],
                            from_existing_data=graphs,
                            task=self.experiment_configuration.get('task', None),
                            )
            return
        path = Path(self.experiment_configuration['paths']['data'])
        if Path(Path(self.experiment_configuration['paths']['data']) / dataset / 'processed').exists() and len(
                list(Path(Path(self.experiment_configuration['paths']['data']) / dataset / 'processed').iterdir())) > 0:
            print(
                f"Dataset {dataset} already exists in {Path(self.experiment_configuration['paths']['data'])} . Skip the data generation.")
            return
        if isinstance(data_generation_type, str):
            if data_generation_type != 'generate_from_function':
                    # download the dataset
                    # create a tmp folder to store the dataset
                    if not Path('tmp').exists():
                        Path('tmp').mkdir()
                    self.graph_data = GraphDataset(root=str(self.experiment_configuration['paths']['data']),
                                                      name=dataset,
                                                      from_existing_data=data_generation_type,
                                                      task=self.experiment_configuration.get('task', None),
                                                      precision=self.experiment_configuration.get('precision', 'float'),
                                                      )
                    if not os.path.exists(path.joinpath(Path(dataset))):
                        os.makedirs(path.joinpath(Path(dataset)))
                    # create processed and raw folders in path+dataset
                    if not os.path.exists(path.joinpath(Path(dataset + "/processed"))):
                        os.makedirs(path.joinpath(Path(dataset + "/processed")))
                    if not os.path.exists(path.joinpath(Path(dataset + "/raw"))):
                        os.makedirs(path.joinpath(Path(dataset + "/raw")))
                    #tu_to_nel(dataset=dataset, out_path=Path(self.experiment_configuration['paths']['data']))
            else:
                print(f'Do not know how to handle data from {data_generation_type}. Do you mean "TUDataset"?')
            pass
        else:
            if data_generation_type is not None:
                if data_generation_args is None:
                    data_generation_args = {}
                try:
                    # generate data
                    graphs, labels =  data_generation_type(**data_generation_args, split_path=Path(self.experiment_configuration['paths']['splits']))
                    # save lists of graphs and labels in the correct graph_format NEL -> Nodes, Edges, Labels
                    save_graphs(Path(self.experiment_configuration['paths']['data']), dataset, graphs, labels, with_degree=False, graph_format='NEL')
                    self.graph_data = GraphDataset(root=str(self.experiment_configuration['paths']['data']),
                                                      name=dataset,
                                                      from_existing_data='NEL',
                                                      task=self.experiment_configuration.get('task', None),
                                                      )
                except:
                    # raise the error that has occurred
                    print(f'Could not generate {dataset} from function {data_generation_type} with arguments {data_generation_args}')

            else:
                try:
                    self.graph_data = GraphDataset(root=str(self.experiment_configuration['paths']['data']),
                                                      name=dataset,
                                                      from_existing_data='NEL',
                                                      task=self.dataset_configuration.get('task', None)
                                                      )
                except:
                    print(f'Could not process the data from {dataset} with the given configuration.')


    def load_data(self):
        # load graph data from pt files if it exists in the processed folder
        if self.graph_data is None and self.experiment_configuration['paths']['data'].joinpath(f'{self.db_name}').joinpath('processed').exists():
            self.graph_data = GraphDataset(root=str(self.experiment_configuration['paths']['data']),
                                              name=self.db_name,
                                              task=self.experiment_configuration.get('task', None),
                                              precision=self.experiment_configuration.get('precision', 'float')
                                              )
            # raise an error if the graph data is still None
        if self.graph_data is None:
            raise ValueError(f'Could not load the graph data for {self.db_name} from {self.experiment_configuration["paths"]["data"]}. Please check the configuration and the data generation function.')


    def load_configuration_splits(self):
        splits_path = self.experiment_configuration['paths']['splits']
        if splits_path.suffix == '.json':
            self.experiment_configuration['splits'] = load_splits(splits_path)
        else:
            self.experiment_configuration['splits'] = load_splits(splits_path.joinpath(f'{self.db_name}_splits.json'))
        if 'pretraining_datasets' in self.experiment_configuration or 'finetuning_datasets' in self.experiment_configuration:
            # check whether the split file exists
            if 'pretraining_datasets' in self.experiment_configuration:
                self.experiment_configuration['split_appendix'] = 'pretraining_' + '_'.join(self.experiment_configuration['pretraining_datasets'])
                split_string = f'{self.db_name}_{self.experiment_configuration["split_appendix"]}_splits.json'
                new_splits_path = splits_path.joinpath(split_string)
                if new_splits_path.exists():
                    return
            elif 'finetuning_datasets' in self.experiment_configuration:
                self.experiment_configuration['split_appendix'] = 'finetuning_' + '_'.join(self.experiment_configuration['finetuning_datasets'])
                split_string = f'{self.db_name}_{self.experiment_configuration["split_appendix"]}_splits.json'
                new_splits_path = splits_path.joinpath(split_string)
                if new_splits_path.exists():
                    return
            # otherwise check whether all split files for the datasets exist
            for dataset in self.experiment_configuration['single_datasets']:
                if not splits_path.joinpath(f'{dataset}_splits.json').exists():
                    self.create_split_file()
            paths = [splits_path for dataset in self.experiment_configuration['single_datasets']]
            datasets = self.experiment_configuration['single_datasets']
            pretraining_ids = []
            finetuning_ids = []
            if 'pretraining_datasets' in self.experiment_configuration:
                # get the ids by positions in the self.experiment_configuration['single_datasets']
                pretraining_ids = [self.experiment_configuration['single_datasets'].index(dataset) for dataset in self.experiment_configuration['pretraining_datasets']]
            if 'finetuning_datasets' in self.experiment_configuration:
                # get the ids by positions in the self.experiment_configuration['single_datasets']
                finetuning_ids = [self.experiment_configuration['single_datasets'].index(dataset) for dataset in self.experiment_configuration['finetuning_datasets']]
            # create the pretraining respective finetuning splits
            pretraining_finetuning(paths, datasets, pretraining_ids=pretraining_ids, finetuning_ids=finetuning_ids)


    def create_split_file(self):
        # create the splits
        if self.experiment_configuration.get('with_splits', True):
            create_splits(self.db_name, Path(self.experiment_configuration['paths']['data']),
                          Path(self.experiment_configuration['paths']['splits']),
                          folds=self.experiment_configuration['validation_folds'], graph_data=self.graph_data)
        else:
            if self.experiment_configuration.get('split_function', None) is not None:
                # generate splits
                self.experiment_configuration['split_function'](self.experiment_configuration['paths']['splits'],
                                                                **self.experiment_configuration['split_function_args'],
                                                                graph_data=self.graph_data)
            else:
                raise ValueError(
                    f'Please specify a split function in the main config file for the dataset {self.db_name} using the key "split_function".')





def load_splits(splits_path: Path) -> dict:
    """
    Load the splits for a given database.
    :param splits_path: Path to the splits file.
    :return: A dictionary containing the train, validation, and test splits.
    """


    with open(splits_path, "rb") as f:
        splits = json.load(f)

    test_indices = [x['test'] for x in splits]
    train_indices = [x['model_selection'][0]['train'] for x in splits]
    vali_indices = [x['model_selection'][0]['validation'] for x in splits]

    # validate that all splits have the same length
    if not (len(test_indices) == len(train_indices) == len(vali_indices)):
        raise ValueError(f'Split file {splits_path} is not correctly formatted. Different number of folds in train, validation and test splits.')
    # validate that all indices are disjoint
    for i in range(len(test_indices)):
        if len(set(test_indices[i]).intersection(set(train_indices[i]))) > 0:
            raise ValueError(f'Split file {splits_path} is not correctly formatted. Overlapping indices in train and test splits in fold {i}.')
        if len(set(test_indices[i]).intersection(set(vali_indices[i]))) > 0:
            raise ValueError(f'Split file {splits_path} is not correctly formatted. Overlapping indices in validation and test splits in fold {i}.')
        if len(set(train_indices[i]).intersection(set(vali_indices[i]))) > 0:
            raise ValueError(f'Split file {splits_path} is not correctly formatted. Overlapping indices in train and validation splits in fold {i}.')

    return {'test': test_indices, 'train': train_indices, 'validation': vali_indices}



def load_preprocessed_data_and_parameters(run_id, validation_id, config_id, validation_folds, graph_data:GraphDataset, run_config, para: Parameters):
    experiment_configuration = run_config.config
    # path do db and db
    draw = False
    print_results = False
    save_weights = False
    save_prediction_values = False
    plot_graphs = False
    print_layer_init = False
    # if debug mode is on, turn on all print and draw options
    if experiment_configuration['mode'] == "debug":
        draw = experiment_configuration['additional_options']['draw']
        print_results = experiment_configuration['additional_options']['print_results']
        save_prediction_values = experiment_configuration['additional_options']['save_prediction_values']
        save_weights = experiment_configuration['additional_options']['save_weights']
        plot_graphs = experiment_configuration['additional_options']['plot_graphs']

    unique_label_dicts = []
    label_layer_ids = []
    unique_properties = []
    properties_layer_ids = []
    for i, l in enumerate(run_config.layers):
        # add the labels to the graph data
        new_unique = l.get_unique_layer_dicts()
        for x in new_unique:
            if x not in unique_label_dicts:
                unique_label_dicts.append(x)
        property_dicts = l.get_unique_property_dicts()
        if property_dicts:
            for x in property_dicts:
                property_dict_name = x.get_property_string()
                if property_dict_name not in unique_properties:
                    unique_properties.append(property_dict_name)

    for label_dict in unique_label_dicts:
        label_path = experiment_configuration['paths']['labels'].joinpath(f'{graph_data.name}').joinpath(f"{graph_data.name}_labels_{get_label_string(label_dict)}.pt")
        if os.path.exists(label_path):
            g_labels = load_labels(path=label_path)
            graph_data.node_labels[get_label_string(label_dict)] = g_labels
        else:
            # raise an error if the file does not exist and add the absolute path to the error message
            raise FileNotFoundError(f"File {label_path} does not exist")

    for prop_name in unique_properties:
        valid_values = {}
        for i, l in enumerate(run_config.layers):
            for j, c in enumerate(l.layer_heads):
                if c.property_dict.property_dict is not None:
                    if c.property_dict.get_property_string() == prop_name:
                        valid_values[(i,j)] = c.property_dict.get_values()

        graph_data.properties[prop_name] = Properties(path=experiment_configuration['paths']['properties'], db_name=graph_data.name,
                                                      property_name=prop_name,
                                                      valid_values=valid_values)

    """
        BenchmarkGraphs parameters
    """
    para.set_data_param(db=graph_data.name,
                        max_coding=1,
                        layers=run_config.layers, node_features=1,
                        run_config=run_config)

    """
        Network parameters
    """
    para.set_evaluation_param(run_id=run_id, n_val_runs=validation_folds,
                              validation_id=validation_id,
                              config_id=config_id,
                              n_epochs=run_config.epochs,
                              learning_rate=run_config.lr,
                              dropout=run_config.dropout,
                              balance_data=run_config.config.get('balance_data', False),
                              convolution_grad=True,
                              resize_graph=True)

    """
    Print, save and draw parameters
    """
    para.set_print_param(no_print=False, print_results=print_results, net_print_weights=True, print_number=1,
                         draw=draw, save_weights=save_weights,
                         save_prediction_values=save_prediction_values, plot_graphs=plot_graphs,
                         print_layer_init=print_layer_init)

    """
        Get the first index in the results directory that is not used
    """
    para.set_file_index(size=6)

    if para.plot_graphs:
        # if not exists create the directory
        if not os.path.exists(experiment_configuration['paths']['results'].joinpath(f"{para.db}/Plots")):
            os.makedirs(experiment_configuration['paths']['results'].joinpath(f"{para.db}/Plots"))
        for i in range(0, len(graph_data.graphs)):
            gdtgl.draw_graph(graph_data.graphs[i], graph_data.graph_labels[i],
                             experiment_configuration['paths']['results'].joinpath(f"{para.db}/Plots/graph_{str(i).zfill(5)}.png"))

