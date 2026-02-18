import json
import os
from pathlib import Path

from simplegnn.datasets.graph_dataset import GraphDataset
from simplegnn.datasets.utils.edge_labeling import Properties
from simplegnn.datasets.utils.node_labeling import load_labels, get_label_string
from simplegnn.framework.utils.parameters import Parameters
from simplegnn.models.ShareGNN.preprocessing.preprocessing import preprocessing_from_config
from simplegnn.utils.utils import save_graphs


class Preprocessing:
    """
    Orchestrates dataset preprocessing, split generation, and folder structure creation.

    This class handles the complete preprocessing pipeline for GNN experiments:
    1. Creates directory structure for data, results, labels, and properties
    2. Generates or downloads graph datasets
    3. Loads or creates train/validation/test splits
    4. Performs model-specific preprocessing (e.g., ShareGNN invariant computation)

    The preprocessing is driven by dataset configurations that specify paths,
    data sources, split strategies, and model requirements.

    Parameters
    ----------
    dataset_configurations : list of dict
        List of dataset configuration dictionaries. Each configuration must contain:
        - 'name' : str
            Dataset name (e.g., 'MUTAG', 'ZINC')
        - 'paths' : dict
            Dictionary with keys 'data', 'results', 'splits', and optionally
            'labels', 'properties'
        - 'data_generation' : str, callable, or list
            Data source specification (e.g., 'TUDataset', custom generation function)
        - Optional: 'task', 'precision', 'with_invariant_layers', etc.

    Attributes
    ----------
    db_name : str
        Name of the current dataset being processed.
    graph_data : GraphDataset or None
        Loaded graph dataset object.
    experiment_configuration : dict
        Current dataset configuration being processed.
    generation_times_labels_path : Path or None
        Path to file tracking label generation timing.
    generation_times_properties_path : Path or None
        Path to file tracking property generation timing.

    Notes
    -----
    The constructor iterates over all dataset configurations and processes each one:
    - Creates folder structure (create_folders)
    - Generates or loads data (generate_data or load_data)
    - Loads splits (load_configuration_splits)
    - Optionally runs model-specific preprocessing

    Directory Structure Created
    ----------------------------
    results/<dataset_name>/
        ├── Plots/
        ├── Weights/
        ├── Models/
        └── Results/
    data/<dataset_name>/
        ├── raw/
        └── processed/
    labels/<dataset_name>/
    properties/<dataset_name>/
    splits/

    Examples
    --------
    >>> config = [{
    ...     'name': 'MUTAG',
    ...     'paths': {
    ...         'data': Path('data'),
    ...         'results': Path('results'),
    ...         'splits': Path('splits')
    ...     },
    ...     'data_generation': 'TUDataset',
    ...     'validation_folds': 10
    ... }]
    >>> prep = Preprocessing(config)
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
        """
        Create directory structure for experiment data, results, and preprocessing outputs.

        Creates the following folder hierarchy if it doesn't exist:
        - Data directory: Root for dataset storage
        - Results directory with subdirectories:
            - <dataset_name>/Plots: For graph visualizations
            - <dataset_name>/Weights: For model weight checkpoints
            - <dataset_name>/Models: For saved model architectures
            - <dataset_name>/Results: For experiment result CSV files
        - Splits directory: For train/val/test split JSON files
        - Labels directory (optional): For precomputed node labels
        - Properties directory (optional): For precomputed edge properties

        Also creates timing log files for tracking label and property generation.

        Notes
        -----
        All directory creation uses exist_ok=True and parents=True, so it is safe
        to call this method multiple times.

        The splits path can be either a directory or a file path. If it ends with
        .json, the parent directory is created instead.

        Labels and properties directories are only created if specified in the
        configuration (paths['labels'] and paths['properties']).

        Two timing log files are created in the results directory:
        - generation_times_labels.txt
        - generation_times_properties.txt
        """
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
            with open(Path(self.experiment_configuration['paths']['results']).joinpath('generation_times_labels.txt'), 'a') as f:
                f.write('Generation times for labels\n')
        if not Path(self.experiment_configuration['paths']['results']).joinpath('generation_times_properties.txt').exists():
            with open(Path(self.experiment_configuration['paths']['results']).joinpath('generation_times_properties.txt'), 'a') as f:
                f.write('Generation times for properties\n')
        self.generation_times_labels_path = self.experiment_configuration['paths']['results'].joinpath('generation_times_labels.txt')
        self.generation_times_properties_path = self.experiment_configuration['paths']['results'].joinpath('generation_times_properties.txt')

    def generate_data(self, dataset, data_generation_type, data_generation_args):
        """
        Generate or download graph dataset based on data generation configuration.

        Supports three data generation modes:
        1. Download from existing sources (e.g., 'TUDataset', 'OGB')
        2. Generate from custom function
        3. Merge multiple datasets (recursive mode for transfer learning)

        Parameters
        ----------
        dataset : str
            Name of the dataset to generate or download.
        data_generation_type : str, callable, list, or None
            - str: Name of existing dataset source ('TUDataset', 'OGB', 'NEL', etc.)
            - callable: Custom generation function that returns (graphs, labels)
            - list: List of generation types for dataset merging (transfer learning)
            - None: Attempt to load from existing NEL format
        data_generation_args : dict, list of dict, or None
            Arguments to pass to generation function. If data_generation_type is a list,
            this should be a list of dicts with matching length.

        Notes
        -----
        **Recursive Merge Mode** (when data_generation_type is a list):
        1. Recursively generates each dataset in the list
        2. Loads each generated dataset as GraphDataset
        3. Merges them into a single unified dataset

        **String Mode** (TUDataset, OGB, etc.):
        1. Downloads dataset using PyTorch Geometric loaders
        2. Creates raw/ and processed/ subdirectories
        3. Stores in data/<dataset_name>/

        **Function Mode** (callable):
        1. Calls generation function with provided arguments
        2. Expects function to return (graphs, labels) tuple
        3. Saves in NEL (Nodes, Edges, Labels) format using save_graphs()

        **NEL Mode** (None or explicit 'NEL'):
        Attempts to load existing data from NEL format files.

        The method skips generation if the processed/ directory already exists
        and contains data.pt file.

        Raises
        ------
        ValueError
            If dataset could not be generated or loaded with given configuration.
        """
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
            if data_generation_type == 'path':
                try:
                    self.graph_data = GraphDataset(root=str(self.experiment_configuration['paths']['data']),
                                                      name=dataset,
                                                      from_existing_data='NEL',
                                                      task=self.dataset_configuration.get('task', None)
                                                      )
                except:
                    print(f'There is no data in the path {self.experiment_configuration["paths"]["data"]} for the dataset {dataset}. Please check the configuration file.')
            elif data_generation_type != 'generate_from_function':
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
        """
        Load preprocessed graph dataset from disk.

        Attempts to load a GraphDataset from the processed/ directory. The dataset
        must have been previously generated and saved as data.pt file.

        Raises
        ------
        ValueError
            If graph data could not be loaded. This indicates either:
            - The processed/ directory doesn't exist
            - The data.pt file is missing or corrupted
            - The dataset configuration is incorrect

        Notes
        -----
        This method checks if self.graph_data is None before attempting to load,
        avoiding redundant loads if data is already in memory.

        The GraphDataset is initialized with:
        - root: Data directory path
        - name: Database name (self.db_name)
        - task: Task type from configuration (optional)
        - precision: 'float' or 'double' (defaults to 'float')
        """
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
        """
        Load train/validation/test splits from JSON files.

        Loads splits from the configured splits path and handles special cases for
        transfer learning (pretraining/finetuning scenarios with multiple datasets).

        Notes
        -----
        **Normal Mode:**
        Loads splits from either:
        - A direct file path (if splits_path ends with .json)
        - <splits_path>/<dataset_name>_splits.json

        **Transfer Learning Mode:**
        If 'pretraining_datasets' or 'finetuning_datasets' is in configuration:
        1. Checks for combined split file with naming pattern:
           <dataset_name>_<mode>_<dataset1>_<dataset2>_..._splits.json
        2. If not found, ensures individual split files exist for each dataset
        3. Calls pretraining_finetuning() to create combined splits

        The loaded splits are stored in self.experiment_configuration['splits']
        as a dictionary with keys 'train', 'validation', 'test', each containing
        lists of indices for each fold.

        See Also
        --------
        load_splits : Function that parses split JSON files
        create_split_file : Creates new split files if they don't exist
        """
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
        """
        Create new train/validation/test split files.

        Generates split files using either the default splitting strategy or
        a custom split function specified in the configuration.

        Notes
        -----
        **Default Mode** (with_splits=True):
        Uses create_splits() function to generate stratified k-fold splits
        with the number of folds specified by 'validation_folds' in configuration.

        **Custom Mode** (with_splits=False):
        Requires 'split_function' and 'split_function_args' in configuration.
        Calls the custom function with the splits path, graph data, and provided
        arguments.

        Split files are saved as JSON in the format expected by load_splits().

        Raises
        ------
        ValueError
            If with_splits=False but no split_function is specified in configuration.

        See Also
        --------
        create_splits : Default split generation function
        load_splits : Loads and validates split files
        """
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
    Load and validate train/validation/test splits from JSON file.

    Reads a split file in the standard format used by the framework and extracts
    train, validation, and test indices for each fold. Performs validation to
    ensure splits are properly formatted and disjoint.

    Parameters
    ----------
    splits_path : Path
        Path to the JSON splits file.

    Returns
    -------
    dict
        Dictionary with keys 'train', 'validation', 'test'. Each value is a list
        of index lists, one per fold. For example:
        {
            'train': [[0,1,2], [3,4,5], ...],       # indices for each fold
            'validation': [[3,4], [0,1], ...],
            'test': [[5,6,7], [2,8,9], ...]
        }

    Raises
    ------
    ValueError
        If split file is incorrectly formatted:
        - Different number of folds in train/validation/test splits
        - Overlapping indices between any pair of splits in any fold

    Notes
    -----
    **Expected JSON Format:**
    ```json
    [
        {
            "test": [5, 6, 7, ...],
            "model_selection": [
                {
                    "train": [0, 1, 2, ...],
                    "validation": [3, 4, ...]
                }
            ]
        },
        ...  # one entry per fold
    ]
    ```

    **Validation Checks:**
    1. All splits (train, validation, test) must have the same number of folds
    2. Within each fold, indices must be disjoint (no overlap between train/val/test)

    The function only uses the first model_selection entry (index 0), as the
    framework typically uses a single train/validation split per fold.
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
    """
    Load preprocessed node labels and edge properties, and configure parameters for training.

    This is a critical setup function that:
    1. Configures output/logging settings based on mode (debug vs. normal)
    2. Identifies required node labels and edge properties from layer configurations
    3. Loads precomputed labels and properties from disk
    4. Attaches them to the GraphDataset object
    5. Initializes all Parameters object settings for the experiment run

    Parameters
    ----------
    run_id : int
        Unique identifier for this experimental run.
    validation_id : int
        Index of the current validation fold (0 to validation_folds-1).
    config_id : int
        Unique identifier for this hyperparameter configuration.
    validation_folds : int
        Total number of cross-validation folds.
    graph_data : GraphDataset
        Graph dataset object that will be augmented with labels and properties.
    run_config : RunConfiguration
        Configuration object containing layer specifications, hyperparameters,
        and experiment settings.
    para : Parameters
        Parameters object to be configured with all experiment settings.

    Notes
    -----
    **Output Configuration:**
    The function sets output flags based on experiment_configuration['mode']:
    - If mode == 'debug': Enables drawing, printing, saving based on 'additional_options'
    - Otherwise: Disables all output (draw, print_results, save_weights, etc.)

    **Label and Property Loading:**
    The function scans all layers in run_config.layers to identify:
    1. Unique node label dictionaries (via get_unique_layer_dicts())
    2. Unique edge property dictionaries (via get_unique_property_dicts())

    Labels are loaded from:
        <labels_path>/<dataset_name>/<dataset_name>_labels_<label_string>.pt

    Properties are loaded from:
        <properties_path>/<dataset_name>/<property_name>/

    **Parameters Configuration:**
    Calls three setter methods on the Parameters object:
    - set_data_param(): Dataset name, layers, node features
    - set_evaluation_param(): Run IDs, epochs, learning rate, dropout
    - set_print_param(): Output and logging flags
    - set_file_index(): Auto-incrementing file index for results

    **Optional Graph Plotting:**
    If para.plot_graphs is True, generates and saves PNG visualizations
    of all graphs in the dataset.

    Raises
    ------
    FileNotFoundError
        If required label files do not exist at the expected paths.

    See Also
    --------
    load_labels : Loads node label tensors from .pt files
    get_label_string : Generates label filename from label dictionary
    Properties : Class for loading and caching edge properties
    """
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

