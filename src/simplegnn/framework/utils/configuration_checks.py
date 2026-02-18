# list of mandatory configuration parameters for the main configuration file
from pathlib import Path

from simplegnn.datasets import custom_datasets

MANDATORY_MAIN_CONFIG_PARAMS = [
    'name',
    'source',
    'task',
    'paths',
]

MANDATORY_MAIN_CONFIG_PATHS_PARAMS = [
    'data',
    'results',
    'splits',
    'models',
    'hyperparameters',
]

MANDATORY_MODELS_CONFIG_PARAMS = [

]

MANDATORY_HYPERPARAMETERS_CONFIG_PARAMS = [

]


def check_main_configuration_file(main_config):
    """
    Check the main configuration file for errors.
    """
    # go through all datasets and mandatory fields
    for key in main_config:
        if key != 'datasets':
            continue
        for dataset in main_config['datasets']:
            # check the name
            if 'name' not in dataset:
                raise ValueError(f'Please specify the name of the dataset in the main configuration file.')
            if 'paths' not in dataset:
                raise ValueError(f'Please specify the paths in the main configuration file.')
            else:
                paths = dataset['paths']
                if 'data' not in paths:
                    raise ValueError(f'Please specify the data path in the main configuration file.')
                else:
                    # set data path to absolute path
                    paths['data'] = Path.absolute(Path(paths['data']))
                if 'results' not in paths:
                    raise ValueError(f'Please specify the results path in the main configuration file.')
                else:
                    # set results path to absolute path
                    paths['results'] = Path.absolute(Path(paths['results']))
                if 'hyperparameters' not in paths:
                    raise ValueError(f'Please specify the hyperparameters path in the main configuration file.')
                else:
                    paths['hyperparameters'] = Path.absolute(Path(paths['hyperparameters']))
                if 'models' not in paths:
                    raise ValueError(f'Please specify the models path in the main configuration file.')
                else:
                    paths['models'] = Path.absolute(Path(paths['models']))
                if 'splits' not in paths:
                    raise ValueError(f'Please specify the splits path in the main configuration file.')
                else:
                    paths['splits'] = Path.absolute(Path(paths['splits']))
                    # check whether the splits file exists
                    split_file_path = dataset['paths']['splits']
                    if not split_file_path.suffix == '.json':
                        split_file_path = split_file_path.joinpath(
                            dataset['name'] + '_splits.json')  # use default naming convention
                        dataset['paths']['splits'] = split_file_path
                    if not split_file_path.is_file():
                        raise FileNotFoundError(
                            f'There is no json file {split_file_path}. Use the script in scripts/generate_splits_files to create the splits file.')

                # set optional paths to absolute paths
                if 'labels' in paths:
                    paths['labels'] = Path.absolute(Path(paths['labels']))
                if 'properties' in paths:
                    paths['properties'] = Path.absolute(Path(paths['properties']))

            if 'task' not in dataset:
                raise ValueError(f'Please specify the task in the main configuration file.'
                                 'Choose between "graph_classification", "graph_regression", "node_classification" and "link_prediction".')
            if 'source' not in dataset:
                raise ValueError(f'Please specify the source of the dataset in the main configuration file.'
                                 'Choose between "TUDataset", "gnn_benchmark", "ZINC" and "generate_from_function".')
            else:
                if isinstance(dataset['source'], list):
                    if len(dataset['source']) != len(dataset.get('single_datasets', 0)):
                        raise ValueError(f'The number of types and datasets do not match.')
                    if dataset.get('data_generation_args', None) is not None:
                        if len(dataset['source']) != len(dataset['data_generation_args']):
                            raise ValueError(f'The number of types and data generation arguments do not match.')
                    for t in dataset['source']:
                        if t not in ['generate_from_function', 'TUDataset', 'gnn_benchmark', 'ZINC', 'planetoid',
                                     'Planetoid', 'Nell', 'ogbn']:
                            raise ValueError(
                                f'The type {t} is not supported. Please use "generate_from_function", "TUDataset", "gnn_benchmark" or "ZINC".')
                else:
                    if dataset['source'] not in ['generate_from_function', 'TUDataset', 'gnn_benchmark', 'ZINC',
                                                 'planetoid', 'Planetoid', 'Nell', 'ogbn', 'MoleculeNet',
                                                 'OGB_GraphProp', 'SubstructureBenchmark', 'NEL', 'QM9', 'QM7', 'path']:
                        raise ValueError(
                            f'The type {dataset["source"]} is not supported. '
                            f'Please use "generate_from_function", "TUDataset", "gnn_benchmark", "ZINC", "planetoid", "Planetoid", "Nell", "ogbn", "MoleculeNet", "OGB_GraphProp", "SubstructureBenchmark", "NEL", "QM9", "QM7" or "path".')

                if 'source' in dataset:
                    data_generation_args = dataset.get('data_generation_args', None)
                    if dataset['source'] == 'generate_from_function':
                        if not hasattr(custom_datasets, dataset['generate_function']):
                            raise ValueError(f"Generate function {dataset['generate_function']} not found")
                        else:
                            data_generation = getattr(custom_datasets, dataset['generate_function'])
                            if not callable(data_generation):
                                raise ValueError(f"Generate function {dataset['generate_function']} is not callable")
                            else:
                                dataset['data_generation'] = data_generation
                    else:
                        dataset['data_generation'] = dataset['source']
                    dataset['data_generation_args'] = data_generation_args


def check_model_configuration_file(dataset_configuration, model_configuration):
    """
    Check the model configuration file for errors.
    """
    if 'models' not in model_configuration:
        raise ValueError(f'Please specify the models in the model configuration file.')
    # Some models need additional preprocessing steps, e.g., invariant-based GNNs need properties and labels (check if the paths are given)
    need_props_and_labels = False
    for network in model_configuration.get('models', []):
        for layer in network:
            if layer.get('layer_type') == 'invariant_based_convolution':
                need_props_and_labels = True
                break
    if need_props_and_labels:
        if 'properties' not in dataset_configuration['paths']:
            raise FileNotFoundError("Properties path is missing")
        else:
            dataset_configuration['with_invariant_layers'] = True
        if 'labels' not in dataset_configuration['paths']:
            raise FileNotFoundError("Labels path is missing")
        else:
            dataset_configuration['with_invariant_layers'] = True
    else:
        dataset_configuration['paths']['properties'] = None
        dataset_configuration['paths']['labels'] = None
        dataset_configuration['with_invariant_layers'] = False


def check_hyperparameter_configuration_file(hyperparameter_configuration):
    """
    Check the hyperparameter configuration file for errors.
    TODO more checks, keep updated with the hyperparameters used in the experiments
    """
    ### check the input features
    if 'input_features' not in hyperparameter_configuration:
        raise ValueError(f'Please specify the input features in the main configuration file.')
    ### TODO per layer weight initialization also possible (if weight initialization is not given default is applied)
    if 'weight_initialization' not in hyperparameter_configuration:
        print(
            "Check the weight initialization of the network. If no weight initialization is given, the default weight initialization of PyTorch is used.")
        # raise ValueError(f'Please specify the weight initialization in the main configuration file.')

    if 'batch_size' not in hyperparameter_configuration:
        raise ValueError(
            f'Please specify the batch size in the experiment configuration file using the key "batch_size".')
    if 'epochs' not in hyperparameter_configuration:
        raise ValueError(
            f'Please specify the number of epochs in the experiment configuration file using the key "epochs".')
    if 'learning_rate' not in hyperparameter_configuration:
        raise ValueError(
            f'Please specify the learning rate in the experiment configuration file using the key "learning_rate".')
    if 'optimizer' not in hyperparameter_configuration:
        raise ValueError(
            f'Please specify the optimizer in the experiment configuration file using the key "optimizer".')
    if 'loss' not in hyperparameter_configuration:
        raise ValueError(
            f'Please specify the loss function in the experiment configuration file using the key "loss".')

    if 'device' not in hyperparameter_configuration:
        print(
            'To use the GPU, please specify the key "device" in the main configuration file. The default value is "cpu".')
        hyperparameter_configuration['device'] = 'cpu'

    if 'precision' not in hyperparameter_configuration:
        print(
            'To use float or double precision, please specify the key "precision" in the main configuration file. The default value is "double".')
        hyperparameter_configuration['precision'] = 'double'

    if 'mode' not in hyperparameter_configuration:
        print(
            'To use the mode, please specify the key "mode" in the main configuration file. The default value is "experiments".'
            'For debugging purposes, set the mode to "debug".')
        hyperparameter_configuration['mode'] = 'experiments'

    if 'early_stopping' not in hyperparameter_configuration:
        print(
            'To use early stopping, please specify the key "early_stopping" in the main configuration file. The default value is False.')
        hyperparameter_configuration['early_stopping'] = {'enabled': False, 'patience': 25}

    if 'rule_occurrence_threshold' not in hyperparameter_configuration:
        print(
            'To use the rule occurrence threshold, please specify the key "rule_occurrence_threshold" in the main configuration file. The default value is 1.')
        hyperparameter_configuration['rule_occurrence_threshold'] = 1
