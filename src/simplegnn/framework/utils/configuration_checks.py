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

# allowed keys and enum values of the `transfer:` block (spec 18 B4); the
# schema is validated statically at config-load time so typos and bad enum
# values fail before any preprocessing or training starts
TRANSFER_CONFIG_KEYS = {'source', 'strategy', 'head', 'invariant_transfer', 'freeze', 'random_init'}
TRANSFER_SOURCE_KEYS = {'results_path', 'dataset', 'select'}
TRANSFER_SOURCE_SELECT_KEYS = {'config_id', 'run_id', 'validation_id'}
TRANSFER_HEAD_KEYS = {'reinit'}
TRANSFER_INVARIANT_KEYS = {'match', 'on_missing', 'allow_non_canonical', 'min_overlap_warn'}
TRANSFER_STRATEGY_VALUES = ('finetune', 'linear_probe')
TRANSFER_HEAD_REINIT_VALUES = ('always', 'never')
TRANSFER_MATCH_VALUES = ('hashes', 'none')
TRANSFER_ON_MISSING_VALUES = ('reinit', 'zero')


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
            'To use float or double precision, please specify the key "precision" in the main configuration file. The default value is "float".')
        hyperparameter_configuration['precision'] = 'float'

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

    # hash-keyed transfer (spec 18 B4): validate the transfer block statically
    if hyperparameter_configuration.get('transfer', None):
        check_transfer_configuration(hyperparameter_configuration['transfer'])
        # one-hot (or any non-constant) input features have dataset-dependent
        # widths, so the transferred linear layers would not shape-match
        input_features = hyperparameter_configuration.get('input_features', None)
        for features in (input_features if isinstance(input_features, list) else [input_features]):
            if isinstance(features, dict) and features.get('name') != 'constant':
                print(f'Warning: a transfer block is configured but input_features uses '
                      f'"{features.get("name")}". Non-constant input features have '
                      f'dataset-dependent widths (e.g. one-hot over dataset labels), so the '
                      f'first linear layer will not shape-match across datasets. '
                      f'Use input_features: {{name: constant, value: 1.0}} for transferable widths.')


def _check_transfer_enum(value, allowed, key):
    if value not in allowed:
        raise ValueError(
            f'transfer.{key} must be one of {list(allowed)}, got {value!r}.')


def check_transfer_configuration(transfer_configuration):
    """
    Statically validate a ``transfer:`` block of the hyperparameter
    configuration (spec 18 B4): allowed keys, enum values, and the required
    source keys. Pure YAML validation — existence checks that depend on the
    source run having happened (checkpoint, sidecar, label hashes) live in
    :func:`check_transfer_runtime_requirements` and run at run start.
    """
    if isinstance(transfer_configuration, list):
        for entry in transfer_configuration:
            check_transfer_configuration(entry)
        return
    if not isinstance(transfer_configuration, dict):
        raise ValueError(
            f'The transfer block must be a mapping with keys {sorted(TRANSFER_CONFIG_KEYS)}, '
            f'got {type(transfer_configuration).__name__}.')
    unknown = set(transfer_configuration) - TRANSFER_CONFIG_KEYS
    if unknown:
        raise ValueError(
            f'Unknown key(s) {sorted(unknown)} in the transfer block. '
            f'Allowed keys: {sorted(TRANSFER_CONFIG_KEYS)}.')

    # source: required, with required results_path/dataset
    source = transfer_configuration.get('source', None)
    if not isinstance(source, dict):
        raise ValueError(
            'Please specify transfer.source as a mapping with the keys "results_path" '
            '(results directory of the pretraining run) and "dataset" (db name of the '
            'source checkpoint).')
    unknown = set(source) - TRANSFER_SOURCE_KEYS
    if unknown:
        raise ValueError(
            f'Unknown key(s) {sorted(unknown)} in transfer.source. '
            f'Allowed keys: {sorted(TRANSFER_SOURCE_KEYS)}.')
    for key in ('results_path', 'dataset'):
        if key not in source:
            raise ValueError(
                f'Please specify transfer.source.{key} in the hyperparameter configuration '
                f'file (the results directory and db name of the pretraining run).')
    select = source.get('select', 'best')
    if isinstance(select, dict):
        unknown = set(select) - TRANSFER_SOURCE_SELECT_KEYS
        if unknown:
            raise ValueError(
                f'Unknown key(s) {sorted(unknown)} in transfer.source.select. '
                f'Allowed keys: {sorted(TRANSFER_SOURCE_SELECT_KEYS)}.')
        for key, value in select.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(
                    f'transfer.source.select.{key} must be an integer, got {value!r}.')
    elif select not in (None, 'best', 'Best', 'best_validation'):
        raise ValueError(
            f"transfer.source.select must be 'best', 'best_validation' or a mapping with "
            f"config_id/run_id/validation_id, got {select!r}.")

    _check_transfer_enum(transfer_configuration.get('strategy', 'finetune'),
                         TRANSFER_STRATEGY_VALUES, 'strategy')

    random_init = transfer_configuration.get('random_init', False)
    if not isinstance(random_init, bool):
        raise ValueError(f'transfer.random_init must be a boolean, got {random_init!r}.')

    head = transfer_configuration.get('head', None) or {}
    if not isinstance(head, dict):
        raise ValueError(f'transfer.head must be a mapping, got {head!r}.')
    unknown = set(head) - TRANSFER_HEAD_KEYS
    if unknown:
        raise ValueError(
            f'Unknown key(s) {sorted(unknown)} in transfer.head. '
            f'Allowed keys: {sorted(TRANSFER_HEAD_KEYS)}.')
    _check_transfer_enum(head.get('reinit', 'always'), TRANSFER_HEAD_REINIT_VALUES, 'head.reinit')

    invariant = transfer_configuration.get('invariant_transfer', None) or {}
    if not isinstance(invariant, dict):
        raise ValueError(f'transfer.invariant_transfer must be a mapping, got {invariant!r}.')
    unknown = set(invariant) - TRANSFER_INVARIANT_KEYS
    if unknown:
        raise ValueError(
            f'Unknown key(s) {sorted(unknown)} in transfer.invariant_transfer. '
            f'Allowed keys: {sorted(TRANSFER_INVARIANT_KEYS)}.')
    _check_transfer_enum(invariant.get('match', 'hashes'),
                         TRANSFER_MATCH_VALUES, 'invariant_transfer.match')
    _check_transfer_enum(invariant.get('on_missing', 'reinit'),
                         TRANSFER_ON_MISSING_VALUES, 'invariant_transfer.on_missing')
    if not isinstance(invariant.get('allow_non_canonical', False), bool):
        raise ValueError(
            f'transfer.invariant_transfer.allow_non_canonical must be a boolean, '
            f'got {invariant["allow_non_canonical"]!r}.')
    min_overlap_warn = invariant.get('min_overlap_warn', 0.10)
    if (isinstance(min_overlap_warn, bool) or not isinstance(min_overlap_warn, (int, float))
            or not 0.0 <= float(min_overlap_warn) <= 1.0):
        raise ValueError(
            f'transfer.invariant_transfer.min_overlap_warn must be a number in [0, 1], '
            f'got {min_overlap_warn!r}.')

    freeze = transfer_configuration.get('freeze', None) or []
    if not isinstance(freeze, list) or any(not isinstance(pattern, str) for pattern in freeze):
        raise ValueError(
            f'transfer.freeze must be a list of state-dict prefix globs '
            f'(e.g. ["net_layers.0*"]), got {freeze!r}.')


def check_transfer_runtime_requirements(transfer_configuration, graph_data=None,
                                        required_labels=None):
    """
    Existence checks of a ``transfer:`` block that can only run once the
    source (pretraining) run has happened and the target dataset is
    preprocessed (spec 18 B4) — called at run start, right before the
    transfer is applied. Returns the resolved source checkpoint path.

    Raises if the resolved source sidecar is missing while
    ``invariant_transfer.match == 'hashes'``, or if a needed target label
    file lacks a hash vocabulary (legacy v1 format). ``required_labels``
    restricts the label check to the label descriptions the invariant layers
    actually consume (default: every entry of ``graph_data.node_labels`` —
    callers with a built model should pass the consumed set, since the
    dataset's built-in in-memory ``'primary'`` labels never carry hashes).
    """
    from simplegnn.framework.utils.transfer import (
        MISSING_HASHES_HINT, MISSING_SIDECAR_HINT, resolve_source_checkpoint,
        sidecar_path_for)

    check_transfer_configuration(transfer_configuration)
    source = transfer_configuration['source']
    checkpoint_path = resolve_source_checkpoint(
        Path(source['results_path']), source['dataset'], source.get('select', 'best'))
    match = (transfer_configuration.get('invariant_transfer', None) or {}).get('match', 'hashes')
    if match == 'hashes':
        sidecar_path = sidecar_path_for(checkpoint_path)
        if not sidecar_path.is_file():
            raise FileNotFoundError(
                f'invariant_transfer.match is "hashes" but the transfer-key sidecar '
                f'{sidecar_path} of the resolved source checkpoint is missing: '
                f'{MISSING_SIDECAR_HINT}.')
        node_labels = getattr(graph_data, 'node_labels', None) or {}
        if required_labels is None:
            required_labels = sorted(node_labels)
        for label_name in required_labels:
            labels = node_labels.get(label_name, None)
            if labels is not None and getattr(labels, 'label_hashes', None) is None:
                raise ValueError(
                    f'invariant_transfer.match is "hashes" but the target label file '
                    f'"{label_name}" of dataset "{getattr(labels, "dataset_name", "?")}" has '
                    f'no hash vocabulary (legacy v1 format): {MISSING_HASHES_HINT}.')
    return checkpoint_path
