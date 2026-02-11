"""
Framework Core: Main experiment orchestration for GNN training and evaluation.

This module provides the main entry point for running GNN experiments in the
SimpleGNN framework. It handles the complete experiment lifecycle including
configuration loading, data preprocessing, hyperparameter grid search,
model training, and result evaluation.

Key Classes
-----------
FrameworkMain : Main experiment orchestrator
    Manages the entire experiment pipeline from YAML configuration to
    final model evaluation and result analysis.

Key Functions
-------------
preprocess_graph_data : Load and preprocess graph datasets
collect_paths : Merge and validate configuration file paths
copy_experiment_config : Archive experiment configuration files

Usage Examples
--------------
Basic experiment workflow:

>>> from pathlib import Path
>>> from simplegnn.framework.core import FrameworkMain
>>> experiment = FrameworkMain(Path('config/main.yml'))
>>> experiment.preprocessing(num_threads=4)
>>> experiment.run_configurations(num_threads=-1)
>>> experiment.evaluate_results()
>>> experiment.run_best_configuration(num_threads=-1)
>>> experiment.evaluate_results(evaluate_best_model=True)

See Also
--------
framework.model_configuration.ModelConfiguration : Single config training
framework.run_configuration.get_run_configs : Grid search generation
"""
import json
import os
import time
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml


import warnings

from simplegnn.datasets.graph_dataset import GraphDataset, get_graph_data
from simplegnn.framework.utils.configuration_checks import check_model_configuration_file, check_hyperparameter_configuration_file, \
    check_main_configuration_file
from simplegnn.framework.model_configuration import ModelConfiguration
from simplegnn.framework.utils.evaluation import model_selection_evaluation
from simplegnn.framework.utils.parameters import Parameters
from simplegnn.framework.utils.preprocessing import Preprocessing, load_preprocessed_data_and_parameters, load_splits
from simplegnn.framework.run_configuration import get_run_configs

warnings.filterwarnings("ignore", category=DeprecationWarning)


class FrameworkMain:
    """
    Main orchestrator for GNN experiments in the SimpleGNN framework.

    This class manages the complete experiment lifecycle for training and
    evaluating Graph Neural Networks. It handles loading and validation of
    three-tier YAML configurations (main config, model architecture config,
    hyperparameter config), coordinates data preprocessing, executes
    hyperparameter grid searches with k-fold cross-validation, and performs
    model selection and final evaluation.

    The framework supports parallel execution via joblib and can handle
    multiple datasets simultaneously. Experiments are fully reproducible
    through seeded random number generation.

    Parameters
    ----------
    main_config_path : Path
        Absolute or relative path to the main YAML configuration file.
        This file specifies datasets, task types, and paths to model and
        hyperparameter configuration files.
    pretrained_network : torch.nn.Module or tuple or str, optional
        Pretrained network to use as initialization for training.
        - None: Train from random initialization (default)
        - torch.nn.Module: Use given model as starting point
        - tuple: (FrameworkMain, experiment_db_id) to load from experiment
        - str: 'best' to load best model from pretraining datasets

    Attributes
    ----------
    main_config_path : Path
        Absolute path to the main configuration file.
    pretrained_network : torch.nn.Module or tuple or str or None
        Pretrained network configuration for transfer learning.
    main_config : dict
        Parsed main YAML configuration containing dataset specifications
        and global settings.
    network_configurations : dict
        Merged configurations for all datasets. Keys are dataset names,
        values are lists of configuration dictionaries containing merged
        main, model, and hyperparameter settings.

    Examples
    --------
    Basic experiment workflow:

    >>> from pathlib import Path
    >>> experiment = FrameworkMain(Path('examples/basic/main.yml'))
    >>> experiment.preprocessing(num_threads=4)
    >>> experiment.run_configurations(num_threads=-1)
    >>> experiment.evaluate_results()

    Transfer learning from pretrained model:

    >>> pretrain_exp = FrameworkMain(Path('pretrain/main.yml'))
    >>> finetune_exp = FrameworkMain(
    ...     Path('finetune/main.yml'),
    ...     pretrained_network=(pretrain_exp, 0)
    ... )
    >>> finetune_exp.run_configurations()

    Notes
    -----
    **Configuration File Structure:**

    The framework uses a three-tier YAML configuration system:

    1. Main config (main.yml): Datasets, task types, config paths
    2. Model config: Layer architecture as list of layer definitions
    3. Hyperparameter config: Training params (optimizer, lr, epochs, etc.)

    Lists in configuration files trigger grid search over all combinations.

    **Parallelization:**

    The framework parallelizes across (validation_fold, run_id, config_id)
    combinations using joblib. Set num_threads=-1 to use all CPU cores.

    See Also
    --------
    framework.model_configuration.ModelConfiguration : Single config training
    framework.utils.preprocessing.Preprocessing : Data preprocessing
    framework.utils.evaluation.model_selection_evaluation : Result analysis
    """

    def __init__(self, main_config_path: Path, pretrained_network=None):
        """
        Initialize FrameworkMain with configuration loading and validation.

        Loads the main YAML configuration file, validates its structure,
        and merges it with model and hyperparameter configurations for each
        dataset specified in the main config.

        Parameters
        ----------
        main_config_path : Path
            Path to the main YAML configuration file. Will be converted to
            absolute path if relative.
        pretrained_network : torch.nn.Module or tuple or str, optional
            Pretrained model initialization (default: None).

        Raises
        ------
        FileNotFoundError
            If main_config_path does not exist.
        ValueError
            If configuration file cannot be loaded or parsed, or if
            validation checks fail.

        Notes
        -----
        This method automatically merges all configuration files for each
        dataset and validates their structure using checks from
        configuration_checks module.
        """

        # get absolute path
        main_config_path = Path.absolute(main_config_path)

        self.main_config_path = main_config_path
        self.pretrained_network = pretrained_network
        if not os.path.exists(main_config_path):
            raise FileNotFoundError(f"Config file {main_config_path} not found")
        try:
            self.main_config = yaml.safe_load(open(main_config_path)) # load the main config file
            check_main_configuration_file(self.main_config)
        except:
            raise ValueError(f"Config file {main_config_path} could not be loaded")
        self.network_configurations = {}
        for dataset in self.main_config['datasets']:
            self.merge_configuration_files(dataset) # merge all information from the main config file, the model config file and the hyperparameter config file



    def run_configurations(self, num_threads=-1):
        """
        Execute hyperparameter grid search for all datasets and configurations.

        Performs a comprehensive grid search over all hyperparameter
        combinations specified in the configuration files. For each dataset,
        this method parallelizes training across (validation_fold, run_id,
        config_id) tuples using joblib. Results are saved incrementally to
        the results directory.

        The method automatically handles:
        - Parallel job distribution across available CPU cores
        - OpenMP thread configuration to avoid conflicts
        - Configuration file archiving to results directory
        - Skipping of already completed runs

        Parameters
        ----------
        num_threads : int, optional
            Number of parallel worker threads for grid search.
            - -1: Use all available CPU cores (default)
            - 1: Sequential execution
            - >1: Use specified number of threads

        Notes
        -----
        **Grid Search Parallelization:**

        Parallelization occurs over the cartesian product of:
        - Validation folds (k-fold cross-validation)
        - Run IDs (random seed variations for statistical significance)
        - Configuration IDs (hyperparameter combinations)

        **OpenMP Handling:**

        Sets OMP_NUM_THREADS=1 when num_threads != 1 to prevent nested
        parallelism conflicts between joblib and PyTorch/NumPy.

        **Results Storage:**

        For each run, results are saved as CSV files in:
        {results_path}/{dataset}/Results/{config_name}_Results_*.csv

        Configuration files are copied to:
        {results_path}/{dataset}/config.yml

        Examples
        --------
        Run grid search using all available cores:

        >>> experiment.run_configurations(num_threads=-1)

        Sequential execution for debugging:

        >>> experiment.run_configurations(num_threads=1)

        See Also
        --------
        run_configuration : Single configuration training execution
        framework.run_configuration.get_run_configs : Generate config grid
        """
        torch.set_warn_always(False)
        # set omp num threads to 1 to avoid conflicts with OpenMP if num_threads is unequal to 1
        if num_threads != 1:
            os.environ['OMP_NUM_THREADS'] = '1'         # set omp_num_threads to 1 to avoid conflicts with OpenMP
        # iterate over all datasets given in the main config file
        for dataset in self.network_configurations.keys():
            # iterate over all configurations for the given dataset
            for i, configuration in enumerate(self.network_configurations[dataset]):
                print(f"Running experiment configuration {i+1}/{len(self.network_configurations[dataset])} for dataset {dataset}")
                max_threads = os.cpu_count()                 # determine the number of parallel jobs
                num_threads = min(configuration.get('num_workers', num_threads), num_threads)
                if num_threads == -1:
                    num_threads = max_threads


                graph_data = preprocess_graph_data(configuration)
                # copy config file to the results directory if it is not already there
                absolute_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                absolute_path = Path(absolute_path)
                copy_experiment_config(absolute_path, configuration,
                                       configuration.get('network_config_file', ''),
                                       dataset)

                # get all possible hyperparameter configurations from the config files
                run_configs = get_run_configs(configuration)
                config_id_names = {}
                for idx, run_config in enumerate(run_configs):
                    config_id = idx + configuration.get('config_id', 0)
                    config_id_names[idx] = f'Configuration_{str(config_id).zfill(6)}'
                print(f"Total number of hyperparameter configurations: {len(run_configs)}")

                # zip all configurations for parallelization and run the grid search
                run_loops = [(validation_id, run_id, c_idx) for validation_id in range(len(configuration.get('splits')['train'])) for run_id in range(configuration.get('num_runs', 1)) for c_idx in range(len(run_configs))]
                num_threads = min(num_threads, len(run_loops))
                print(f"Run the grid search for dataset {dataset} using {len(configuration.get('splits')['train'])}-fold cross-validation and {num_threads} number of parallel jobs")
                joblib.Parallel(n_jobs=num_threads)(
                    joblib.delayed(self.run_configuration)(graph_data=graph_data,
                                                           run_config=run_configs[run_loops[i][2]],
                                                           validation_id=run_loops[i][0],
                                                           run_id=run_loops[i][1],
                                                           config_id=config_id_names[run_loops[i][2]]) for i in range(len(run_loops)))

    def evaluate_results(self, evaluate_best_model=False,
                         evaluate_validation_only=False):
        """
        Analyze experiment results and perform model selection.

        Aggregates results from all hyperparameter configurations and
        validation folds to identify the best performing model. Generates
        summary statistics (mean, std) across runs and saves them to CSV
        files. Can operate in two modes: grid search evaluation (default)
        or final test set evaluation (when evaluate_best_model=True).

        Parameters
        ----------
        evaluate_best_model : bool, optional
            Whether to evaluate the best model on the test set.
            - False: Perform model selection on validation results (default)
            - True: Evaluate best configuration on test set
        evaluate_validation_only : bool, optional
            Return only validation set performance metrics.
            - False: Return validation and test metrics (default)
            - True: Return only validation metrics (for hyperparameter tuning)

        Notes
        -----
        **Evaluation Modes:**

        Grid Search Evaluation (evaluate_best_model=False):
        - Aggregates results across all configurations
        - Selects best config based on validation performance
        - Saves summary to: {results_path}/{dataset}/summary.csv
        - Skips if summary.csv already exists

        Final Test Evaluation (evaluate_best_model=True):
        - Evaluates best model from grid search on test set
        - Saves to: {results_path}/{dataset}/summary_best.csv
        - Requires prior grid search evaluation

        **Output CSV Columns:**

        Summary files contain: config_id, mean_train_acc, std_train_acc,
        mean_val_acc, std_val_acc, mean_test_acc, std_test_acc, etc.

        Examples
        --------
        Select best configuration from grid search:

        >>> experiment.run_configurations()
        >>> experiment.evaluate_results()

        Evaluate best model on test set:

        >>> experiment.run_best_configuration()
        >>> experiment.evaluate_results(evaluate_best_model=True)

        See Also
        --------
        framework.utils.evaluation.model_selection_evaluation : Core evaluator
        run_best_configuration : Re-run best configuration
        """
        # set omp_num_threads to 1 to avoid conflicts with OpenMP
        os.environ['OMP_NUM_THREADS'] = '1'
        # iterate over the databases
        # iterate over the databases
        for dataset in self.network_configurations.keys():
            for i, configuration in enumerate(self.network_configurations[dataset]):
                if evaluate_best_model:
                    # check whether evaluation has been done before
                    out_path = configuration['paths']['results'].joinpath(dataset).joinpath('summary_best.csv')
                    if out_path.exists():
                        print(f"Evaluation for the best model of dataset {dataset} already exists. Skipping the evaluation.")
                        continue
                    else:
                        print(f"Evaluate the best model of the experiment for dataset {dataset}")
                else:
                    out_path = configuration['paths']['results'].joinpath(dataset).joinpath('summary.csv')
                    if out_path.exists():
                        print(f"Evaluation for the experiment of dataset {dataset} already exists. Skipping the evaluation.")
                        continue
                    else:
                        print(f"Evaluate the results of the experiment for dataset {dataset}")

                model_selection_evaluation(db_name = dataset,
                                           evaluate_best_model=evaluate_best_model,
                                           experiment_config=configuration,
                                       evaluate_validation_only=evaluate_validation_only)

    def run_best_configuration(self, num_threads=-1):
        """
        Re-run the best configuration from grid search for final evaluation.

        After identifying the best hyperparameter configuration via
        evaluate_results(), this method re-trains that configuration
        multiple times (default: 3 runs) to obtain statistically robust
        final test set performance estimates. Parallelizes across
        (run_id, validation_fold) pairs.

        Parameters
        ----------
        num_threads : int, optional
            Number of parallel worker threads.
            - -1: Use all available CPU cores (default)
            - 1: Sequential execution
            - >1: Use specified number of threads

        Notes
        -----
        **Workflow:**

        1. Load best config_id from grid search evaluation
        2. Train best config with multiple runs (evaluation_run_number)
        3. Save models with prefix 'Best_Configuration_' in Models/
        4. Save results in Results/ directory

        **Number of Runs:**

        Controlled by config parameter 'evaluation_run_number' (default: 3).
        Multiple runs provide mean and std estimates for final performance.

        **Model Naming:**

        Best models are saved as:
        model_Best_Configuration_{config_id}_run_{run_id}_val_step_{val_id}.pt

        Examples
        --------
        Standard workflow:

        >>> experiment.run_configurations()
        >>> experiment.evaluate_results()
        >>> experiment.run_best_configuration(num_threads=-1)
        >>> experiment.evaluate_results(evaluate_best_model=True)

        See Also
        --------
        evaluate_results : Model selection and evaluation
        run_configuration : Single configuration execution
        """
        # set omp_num_threads to 1 to avoid conflicts with OpenMP
        os.environ['OMP_NUM_THREADS'] = '1'
        # iterate over the databases
        for dataset in self.network_configurations.keys():
            for i, configuration in enumerate(self.network_configurations[dataset]):
                print(f"Running experiment for dataset {dataset}")
                # derive validation folds from the configuration file using the splits
                validation_folds = len(configuration.get('splits', {}).get('test', None))

                # load the config file
                # run the best models
                # parallelize over (run_id, validation_id) pairs
                evaluation_run_number = configuration.get('evaluation_run_number', 3)

                # determine the number of parallel jobs
                max_threads = os.cpu_count()
                num_threads = min(configuration.get('num_workers', num_threads), num_threads)
                if num_threads == -1:
                    num_threads = max_threads

                parallelization_pairs = [(run_id, validation_id) for run_id in range(evaluation_run_number) for validation_id in range(validation_folds)]
                num_threads = min(num_threads, len(parallelization_pairs))
                graph_data = preprocess_graph_data(configuration)
                best_config_id = None
                configuration['best_model'] = True
                # get the best configuration and run it
                best_config_id = model_selection_evaluation(db_name=dataset,
                                                            get_best_model=True,
                                                            experiment_config=configuration)
                run_configs = get_run_configs(configuration)
                config_id = f'Best_Configuration_{str(best_config_id).zfill(6)}'
                print(f"Run the best model of dataset {dataset} using {evaluation_run_number} different runs. The number of parallel jobs is {num_threads}")
                joblib.Parallel(n_jobs=num_threads)(joblib.delayed(self.run_configuration)(
                                                            graph_data=graph_data,
                                                            run_config=run_configs[best_config_id],
                                                            validation_id=validation_id,
                                                            run_id=run_id,
                                                            config_id=config_id)
                                                 for run_id, validation_id in parallelization_pairs)

    def merge_configuration_files(self, dataset_configuration):
        """
        Merge main, model, and hyperparameter configs for a single dataset.

        Loads model architecture and hyperparameter YAML files specified in
        the dataset configuration, validates them, and merges all settings
        into a single unified configuration dictionary. Handles both single
        datasets and concatenated dataset lists.

        Parameters
        ----------
        dataset_configuration : dict
            Dataset-specific configuration from main config file containing
            'name', 'paths', and other dataset settings.

        Notes
        -----
        **Merge Order:**

        1. Start with dataset_configuration from main config
        2. Load and merge model_configuration from paths['models']
        3. Load and merge hyperparameter_configuration from
           paths['hyperparameters']

        Later configs override earlier ones for duplicate keys.

        **Dataset Name Handling:**

        - Single dataset: name is string, used as-is
        - Multiple datasets: name is list, concatenated with underscores
          (e.g., ['MUTAG', 'PROTEINS'] → 'MUTAG_PROTEINS')

        **Validation:**

        Calls check_model_configuration_file() and
        check_hyperparameter_configuration_file() to ensure all
        mandatory parameters are present.

        See Also
        --------
        framework.utils.configuration_checks : Config validation functions
        """

        # First load the model configuration file and check it for errors, if no errors found, merge it with the main configuration file
        model_config_path = dataset_configuration.get('paths', {}).get('models', None)
        # load the config file
        model_configuration = yaml.load(open(model_config_path), Loader=yaml.FullLoader)
        check_model_configuration_file(dataset_configuration, model_configuration)
        for key in model_configuration:
            dataset_configuration[key] = model_configuration[key]

        # load hyperparameter configuration file and check it for errors, if no errors found, merge it with the main configuration file
        hyperparameter_config_path = dataset_configuration.get('paths', {}).get('hyperparameters', None)
        # load the config file
        hyperparameter_configuration = yaml.load(open(hyperparameter_config_path), Loader=yaml.FullLoader)
        check_hyperparameter_configuration_file(hyperparameter_configuration)
        for key in hyperparameter_configuration:
            dataset_configuration[key] = hyperparameter_configuration[key]

        dataset_string_name = None
        if isinstance(dataset_configuration['name'], list):
            str_concatenation = ''
            for i, name in enumerate(dataset_configuration['name']):
                str_concatenation += name
                if i < len(dataset_configuration['name']) - 1:
                    str_concatenation += '_'
            dataset_configuration['name'] = str_concatenation
            if str_concatenation not in self.network_configurations:
                self.network_configurations[str_concatenation] = [dataset_configuration.copy()]
            else:
                self.network_configurations[str_concatenation].append(dataset_configuration.copy())
            self.network_configurations[str_concatenation][-1]['single_datasets'] = dataset_configuration['name']

        else:
            dataset_string_name = dataset_configuration['name']
            if dataset_string_name not in self.network_configurations:
                self.network_configurations[dataset_string_name] = [dataset_configuration.copy()]
            else:
                self.network_configurations[dataset_string_name].append(dataset_configuration.copy())

    def get_configuration_list(self):
        """
        Get flattened list of all dataset configurations.

        Returns
        -------
        list of dict
            All configuration dictionaries across all datasets, flattened
            from the network_configurations nested structure.

        Notes
        -----
        Useful for iterating over all configurations regardless of which
        dataset they belong to.
        """
        all_configurations = []
        for key in self.network_configurations:
            for configuration in self.network_configurations[key]:
                all_configurations.append(configuration)
        return all_configurations

    def preprocessing(self, num_threads=-1):
        """
        Preprocess all datasets in parallel.

        Executes dataset-specific preprocessing including:
        - Loading raw graph data
        - Computing node/edge labels (BFS, automorphism, centrality, etc.)
        - Generating pairwise properties for ShareGNN
        - Creating/validating train/val/test splits
        - Saving preprocessed data to disk

        Parameters
        ----------
        num_threads : int, optional
            Number of parallel preprocessing workers.
            - -1: Use min(num_datasets, CPU_count) (default)
            - 1: Sequential preprocessing
            - >1: Use specified number of threads

        Notes
        -----
        **Preprocessing Steps:**

        For each dataset:
        1. Load raw graph data from data_path
        2. Compute required node/edge labels based on model config
        3. Generate splits if not already present
        4. For ShareGNN: compute pairwise distance/property matrices
        5. Save all preprocessed data to paths['data']

        **Parallelization:**

        Preprocessing is parallelized across datasets (not within a single
        dataset). Each dataset's preprocessing runs independently.

        **Caching:**

        Preprocessed data is cached on disk. Re-running preprocessing will
        skip datasets that already have valid cached data.

        Examples
        --------
        Preprocess all datasets using 4 threads:

        >>> experiment.preprocessing(num_threads=4)

        Sequential preprocessing for debugging:

        >>> experiment.preprocessing(num_threads=1)

        See Also
        --------
        framework.utils.preprocessing.Preprocessing : Core preprocessing class
        datasets.graph_dataset.GraphDataset : Dataset container
        """
        num_datasets = len(self.network_configurations)
        # parallelize over the datasets
        if num_threads == -1:
            num_threads = min(num_datasets, os.cpu_count())
            num_threads = self.main_config.get('num_workers', num_threads)
            num_threads = min(num_threads, num_datasets)
        joblib.Parallel(n_jobs=num_threads)(joblib.delayed(Preprocessing)(self.network_configurations[key]) for key in self.network_configurations.keys())


    def run_configuration(self, graph_data: GraphDataset, run_config,
                          validation_id: int = 0, run_id: int = 0,
                          config_id: int = None):
        """
        Execute training for a single hyperparameter configuration.

        Trains a GNN model with specified hyperparameters on a particular
        train/validation/test split. This is the atomic unit of execution
        called in parallel by run_configurations() and
        run_best_configuration().

        Parameters
        ----------
        graph_data : GraphDataset
            Preprocessed graph dataset containing node features, edge
            indices, labels, and optional node/edge properties.
        run_config : RunConfiguration
            Configuration object containing hyperparameters (optimizer, lr,
            batch size, etc.) and model architecture specification.
        validation_id : int, optional
            Index of the validation fold to use from the splits (default: 0).
            Determines train/val/test split assignment.
        run_id : int, optional
            Random seed variation index for statistical robustness
            (default: 0). Seed = 42 + validation_id + n_val_runs * run_id.
        config_id : int or None, optional
            Hyperparameter configuration identifier for naming saved results
            (default: None). Used in output filenames.

        Notes
        -----
        **Random Seed:**

        Deterministic seed computed as:
        seed = 42 + validation_id + para.n_val_runs * run_id

        This ensures reproducibility while varying across runs and folds.

        **Output Files:**

        Results saved to:
        {results_path}/{dataset}/Results/{name}_{config_id}_Results_run_id_{run_id}_validation_step_{validation_id}.json

        Models saved to:
        {results_path}/{dataset}/Models/model_{config_id}_run_{run_id}_val_step_{validation_id}.pt

        **Skip Logic:**

        If result JSON already exists, skips training to avoid redundant
        computation. Useful for resuming interrupted grid searches.

        **Pretrained Networks:**

        If self.pretrained_network is set, uses it to initialize model
        weights before training (transfer learning).

        See Also
        --------
        framework.model_configuration.ModelConfiguration : Training executor
        framework.utils.preprocessing.load_preprocessed_data_and_parameters
        """
        final_path = run_config.config['paths']['results'].joinpath(f'{graph_data.name}/Results/')
        configuration_file_name = f'{run_config.config["name"]}_{str(config_id).zfill(6)}_Results_run_id_{run_id}_validation_step_{validation_id}.json'
        # check if the configuration file exists
        if not final_path.joinpath(configuration_file_name).exists():
            print(f"Run the model for dataset {run_config.config['name']} with config_id {config_id}, run_id {run_id} and validation_id {validation_id}")
            para = Parameters()
            load_preprocessed_data_and_parameters(config_id=config_id,
                                                  run_id=run_id,
                                                  validation_id=validation_id,
                                                  validation_folds=run_config.config.get('validation_folds', 10),
                                                  graph_data=graph_data, run_config=run_config, para=para)

            # split the data into training, validation and test data
            seed = 42 + validation_id + para.n_val_runs * run_id
            # load the data splits
            split_data = run_config.config['splits']
            test_data = split_data['test'][validation_id]
            train_data = split_data['train'][validation_id]
            validation_data = split_data['validation'][validation_id]

            # create the model configuration object
            graph_model = ModelConfiguration(run_id, validation_id, graph_data, (np.array(train_data), np.array(validation_data), np.array(test_data)), seed, para)

            # run the model, if a pretrained network is given, use it
            if isinstance(self.pretrained_network, tuple):
                # tuple FrameworkMain object and experiment_db_id
                graph_model.train_configuration(pretrained_network=self.pretrained_network[0].load_model(db_name=para.run_config.config['name'], run_id=run_id, validation_id=validation_id, best=True, experiment_db_id=self.pretrained_network[1]))
            elif isinstance(self.pretrained_network, str):
                if self.pretrained_network in ['best', 'Best']:
                    # TODO load only the best model that achieved the best test accuracy on the pretraining datasets
                    pass
            else:
                graph_model.train_configuration(pretrained_network=self.pretrained_network)


            # create a configuration file TODO fill the configuration file with more infos
            with open(final_path.joinpath(configuration_file_name), 'w') as f:
                # add train_validation_test_data to the configuration file
                # get current time in yyyy-mm-dd HH:MM:SS format
                current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                f.write(json.dumps({
                    'experiment_time': current_time,
                    'config_id': config_id,
                    'run_id': run_id,
                    'validation_id': validation_id,
                }, indent=4))


        else:
            print(f"Configuration file {configuration_file_name} already exists. Skipping the run for dataset {run_config.config['name']} with config_id {config_id}, run_id {run_id} and validation_id {validation_id}")

    def load_ordinary_model(self, db_name, config_id=0, run_id=0,
                            validation_id=0, best=True, experiment_db_id=0):
        """
        Load a trained non-ShareGNN model from disk.

        Parameters
        ----------
        db_name : str
            Dataset name to load model for.
        config_id : int, optional
            Configuration ID (default: 0). Ignored if best=True.
        run_id : int, optional
            Run ID for random seed variation (default: 0).
        validation_id : int, optional
            Validation fold index (default: 0).
        best : bool, optional
            Load best configuration from grid search (default: True).
        experiment_db_id : int, optional
            Index into network_configurations[db_name] list (default: 0).

        Returns
        -------
        torch.nn.Module
            Loaded model with restored state_dict.

        Raises
        ------
        FileNotFoundError
            If model file or model directory does not exist.

        Notes
        -----
        This method is for loading classical GNN models (GCN, GIN, GAT, etc.).
        For ShareGNN models, use load_model() instead.
        """
        experiment_configuration = self.network_configurations[db_name][experiment_db_id]
        graph_data = preprocess_graph_data(experiment_configuration)
        run_configs = get_run_configs(experiment_configuration)
        # get the path to the model
        path_to_models = experiment_configuration['paths']['results'].joinpath(db_name).joinpath('Models')

        if best:
            if path_to_models.exists():
                # get one file that contais the string 'Best_Configuration' in the name
                curr_path = next(path_to_models.glob('*Best_Configuration*'))
            else:
                raise FileNotFoundError(f"Model directory {path_to_models} not found")
            config_id = int(curr_path.name.split('_')[3])
            model_path = path_to_models.joinpath(
                f'model_Best_Configuration_{str(config_id).zfill(6)}_run_{run_id}_val_step_{validation_id}.pt')
        else:
            if not path_to_models.exists():
                raise FileNotFoundError(f"Model directory {path_to_models} not found")
            model_path = path_to_models.joinpath(
                f'model_Configuration_{str(config_id).zfill(6)}_run_{run_id}_val_step_{validation_id}.pt')

        run_config = run_configs[config_id]
        # check if the model exists
        if model_path.exists():
            with open(model_path, 'r'):
                para = Parameters()
                load_preprocessed_data_and_parameters(config_id=config_id,
                                                      run_id=run_id,
                                                      validation_id=validation_id,
                                                      graph_data=graph_data,
                                                      run_config=run_config,
                                                      para=para,
                                                      validation_folds=experiment_configuration.get('validation_folds',
                                                                                                    10))

                """
                    Get the first index in the results directory that is not used
                """
                para.set_file_index(size=6)
                net = OrdinaryGNN(graph_data=graph_data,
                                        para=para,
                                        seed=0, device=run_config.config.get('device', 'cpu'))

                net.load_state_dict(torch.load(model_path, weights_only=True))
            return net
        else:
            raise FileNotFoundError(f"Model {model_path} not found")

    def load_model(self, db_name, config_id=0, run_id=0, validation_id=0,
                   best=True, experiment_db_id=0):
        """
        Load a trained ShareGNN model from disk.

        Parameters
        ----------
        db_name : str
            Dataset name to load model for.
        config_id : int, optional
            Configuration ID (default: 0). Ignored if best=True.
        run_id : int, optional
            Run ID for random seed variation (default: 0).
        validation_id : int, optional
            Validation fold index (default: 0).
        best : bool, optional
            Load best configuration from grid search (default: True).
            If True, automatically finds config_id from saved best model.
        experiment_db_id : int, optional
            Index into network_configurations[db_name] list (default: 0).

        Returns
        -------
        torch.nn.Module
            Loaded ShareGNN model with restored state_dict.

        Raises
        ------
        FileNotFoundError
            If model file or model directory does not exist.

        Notes
        -----
        This method is specifically for ShareGNN models. For classical
        GNN models (GCN, GIN, etc.), use load_ordinary_model() instead.

        Model files are located at:
        {results_path}/{db_name}/Models/model_{prefix}_{config_id}_run_{run_id}_val_step_{validation_id}.pt

        where prefix is 'Best_Configuration' if best=True, else
        'Configuration'.
        """
        experiment_configuration = self.network_configurations[db_name][experiment_db_id]
        graph_data = preprocess_graph_data(experiment_configuration)
        run_configs = get_run_configs(experiment_configuration)
        # get the path to the model
        path_to_models = experiment_configuration['paths']['results'].joinpath(db_name).joinpath('Models')

        if best:
            if path_to_models.exists():
                # get one file that contais the string 'Best_Configuration' in the name
                curr_path = next(path_to_models.glob('*Best_Configuration*'))
            else:
                raise FileNotFoundError(f"Model directory {path_to_models} not found")
            config_id = int(curr_path.name.split('_')[3])
            model_path = path_to_models.joinpath(f'model_Best_Configuration_{str(config_id).zfill(6)}_run_{run_id}_val_step_{validation_id}.pt')
        else:
            if not path_to_models.exists():
                raise FileNotFoundError(f"Model directory {path_to_models} not found")
            model_path = path_to_models.joinpath(f'model_Configuration_{str(config_id).zfill(6)}_run_{run_id}_val_step_{validation_id}.pt')

        run_config = run_configs[config_id]
        # check if the model exists
        if model_path.exists():
            with open(model_path, 'r'):
                para = Parameters()
                load_preprocessed_data_and_parameters(config_id=config_id,
                                                      run_id=run_id,
                                                      validation_id=validation_id,
                                                      graph_data=graph_data,
                                                      run_config=run_config,
                                                      para=para,
                                                      validation_folds=len(experiment_configuration.get('splits')['train']),
                                                        )

                """
                    Get the first index in the results directory that is not used
                """
                para.set_file_index(size=6)
                net = ShareGNN.ShareGNN(graph_data=graph_data,
                                       para=para,
                                       seed=0, device=run_config.config.get('device', 'cpu'))

                net.load_state_dict(torch.load(model_path, weights_only=True))
            return net
        else:
            raise FileNotFoundError(f"Model {model_path} not found")

    def evaluate_model_on_graphs(self, db_name, db_id=0, graph_ids=[],
                                 config_id=0, run_id=0, validation_id=0,
                                 best=True):
        """
        Evaluate a trained model on specific graphs.

        Loads a trained model and evaluates it on a custom set of graph
        indices. Useful for analyzing model behavior on specific examples
        or performing custom evaluations beyond standard train/val/test
        splits.

        Parameters
        ----------
        db_name : str
            Dataset name containing the graphs.
        db_id : int, optional
            Index into network_configurations[db_name] list (default: 0).
        graph_ids : list of int, optional
            Indices of graphs to evaluate on (default: []).
        config_id : int, optional
            Configuration ID (default: 0). Ignored if best=True.
        run_id : int, optional
            Run ID for random seed variation (default: 0).
        validation_id : int, optional
            Validation fold index (default: 0).
        best : bool, optional
            Load best configuration from grid search (default: True).

        Returns
        -------
        outputs : torch.Tensor
            Model predictions for each graph.
            Shape: (len(graph_ids), num_classes)
        labels : torch.Tensor
            True labels for each graph.
            Shape: (len(graph_ids),)
        accuracy : float
            Classification accuracy on the specified graphs.

        Notes
        -----
        Evaluation is performed with torch.no_grad() for efficiency.
        Model is automatically set to evaluation mode.
        """
        # evaluate the performance of the model on the test data
        experiment_configuration = self.network_configurations[db_name][db_id]
        graph_data = preprocess_graph_data(experiment_configuration)
        data = np.asarray(graph_ids, dtype=int)
        outputs = torch.zeros((len(data), graph_data.num_classes), dtype=torch.double)
        # load the model
        net = self.load_model(db_name, config_id=config_id, run_id=run_id, validation_id=validation_id, best=best)
        with torch.no_grad():
            for j, data_pos in enumerate(data, 0):
                outputs[j] = net(graph_data[data_pos].x, data_pos)
            labels = graph_data.y[data]
            # calculate the errors between the outputs and the labels by getting the argmax of the outputs and the labels
            arg_max_outputs = torch.argmax(outputs, dim=1)
            correct_outputs = torch.eq(arg_max_outputs, labels)
            num_correct = torch.sum(correct_outputs).item()
            accuracy = num_correct / len(correct_outputs)
            print(f"Dataset: {db_name}, Run Id: {run_id}, Validation Split Id: {validation_id}, Accuracy: {accuracy}")
        return outputs, labels, accuracy

    def evaluate_model(self, db_name, db_id=0, config_id=0, run_id=0,
                      validation_id=0, best=True):
        """
        Evaluate a trained model on the test set.

        Loads a trained model and evaluates it on the test split specified
        by validation_id. Computes predictions and accuracy for all test
        graphs.

        Parameters
        ----------
        db_name : str
            Dataset name to evaluate on.
        db_id : int, optional
            Index into network_configurations[db_name] list (default: 0).
        config_id : int, optional
            Configuration ID (default: 0). Ignored if best=True.
        run_id : int, optional
            Run ID for random seed variation (default: 0).
        validation_id : int, optional
            Validation fold index determining test split (default: 0).
        best : bool, optional
            Load best configuration from grid search (default: True).

        Returns
        -------
        outputs : torch.Tensor
            Model predictions for each test graph.
            Shape: (num_test_graphs, num_classes)
        labels : torch.Tensor
            True labels for each test graph.
            Shape: (num_test_graphs,)
        accuracy : float
            Classification accuracy on the test set.

        Notes
        -----
        Test split is loaded from splits file at:
        {paths['splits']}/splits.json

        Evaluation uses test_data = splits[0][validation_id].

        Evaluation is performed with torch.no_grad() for efficiency.
        """
        # evaluate the performance of the model on the test data
        experiment_configuration = self.network_configurations[db_name][db_id]
        graph_data = preprocess_graph_data(experiment_configuration)
        split_data = load_splits(experiment_configuration['paths']['splits'])
        test_data = np.asarray(split_data[0][validation_id], dtype=int)
        outputs = torch.zeros((len(test_data), graph_data.num_classes), dtype=torch.double)
        # load the model
        net = self.load_model(db_name, config_id=config_id, run_id=run_id, validation_id=validation_id, best=best)
        with torch.no_grad():
            for j, data_pos in enumerate(test_data, 0):
                outputs[j] = net(graph_data[data_pos].x, data_pos)
            labels = graph_data.y[test_data]
            # calculate the errors between the outputs and the labels by getting the argmax of the outputs and the labels
            counter = 0
            correct = 0
            for i, x in enumerate(outputs, 0):
                if torch.argmax(x) == torch.argmax(labels[i]):
                    correct += 1
                counter += 1
            accuracy = correct / counter
            print(f"Dataset: {db_name}, Run Id: {run_id}, Validation Split Id: {validation_id}, Accuracy: {accuracy}")
        return outputs, labels, accuracy

def collect_paths(main_configuration, model_configuration,
                  dataset_configuration=None):
    """
    Merge and validate file paths from configuration hierarchy.

    Combines paths from main config, model config, and dataset config with
    proper precedence handling. Validates that all required paths are
    present and checks whether ShareGNN-specific paths (properties, labels)
    are needed based on model architecture.

    Parameters
    ----------
    main_configuration : dict
        Main YAML configuration containing global path defaults.
    model_configuration : dict
        Model architecture configuration that may override paths.
    dataset_configuration : dict or None, optional
        Dataset-specific configuration (default: None).

    Returns
    -------
    dict
        Validated paths dictionary containing 'data', 'results', 'splits',
        and optionally 'properties' and 'labels' keys.

    Raises
    ------
    ValueError
        If paths not found in dataset configuration.
    FileNotFoundError
        If required path keys ('data', 'results', 'splits') are missing,
        or if ShareGNN models require 'properties'/'labels' but they are
        absent.

    Notes
    -----
    **Path Precedence:**

    1. Dataset configuration paths (highest priority)
    2. Model configuration paths
    3. Main configuration paths (lowest priority)

    **ShareGNN Detection:**

    Automatically detects ShareGNN models by checking for layers with
    layer_type='invariant_based_convolution'. If found, requires
    'properties' and 'labels' paths.

    **Results Appendix:**

    If dataset_configuration contains 'pretraining_datasets' or
    'finetuning_datasets', appends suffix to results path for organization.
    """
    # first look into the main config file
    paths = deepcopy(main_configuration.get('paths', {}))
    # copy to dataset configuration if it does not exist TODO use only the paths from the dataset configuration
    if 'paths' not in dataset_configuration:
        dataset_configuration['paths'] = paths

    if 'pretraining_datasets' in dataset_configuration:
        dataset_configuration['results_appendix'] = 'pretraining_' + '_'.join(dataset_configuration['pretraining_datasets'])
    if 'finetuning_datasets' in dataset_configuration:
        dataset_configuration['results_appendix'] = 'finetuning_' + '_'.join(dataset_configuration['finetuning_datasets'])
    if 'results_appendix' in dataset_configuration:
        dataset_configuration['paths']['results'] = dataset_configuration['paths']['results'] + dataset_configuration['results_appendix'] + '/'

    # if there are paths in the experiment config file, overwrite the paths TODO change this experiment config should only be for network definition
    if model_configuration.get('paths', None) is not None:
        if model_configuration['paths'].get('data', None) is not None:
            paths['data'] = model_configuration['paths']['data']
        if model_configuration['paths'].get('results', None) is not None:
            paths['results'] = model_configuration['paths']['results']
        if model_configuration['paths'].get('splits', None) is not None:
            paths['splits'] = model_configuration['paths']['splits']
        if model_configuration['paths'].get('properties', None) is not None:
            paths['properties'] = model_configuration['paths']['properties']
        if model_configuration['paths'].get('labels', None) is not None:
            paths['labels'] = model_configuration['paths']['labels']

    # get the paths from the dataset configuration
    paths = dataset_configuration.get('paths', None)
    if paths is None:
        raise ValueError("Paths not found in the dataset configuration. Please specify the paths in the dataset configuration file.")



    # check whether one of the paths is missing
    if 'data' not in paths:
        raise FileNotFoundError("Data path is missing")
    if 'results' not in paths:
        raise FileNotFoundError("Results path is missing")
    if 'splits' not in paths:
        raise FileNotFoundError("Splits path is missing")
    # Not all GNNs need properties and labels go over the networks in experiment configuration and check whether properties and labels are needed
    need_props_and_labels = False
    for network in model_configuration.get('models', []):
        for layer in network:
            if layer.get('layer_type') == 'invariant_based_convolution':
                need_props_and_labels = True
                break
    if need_props_and_labels:
        if 'properties' not in paths:
            raise FileNotFoundError("Properties path is missing")
        if 'labels' not in paths:
            raise FileNotFoundError("Labels path is missing")

    return paths

def copy_experiment_config(absolute_path, experiment_configuration,
                           experiment_configuration_path, graph_db_name):
    """
    Archive experiment configuration to results directory.

    Copies the experiment configuration YAML file to the results directory
    for reproducibility. Ensures that experiment settings are preserved
    alongside results.

    Parameters
    ----------
    absolute_path : Path
        Absolute path to the src directory.
    experiment_configuration : dict
        Merged experiment configuration containing paths.
    experiment_configuration_path : str or Path
        Relative path to the configuration file to copy.
    graph_db_name : str
        Dataset name for organizing results directory.

    Notes
    -----
    Configuration is copied to:
    {results_path}/{graph_db_name}/config.yml

    If the file already exists, it is not overwritten to preserve original
    experiment settings.
    """
    import shutil
    results_path = experiment_configuration['paths']['results']
    if not results_path.joinpath(f"{graph_db_name}/config.yml"):
        source_path = Path(absolute_path).joinpath(experiment_configuration_path)
        destination_path = results_path.joinpath(f"{graph_db_name}/config.yml")
        shutil.copy2(source_path, destination_path)


def preprocess_graph_data(experiment_configuration: dict):
    """
    Load and prepare graph dataset for training.

    Loads the graph dataset from disk, applies task-specific preprocessing,
    sets input/output features, and moves data to the specified device.
    This function wraps get_graph_data() with experiment configuration.

    Parameters
    ----------
    experiment_configuration : dict
        Merged configuration containing dataset name, paths, task type,
        features, format, precision, and device.

    Returns
    -------
    GraphDataset
        Preprocessed graph dataset ready for training. Dataset is moved to
        the device specified in configuration (CPU or CUDA).

    Notes
    -----
    **Configuration Keys Used:**

    - name: Dataset name (e.g., 'MUTAG', 'ZINC')
    - paths['data']: Path to raw/preprocessed data
    - task: 'graph_classification', 'graph_regression', 'node_classification'
    - input_features: Node feature type (e.g., 'one_hot_degree', 'atom_type')
    - output_features: Target feature type
    - format: Dataset format ('RuleGNNDataset', 'TUDataset', 'ZINC', etc.)
    - precision: 'float' or 'double' for tensor dtype
    - device: 'cpu' or 'cuda'

    **Device Handling:**

    Dataset is automatically moved to device specified in config. Ensures
    all tensors (features, labels, properties) are on correct device before
    training.

    See Also
    --------
    datasets.graph_dataset.get_graph_data : Core dataset loading function
    """
    graph_data = get_graph_data(db_name=experiment_configuration['name'], data_path=experiment_configuration['paths']['data'],
                                task=experiment_configuration.get('task', 'graph_classification'),
                                input_features=experiment_configuration.get('input_features', None),
                                output_features=experiment_configuration.get('output_features', None),
                                graph_format=experiment_configuration.get('format', 'RuleGNNDataset'),
                                precision=experiment_configuration.get('precision', 'double'),
                                experiment_config=experiment_configuration)
    # move the dataset to the device
    graph_data.to(experiment_configuration.get('device', 'cpu'))
    return graph_data


