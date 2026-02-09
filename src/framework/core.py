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

from datasets.graph_dataset import GraphDataset, get_graph_data
from framework.utils.configuration_checks import check_model_configuration_file, check_hyperparameter_configuration_file, \
    check_main_configuration_file
from framework.model_configuration import ModelConfiguration
from framework.utils.evaluation import model_selection_evaluation
from framework.utils.parameters import Parameters
from framework.utils.preprocessing import Preprocessing, load_preprocessed_data_and_parameters, load_splits
from framework.run_configuration import get_run_configs

warnings.filterwarnings("ignore", category=DeprecationWarning)


class FrameworkMain:
    """
    This is the main class to run the all the GNN experiments.
    All experiment parameters are defined in two different configuration files:
    1) main config file: contains general information about the datasets, paths, training parameters, contains also the paths to the network config files
    2) network config file: contains information about the neural network architecture, hyperparameters
    """

    def __init__(self, main_config_path: Path, pretrained_network=None):
        """
        Initialization of the FrameworkMain class. Loads the configuration files and checks their consistency. If given, load a pretrained network to run the experiments.
        :param main_config_path: path to the main config file (contains the path to the experiment config file)
        :param pretrained_network: neural network model to run the experiments. Default is None. Otherwise, use the given model as starting point.
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
        Run all configurations for all the datasets defined in the main config file (default) and save the results in the results directory defined in the config file.
        :param num_threads: number of threads to use for the grid search. Default is -1. If -1, use all available threads.
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

    def evaluate_results(self, evaluate_best_model=False, evaluate_validation_only=False):
        """
        Evaluate the results of the experiments for all the datasets defined in the main config file (default) or only over the datasets defined in the dataset_names list.
        parameters:
        - dataset_names: list of strings with the names of the datasets to evaluate
        - evaluation_type: string with the type of evaluation to perform. Default is 'accuracy'. Other options are 'loss'. For accuracy, take the best model according to the validation accuracy. For loss, take the best model according to the validation loss.
        - evaluate_best_model: boolean to evaluate the best model of the experiment. Default is False. If False, evaluate the results of the experiment.
        - evaluate_validation_only: boolean to evaluate only on the validation set. Returns the epoch with the best validation accuracy/loss. Default is False.
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
        Run over all the datasets defined in the main config file (default) or only over the datasets defined in the dataset_names list.
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
        Merge the configuration files defined per dataset in the main config file with the model configuration file and hyperparameter configuration file.
        :param dataset_configuration: dictionary with the dataset configuration
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
        all_configurations = []
        for key in self.network_configurations:
            for configuration in self.network_configurations[key]:
                all_configurations.append(configuration)
        return all_configurations

    def preprocessing(self, num_threads=-1):
        num_datasets = len(self.network_configurations)
        # parallelize over the datasets
        if num_threads == -1:
            num_threads = min(num_datasets, os.cpu_count())
            num_threads = self.main_config.get('num_workers', num_threads)
            num_threads = min(num_threads, num_datasets)
        joblib.Parallel(n_jobs=num_threads)(joblib.delayed(Preprocessing)(self.network_configurations[key]) for key in self.network_configurations.keys())


    def run_configuration(self, graph_data: GraphDataset, run_config, validation_id:int=0, run_id:int=0, config_id:int=None):
        """
        Run the experiment for a given configuration, a fixed validation split and a fixed run id determining the seed for the random number generator.
        parameters:
        :param graph_data: graph data object containing all the information about the graph(s)
        :param run_config: run configuration object containing all the information about the hyperparameters
        :param validation_id: integer with the validation id, i.e., which validation split to use
        :param run_id: integer with the run id, i.e., which run to use. The id determines the seed for the random number generator
        :param config_id: integer with the configuration id, i.e., which hyperparameter configuration to use
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

    def load_ordinary_model(self, db_name, config_id=0, run_id=0, validation_id=0, best=True, experiment_db_id=0):
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

    def load_model(self, db_name, config_id=0, run_id=0, validation_id=0, best=True, experiment_db_id=0):
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

    def evaluate_model_on_graphs(self, db_name, db_id=0, graph_ids=[], config_id=0, run_id=0, validation_id=0, best=True):
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

    # evaluate the model on the test data
    def evaluate_model(self, db_name, db_id=0, config_id=0, run_id=0, validation_id=0, best=True):
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

def collect_paths(main_configuration, model_configuration, dataset_configuration=None):
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

def copy_experiment_config(absolute_path, experiment_configuration, experiment_configuration_path,
                           graph_db_name):
    import shutil
    results_path = experiment_configuration['paths']['results']
    if not results_path.joinpath(f"{graph_db_name}/config.yml"):
        source_path = Path(absolute_path).joinpath(experiment_configuration_path)
        destination_path = results_path.joinpath(f"{graph_db_name}/config.yml")
        shutil.copy2(source_path, destination_path)


def preprocess_graph_data(experiment_configuration:dict):
    """
            Create Input data, information and labels from the graphs for training and testing
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


