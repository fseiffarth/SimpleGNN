import itertools
from typing import List

from models.ShareGNN.utils import Layer
from models.layers.utils.layer_loader import layer_from_yml, check_network_architectures


class RunConfiguration:
    def __init__(self, config, network_architecture, layers, batch_size, lr, epochs, dropout, optimizer, weight_decay, loss, task="classification"):
        self.config = config
        self.network_architecture = network_architecture.copy()
        self.layers = layers.copy()
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.dropout = dropout
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.loss = loss
        self.task = task

    def print(self):
        print(f"Network architecture: {self.network_architecture}")
        print(f"Layers: {self.layers}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.lr}")
        print(f"Epochs: {self.epochs}")
        print(f"Dropout: {self.dropout}")
        print(f"Optimizer: {self.optimizer}")
        print(f"Weight decay: {self.weight_decay}")
        print(f"Loss: {self.loss}")

def generate_layer_options(layer_dict):
    options = []
    for label_type in layer_dict['labels']:
        properties_dict = []
        base_dict = {}
        for key, value in layer_dict.items():
            if key != 'labels' and key != 'properties':
                base_dict[key] = value
        if layer_dict.get('properties', None) is not None:
            for prop_val in layer_dict['properties']:
                properties_dict.append(prop_val)
        key_list = list(label_type.keys())
        # remove label_type from key list
        key_list.remove('label_type')
        # remove all keys where the value is None or an empty list
        for key in key_list:
            if label_type[key] is None or (type(label_type[key]) == list and len(label_type[key]) == 0):
                key_list.remove(key)
        if len(key_list) == 0:
            if properties_dict is None or len(properties_dict) == 0:
                options.append({'layer_type': layer_dict['layer_type'], 'channels': [{'labels': label_type}]})
            else:
                for prop_val in properties_dict:
                    options.append({'layer_type': layer_dict['layer_type'], 'channels': [{'labels': label_type, 'properties': prop_val.copy()}]})
        else:
            value_list = []
            for key in key_list:
                if type(label_type[key]) != list:
                    value_list.append([label_type[key]])
                else:
                    value_list.append(label_type[key])
            # get all value combinations as tuples over the value lists
            value_combinations = []
            for i in range(len(value_list)):
                if len(value_combinations) == 0:
                    for v in value_list[i]:
                        value_combinations.append([v])
                else:
                    new_combinations = []
                    for c in value_combinations:
                        for v in value_list[i]:
                            new_combinations.append(c + [v])
                    value_combinations = new_combinations
            for i, values in enumerate(value_combinations):
                if properties_dict is None or len(properties_dict) == 0:
                    curr_layer_dict = base_dict
                    channels_list = []
                    label_dict = {'label_type': label_type['label_type']}
                    for j, value in enumerate(values):
                        label_dict[key_list[j]] = value
                    channels_list.append({'labels' : label_dict})
                    curr_layer_dict['channels'] = channels_list
                    options.append(curr_layer_dict)
                else:
                    for prop_val in properties_dict:
                        curr_layer_dict = base_dict
                        channels_list = []
                        label_dict = {'label_type': label_type['label_type']}
                        for j, value in enumerate(values):
                            label_dict[key_list[j]] = value
                        channels_list.append({'labels' : label_dict, 'properties': prop_val.copy()})
                        curr_layer_dict['channels'] = channels_list
                        options.append(curr_layer_dict)
    return options


def preprocess_network_architectures(network_architectures_dict)-> List[List[dict]]:
    """
    Preprocesses the network architectures from the config file into a list of network architectures
    :param network_architectures_dict:
    :return: list of network architectures in the correct format
    """
    network_architectures = []
    for network_id, network_architecture in enumerate(network_architectures_dict):
        layers_per_architecture = []
        for i, layer in enumerate(network_architecture):
            layer_from_yml(i, layer, layers_per_architecture, network_architecture)
        # cartesian product over all entries of layers_per_architecture
        combinations = [x for x in itertools.product(*layers_per_architecture)]

        network_architectures += [list(x) for x in combinations]
    if not check_network_architectures(network_architectures, print_errors=True):
        raise ValueError('Network architecture not correctly defined')
    return network_architectures





def get_run_configs(experiment_configuration):
    # define the network type from the config file
    run_configs = []
    task = "graph_classification" #default task is graph classification
    if 'task' in experiment_configuration:
        task = experiment_configuration['task']
    # get networks from the config file and preprocess them
    # bring the config file network architecture into the correct format
    # TODO: Handle multiple layers
    network_architectures = preprocess_network_architectures(experiment_configuration['models'])
    # iterate over all network architectures
    for network_architecture in network_architectures:
        layers = []
        # get all different run configurations
        for i, l in enumerate(network_architecture):
            layers.append(Layer(l, i))
        for b in experiment_configuration.get('batch_size', [128]):
            for lr in experiment_configuration.get('learning_rate', [0.001]):
                for e in experiment_configuration.get('epochs', [100]):
                    for d in experiment_configuration.get('dropout', [0.0]):
                        for o in experiment_configuration.get('optimizer', ['Adam']):
                            for w in experiment_configuration.get('weight_decay', [0.0]):
                                for loss in experiment_configuration.get('loss', ['CrossEntropyLoss']):
                                    run_configs.append(
                                        RunConfiguration(experiment_configuration, network_architecture, layers, b, lr, e, d, o, w, loss, task))
    return run_configs
