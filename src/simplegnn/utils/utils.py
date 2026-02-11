import copy
import os
from pathlib import Path
from typing import List, Any
import networkx as nx

from simplegnn.framework.utils.parameters import Parameters


def get_k_lowest_nonzero_indices(tensor, k):
    # Flatten the tensor
    flat_tensor = tensor.flatten()

    # Get the indices of non-zero elements
    non_zero_indices = torch.nonzero(flat_tensor, as_tuple=True)[0]

    # Select the non-zero elements
    non_zero_elements = torch.index_select(flat_tensor, 0, non_zero_indices)

    # Get the indices of the k lowest elements
    k_lowest_values, k_lowest_indices = torch.topk(non_zero_elements, k, largest=False)

    # Get the original indices
    k_lowest_original_indices = non_zero_indices[k_lowest_indices]

    return k_lowest_original_indices


def save_graphs(path: Path, db_name, graphs: List[nx.Graph], labels: List[int] = None, with_degree=False, graph_format=None):
    # save in two files DBName_Nodes.txt and DBName_Edges.txt
    # DBName_Nodes.txt has the following structure GraphId NodeId Feature1 Feature2 ...
    # DBName_Edges.txt has the following structure GraphId Node1 Node2 Feature1 Feature2 ...
    # DBName_Labels.txt has the following structure GraphId Label
    # if not folder db_name exists in path create it
    if not os.path.exists(path.joinpath(Path(db_name))):
        os.makedirs(path.joinpath(Path(db_name)))
    # create processed and raw folders in path+db_name
    if not os.path.exists(path.joinpath(Path(db_name + "/processed"))):
        os.makedirs(path.joinpath(Path(db_name + "/processed")))
    if not os.path.exists(path.joinpath(Path(db_name + "/raw"))):
        os.makedirs(path.joinpath(Path(db_name + "/raw")))
    # update path to write into raw folder
    path = path.joinpath(Path(db_name + "/raw/"))
    with open(path.joinpath(Path(db_name + "_Nodes.txt")), "w") as f:
        for i, graph in enumerate(graphs):
            for node in graph.nodes(data=True):
                # get list of all data entries of the node, first label then the rest
                if 'primary_node_labels' not in node[1]:
                    data_list = [0]
                    if with_degree:
                        data_list.append(graph.degree(node[0]))
                elif type(node[1]['primary_node_labels']) == np.ndarray or type(node[1]['primary_node_labels']) == list:
                        data_list = [int(node[1]['primary_node_labels'][0])]
                        for v in node[1]['primary_node_labels'][1:]:
                            data_list.append(v)
                else:
                    data_list = [int(node[1]['primary_node_labels'])]
                # append all the other features
                for key, value in node[1].items():
                    if key != 'primary_node_labels':
                        if type(value) == int:
                            data_list.append(value)
                        elif type(value) == np.ndarray or type(value) == list:
                            for v in value:
                                data_list.append(v)
                f.write(str(i) + " " + str(node[0]) + " " + " ".join(map(str, data_list)) + "\n")
        # remove last empty line
        f.seek(f.tell() - 1, 0)
        f.truncate()
    with open(path.joinpath(db_name + "_Edges.txt"), "w") as f:
        for i, graph in enumerate(graphs):
            for edge in graph.edges(data=True):
                # get list of all data entries of the node, first label then the rest
                if 'primary_edge_labels' not in edge[2]:
                    data_list = [0]
                else:
                    if type(edge[2]['primary_edge_labels']) == np.ndarray or type(edge[2]['primary_edge_labels']) == list:
                        if len(edge[2]['primary_edge_labels']) == 1:
                            data_list = [int(edge[2]['primary_edge_labels'][0])]
                        else:
                            # raise an error as the label must be a single value
                            raise ValueError("Edge label must be a single value")
                    else:
                        data_list = [int(edge[2]['primary_edge_labels'])]
                # append all the other features
                for key, value in edge[2].items():
                    if key != 'primary_edge_labels':
                        if type(value) == int:
                            data_list.append(value)
                        elif type(value) == np.ndarray or type(value) == list:
                            for v in value:
                                data_list.append(v)
                f.write(str(i) + " " + str(edge[0]) + " " + str(edge[1]) + " " + " ".join(map(str, data_list)) + "\n")
        # remove last empty line
        f.seek(f.tell() - 1, 0)
        f.truncate()
    if graph_format == 'NEL':
        with open(path.joinpath(db_name + "_Labels.txt"), "w") as f:
            if labels is not None:
                for i, label in enumerate(labels):
                    if type(label) == int:
                        f.write(db_name + " " + str(i) + " " + str(label) + "\n")
                    elif type(label) == np.ndarray or type(label) == list:
                        f.write(db_name + " " + str(i) + " " + " ".join(map(str, label)) + "\n")
                    else:
                        f.write(db_name + " " + str(i) + " " + str(label) + "\n")
            else:
                for i in range(len(graphs)):
                    f.write(db_name + " " + str(i) + " " + str(0) + "\n")
            # remove last empty line
            if f.tell() > 0:
                f.seek(f.tell() - 1, 0)
                f.truncate()
    else:
        with open(path.joinpath(db_name + "_Labels.txt"), "w") as f:
            if labels is not None:
                for i, label in enumerate(labels):
                    if type(label) == int:
                        f.write(str(i) + " " + str(label) + "\n")
                    elif type(label) == np.ndarray or type(label) == list:
                        f.write(str(i) + " " + " ".join(map(str, label)) + "\n")
                    else:
                        f.write(str(i) + " " + str(label) + "\n")
            else:
                for i in range(len(graphs)):
                    f.write(str(i) + " " + str(0) + "\n")
            # remove last empty line
            if f.tell() > 0:
                f.seek(f.tell() - 1, 0)
                f.truncate()


def load_graphs(path: Path, db_name: str, graph_format=None):
    graphs = []
    labels = []
    with open(path.joinpath(db_name + "_Nodes.txt"), "r") as f:
        lines = f.readlines()
        for line in lines:
            data = line.strip().split(" ")
            graph_id = int(data[0])
            node_id = int(data[1])
            feature = list(map(float, data[2:]))
            while len(graphs) <= graph_id:
                graphs.append(nx.Graph())
            graphs[graph_id].add_node(node_id, label=feature)
    with open(path.joinpath(db_name + "_Edges.txt"), "r") as f:
        lines = f.readlines()
        for line in lines:
            data = line.strip().split(" ")
            graph_id = int(data[0])
            node1 = int(data[1])
            node2 = int(data[2])
            feature = list(map(float, data[3:]))
            graphs[graph_id].add_edge(node1, node2, label=feature)
    if graph_format == 'NEL':
        with open(path.joinpath(db_name + "_Labels.txt"), "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                data = line.strip().split(" ")
                graph_name = data[0]
                graphs[i].name = graph_name
                graph_id = int(data[1])
                if len(data) == 3:
                    # first try to convert to int
                    try:
                        label = int(data[2])
                    except:
                        # if it fails convert to float
                        try:
                            label = float(data[2])
                        except:
                            # if it fails raise an error
                            raise ValueError("Label is not in the correct format")

                else:
                    label = list(map(float, data[2:]))

                while len(labels) <= graph_id:
                    labels.append(label)
                labels[graph_id] = label
    else:
        with open(path.joinpath(db_name + "_Labels.txt"), "r") as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip().split(" ")
                graph_id = int(data[0])
                if len(data) == 2:
                    # first try to convert to int
                    try:
                        label = int(data[1])
                    except:
                        # if it fails convert to float
                        try:
                            label = float(data[1])
                        except:
                            # if it fails raise an error
                            raise ValueError("Label is not in the correct format")

                else:
                    label = list(map(float, data[1:]))

                while len(labels) <= graph_id:
                    labels.append(label)
                labels[graph_id] = label
    return graphs, labels


def valid_pruning_configuration(para: Parameters, epoch: int) -> bool:
    if 'prune' in para.run_config.config and 'enabled' in para.run_config.config['prune'] and 'epochs' in para.run_config.config[
        'prune'] and 'percentage' in para.run_config.config['prune'] and para.run_config.config['prune']['enabled']:
        if (epoch + 1) % para.run_config.config['prune']['epochs'] == 0 and 0 < epoch + 1 < para.n_epochs and len(
                para.run_config.config['prune']['percentage']) == len(para.run_config.layers):
            return True
    if 'prune' in para.run_config.config and 'enabled' in para.run_config.config['prune'] and para.run_config.config['prune']['enabled'] and (
            epoch + 1) % para.n_epochs == 0:
        print("Pruning is enabled but the configuration is not correct")
    return False


def is_pruning(config:None) -> bool:
    if 'prune' in config and 'enabled' in config['prune'] and config['prune']['enabled']:
        return True
    return False




def reshape_indices(a, b):
    reshape_dict = {}
    ita = np.nditer(a, flags=['multi_index'])
    itb = np.nditer(b, flags=['multi_index'])
    while not ita.finished:
        reshape_dict[ita.multi_index] = itb.multi_index
        ita.iternext()
        itb.iternext()

    return reshape_dict


def convert_to_tuple(value: List):
    """
    Convert the value to a tuple and each element of the value to a tuple if it is a list
    """
    # create copy of value
    new_value = copy.deepcopy(value)
    for i, v in enumerate(new_value):
        if type(v) == list:
            new_value[i] = tuple(v)
    return tuple(new_value)


def convert_to_list(value: Any):
    if type(value) == int:
        return value
    elif type(value) == tuple:
        value = list(value)
        for i, v in enumerate(value):
            if type(v) == tuple:
                value[i] = list(v)
        return value


'''
Created on 14.03.2019

@author:
'''
import random
import matplotlib.pyplot as plt
import numpy as np
import torch

def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]


def get_train_test_list(size, divide=0.9, seed=10):
    random.seed(seed)
    data = [i for i in range(0, size)]
    train_data = random.sample(data, int(len(data) * divide))
    test_data = diff(data, train_data)
    return train_data, test_data


def get_train_validation_test_list(test_indices, validation_step, seed=10, balanced=False, graph_labels=[], val_size=0):
    test_data = test_indices[validation_step]
    train_data_unb = np.concatenate([x for i, x in enumerate(test_indices) if i != validation_step])
    np.random.seed(seed)
    np.random.shuffle(train_data_unb)
    val_data = []
    if val_size > 0:
        val_data, train_data_unb = np.split(train_data_unb, [int(val_size * train_data_unb.size)])
    # sort validation data
    val_data = np.sort(val_data)
    train_data_b = train_data_unb.copy()

    # create balanced training set
    if balanced and graph_labels:
        label_dict = {}
        max_class = 0

        # get all classes of labels by name
        for x in train_data_unb:
            if str(graph_labels[x]) in label_dict.keys():
                label_dict[str(graph_labels[x])] += 1
            else:
                label = str(graph_labels[x])
                label_dict[label] = 1

        # get biggest class of labels
        for x in label_dict.keys():
            if label_dict[x] > max_class:
                max_class = label_dict[x]

        # dublicate samples from smaller classes
        for x in label_dict.keys():
            if label_dict[x] < max_class:
                number = max_class - label_dict[x]
                # get all indices of train_data_unb with graph_labels == x
                train_data_class_x = [i for i in train_data_b if graph_labels[i] == int(x)]
                # get new_array size random values from train_data_class_x
                np.random.seed(seed)
                new_array = np.random.choice(train_data_class_x, number, replace=True)
                # add the new array to the training data
                train_data_b = np.append(train_data_b, new_array)
        # sort training data
        train_data_b = np.sort(train_data_b)
        return np.asarray(train_data_b), np.asarray(val_data, dtype=int), test_data
    else:
        # sort training data
        train_data_unb = np.sort(train_data_unb)
        return train_data_unb, np.asarray(val_data, dtype=int), test_data


def balance_data(data, labels):
    train_data_b = data.copy()
    label_dict = {}
    max_class = 0

    # get all classes of labels by name
    for x in data:
        if str(labels[x]) in label_dict.keys():
            label_dict[str(labels[x])] += 1
        else:
            label = str(labels[x])
            label_dict[label] = 1

    # get biggest class of labels
    for x in label_dict.keys():
        if label_dict[x] > max_class:
            max_class = label_dict[x]

    # dublicate samples from smaller classes
    for x in label_dict.keys():
        if label_dict[x] < max_class:
            number = max_class - label_dict[x]
            counter = 0
            while counter < number:
                for t in data:
                    if counter < number and str(labels[t]) == x:
                        train_data_b.append(t)
                        counter += 1
    return train_data_b


def get_training_batch(training_data, batch_size):
    data = [training_data[x:min(x + batch_size, len(training_data))] for x in range(0, len(training_data), batch_size)]
    return data


def get_accuracy(output, labels, one_hot_encoding=True, zero_one=False):
    counter = 0
    correct = 0
    if one_hot_encoding:
        for i, x in enumerate(output, 0):
            if torch.argmax(x) == torch.argmax(labels[i]):
                correct += 1
            counter += 1
    else:
        if zero_one:
            for i, x in enumerate(output, 0):
                if abs(x - labels[i]) < 0.5:
                    correct += 1
                counter += 1
        else:
            for i, x in enumerate(output, 0):
                if x * labels[i] > 0:
                    correct += 1
                counter += 1
    return correct / counter


def live_plotter(x_vec, y1_data, line1, identifier='', pause_time=0.1):
    if line1 == []:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(13, 6))
        ax = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        line1, = ax.plot(x_vec, y1_data, '-o', alpha=0.8)
        # update plot label/title
        plt.ylabel('Y Label')
        plt.title('Title: {}'.format(identifier))
        plt.show()

    # after the figure, axis, and line are created, we only need to update the y-data
    line1.set_ydata(y1_data)
    # adjust limits if new data goes beyond bounds
    if np.min(y1_data) <= line1.axes.get_ylim()[0] or np.max(y1_data) >= line1.axes.get_ylim()[1]:
        plt.ylim([np.min(y1_data) - np.std(y1_data), np.max(y1_data) + np.std(y1_data)])
    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)

    # return line so we can update it again in the next iteration
    return line1


def plot_init(line_num, identifier='', epochs=100):
    coordinates = [[], []]
    lines = []
    # this is the call to matplotlib that allows dynamic plotting
    plt.ion()
    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(111)
    # update plot label/title
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    # set y range min and max
    plt.ylim([0, 1])
    # set x range min and max
    plt.xlim([1, epochs])

    plt.title('Title: {}'.format(identifier))
    plt.show()

    for i in range(0, line_num):
        # create a variable for the line so we can later update it
        line, = ax.plot(0, 0, '-o', alpha=0.8)
        lines.append(line)
        coordinates[0].append(np.zeros(1))
        coordinates[1].append(np.zeros(1))
    return lines, coordinates


def plot_learning_data(new_x, new_y, data, epochs, title=''):
    plt.ion()
    plt.clf()

    # set y range min and max
    plt.ylim([0, 100])
    # set x range min and max
    plt.xlim([1, new_x])

    plt.title(f"Title: {title}")
    # add new_x new_y to data which is a map of lists
    for i, y in enumerate(new_y, 0):
        if len(data['Y']) <= i:
            data['Y'].append([])
            data['X'].append([])
        data['Y'][i].append(y)
        data['X'][i].append(new_x)
    # plot data as multiple lines with different colors in one plot
    for i, x in enumerate(data['X'], 0):
        plt.plot(x, data['Y'][i])

    # add legend to the lines epoch accuracy, validation accuracy and test accuracy
    plt.legend(['Train', 'Validation', 'Test', 'Loss'], loc='lower right')

    plt.draw()
    plt.pause(0.001)
    return data


def add_values(xvalues, yvalues, coordinates):
    for i, x in enumerate(xvalues, 0):
        coordinates[0][i] = np.append(coordinates[0][i], x)
        coordinates[1][i] = np.append(coordinates[1][i], yvalues[i])

    return coordinates


def live_plotter_lines(coordinates, lines):
    for i, line in enumerate(lines, 0):
        # after the figure, axis, and line are created, we only need to update the y-data
        line.set_ydata(coordinates[1][i])
        line.set_xdata(coordinates[0][i])
    # return line so we can update it again in the next iteration
    return lines


def get_data_indices(size, seed, kFold):
    random.seed(seed)
    np.random.seed(seed)
    data = np.arange(0, size)
    np.random.shuffle(data)
    data = np.array_split(data, kFold)
    # sort the data
    for i in range(0, len(data)):
        data[i] = np.sort(data[i])
    return data

