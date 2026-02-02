# generate WL labels for the graph data and save them to a file
import abc
import time
from pathlib import Path
from typing import List, Optional, Union, Tuple

import networkx as nx
import torch
from networkx.algorithms.isomorphism import GraphMatcher
from numpy import sort
from torch_geometric.io import fs

from datasets.graph_dataset import GraphDataset
from datasets.utils.NodeLabels import NodeLabels
from datasets.utils.node_labeling_functions import weisfeiler_lehman_node_labeling




def load_labels(path='') -> NodeLabels:
    """
    Load the labels from a file.
    :param path: Path to the file
    :return: NodeLabels object
    """
    dataset_name, label_name, node_labels = torch.load(path, weights_only=True)
    return NodeLabels(dataset_name, label_name, node_labels)

def combine_node_labels(labels: List[NodeLabels]):
    graph_name = labels[0].dataset_name
    combined_label_name = '_'.join([l.label_name for l in labels])
    # stack all the node labels
    stacked_labels = torch.stack([l.node_labels for l in labels], dim=1)
    # get all indices of the unique labels tensors containing -1
    first_entry = stacked_labels[:, 0] == -1
    second_entry = stacked_labels[:, 1] == -1
    # compute or between the two tensors
    or_entries = torch.logical_or(first_entry, second_entry)
    # get the indices of True values
    invalid_indices = torch.where(or_entries)[0]
    # set stack_labels to max, max
    max_first = torch.max(stacked_labels[:, 0] + 1)
    max_second = torch.max(stacked_labels[:, 1] + 1)
    stacked_labels[invalid_indices] = torch.tensor([max_first, max_second])
    # get all unique rows
    unique_labels, new_labels = torch.unique(stacked_labels, return_inverse=True, dim=0)

    artificial_label = len(unique_labels) - 1
    # get frequency of each value in the new labels
    unique_labels_count = torch.bincount(new_labels)
    # set count for invalid indices to 0
    unique_labels_count[artificial_label] = 0
    # sort the unique labels by the frequency and keep the indices
    sorted_indices = torch.argsort(unique_labels_count, descending=True, stable=True)
    # reindex the unique labels: most frequent label is 0, second most frequent is 1, ...
    frequency_sorted_labels = new_labels.new(sorted_indices).argsort()[new_labels]
    new_labels[invalid_indices] = -1
    frequency_sorted_labels[invalid_indices] = -1
    return NodeLabels(graph_name, combined_label_name, torch.stack([new_labels, frequency_sorted_labels], dim=1))

def get_label_string(label_dict: dict) -> str:
    """
    converts a label dictionary to a unique string representation
    :param label_dict: the dictionary that contains the information of the labels
    :return: the unique string representation of the corresponding label dictionary
    """
    label_type = label_dict.get('label_type', None)
    if label_type is None:
        raise ValueError("Label type is not specified")

    if isinstance(label_type, list):
        l_string = ""
        for i, l in enumerate(label_type):
            new_label_dict = label_dict.copy()
            new_label_dict['label_type'] = l
            if i > 0:
                l_string += "_"
            l_string += get_label_string(new_label_dict)
        max_labels = label_dict.get('max_labels', None)
        if max_labels is not None:
            l_string = f"{l_string}_{max_labels}"
        return l_string

    if label_type == "primary":
        l_string = "primary"
        if 'max_labels' in label_dict:
            max_labels = label_dict['max_labels']
            l_string = f"primary_{max_labels}"
    elif label_type == "index":
        l_string = "index"
        max_labels = label_dict.get('max_labels', None)
        if max_labels is not None:
            l_string = f"index_{max_labels}"
    elif label_type == "index_text":
        l_string = "index_text"
        max_labels = label_dict.get('max_labels', None)
        if max_labels is not None:
            l_string = f"index_text_{max_labels}"
    elif label_type == "wl":
        iterations = label_dict.get('depth', 3)
        l_string = f"wl_{iterations}"
        max_labels = label_dict.get('max_labels', None)
        if max_labels is not None:
            l_string = f"{l_string}_{max_labels}"
    elif label_type == "wl_labeled":
        l_string = 'wl_labeled'
        if 'base_labels' in label_dict:
            l_string = f"{l_string}_{get_label_string(label_dict['base_labels'])}_base_labels"
        iterations = label_dict.get('depth', 3)
        l_string = f"{l_string}_{iterations}"
        max_labels = label_dict.get('max_labels', None)
        if max_labels is not None:
            l_string = f"{l_string}_{max_labels}"
    elif label_type == "wl_labeled_edges":
        l_string = 'wl_labeled_edges'
        if 'base_labels' in label_dict:
            l_string = f"{l_string}_{get_label_string(label_dict['base_labels'])}_base_labels"
        iterations = label_dict.get('depth', 3)
        l_string = f"{l_string}_{iterations}"
        max_labels = label_dict.get('max_labels', None)
        if max_labels is not None:
            l_string = f"{l_string}_{max_labels}"
    elif label_type == "degree":
        l_string = "wl_0"
        max_labels = label_dict.get('max_labels', None)
        if max_labels is not None:
            l_string = f"{l_string}_{max_labels}"
    elif label_type == "simple_cycles":
        l_string = "simple_cycles"
        if 'min_cycle_length' in label_dict:
            min_cycle_length = label_dict['min_cycle_length']
            l_string = f"{l_string}_{min_cycle_length}"
        if 'max_cycle_length' in label_dict:
            max_cycle_length = label_dict['max_cycle_length']
            l_string = f"{l_string}_{max_cycle_length}"
        else:
            l_string = "simple_cycles_max"
        max_labels = label_dict.get('max_labels', None)
        if max_labels is not None:
            l_string = f"{l_string}_{max_labels}"
    elif label_type == "induced_cycles":
        l_string = "induced_cycles"
        if 'min_cycle_length' in label_dict:
            min_cycle_length = label_dict['min_cycle_length']
            l_string = f"{l_string}_{min_cycle_length}"
        if 'max_cycle_length' in label_dict:
            max_cycle_length = label_dict['max_cycle_length']
            l_string = f"{l_string}_{max_cycle_length}"
        else:
            l_string = "induced_cycles_max"
        max_labels = label_dict.get('max_labels', None)
        if max_labels is not None:
            l_string = f"{l_string}_{max_labels}"
    elif label_type == "cliques":
        l_string = f"cliques"
        if 'max_clique_size' in label_dict:
            max_clique_size = label_dict['max_clique_size']
            l_string = f"cliques_{max_clique_size}"
        max_labels = label_dict.get('max_labels', None)
        if max_labels is not None:
            l_string = f"{l_string}_{max_labels}"
    elif label_type == "subgraph":
        l_string = f"subgraph"
        if 'id' in label_dict:
            subgraph_id = label_dict['id']
            l_string = f"{l_string}_{subgraph_id}"
        max_labels = label_dict.get('max_labels', None)
        if max_labels is not None:
            l_string = f"{l_string}_{max_labels}"
    elif label_type == "trivial":
        l_string = "trivial"
    else:
        raise ValueError(f"Layer type {label_type} is not supported")

    return l_string


# Todo replace the save functions by classes (the base class should be the following NodeLabelingBase)
class NodeLabelingBase(abc.ABC):
    def __init__(self,
                 base_name,
                 graph_data: GraphDataset,
                 label_path: Optional[Path] = None,
                 max_labels: Optional[int] = None,
                 optional_parameters: List[Tuple[str, int]] = None,
                 save_times: Optional[Path] = None):
        self.base_name = base_name
        self.graph_data = graph_data
        self.label_path = label_path
        self.max_labels = max_labels
        self.optional_parameters = optional_parameters
        self.save_times = save_times
        self.string_label_name = self.base_name
        self.set_string_label_name()
        self.file_path = None

    def create_and_save_labels(self):
        if self.label_path is None:
            raise ValueError("No label path given")
        else:
            self.file_path = self.label_path.joinpath(f"{self.graph_data.name}_labels_{self.string_label_name}.pt")
        if not self.file_path.exists():
            print(f"Saving {self.string_label_name} labels for {self.graph_data.name} to {self.file_path}")
            start_time = time.time()
            # create the labels
            graph_node_labels = self.generate()
            self.save_labels_to_file(graph_node_labels)
            if self.save_times is not None:
                try:
                    with open(self.save_times, 'a') as f:
                        f.write(f"{self.graph_data.name}, {self.string_label_name}, {time.time() - start_time}\n")
                except:
                    raise ValueError("No save time path given")
        else:
            print(f"File {self.file_path} already exists. Skipping.")

    @abc.abstractmethod
    def generate(self) -> Optional[Union[List[List[int]], torch.Tensor]]:
        """
        Create the labels for the graph data.
        This method should be implemented by subclasses.
        It return the graph_node labels
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    def set_string_label_name(self):
        # take base name and append first max_labels if it is not None and then all the parameters in the optional_parameters list
        if self.max_labels is not None:
            self.string_label_name = f"{self.string_label_name}_max_labels_{self.max_labels}"
        if self.optional_parameters is not None:
            for param in self.optional_parameters:
                self.string_label_name = f"{self.string_label_name}_{param[0]}_{param[1]}"
        return


    def save_labels_to_file(self, graph_node_labels: Optional[Union[List[List[int]], torch.Tensor]]):
        """
        Save the node labels to a file
        :param graph_node_labels: List of lists with the node labels for each graph or torch.Tensor with the node labels
        """
        if isinstance(graph_node_labels, torch.Tensor):
            pass
        elif isinstance(graph_node_labels, list):
            # flatten the node labels
            graph_node_labels = torch.tensor([label for graph_labels in graph_node_labels for label in graph_labels])
        else:
            raise ValueError("graph_node_labels must be either a torch.Tensor or a list of lists")
        # save the node labels to a file as torch tensor with the original labels as first column and the new labels as second column
        fs.torch_save(
            (self.graph_data.name, self.string_label_name, relabel_node_labels(graph_node_labels, self.max_labels)), str(self.file_path)
        )
        raise NotImplementedError("This method should be implemented by subclasses")


class TrivialNodeLabeling(NodeLabelingBase):
    def __init__(self, graph_data: GraphDataset, label_path: Optional[Path] = None, max_labels: Optional[int] = None, save_times: Optional[Path] = None):
        super().__init__(base_name='trivial', graph_data=graph_data, label_path=label_path, max_labels=max_labels, save_times=save_times)

    def generate(self) -> Optional[Union[List[List[int]], torch.Tensor]]:
        # label 0 for all nodes
        return torch.zeros(len(self.graph_data.data.x), dtype=torch.long)

class IndexNodeLabeling(NodeLabelingBase):
    def __init__(self, graph_data: GraphDataset, label_path: Optional[Path] = None, max_labels: Optional[int] = None, save_times: Optional[Path] = None):
        super().__init__(base_name='index', graph_data=graph_data, label_path=label_path, max_labels=max_labels, save_times=save_times)

    def generate(self) -> Optional[Union[List[List[int]], torch.Tensor]]:
        node_labels = []
        if self.graph_data.nx_graphs is None:
            self.graph_data.create_nx_graphs(directed=False)
        for graph in self.graph_data.nx_graphs:
            node_labels.append([index for index, node in enumerate(graph.nodes())])
        return node_labels

class IndexTextNodeLabeling(NodeLabelingBase):
    def __init__(self, graph_data: GraphDataset, label_path: Optional[Path] = None, max_labels: Optional[int] = None, save_times: Optional[Path] = None):
        super().__init__(base_name='index_text', graph_data=graph_data, label_path=label_path, max_labels=max_labels, save_times=save_times)

    def generate(self) -> Optional[Union[List[List[int]], torch.Tensor]]:
        node_labels = []
        if self.graph_data.nx_graphs is None:
            self.graph_data.create_nx_graphs(directed=False)
        for graph in self.graph_data.nx_graphs:
            node_labels.append([index for index, node in enumerate(graph.nodes())])
            # define index -1 and -2 for the first and last entry
            node_labels[-1][0] = -1  # first entry
            node_labels[-1][-1] = -2  # last entry
        return node_labels

class PrimaryNodeLabeling(NodeLabelingBase):
    def __init__(self, graph_data: GraphDataset, label_path: Optional[Path] = None, max_labels: Optional[int] = None, save_times: Optional[Path] = None):
        super().__init__(base_name='primary', graph_data=graph_data, label_path=label_path, max_labels=max_labels, save_times=save_times)

    def generate(self) -> Optional[Union[List[List[int]], torch.Tensor]]:
        return self.graph_data.node_labels['primary']

class DegreeNodeLabeling(NodeLabelingBase):
    def __init__(self, graph_data: GraphDataset, label_path: Optional[Path] = None, max_labels: Optional[int] = None, save_times: Optional[Path] = None):
        super().__init__(base_name='wl_0', graph_data=graph_data, label_path=label_path, max_labels=max_labels, save_times=save_times)

    def generate(self) -> Optional[Union[List[List[int]], torch.Tensor]]:
        if self.graph_data.nx_graphs is None:
            self.graph_data.create_nx_graphs(directed=False)
        # iterate over the graphs and get the degree of each node
        node_labels = []
        for i, graph in enumerate(self.graph_data.nx_graphs):
            node_labels.append([0 for _ in range(len(graph.nodes()))])
            for node in graph.nodes():
                node_labels[-1][node] = graph.degree(node)
        return node_labels

class LabeledDegreeNodeLabeling(NodeLabelingBase):
    def __init__(self, graph_data: GraphDataset, label_path: Optional[Path] = None, max_labels: Optional[int] = None, save_times: Optional[Path] = None):
        super().__init__(base_name='wl_labeled_0', graph_data=graph_data, label_path=label_path, max_labels=max_labels, save_times=save_times)

    def generate(self) -> Optional[Union[List[List[int]], torch.Tensor]]:
        if self.graph_data.nx_graphs is None:
            self.graph_data.create_nx_graphs(directed=False)
        # iterate over the graphs and get the degree of each node
        node_labels = []
        unique_neighbor_labels = set()
        node_to_hash = dict()
        for graph_id, graph in enumerate(self.graph_data.nx_graphs):
            for i, node in enumerate(graph.nodes(data=True)):
                neighbors = list(graph.neighbors(node[0]))
                node_identifier = [node[1]['primary_node_labels']]
                node_identifier += [graph.nodes[neighbor]['primary_node_labels'] for neighbor in neighbors]
                # convert to tuple and add to set
                node_identifier = tuple(node_identifier)
                unique_neighbor_labels.add(node_identifier)
                node_to_hash[node[0]] = node_identifier
        # convert the unique neighbor labels to a dict
        unique_neighbor_label_dict = {label: i for i, label in enumerate(unique_neighbor_labels)}
        for graph in self.graph_data.nx_graphs:
            node_labels.append([unique_neighbor_label_dict[node_to_hash[node]] for node in graph.nodes()])
        return node_labels

class WeisfeilerLehmanNodeLabeling(NodeLabelingBase):
    def __init__(self, graph_data: GraphDataset, depth, max_labels: Optional[int] = None, label_path: Optional[Path] = None, save_times: Optional[Path] = None):
        super().__init__(base_name='wl', graph_data=graph_data, label_path=label_path, max_labels=max_labels, optional_parameters=[('depth', depth)], save_times=save_times)

    def generate(self) -> Optional[Union[List[List[int]], torch.Tensor]]:
        if self.graph_data.nx_graphs is None:
            self.graph_data.create_nx_graphs(directed=False)
        node_labels, unique_node_labels, db_unique_node_labels = weisfeiler_lehman_node_labeling(self.graph_data.nx_graphs, depth=self.optional_parameters['depth'], labeled=False)
        return node_labels

class WeisfeilerLehmanLabeledNodeLabeling(NodeLabelingBase):
    def __init__(self, graph_data: GraphDataset, depth, max_labels: Optional[int] = None, label_path: Optional[Path] = None, base_labels: Optional[dict] = None, save_times: Optional[Path] = None):
        super().__init__(base_name='wl_labeled', graph_data=graph_data, label_path=label_path, max_labels=max_labels, optional_parameters=[('depth', depth), ('base_labels', get_label_string(base_labels['layer_dict']))], save_times=save_times)
        self.base_labels = base_labels

    def generate(self) -> Optional[Union[List[List[int]], torch.Tensor]]:
        if self.graph_data.nx_graphs is None:
            self.graph_data.create_nx_graphs(directed=False)
        node_labels, unique_node_labels, db_unique_node_labels = weisfeiler_lehman_node_labeling(self.graph_data.nx_graphs, depth=self.optional_parameters['depth'], labeled=True, base_labels=self.base_labels)
        return node_labels


def save_labels_to_file(file:Path, dataset_name:str, label_name:str, graph_node_labels:Optional[Union[List[List[int]], torch.Tensor]], max_labels:None):
    """
    Save the node labels to a file
    :param file: Path to the file
    :param dataset_name: Name of the dataset
    :param label_name: Name of the labels
    :param graph_node_labels: List of lists with the node labels for each graph or torch.Tensor with the node labels
    :param max_labels: Maximum number of labels to use
    """
    if isinstance(graph_node_labels, torch.Tensor):
        pass
    elif isinstance(graph_node_labels, list):
        # flatten the node labels
        graph_node_labels = torch.tensor([label for graph_labels in graph_node_labels for label in graph_labels])
    else:
        raise ValueError("graph_node_labels must be either a torch.Tensor or a list of lists")
    # save the node labels to a file as torch tensor with the original labels as first column and the new labels as second column
    fs.torch_save(
        (dataset_name, label_name, relabel_node_labels(graph_node_labels, max_labels)), str(file)
    )

def save_primary_labels(graph_data:GraphDataset, label_path=None, max_labels=None, save_times=None) -> str:
    l = f'primary'
    if max_labels is not None:
        l = f'{l}_{max_labels}'
    # save the node labels to a file
    if label_path is None:
        raise ValueError("No label path given")
    else:
        file = label_path.joinpath(f"{graph_data.name}_labels_{l}.pt")
    if not file.exists():
        print(f"Saving {l} labels for {graph_data.name} to {file}")
        start_time = time.time()
        save_labels_to_file(file,graph_data.name, l, graph_data.node_labels['primary'], max_labels)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{graph_data.name}, {l}, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")
    return file



def save_degree_labels(graph_data:GraphDataset, label_path=None, max_labels=None, save_times=None)->str:
    #save the node labels to a file
    l = 'wl_0'
    if max_labels is not None:
        l = f'{l}_{max_labels}'
    if label_path is None:
        raise ValueError("No label path given")
    else:
        file = label_path.joinpath(f"{graph_data.name}_labels_{l}.pt")
    if not file.exists():
        print(f"Saving {l} for {graph_data.name} to {file}")
        if graph_data.nx_graphs is None:
            graph_data.create_nx_graphs(directed=False)
        start_time = time.time()
        # iterate over the graphs and get the degree of each node
        node_labels = []
        for i,graph in enumerate(graph_data.nx_graphs):
            node_labels.append([0 for _ in range(len(graph.nodes()))])
            for node in graph.nodes():
                node_labels[-1][node] = graph.degree(node)
        save_labels_to_file(file, graph_data.name, l, node_labels, max_labels=max_labels)
        #write_node_labels(file, node_labels)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{graph_data.name}, {l}, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")
    return file

def save_labeled_degree_labels(graph_data:GraphDataset, label_path=None, max_labels=None, save_times=None)->str:
    # save the node labels to a file
    l = 'wl_labeled_0'
    if max_labels is not None:
        l = f'{l}_{max_labels}'
    if label_path is None:
        raise ValueError("No label path given")
    else:
        file = label_path.joinpath(f"{graph_data.name}_labels_{l}.pt")
    # check whether the file already exists
    if not file.exists():
        print(f"Saving {l} labels for {graph_data.name} to {file}")
        if graph_data.nx_graphs is None:
            graph_data.create_nx_graphs(directed=False)
        start_time = time.time()
        # iterate over the graphs and get the degree of each node
        node_labels = []
        unique_neighbor_labels = set()
        node_to_hash = []
        for graph_id, graph in enumerate(graph_data.nx_graphs):
            node_to_hash.append(dict())
            for node in graph.nodes(data=True):
                neighbors = list(graph.neighbors(node[0]))
                node_identifier = str(node[1]['primary_node_labels'])
                neighbor_identifier = sort([graph.nodes[neighbor]['primary_node_labels'] for neighbor in neighbors])
                string_neighbor_identifier = '_'.join([str(n) for n in neighbor_identifier])
                node_identifier = f'{node_identifier}|{string_neighbor_identifier}'
                # convert to tuple and add to set
                unique_neighbor_labels.add(node_identifier)
                node_to_hash[graph_id][node[0]] = node_identifier
        # convert the unique neighbor labels to a dict
        unique_neighbor_label_dict = {label: i for i, label in enumerate(unique_neighbor_labels)}
        for graph_id, graph in enumerate(graph_data.nx_graphs):
            node_labels.append([unique_neighbor_label_dict[node_to_hash[graph_id][node]] for node in graph.nodes()])
        save_labels_to_file(file, graph_data.name, l, node_labels, max_labels)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{graph_data.name}, {l}, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")
    return file


def save_trivial_labels(graph_data:GraphDataset, label_path=None, save_times=None)->str:
    # save the node labels to a file
    l = 'trivial'
    if label_path is None:
        raise ValueError("No label path given")
    else:
        file = label_path.joinpath(f'{graph_data.name}_labels_{l}.pt')
    if not file.exists():
        print(f"Saving {l} labels for {graph_data.name} to {file}")
        start_time = time.time()
        # label 0 for all nodes
        trivial_labels = torch.zeros(len(graph_data.data.x), dtype=torch.long)
        save_labels_to_file(file, graph_data.name, l, trivial_labels, max_labels=None)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{graph_data.name}, {l}, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")
    return file

def save_index_labels(graph_data:GraphDataset, max_labels=None, label_path=None, index_text=False, save_times=None)->str:
    l = 'index'
    if index_text:
        l = f'{l}_text'
    if max_labels is not None:
        l = f'{l}_{max_labels}'
    if label_path is None:
        raise ValueError("No label path given")
    else:
        file = label_path.joinpath(f"{graph_data.name}_labels_{l}.pt")
    # check whether the file already exists
    if not file.exists():
        print(f"Saving {l} labels for {graph_data.name} to {file}")
        node_labels = []
        if graph_data.nx_graphs is None:
            graph_data.create_nx_graphs(directed=False)
        start_time = time.time()
        for graph in graph_data.nx_graphs:
            node_labels.append([index for index, node in enumerate(graph.nodes())])
            if index_text:
                # define index -1 and -2 for the first and last entry
                node_labels[-1][0] = -1  # first entry
                node_labels[-1][-1] = -2
        save_labels_to_file(file, graph_data.name, l, node_labels, max_labels)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{graph_data.name}, {l}, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")
    return file



def save_wl_labels(graph_data:GraphDataset, depth, max_labels=None, label_path=None, save_times=None)->str:
    # save the node labels to a file
    l = f'wl_{depth}'
    if max_labels is not None:
        l = f'{l}_{max_labels}'
    if label_path is None:
        raise ValueError("No label path given")
    else:
        file = label_path.joinpath(f'{graph_data.name}_labels_{l}.pt')
    if not file.exists():
        print(f"Saving {l} labels for {graph_data.name} to {file}")
        if graph_data.nx_graphs is None:
            graph_data.create_nx_graphs(directed=False)
        start_time = time.time()
        graph_node_labels, unique_node_labels, db_unique_node_labels = weisfeiler_lehman_node_labeling(graph_data.nx_graphs, depth=depth, labeled=False)
        save_labels_to_file(file, graph_data.name, l, graph_node_labels, max_labels)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{graph_data.name}, {l}_{max_labels}, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")
    return file

def save_wl_labeled_labels(graph_data:GraphDataset, depth, max_labels=None, label_path=None, base_labels:Optional[dict]=None, save_times=None)->str:
    # save the node labels to a file
    l = 'wl_labeled'
    if base_labels is not None and base_labels['layer_dict']['label_type'] != 'primary':
        l = f'{l}_{get_label_string(base_labels["layer_dict"])}_base_labels'
    l = f'{l}_{depth}'
    if max_labels is not None:
        l = f'{l}_{max_labels}'
    if label_path is None:
        raise ValueError("No label path given")
    else:
        file = label_path.joinpath(f'{graph_data.name}_labels_{l}.pt')
    if not file.exists():
        print(f"Saving {l} labels for {graph_data.name} to {file}")
        if graph_data.nx_graphs is None:
            graph_data.create_nx_graphs(directed=False)
        start_time = time.time()
        node_labels, unique_node_labels, db_unique_node_labels = weisfeiler_lehman_node_labeling(graph_data.nx_graphs, depth=depth, labeled=True, base_labels=base_labels)
        save_labels_to_file(file, graph_data.name, l, node_labels, max_labels)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{graph_data.name}, {l}_{max_labels}, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")
    return file


def save_wl_labeled_edges_labels(graph_data:GraphDataset, depth, max_labels=None, label_path=None, base_labels:Optional[dict]=None, save_times=None)->str:
    # save the node labels to a file
    l = 'wl_labeled_edges'
    if base_labels is not None and base_labels['layer_dict']['label_type'] != 'primary':
        l = f'{l}_{get_label_string(base_labels["layer_dict"])}_base_labels'
    l = f'{l}_{depth}'
    if max_labels is not None:
        l = f'{l}_{max_labels}'
    if label_path is None:
        raise ValueError("No label path given")
    else:
        file = label_path.joinpath(f'{graph_data.name}_labels_{l}.pt')
    if not file.exists():
        print(f"Saving {l} labels for {graph_data.name} to {file}")
        if graph_data.nx_graphs is None:
            graph_data.create_nx_graphs(directed=False)
        start_time = time.time()
        node_labels, unique_node_labels, db_unique_node_labels = weisfeiler_lehman_node_labeling(graph_data.nx_graphs, depth=depth, labeled=True, base_labels=base_labels, with_edge_labels=True)
        save_labels_to_file(file, graph_data.name, l, node_labels, max_labels)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{graph_data.name}, {l}_{max_labels}, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")
    return file

def save_cycle_labels(graph_data:GraphDataset, min_cycle_length=None, max_cycle_length=6, max_labels=None, cycle_type='simple', label_path=None, save_times=None)->str:
    if cycle_type not in ['simple', 'induced']:
        raise ValueError("Cycle type must be either 'simple' or 'induced'")
    l = 'simple_cycles'
    if cycle_type == 'induced':
        l = 'induced_cycles'
    if min_cycle_length is not None:
        l = f'{l}_{min_cycle_length}'
    l = f'{l}_{max_cycle_length}'
    if max_labels is not None:
        l = f"{l}_{max_labels}"
    if label_path is None:
        raise ValueError("No label path given")
    else:
        file = label_path.joinpath(f'{graph_data.name}_labels_{l}.pt')
    if not file.exists():
        print(f"Saving {cycle_type} cycles for {graph_data.name} to {file}")
        if graph_data.nx_graphs is None:
            graph_data.create_nx_graphs(directed=False)
        start_time = time.time()
        cycle_dict = []
        for i, graph in enumerate(graph_data.nx_graphs):
            if i % (len(graph_data.nx_graphs) // 10) == 0:
                print(f"Graph {graph_data.name} {i + 1}/{len(graph_data.nx_graphs)} Labels: {l}")
            cycle_dict.append({})
            if cycle_type == 'simple':
                cycles = nx.simple_cycles(graph, max_cycle_length)
                if min_cycle_length is not None:
                    cycles = [cycle for cycle in cycles if len(cycle) >= min_cycle_length]
            elif cycle_type == 'induced':
                cycles = nx.chordless_cycles(graph, max_cycle_length)
                if min_cycle_length is not None:
                    cycles = [cycle for cycle in cycles if len(cycle) >= min_cycle_length]
            for cycle in cycles:
                for node in cycle:
                    if node in cycle_dict[-1]:
                        if len(cycle) in cycle_dict[-1][node]:
                            cycle_dict[-1][node][len(cycle)] += 1
                        else:
                            cycle_dict[-1][node][len(cycle)] = 1
                    else:
                        cycle_dict[-1][node] = {}
                        cycle_dict[-1][node][len(cycle)] = 1

        # get all unique dicts of cycles
        dict_list = []
        for g in cycle_dict:
            for node_id, c_dict in g.items():
                dict_list.append(c_dict)

        dict_list = list({str(i) for i in dict_list})
        # sort the dict_list
        dict_list = sorted(dict_list)
        label_dict = {key: value for key, value in zip(dict_list, range(len(dict_list)))}

        # set the node labels
        labels = []
        for graph_id, graph in enumerate(graph_data.nx_graphs):
            labels.append([])
            for node in graph.nodes():
                if node in cycle_dict[graph_id]:
                    cycle_d = str(cycle_dict[graph_id][node])
                    labels[-1].append(label_dict[cycle_d])
                else:
                    labels[-1].append(len(label_dict))

        save_labels_to_file(file, graph_data.name, l, labels, max_labels)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{graph_data.name}, {cycle_type}_cycles_{max_cycle_length}{l}, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")
    return file


def save_in_circle_labels(graph_data:GraphDataset, length_bound=6, max_labels=None, label_path=None, save_times=None)->str:
    l = f'in_cycle_{length_bound}'
    if max_labels is not None:
        l = f"{l}_{max_labels}"
    if label_path is None:
        raise ValueError("No label path given")
    else:
        file = label_path.joinpath(f'{graph_data.name}_labels_{l}.pt')
    if not file.exists():
        print(f"Saving in circle labels for {graph_data.name} to {file}")
        if graph_data.nx_graphs is None:
            graph_data.create_nx_graphs(directed=False)
        start_time = time.time()
        node_in_cycle = []
        for graph in graph_data.nx_graphs:
            node_in_cycle.append({})
            cycles = nx.chordless_cycles(graph, length_bound)
            for cycle in cycles:
                for node in cycle:
                    node_in_cycle[-1][node] = 1


        # set the node labels, if node is in a cycle label 1, else 0
        labels = []
        for graph_id, graph in enumerate(graph_data.nx_graphs):
            labels.append([])
            for node in graph.nodes():
                if node in node_in_cycle[graph_id]:
                    labels[-1].append(1)
                else:
                    labels[-1].append(0)

        save_labels_to_file(file, graph_data.name, l, labels, max_labels=None)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{graph_data.name}, {l}, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")
    return file



def save_subgraph_labels(graph_data:GraphDataset, subgraphs=List[nx.Graph], name='subgraph', subgraph_id=0, max_labels=None, label_path=None, save_times=None)->str:
    l = f'{name}_{subgraph_id}'
    if max_labels is not None:
        l = f"{l}_{max_labels}"
    if label_path is None:
        raise ValueError("No label path given")
    else:
        file = label_path.joinpath(f'{graph_data.name}_labels_{l}.pt')
    if not file.exists():
        print(f"Saving {l} labels for {graph_data.name} to {file}")
        if graph_data.nx_graphs is None:
            graph_data.create_nx_graphs(directed=False)
        start_time = time.time()
        subgraph_dict = []
        for i, graph in enumerate(graph_data.nx_graphs):
            # print the progress
            print(f"Graph {graph_data.name} {i + 1}/{len(graph_data.nx_graphs)} Labels: {l}")
            subgraph_dict.append({})
            for i, subgraph in enumerate(subgraphs):
                GM = GraphMatcher(graph, subgraph)
                for x in GM.subgraph_isomorphisms_iter():
                    for node in x:
                        if node in subgraph_dict[-1]:
                            if i in subgraph_dict[-1][node]:
                                subgraph_dict[-1][node][i] += 1
                            else:
                                subgraph_dict[-1][node][i] = 1
                        else:
                            subgraph_dict[-1][node] = {}
                            subgraph_dict[-1][node][i] = 1

        # get all unique dicts of cycles
        dict_list = []
        for g in subgraph_dict:
            for node_id, c_dict in g.items():
                dict_list.append(c_dict)

        dict_list = list({str(i) for i in dict_list})
        # sort the dict_list
        dict_list = sorted(dict_list)
        label_dict = {key: value for key, value in zip(dict_list, range(len(dict_list)))}

        # set the node labels
        labels = []
        for graph_id, graph in enumerate(graph_data.nx_graphs):
            labels.append([])
            for node in graph.nodes():
                if node in subgraph_dict[graph_id]:
                    cycle_d = str(subgraph_dict[graph_id][node])
                    labels[-1].append(label_dict[cycle_d])
                else:
                    labels[-1].append(len(label_dict))

        save_labels_to_file(file, graph_data.name, l, labels, max_labels=max_labels)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{graph_data.name}, {l}, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")
    return file


def save_clique_labels(graph_data:GraphDataset, max_clique=6, max_labels=None, label_path=None, save_times=None)->str:
    l = f'cliques_{max_clique}'
    if max_labels is not None:
        l = f"{l}_{max_labels}"
    if label_path is None:
        raise ValueError("No label path given")
    else:
        file = label_path.joinpath(f'{graph_data.name}_labels_{l}.pt')
    if not file.exists():
        print(f"Saving {l} labels for {graph_data.name} to {file}")
        if graph_data.nx_graphs is None:
            graph_data.create_nx_graphs(directed=False)
        start_time = time.time()
        clique_dict = []
        for i,graph in enumerate(graph_data.nx_graphs):
            print(f"Graph {graph_data.name} {i + 1}/{len(graph_data.nx_graphs)} Labels: {l}")
            clique_dict.append({})
            cliques = list(nx.find_cliques(graph))
            for clique in cliques:
                if len(clique) <= max_clique:
                    for node in clique:
                        if node in clique_dict[-1]:
                            if len(clique) in clique_dict[-1][node]:
                                clique_dict[-1][node][len(clique)] += 1
                            else:
                                clique_dict[-1][node][len(clique)] = 1
                        else:
                            clique_dict[-1][node] = {}
                            clique_dict[-1][node][len(clique)] = 1

        # get all unique dicts of cycles
        dict_list = []
        for g in clique_dict:
            for node_id, c_dict in g.items():
                dict_list.append(c_dict)

        dict_list = list({str(i) for i in dict_list})
        # sort the dict_list
        dict_list = sorted(dict_list)
        label_dict = {key: value for key, value in zip(dict_list, range(len(dict_list)))}

        # set the node labels
        labels = []
        for graph_id, graph in enumerate(graph_data.nx_graphs):
            labels.append([])
            for node in graph.nodes():
                if node in clique_dict[graph_id]:
                    cycle_d = str(clique_dict[graph_id][node])
                    labels[-1].append(label_dict[cycle_d])
                else:
                    labels[-1].append(len(label_dict))

        save_labels_to_file(file, graph_data.name, l, labels, max_labels)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{graph_data.name}, {l}, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")
    return file

def relabel_node_labels(node_labels: torch.Tensor, max_number_labels:Optional[int]) -> torch.Tensor:
    '''
    Relabel the original labels by mapping them to 0, 1, 2, ... where 0 is the most frequent label of the original labels
    param node_labels: torch.Tensor with the original node labels
    param max_number_labels: Optional[int]
    return: n x 2 torch.Tensor with the original labels as first column and the new labels as second column
    '''
    # get frequency of each value in the new labels, first flatten the tensor
    node_labels = node_labels.flatten()
    max_id = torch.max(node_labels) + 1
    # set negative values to torch.max
    node_labels = torch.where(node_labels < 0, max_id, node_labels)
    unique_labels_count = torch.bincount(node_labels)
    unique_labels_count[-1] = 0
    # sort the unique labels by the frequency and keep the indices
    sorted_indices = torch.argsort(unique_labels_count, descending=True, stable=True)
    # if max_number_labels is given, set sorted_indices after max_number_labels to max_number_labels - 1
    # reindex the unique labels: most frequent label is 0, second most frequent is 1, ...
    frequency_sorted_labels = node_labels.new(sorted_indices).argsort()[node_labels]
    if max_number_labels is not None:
        frequency_sorted_labels = torch.where(frequency_sorted_labels >= max_number_labels, max_number_labels - 1, frequency_sorted_labels)
    # set max_id to -1
    frequency_sorted_labels = torch.where(frequency_sorted_labels == max_id, -1, frequency_sorted_labels)
    node_labels = torch.where(node_labels == max_id, -1, node_labels)
    return torch.stack([node_labels, frequency_sorted_labels], dim=1)


