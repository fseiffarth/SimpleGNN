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

from simplegnn.datasets.graph_dataset import GraphDataset
from simplegnn.datasets.utils.NodeLabels import NodeLabels
from simplegnn.datasets.utils.node_labeling_functions import weisfeiler_lehman_node_labeling




def load_labels(path='') -> NodeLabels:
    """
    Load precomputed node labels from a .pt file.

    Parameters
    ----------
    path : str or Path, optional
        Path to the .pt file containing saved labels (default: '').

    Returns
    -------
    NodeLabels
        Loaded node labels object containing dataset name, label name, and
        label tensor.

    Notes
    -----
    The file must have been saved using torch.save() with the format:
    (dataset_name, label_name, node_labels)

    where node_labels is a torch.Tensor of shape (N, 2):
    - Column 0: Original label indices
    - Column 1: Frequency-sorted label indices (-1 for invalid/padding)

    See Also
    --------
    save_labels_to_file : Saves labels in the expected format
    NodeLabels : Container class for node labels
    """
    dataset_name, label_name, node_labels = torch.load(path, weights_only=True)
    return NodeLabels(dataset_name, label_name, node_labels)

def combine_node_labels(labels: List[NodeLabels]):
    """
    Combine multiple node label sets into a single joint labeling.

    Creates a new labeling by taking the Cartesian product of two label sets,
    assigning unique indices to each label pair. Invalid labels (-1) are handled
    specially and sorted by frequency to optimize downstream usage.

    Parameters
    ----------
    labels : list of NodeLabels
        List of exactly 2 NodeLabels objects to combine. Must have the same
        dataset_name and compatible dimensions (same number of nodes).

    Returns
    -------
    NodeLabels
        Combined node labels with:
        - dataset_name: Same as input labels
        - label_name: Concatenation of input label names (e.g., 'wl_3_degree')
        - node_labels: torch.Tensor of shape (N, 2) where:
            - Column 0: Unique index for each label pair (original order)
            - Column 1: Frequency-sorted indices (most frequent pair = 0)

    Notes
    -----
    **Algorithm:**
    1. Stack label tensors along dimension 1: (N,) + (N,) → (N, 2)
    2. Identify rows containing -1 in either column
    3. Replace invalid rows with (max+1, max+1) to separate from valid labels
    4. Extract unique label pairs and assign indices
    5. Compute label frequency across all nodes
    6. Create frequency-sorted version (most common label → 0)
    7. Restore -1 for originally invalid labels

    **Frequency Sorting:**
    The frequency-sorted column (column 1) assigns index 0 to the most frequent
    label combination, 1 to the second most frequent, etc. This is useful for
    ShareGNN layers that benefit from label frequency information.

    **Invalid Label Handling:**
    Labels with value -1 are preserved in the output but excluded from frequency
    counting. The artificial max label is removed after unique pair extraction.

    Examples
    --------
    >>> wl_labels = NodeLabels('MUTAG', 'wl_3', torch.tensor([0, 1, 0, 2, -1]))
    >>> degree_labels = NodeLabels('MUTAG', 'degree', torch.tensor([2, 2, 1, 3, -1]))
    >>> combined = combine_node_labels([wl_labels, degree_labels])
    >>> combined.label_name
    'wl_3_degree'
    >>> combined.node_labels.shape
    torch.Size([5, 2])  # (original indices, frequency-sorted indices)

    See Also
    --------
    get_label_string : Generates label names from label dictionaries
    NodeLabels : Container for node label data
    """
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
    Convert a label configuration dictionary to a unique filename string.

    Generates a canonical string representation of a label configuration for use
    in filenames, ensuring that equivalent configurations always produce the same
    string. Supports all label types in the framework and handles recursive
    composition for multi-label combinations.

    Parameters
    ----------
    label_dict : dict
        Label configuration dictionary. Must contain 'label_type' key. Additional
        keys depend on the label type:

        Common keys:
        - 'label_type' : str or list of str
            Type of labeling (see Notes for supported types).
        - 'max_labels' : int, optional
            Maximum number of distinct labels to retain.

        Type-specific keys:
        - 'depth' : int
            WL iteration depth (for 'wl', 'wl_labeled', 'wl_labeled_edges').
        - 'base_labels' : dict
            Base labeling for labeled WL variants.
        - 'min_cycle_length', 'max_cycle_length' : int
            Cycle bounds (for 'simple_cycles', 'induced_cycles').
        - 'max_clique_size' : int
            Maximum clique size (for 'cliques').
        - 'id' : int
            Subgraph identifier (for 'subgraph').

    Returns
    -------
    str
        Unique string identifier for the label configuration. Format examples:
        - 'primary' or 'primary_100' (with max_labels)
        - 'wl_3' or 'wl_3_50' (WL with depth 3, optionally capped at 50 labels)
        - 'wl_labeled_primary_base_labels_3_100' (WL labeled, base=primary, depth=3, max=100)
        - 'simple_cycles_3_6' (simple cycles with length 3-6)
        - 'wl_3_degree' (combination of two label types)

    Raises
    ------
    ValueError
        If 'label_type' is missing from label_dict or if an unsupported label
        type is specified.

    Notes
    -----
    **Supported Label Types:**
    - 'trivial': All nodes get label 0
    - 'index': Nodes labeled by their index (0, 1, 2, ...)
    - 'index_text': Same as index but with text representation
    - 'primary': Nodes labeled by their original graph label
    - 'degree': Node degree (equivalent to wl_0)
    - 'wl': Weisfeiler-Lehman labeling (requires 'depth')
    - 'wl_labeled': WL using base_labels as initial coloring
    - 'wl_labeled_edges': WL with edge labels (requires 'base_labels')
    - 'simple_cycles': Simple cycle membership (requires 'min/max_cycle_length')
    - 'induced_cycles': Induced cycle membership
    - 'cliques': Clique membership (requires 'max_clique_size')
    - 'subgraph': Subgraph pattern matching (requires 'id')

    **Recursive Composition:**
    If label_type is a list, the function recursively generates strings for each
    type and joins them with underscores. This creates combined labels like:
    ['wl', 'degree'] → 'wl_3_degree'

    **Filename Safety:**
    The generated strings use only alphanumeric characters and underscores,
    making them safe for use in filenames across all platforms.

    Examples
    --------
    >>> get_label_string({'label_type': 'primary', 'max_labels': 100})
    'primary_100'

    >>> get_label_string({'label_type': 'wl', 'depth': 3, 'max_labels': 50})
    'wl_3_50'

    >>> get_label_string({
    ...     'label_type': 'wl_labeled',
    ...     'base_labels': {'label_type': 'primary'},
    ...     'depth': 3
    ... })
    'wl_labeled_primary_base_labels_3'

    >>> get_label_string({
    ...     'label_type': ['wl', 'degree'],
    ...     'depth': 3,
    ...     'max_labels': 100
    ... })
    'wl_3_degree_100'

    See Also
    --------
    combine_node_labels : Combines multiple label sets
    save_*_labels : Functions that use these strings for filenames
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
    """
    Abstract base class for node labeling strategies.

    Provides a common interface and workflow for generating, processing, and
    saving node labels for graph datasets. Subclasses implement specific labeling
    algorithms (WL, degree, cycles, etc.) by overriding the generate() method.

    Parameters
    ----------
    base_name : str
        Base name for this labeling type (e.g., 'wl', 'degree', 'trivial').
    graph_data : GraphDataset
        Graph dataset to label.
    label_path : Path, optional
        Directory where label files will be saved (default: None).
        If None, create_and_save_labels() will raise an error.
    max_labels : int, optional
        Maximum number of distinct labels to retain. If specified, less frequent
        labels will be merged into a single "other" category (default: None).
    optional_parameters : list of tuple, optional
        List of (name, value) pairs for algorithm-specific parameters
        (e.g., [('depth', 3), ('min_cycle_length', 3)]).
        These are appended to the label filename string (default: None).
    save_times : Path, optional
        Path to a file for logging generation times. Each labeling operation
        appends a line with format: "dataset_name, label_name, time_seconds"
        (default: None).

    Attributes
    ----------
    base_name : str
        Base labeling type name.
    graph_data : GraphDataset
        Graph dataset being labeled.
    label_path : Path or None
        Directory for saving labels.
    max_labels : int or None
        Label count cap.
    optional_parameters : list of tuple or None
        Algorithm-specific parameters.
    save_times : Path or None
        Timing log file path.
    string_label_name : str
        Full label identifier string constructed from base_name, max_labels,
        and optional_parameters. Used in filenames.
    file_path : Path or None
        Full path to the label file (set by create_and_save_labels()).

    Methods
    -------
    create_and_save_labels()
        Main entry point: generates labels and saves to disk if not already cached.
    generate()
        Abstract method to be implemented by subclasses. Creates the actual labels.
    set_string_label_name()
        Constructs the full label name string from parameters.
    save_labels_to_file(graph_node_labels)
        Saves label tensor to disk in standard format.

    Notes
    -----
    **Label Generation Workflow:**
    1. Call create_and_save_labels()
    2. Construct filename from string_label_name
    3. Check if file already exists (caching)
    4. If not, call generate() to create labels
    5. Apply max_labels capping via relabel_node_labels()
    6. Save to disk using torch.save()
    7. Optionally log generation time

    **File Format:**
    Labels are saved as .pt files with naming pattern:
        <dataset_name>_labels_<string_label_name>.pt

    Containing:
        (dataset_name, label_name, node_labels)

    Where node_labels is a torch.Tensor of shape (N, 2):
    - Column 0: Original label indices
    - Column 1: Frequency-sorted label indices (most frequent → 0)

    **Caching:**
    If the target file already exists, generation is skipped. This allows
    expensive label computations to be reused across experiments.

    **Label Capping:**
    When max_labels is set, only the most frequent labels are kept. Less
    frequent labels are remapped to a single "other" label (-1).

    Raises
    ------
    ValueError
        If label_path is None when create_and_save_labels() is called.
    NotImplementedError
        If generate() is not implemented by a subclass.

    See Also
    --------
    TrivialNodeLabeling : All nodes labeled 0
    WeisfeilerLehmanNodeLabeling : WL algorithm
    DegreeNodeLabeling : Node degree labeling
    relabel_node_labels : Applies max_labels capping

    Examples
    --------
    >>> # Subclass example
    >>> class CustomLabeling(NodeLabelingBase):
    ...     def __init__(self, graph_data, label_path):
    ...         super().__init__('custom', graph_data, label_path)
    ...
    ...     def generate(self):
    ...         # Custom labeling logic
    ...         return torch.zeros(graph_data.num_nodes, dtype=torch.long)
    ...
    >>> labeler = CustomLabeling(my_graph_data, Path('labels/'))
    >>> labeler.create_and_save_labels()
    # Saves to labels/MUTAG_labels_custom.pt (if doesn't exist)
    """
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
        """
        Generate and save node labels if not already cached.

        Main entry point for label creation. Checks if labels already exist on
        disk (caching). If not, generates labels via generate(), applies label
        capping, and saves to file. Optionally logs generation time.

        Raises
        ------
        ValueError
            If self.label_path is None or self.save_times is specified but invalid.

        Notes
        -----
        File path format: <label_path>/<dataset_name>_labels_<string_label_name>.pt

        If the file exists, prints a message and skips generation. This caching
        behavior allows expensive computations (e.g., WL with depth 5) to be
        reused across multiple experiments.

        Generation time is appended to save_times file if specified, in format:
        "dataset_name, label_name, time_seconds"
        """
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
        """
        Construct the full label name string from configuration.

        Builds string_label_name by concatenating:
        1. base_name
        2. '_max_labels_<N>' if max_labels is not None
        3. '_<param_name>_<param_value>' for each optional_parameter

        Example: base_name='wl', max_labels=100, optional_parameters=[('depth', 3)]
        → string_label_name='wl_max_labels_100_depth_3'

        This string is used in filenames to uniquely identify label configurations.
        """
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
    """
    Trivial node labeling: assigns label 0 to all nodes.

    This is the simplest labeling strategy, providing no structural information.
    Useful as a baseline or when node features alone are sufficient.

    See Also
    --------
    NodeLabelingBase : Base class with full documentation
    IndexNodeLabeling : Assigns unique indices to each node
    """
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
    """
    Degree-based node labeling: labels nodes by their degree.

    Each node is assigned a label equal to its degree (number of neighbors).
    This is equivalent to 0-iteration Weisfeiler-Lehman (wl_0) labeling.

    Notes
    -----
    - Label value equals node degree directly
    - Undirected graphs: degree = number of incident edges
    - Provides local structural information without graph-wide context
    - Commonly used as base labeling for more complex strategies

    See Also
    --------
    NodeLabelingBase : Base class with full documentation
    WeisfeilerLehmanNodeLabeling : WL algorithm (iterative refinement of degree)
    """
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
    """
    Weisfeiler-Lehman node labeling with configurable depth.

    Implements the classic WL algorithm for graph isomorphism testing. Each iteration
    refines node labels based on the multiset of neighbor labels, capturing
    increasingly global structural patterns.

    Parameters
    ----------
    depth : int
        Number of WL iterations. Higher depth captures longer-range dependencies:
        - depth=0: Equivalent to degree labeling
        - depth=1: Degree + immediate neighbor degrees
        - depth=k: Information from k-hop neighborhood

    Notes
    -----
    The WL algorithm is a powerful graph hashing technique that distinguishes most
    non-isomorphic graphs. It's the theoretical foundation for many GNN architectures.

    Algorithm:
    1. Initialize with degree labels (or trivial labels)
    2. For each iteration:
        a. Collect multiset of neighbor labels for each node
        b. Hash (node_label, sorted_neighbor_labels) to create new label
        c. Compress label space if needed
    3. Return final labels after 'depth' iterations

    Computational complexity: O(depth × |E|) where |E| is the number of edges.

    See Also
    --------
    NodeLabelingBase : Base class with full documentation
    DegreeNodeLabeling : Equivalent to WL with depth=0
    WeisfeilerLehmanLabeledNodeLabeling : WL variant using initial node features
    """
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


class BetweennessCentralityNodeLabeling(NodeLabelingBase):
    """
    Node labeling based on betweenness centrality.

    Betweenness centrality measures how often a node lies on shortest paths
    between other nodes. High values indicate "bridge" nodes that connect
    different parts of the graph.

    The continuous centrality values are discretized into bins using
    percentile-based binning for meaningful distribution across labels.

    Parameters
    ----------
    graph_data : GraphDataset
        The dataset to label.
    label_path : Path, optional
        Directory where labels will be saved.
    max_labels : int, optional
        Maximum number of distinct labels (for frequency-based filtering after binning).
    num_bins : int, optional
        Number of bins for discretization. If not specified, uses max_labels
        or defaults to 10.
    save_times : Path, optional
        Path to file for logging generation times.

    Notes
    -----
    - Centrality values range from 0.0 (peripheral) to 1.0 (central bridge)
    - Percentile-based binning ensures balanced label distribution
    - Nodes with same centrality receive same label
    - Computational complexity: O(n³) for dense graphs, O(nm) for sparse graphs

    See Also
    --------
    NodeLabelingBase : Base class with full documentation
    DegreeNodeLabeling : Simpler local structural measure
    """
    def __init__(self, graph_data: GraphDataset, label_path: Optional[Path] = None,
                 max_labels: Optional[int] = None, num_bins: Optional[int] = None,
                 save_times: Optional[Path] = None):
        optional_params = []
        if num_bins is not None:
            optional_params.append(('bins', num_bins))

        super().__init__(
            base_name='betweenness',
            graph_data=graph_data,
            label_path=label_path,
            max_labels=max_labels,
            optional_parameters=optional_params,
            save_times=save_times
        )

        self.num_bins = num_bins or max_labels or 10

    def generate(self) -> List[List[int]]:
        """
        Compute betweenness centrality labels for all nodes.

        Returns
        -------
        List[List[int]]
            List of discretized centrality labels, one list per graph.
        """
        if self.graph_data.nx_graphs is None:
            self.graph_data.create_nx_graphs(directed=False)

        # Step 1: Collect all betweenness centrality values across dataset
        all_centralities = []
        graph_centralities = []

        for graph in self.graph_data.nx_graphs:
            # Compute betweenness centrality for all nodes
            centrality = nx.betweenness_centrality(graph, normalized=True)
            # Store as list maintaining node order
            cent_list = [centrality[node] for node in graph.nodes()]
            graph_centralities.append(cent_list)
            all_centralities.extend(cent_list)

        # Step 2: Compute percentile-based bins across entire dataset
        import numpy as np
        percentiles = np.linspace(0, 100, self.num_bins + 1)
        bins = np.percentile(all_centralities, percentiles)

        # Handle edge case where all values are identical
        if len(set(bins)) == 1:
            return [[0] * len(gc) for gc in graph_centralities]

        # Step 3: Discretize each graph using the global bins
        node_labels = []
        for cent_list in graph_centralities:
            # digitize returns 1-indexed bins, subtract 1 for 0-indexing
            # Example: bins=[0, 0.2, 0.5, 1.0] with num_bins=3 gives labels [0, 1, 2]
            labels = np.digitize(cent_list, bins[1:-1], right=False)
            node_labels.append(labels.tolist())

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


def save_betweenness_centrality_labels(graph_data: GraphDataset, label_path: Optional[Path] = None,
                                       max_labels: Optional[int] = None, num_bins: Optional[int] = None,
                                       save_times: Optional[Path] = None) -> str:
    """
    Generate and save betweenness centrality-based node labels.

    Parameters
    ----------
    graph_data : GraphDataset
        Graph dataset to label.
    label_path : Path, optional
        Directory where labels will be saved.
    max_labels : int, optional
        Maximum number of distinct labels (used as num_bins if num_bins not set).
    num_bins : int, optional
        Number of bins for discretization (overrides max_labels if set).
    save_times : Path, optional
        Path to file for logging generation times.

    Returns
    -------
    str
        Path to the saved .pt file.

    See Also
    --------
    BetweennessCentralityNodeLabeling : Implementation details
    """
    labeling = BetweennessCentralityNodeLabeling(
        graph_data, label_path, max_labels, num_bins, save_times
    )
    return labeling.create_and_save_labels()


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


