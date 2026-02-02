# abstract class for graph data preprocessing
import abc
from pathlib import Path

import numpy as np
from torch_geometric.data import InMemoryDataset, Data
import torch
import torch_geometric
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.datasets import ZINC, TUDataset


class GraphDataPreprocessing(abc.ABC):
    def __init__(self, name, tmp_dir="/tmp"):
        self.name = name
        self.tmp_dir = tmp_dir
        self.processed_dataset = None
        self.slices = None
        self.sizes =  {'num_edge_attributes': None,
                         'num_edge_labels': None,
                         'num_node_attributes': None,
                         'num_node_labels': None
                }


    @abc.abstractmethod
    def preprocess(self, *args, **kwargs):
        """
        Abstract method to preprocess the raw dataset.
        This method should be implemented by subclasses to perform specific preprocessing tasks.

        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        :return: Processed graph data.
        """
        return NotImplementedError("Subclasses should implement this method.")


    def set_sizes(self):
        if self.processed_dataset is None:
            raise ValueError("Processed dataset is not set. Please run preprocess() first.")

        self.sizes = {'num_edge_attributes': self.processed_dataset.edge_attributes.shape[-1],
                 'num_edge_labels': len(torch.unique(self.processed_dataset.primary_edge_labels)),
                 'num_node_attributes': self.processed_dataset.node_attributes.shape[-1],
                 'num_node_labels': len(torch.unique(self.processed_dataset.primary_node_labels))
                 }
class MergedGraphDataPreprocessing(GraphDataPreprocessing):
    def __init__(self, name, tmp_dir="/tmp", datasets_list=None):
        super().__init__(name, tmp_dir)
        self.preprocess(datasets_list=datasets_list)


    def preprocess(self, datasets_list=None, *args, **kwargs):
        """
        Preprocess the merged dataset.

        :param datasets_list:
        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        :return: Processed graph data.
        """
        all_x = None
        all_edge_indices = None
        all_edge_atr = None
        all_y = None
        all_num_nodes = None
        if datasets_list is None or len(datasets_list) == 0:
            raise ValueError("datasets_list must be provided and non-empty")
        self.slices = {
            'x': [0],
            'primary_node_labels': [0],
            'primary_edge_labels': [0],
            'node_attributes': [0],
            'edge_index': [0],
            'edge_attributes': [0],
            'y': [0]
        }
        self.sizes = {
            'num_node_labels': 0,
            'num_node_attributes': 0,
            'num_edge_labels': 0,
            'num_edge_attributes': 0
        }
        for i, dataset in enumerate(datasets_list):
            current_x = dataset.data.x
            current_edge_indices = dataset.data.edge_index
            current_edge_attributes = dataset.data.edge_attributes
            current_node_attributes = dataset.data.node_attributes
            current_primary_node_labels = dataset.data.primary_node_labels
            current_primary_edge_labels = dataset.data.primary_edge_labels
            current_y = dataset.data.y
            current_num_nodes = dataset.data.num_nodes
            if i == 0:
                all_x = current_x
                all_node_attributes = current_node_attributes
                all_primary_node_labels = current_primary_node_labels
                all_edge_indices = current_edge_indices
                all_edge_attributes = current_edge_attributes
                all_primary_edge_labels = current_primary_edge_labels
                all_y = current_y
                all_num_nodes = current_num_nodes
                self.slices = {
                    'x': dataset.slices['x'],
                    'primary_node_labels': dataset.slices['primary_node_labels'],
                    'edge_index': dataset.slices['edge_index'],
                    'y': dataset.slices['y'],
                    'names': [dataset.name] * len(dataset)
                }
                if 'node_attributes' in dataset.slices:
                    self.slices['node_attributes'] = dataset.slices['node_attributes']
                if 'edge_attributes' in dataset.slices:
                    self.slices['edge_attributes'] = dataset.slices['edge_attributes']
                if 'primary_edge_labels' in dataset.slices:
                    self.slices['primary_edge_labels'] = dataset.slices['primary_edge_labels']
                self.sizes = {
                    'num_node_labels': dataset.num_node_labels,
                    'num_node_attributes': dataset.num_node_attributes,
                    'num_edge_labels': dataset.num_edge_labels,
                    'num_edge_attributes': dataset.num_edge_attributes,
                }
            else:
                max_node_labels = max(self.sizes['num_node_labels'], dataset.num_node_labels)
                max_node_attrs = max(self.sizes['num_node_attributes'], dataset.num_node_attributes)
                max_edge_labels = max(self.sizes['num_edge_labels'], dataset.num_edge_labels)
                max_edge_attrs = max(self.sizes['num_edge_attributes'], dataset.num_edge_attributes)

                self.slices['x'] = torch.cat((self.slices['x'], dataset.slices['x'][1:] + all_x.shape[0]), dim=0)
                self.slices['primary_node_labels'] = torch.cat((self.slices['primary_node_labels'],
                                                                dataset.slices['primary_node_labels'][1:] +
                                                                all_primary_node_labels.shape[0]), dim=0)

                self.slices['edge_index'] = torch.cat(
                    (self.slices['edge_index'], dataset.slices['edge_index'][1:] + all_edge_indices.shape[1]), dim=0)

                if 'node_attributes' in self.slices:
                    self.slices['node_attributes'] = torch.cat((self.slices['node_attributes'],
                                                                dataset.slices['node_attributes'][1:] +
                                                                all_node_attributes.shape[0]), dim=0)
                elif 'node_attributes' in dataset.slices:
                    pass  # TODO: handle the case where the first dataset has no node attributes but the second has

                if 'edge_attributes' in self.slices:
                    self.slices['edge_attr'] = torch.cat(
                        (self.slices['edge_attr'], dataset.slices['edge_attr'][1:] + all_edge_atr.shape[0]), dim=0)
                elif 'edge_attributes' in dataset.slices:
                    pass  # TODO: handle the case where the first dataset has no edge attributes but the second has

                if 'primary_edge_labels' in self.slices:
                    self.slices['primary_edge_labels'] = torch.cat((self.slices['primary_edge_labels'],
                                                                   dataset.slices['primary_edge_labels'][1:] +
                                                                   all_primary_edge_labels.shape[0]), dim=0)
                elif 'primary_edge_labels' in dataset.slices:
                    pass # TODO: handle the case where the first dataset has no primary edge labels but the second has

                self.slices['y'] = torch.cat((self.slices['y'], dataset.slices['y'][1:] + all_y.shape[0]), dim=0)
                self.slices['names'] += [dataset.name] * len(dataset)

                # bring all tensors to the same size
                max_size = max(current_x.shape[1], all_x.shape[1])
                if all_x.shape[1] < max_size:
                    all_x = torch.cat((all_x, torch.zeros(all_x.shape[0], max_size - all_x.shape[1])), dim=1)
                if current_x.shape[1] < max_size:
                    current_x = torch.cat((current_x, torch.zeros(current_x.shape[0], max_size - current_x.shape[1])), dim=1)
                # merge the x tensors
                all_x = torch.cat((all_x, current_x), dim=0)
                all_primary_node_labels = torch.cat((all_primary_node_labels, current_primary_node_labels), dim=0)

                if 'node_attributes' in self.slices:
                    all_node_attributes = torch.cat((all_node_attributes, current_node_attributes), dim=0)

                if 'edge_attributes' in self.slices:
                    all_edge_attributes = torch.cat((all_edge_attributes, current_edge_attributes), dim=0)

                all_edge_indices = torch.cat((all_edge_indices, current_edge_indices), dim=1)

                if 'primary_edge_labels' in self.slices:
                    all_primary_edge_labels = torch.cat((all_primary_edge_labels, current_primary_edge_labels), dim=0)

                all_y = torch.cat((all_y, current_y), dim=0)
                all_num_nodes = torch.cat((all_num_nodes, current_num_nodes), dim=0)

                self.sizes['num_node_labels'] = max_node_labels
                self.sizes['num_node_attributes'] = max_node_attrs
                self.sizes['num_edge_labels'] = max_edge_labels
                self.sizes['num_edge_attributes'] = max_edge_attrs
        # make self data from all_x, all_edge_indices, all_edge_atr, all_y
        self.processed_dataset = Data(x=all_x, node_attributes=all_node_attributes, primary_node_labels=all_primary_node_labels, edge_index=all_edge_indices, primary_edge_labels=all_primary_edge_labels, edge_attributes=all_edge_attributes, y=all_y, num_nodes=all_num_nodes)
        return self.processed_dataset, self.slices, self.sizes

class ZINCGraphDataPreprocessing(GraphDataPreprocessing):
    def __init__(self, name, tmp_dir="/tmp"):
        super().__init__(name, tmp_dir)
        self.preprocess()


    def preprocess(self, *args, **kwargs):
        """
        Preprocess the ZINC dataset.

        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        :return: Processed graph data.
        """
        subset = True
        if self.name in ['ZINC-full', 'ZINC-Full', 'ZINCFull', 'ZINC-250k']:
            subset = False
        train_data = ZINC(root=self.tmp_dir, subset=subset, split='train')
        validation_data = ZINC(root=self.tmp_dir, subset=subset, split='val')
        test_data = ZINC(root=self.tmp_dir, subset=subset, split='test')
        # merge train_data._data, validation_data._data and test_data._data
        all_data = torch_geometric.data.InMemoryDataset.collate(
            [train_data._data, validation_data._data, test_data._data])

        self.processed_dataset = all_data[0]

        # merge the slices
        self.slices = dict()
        for key in train_data.slices.keys():
            validation_data.slices[key] += train_data.slices[key][-1]
            test_data.slices[key] += validation_data.slices[key][-1]
            self.slices[key] = torch.cat(
                (train_data.slices[key], validation_data.slices[key][1:], test_data.slices[key][1:]))

        self.processed_dataset.primary_node_labels = self.processed_dataset.x
        # flatten
        self.processed_dataset.primary_node_labels = self.processed_dataset.primary_node_labels.view(-1)
        self.slices['primary_node_labels'] = self.slices['x']
        self.processed_dataset.node_attributes = torch.Tensor()
        self.processed_dataset.primary_edge_labels = self.processed_dataset.edge_attr
        self.slices['primary_edge_labels'] = self.slices['edge_attr']
        self.processed_dataset.edge_attributes = torch.Tensor()


        self.set_sizes()

        return self.processed_dataset, self.slices, self.sizes


class QMGraphDataPreprocessing(GraphDataPreprocessing):
    def __init__(self, name, tmp_dir="/tmp"):
        super().__init__(name, tmp_dir)
        self.preprocess()


    def preprocess(self, *args, **kwargs):
        """
        Preprocess the QM9 dataset.

        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        :return: Processed graph data.
        """
        if self.name in ['QM9', 'qm9', 'QM', 'qm']:
            dataset = torch_geometric.datasets.QM9(root=self.tmp_dir)
        elif self.name in ['QM7', 'qm7', 'QM7b', 'qm7b']:
            dataset = torch_geometric.datasets.QM7b(root=self.tmp_dir)
        dataset_node_labels = dataset.data.z
        dataset_node_attributes = dataset.data.x[:, [6, 7, 8, 9]]
        # one hot over the edge_attr
        dataset_edge_labels = torch.argmax(dataset.data.edge_attr, dim=1)

        dataset.data.primary_node_labels = dataset_node_labels
        dataset.data.primary_edge_labels = dataset_edge_labels
        dataset.data.node_attributes = torch.cat((dataset_node_attributes, dataset.pos), dim=1)
        dataset.data.edge_attributes = torch.Tensor()

        self.processed_dataset = dataset.data
        self.slices = dataset.slices
        self.slices['primary_node_labels'] = self.slices['x']
        self.slices['node_attributes'] = self.slices['x']
        self.slices['primary_edge_labels'] = self.slices['edge_attr']
        self.set_sizes()
        return self.processed_dataset, self.slices, self.sizes


class OGBGraphPropertyGraphDataPreprocessing(GraphDataPreprocessing):
    def __init__(self, name, tmp_dir="/tmp"):
        super().__init__(name, tmp_dir)
        self.preprocess()


    def preprocess(self, *args, **kwargs):
        """
        Preprocess the OGB dataset.

        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        :return: Processed graph data.
        """
        dataset_ogb = PygGraphPropPredDataset(name=self.name, root=self.tmp_dir)
        split_idx = dataset_ogb.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        self.processed_dataset = dataset_ogb.data
        self.processed_dataset.primary_node_labels = dataset_ogb.x[:, 0]  # first column is the primary node label
        self.processed_dataset.node_attributes = dataset_ogb.x[:, 1:9]  # next 8 columns are node attributes
        self.processed_dataset.primary_edge_labels = dataset_ogb.edge_attr[:, 0]  # first column is the primary edge label
        self.processed_dataset.edge_attributes = dataset_ogb.edge_attr[:, 1:3]  # next 2 columns are edge attributes

        # if second dimension of y is 1, flatten it
        if self.processed_dataset.y.dim() == 2 and self.processed_dataset.y.shape[1] == 1:
            self.processed_dataset.y = self.processed_dataset.y.view(-1)

        self.slices = dataset_ogb.slices
        self.slices['primary_node_labels'] = self.slices['x']
        self.slices['node_attributes'] = self.slices['x']

        self.slices['primary_edge_labels'] = self.slices['edge_attr']
        self.slices['edge_attributes'] = self.slices['edge_attr']

        self.set_sizes()

        return self.processed_dataset, self.slices, self.sizes

class SubstructureBenchmarkPreprocessing(GraphDataPreprocessing):
    def __init__(self, name, tmp_dir="/tmp"):
        super().__init__(name, tmp_dir)
        self.preprocess()
    def preprocess(self, *args, **kwargs):
        """
        Preprocess the Substructure Benchmark dataset.
        """
        # relative path to project root
        root_path = Path(__file__).parent.parent.parent.parent
        # root_path to string
        root_path = str(root_path)
        train_data = GraphCount(root=root_path, split="train", task=self.name)
        validation_data = GraphCount(root=root_path, split="val", task=self.name)
        test_data = GraphCount(root=root_path, split="test", task=self.name)
        all_data = torch_geometric.data.InMemoryDataset.collate([train_data._data, validation_data._data, test_data._data])
        self.processed_dataset = all_data[0]
        # flatten y if self.name is not 'multi'
        if self.name != 'multi':
            self.processed_dataset.y = self.processed_dataset.y.view(-1)
        # merge the slices
        self.slices = dict()
        for key in train_data.slices.keys():
            validation_data.slices[key] += train_data.slices[key][-1]
            test_data.slices[key] += validation_data.slices[key][-1]
            self.slices[key] = torch.cat((train_data.slices[key], validation_data.slices[key][1:], test_data.slices[key][1:]))

        self.processed_dataset.primary_node_labels = self.processed_dataset.x
        self.processed_dataset.node_attributes = torch.Tensor()
        self.processed_dataset.primary_edge_labels = torch.Tensor()
        self.processed_dataset.edge_attributes = torch.Tensor()

        self.slices['primary_node_labels'] = self.slices['x']


        sizes = {'num_edge_attributes': 0,
                 'num_edge_labels': 0,
                 'num_node_attributes': 0,
                 'num_node_labels': 0
                 }
        return self.processed_dataset, self.slices, sizes


class TUDatasetPreprocessing(GraphDataPreprocessing):
    def __init__(self, name, tmp_dir="/tmp"):
        super().__init__(name, tmp_dir)
        self.preprocess()

    def preprocess(self, *args, **kwargs):
        tu_dataset = TUDataset(root='tmp/', name=self.name, use_node_attr=True, use_edge_attr=True)
        if 'x' not in tu_dataset.data:
            tu_dataset.data.x = torch.zeros((tu_dataset.num_nodes,1), dtype=torch.float)
            tu_dataset.data.primary_node_labels = torch.zeros(tu_dataset.num_nodes, dtype=torch.long)
            tu_dataset.slices['x'] = torch.zeros(len(tu_dataset)+1, dtype=torch.long)
            # get slices from edge_index_slices
            edge_slices = tu_dataset.slices['edge_index']
            for i, edge_slice in enumerate(edge_slices):
                if i > 0:
                    start = edge_slices[i-1]
                    end = edge_slices[i]
                    num_nodes = torch.max(tu_dataset.edge_index[:, start:end]) - torch.min(tu_dataset.edge_index[:, start:end]) + 1
                    tu_dataset.slices['x'][i] = tu_dataset.slices['x'][i-1] + num_nodes
        else:
            tu_dataset.data.primary_node_labels = torch.argmax(tu_dataset.data.x[:,tu_dataset.sizes['num_node_attributes']:], dim=1)
        tu_dataset.slices['primary_node_labels'] = tu_dataset.slices['x']
        if tu_dataset.sizes['num_node_attributes'] > 0:
            tu_dataset.data.node_attributes = tu_dataset.data.x[:,:tu_dataset.sizes['num_node_attributes']]
            tu_dataset.slices['node_attributes'] = tu_dataset.slices['x']
        else:
            tu_dataset.data.node_attributes = torch.Tensor()
        if tu_dataset.data.edge_attr is None:
            tu_dataset.data.primary_edge_labels = torch.Tensor()
            tu_dataset.data.edge_attributes = torch.Tensor()
        else:
            tu_dataset.data.primary_edge_labels = torch.argmax(tu_dataset.data.edge_attr[:,tu_dataset.sizes['num_edge_attributes']:], dim=1)
            tu_dataset.slices['primary_edge_labels'] = tu_dataset.slices['edge_attr']
            if tu_dataset.sizes['num_edge_attributes'] > 0:
                tu_dataset.data.edge_attributes = tu_dataset.data.edge_attr[:,:tu_dataset.sizes['num_edge_attributes']]
            else:
                tu_dataset.data.edge_attributes = tu_dataset.data.edge_attr
            tu_dataset.slices['edge_attributes'] = tu_dataset.slices['edge_attr']
            # remove edge_attr from data, slices and sizes
            tu_dataset.data.edge_attr = None
            tu_dataset.slices.pop('edge_attr', None)

        self.processed_dataset = tu_dataset.data
        self.slices = tu_dataset.slices
        self.set_sizes()

        return self.processed_dataset, self.slices, self.sizes



class GraphCount(InMemoryDataset):

    task_index = dict(
        triangle=0,
        tri_tail=1,
        star=2,
        cycle4=3,
        cycle5=4,
        cycle6=5,
        multi = -1,
    )

    def __init__(self, root:str, split:str, task:str, **kwargs):
        super().__init__(root=root, **kwargs)

        _pt = dict(zip(["train", "val", "test"], self.processed_paths))
        self.data, self.slices = torch.load(_pt[split])

        index = self.task_index[task]
        if index != -1:
            self.data.y = self.data.y[:, index:index+1]

    @property
    def raw_file_names(self):
        return ["Data/GraphDatasets/SubstructureCountingBenchmark.pt"]

    @property
    def processed_dir(self):
        return f"{self.root}/randomgraph"

    @property
    def processed_file_names(self):
        return ["train.pt", "val.pt", "test.pt"]

    def process(self):

        _pt, = self.raw_file_names
        raw = torch.load(f"{self.root}/{_pt}")

        def to(graph):

            A = graph["A"]
            y = graph["y"]

            return Data(
                x=torch.ones(A.shape[0], 1, dtype=torch.int64), y=y,
                edge_index=torch.Tensor(np.vstack(np.where(graph["A"] > 0)))
                     .type(torch.int64),
            )

        data = [to(graph) for graph in raw["data"]]

        if self.pre_filter is not None:
            data = filter(self.pre_filter, data)

        if self.pre_transform is not None:
            data = map(self.pre_transform, data)

        data_list = list(data)
        normalize = torch.std(torch.stack([data.y for data in data_list]), dim=0)

        for split in ["train", "val", "test"]:

            from operator import itemgetter
            split_idx = raw["index"][split]
            splits = itemgetter(*split_idx)(data_list)

            data, slices = self.collate(splits)
            data.y = data.y / normalize

            torch.save((data, slices), f"{self.processed_dir}/{split}.pt")
