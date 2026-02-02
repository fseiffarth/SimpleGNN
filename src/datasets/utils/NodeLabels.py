import torch


class NodeLabels:
    def __init__(self, dataset_name:str, label_name:str, node_labels:torch.Tensor):
        # first column are the original node labels, the second column are the relabeled node labels
        self.dataset_name = dataset_name
        self.label_name = label_name
        self.original_node_labels = node_labels[:, 0]
        self.node_labels = node_labels[:, 1]
        self.unique_node_labels, self.unique_node_labels_count = torch.unique(self.node_labels, return_counts=True)
        self.num_unique_node_labels = len(self.unique_node_labels)

    def __iadd__(self, other):
        pass