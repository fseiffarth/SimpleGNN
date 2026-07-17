import torch

from simplegnn.datasets.utils.label_hashing import HASH_SCHEMA_VERSION


class NodeLabels:
    def __init__(self, dataset_name:str, label_name:str, node_labels:torch.Tensor, label_hashes:torch.Tensor=None, hash_meta:dict=None):
        # first column are the original node labels, the second column are the relabeled node labels
        self.dataset_name = dataset_name
        self.label_name = label_name
        self.original_node_labels = node_labels[:, 0]
        self.node_labels = node_labels[:, 1]
        self.unique_node_labels, self.unique_node_labels_count = torch.unique(self.node_labels, return_counts=True)
        self.num_unique_node_labels = len(self.unique_node_labels)
        # canonical hash vocabulary (format v2 label files): frequency-sorted id -> int64 hash
        self.label_hashes = label_hashes
        self.hash_meta = hash_meta
        self.has_canonical_hashes = bool(
            label_hashes is not None
            and hash_meta is not None
            and hash_meta.get('schema') == HASH_SCHEMA_VERSION
            and hash_meta.get('canonical', False)
        )

    def __iadd__(self, other):
        pass
