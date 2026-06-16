from __future__ import annotations

import networkx as nx
import pytest


torch = pytest.importorskip("torch")

from simplegnn.datasets.utils.NodeLabels import NodeLabels
from simplegnn.datasets.utils.node_labeling import combine_node_labels, get_label_string, relabel_node_labels
from simplegnn.datasets.utils.node_labeling_functions import (
    degree_node_labeling,
    standard_node_labeling,
    weisfeiler_lehman_node_labeling,
)


def make_graphs():
    g1 = nx.Graph()
    g1.add_node(0, primary_node_labels=1)
    g1.add_node(1, primary_node_labels=2)
    g1.add_edge(0, 1)

    g2 = nx.Graph()
    g2.add_node(0, primary_node_labels=1)
    g2.add_node(1, primary_node_labels=1)
    g2.add_edge(0, 1)
    return [g1, g2]


def test_standard_and_degree_labeling_shapes():
    graphs = make_graphs()

    standard = standard_node_labeling(graphs)
    degree = degree_node_labeling(graphs)

    assert len(standard[0]) == len(graphs)
    assert len(degree[0]) == len(graphs)


def test_wl_labeling_returns_labels_for_each_node():
    graphs = make_graphs()
    labels, unique, db_unique = weisfeiler_lehman_node_labeling(graphs, depth=2)

    assert sum(len(x) for x in labels) == sum(g.number_of_nodes() for g in graphs)
    assert len(unique) == len(graphs)
    assert len(db_unique) > 0


def test_combine_node_labels_returns_two_column_tensor():
    l1 = NodeLabels("MUTAG", "a", torch.tensor([[0, 0], [1, 1], [2, 2], [-1, -1]]))
    l2 = NodeLabels("MUTAG", "b", torch.tensor([[2, 2], [1, 1], [0, 0], [-1, -1]]))

    combined = combine_node_labels([l1, l2])

    assert combined.dataset_name == "MUTAG"
    assert combined.node_labels.ndim == 2
    assert combined.node_labels.shape[1] == 2


def test_get_label_string_for_wl_and_primary():
    assert get_label_string({"label_type": "primary"}) == "primary"
    assert get_label_string({"label_type": "wl", "depth": 3}) == "wl_3"


def test_relabel_node_labels_handles_cap_and_invalids():
    node_labels = torch.tensor([0, 0, 1, 2, -1], dtype=torch.long)

    relabeled = relabel_node_labels(node_labels, max_number_labels=2)

    assert relabeled.shape == (5, 2)
    assert relabeled[-1, 0].item() == -1
    assert relabeled[:, 1].max().item() <= 1
