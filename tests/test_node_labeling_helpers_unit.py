from __future__ import annotations

import networkx as nx
import pytest


torch = pytest.importorskip("torch")

from simplegnn.datasets.utils.NodeLabels import NodeLabels
from simplegnn.datasets.utils.node_labeling import (
    ClosedWalkNodeLabeling,
    combine_node_labels,
    get_label_string,
    relabel_node_labels,
    save_closed_walk_labels,
)
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


def test_combine_node_labels_returns_per_node_labels():
    # NodeLabels takes a two-column (original, relabeled) tensor and stores
    # each column as a 1-D per-node tensor; combine_node_labels feeds it the
    # (unique-pair index, frequency-sorted index) columns.
    l1 = NodeLabels("MUTAG", "a", torch.tensor([[0, 0], [1, 1], [2, 2], [-1, -1]]))
    l2 = NodeLabels("MUTAG", "b", torch.tensor([[2, 2], [1, 1], [0, 0], [-1, -1]]))

    combined = combine_node_labels([l1, l2])

    assert combined.dataset_name == "MUTAG"
    assert combined.label_name == "a_b"
    assert combined.node_labels.ndim == 1
    assert combined.original_node_labels.ndim == 1
    # pairs (0,2), (1,1), (2,0) are distinct; (-1,-1) stays invalid
    assert torch.equal(combined.node_labels, torch.tensor([0, 1, 2, -1]))


def test_get_label_string_for_wl_and_primary():
    assert get_label_string({"label_type": "primary"}) == "primary"
    assert get_label_string({"label_type": "wl", "depth": 3}) == "wl_3"


class _StubGraphData:
    def __init__(self, graphs):
        self.name = "stub"
        self.nx_graphs = graphs


def test_get_label_string_for_closed_walks():
    assert get_label_string({"label_type": "closed_walks", "max_walk_length": 4}) == "closed_walks_4"
    assert get_label_string({"label_type": "closed_walks"}) == "closed_walks_6"
    assert (
        get_label_string({"label_type": "closed_walks", "min_walk_length": 3, "max_walk_length": 3, "max_labels": 10})
        == "closed_walks_3_3_10"
    )


def test_closed_walk_labeling_distinguishes_cycle_structure():
    # triangle: every node has profile ((A^2)_ii, (A^3)_ii) = (2, 2)
    # path 0-1-2: ends have (1, 0), middle has (2, 0)
    triangle = nx.cycle_graph(3)
    path = nx.path_graph(3)
    labeling = ClosedWalkNodeLabeling(_StubGraphData([triangle, path]), max_walk_length=3)

    labels = labeling.generate()

    tri_labels, path_labels = labels
    assert tri_labels[0] == tri_labels[1] == tri_labels[2]
    assert path_labels[0] == path_labels[2]
    assert path_labels[1] != path_labels[0]
    # triangle nodes and path middle both have degree 2, but only the walk
    # profile separates them (degree labeling would merge them)
    assert tri_labels[0] != path_labels[1]


def test_closed_walk_single_length_equals_diagonal_of_matrix_power():
    # min == max == 3 labels nodes by (A^3)_ii alone: 2 for triangle nodes, 0 for path nodes
    triangle = nx.cycle_graph(3)
    path = nx.path_graph(3)
    labeling = ClosedWalkNodeLabeling(_StubGraphData([triangle, path]), min_walk_length=3, max_walk_length=3)

    tri_labels, path_labels = labeling.generate()

    assert len(set(tri_labels)) == 1
    assert len(set(path_labels)) == 1
    assert tri_labels[0] != path_labels[0]


def test_save_closed_walk_labels_writes_expected_file(tmp_path):
    graph_data = _StubGraphData([nx.cycle_graph(4)])

    file = save_closed_walk_labels(graph_data, max_walk_length=4, label_path=tmp_path)

    assert file.name == "stub_labels_closed_walks_4.pt"
    assert file.exists()
    payload = torch.load(file, weights_only=True)
    assert payload["dataset_name"] == "stub"
    assert payload["label_name"] == "closed_walks_4"
    assert payload["node_labels"].shape == (4, 2)
    assert payload["label_hashes"].dtype == torch.int64
    assert payload["hash_meta"]["canonical"] is True


def test_relabel_node_labels_handles_cap_and_invalids():
    node_labels = torch.tensor([0, 0, 1, 2, -1], dtype=torch.long)

    relabeled = relabel_node_labels(node_labels, max_number_labels=2)

    assert relabeled.shape == (5, 2)
    assert relabeled[-1, 0].item() == -1
    assert relabeled[:, 1].max().item() <= 1
