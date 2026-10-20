from __future__ import annotations

import networkx as nx
import pytest


torch = pytest.importorskip("torch")

from simplegnn.datasets.utils.NodeLabels import NodeLabels
from simplegnn.datasets.utils.node_labeling import (
    ClosedWalkEdgeLabeledNodeLabeling,
    ClosedWalkNodeLabeling,
    ClosedWalkWeightedNodeLabeling,
    combine_node_labels,
    get_label_string,
    relabel_node_labels,
    save_closed_walk_edges_labels,
    save_closed_walk_labels,
    save_closed_walk_weighted_labels,
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


def make_labeled_cycles():
    # both 4-cycles carry two edges of each label, but only in the second one are
    # the equal labels adjacent -- the unlabeled walk counts cannot see that
    alternating = nx.cycle_graph(4)
    nx.set_edge_attributes(
        alternating, {(0, 1): 0, (1, 2): 1, (2, 3): 0, (3, 0): 1}, 'primary_edge_labels')
    grouped = nx.cycle_graph(4)
    nx.set_edge_attributes(
        grouped, {(0, 1): 0, (1, 2): 0, (2, 3): 1, (3, 0): 1}, 'primary_edge_labels')
    return [alternating, grouped]


def test_get_label_string_for_edge_labeled_closed_walks():
    assert get_label_string({"label_type": "closed_walks_edges", "max_walk_length": 4}) == "closed_walks_edges_4"
    assert get_label_string({"label_type": "closed_walks_weighted"}) == "closed_walks_weighted_6"
    assert (
        get_label_string({"label_type": "closed_walks_edges", "min_walk_length": 3,
                          "max_walk_length": 5, "max_labels": 10})
        == "closed_walks_edges_3_5_10"
    )


def test_edge_labeled_closed_walks_separate_what_unlabeled_counts_merge():
    graphs = make_labeled_cycles()
    stub = _StubGraphData(graphs)

    unlabeled = [label for labels in ClosedWalkNodeLabeling(stub, min_walk_length=2, max_walk_length=2).generate()
                 for label in labels]
    refined = [label for labels in
               ClosedWalkEdgeLabeledNodeLabeling(stub, min_walk_length=2, max_walk_length=2).generate()
               for label in labels]

    # every node has degree 2, so (A^2)_ii == 2 everywhere
    assert len(set(unlabeled)) == 1
    # the length-2 walks split by the label of the edge they traverse: nodes
    # incident to one edge of each label, to two 0-edges, and to two 1-edges
    assert len(set(refined)) == 3
    alternating_labels, grouped_labels = ClosedWalkEdgeLabeledNodeLabeling(
        stub, min_walk_length=2, max_walk_length=2).generate()
    assert len(set(alternating_labels)) == 1
    assert grouped_labels[1] != grouped_labels[3] != grouped_labels[0]
    # the mixed nodes of the grouped cycle match the alternating cycle's nodes
    assert grouped_labels[0] == grouped_labels[2] == alternating_labels[0]


def test_edge_labeled_closed_walks_refine_the_unlabeled_labels():
    graphs = make_labeled_cycles() + [nx.path_graph(3)]
    nx.set_edge_attributes(graphs[-1], {(0, 1): 0, (1, 2): 1}, 'primary_edge_labels')
    stub = _StubGraphData(graphs)

    unlabeled = [label for labels in ClosedWalkNodeLabeling(stub, max_walk_length=4).generate() for label in labels]
    refined = [label for labels in ClosedWalkEdgeLabeledNodeLabeling(stub, max_walk_length=4).generate()
               for label in labels]

    # equal refined label => equal unlabeled label (the refinement never merges)
    by_refined = {}
    for refined_label, unlabeled_label in zip(refined, unlabeled):
        assert by_refined.setdefault(refined_label, unlabeled_label) == unlabeled_label
    assert len(set(refined)) > len(set(unlabeled))


def test_weighted_closed_walks_separate_the_same_cycles():
    stub = _StubGraphData(make_labeled_cycles())

    alternating_labels, grouped_labels = ClosedWalkWeightedNodeLabeling(
        stub, min_walk_length=2, max_walk_length=2).generate()

    # primes 2 and 3: mixed node 2*2 + 3*3 = 13, two 0-edges 8, two 1-edges 18
    assert len(set(alternating_labels)) == 1
    assert grouped_labels[0] == grouped_labels[2] == alternating_labels[0]
    assert len({grouped_labels[1], grouped_labels[3], grouped_labels[0]}) == 3


def test_weighted_closed_walks_assign_primes_by_sorted_edge_label():
    stub = _StubGraphData(make_labeled_cycles())
    labeling = ClosedWalkWeightedNodeLabeling(stub, max_walk_length=2)

    labeling.generate()

    assert labeling.label_weights == {0: 2, 1: 3}


def test_weighted_closed_walks_raise_instead_of_rounding():
    graph = nx.complete_graph(10)
    nx.set_edge_attributes(graph, 1, 'primary_edge_labels')
    labeling = ClosedWalkWeightedNodeLabeling(_StubGraphData([graph]), max_walk_length=20)

    with pytest.raises(ValueError, match="exact float64 range"):
        labeling.generate()


def test_edge_labeled_closed_walks_require_edge_labels():
    labeling = ClosedWalkEdgeLabeledNodeLabeling(_StubGraphData([nx.cycle_graph(4)]), max_walk_length=2)

    with pytest.raises(ValueError, match="primary_edge_labels"):
        labeling.generate()


def test_save_edge_labeled_closed_walk_labels_write_expected_files(tmp_path):
    graph_data = _StubGraphData(make_labeled_cycles())

    edges_file = save_closed_walk_edges_labels(graph_data, max_walk_length=4, label_path=tmp_path)
    weighted_file = save_closed_walk_weighted_labels(graph_data, max_walk_length=4, label_path=tmp_path)

    assert edges_file.name == "stub_labels_closed_walks_edges_4.pt"
    assert weighted_file.name == "stub_labels_closed_walks_weighted_4.pt"
    for file, kind in ((edges_file, "closed_walks_edges"), (weighted_file, "closed_walks_weighted")):
        payload = torch.load(file, weights_only=True)
        assert payload["dataset_name"] == "stub"
        assert payload["label_name"] == f"{kind}_4"
        assert payload["node_labels"].shape == (8, 2)
        assert payload["label_hashes"].dtype == torch.int64
        assert payload["hash_meta"]["canonical"] is True
        assert payload["hash_meta"]["kind"] == kind


def test_relabel_node_labels_handles_cap_and_invalids():
    node_labels = torch.tensor([0, 0, 1, 2, -1], dtype=torch.long)

    relabeled = relabel_node_labels(node_labels, max_number_labels=2)

    assert relabeled.shape == (5, 2)
    assert relabeled[-1, 0].item() == -1
    assert relabeled[:, 1].max().item() <= 1
