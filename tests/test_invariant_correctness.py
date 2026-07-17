"""Correctness tests for the node label invariants and pairwise properties.

These tests pin down the semantic guarantees every labeling must satisfy:
- labels are permutation/isomorphism invariant,
- labels computed for a graph do not depend on which other graphs are in the
  dataset (no cross-graph contamination),
- the filename written by each save_* function matches what
  get_label_string() (and hence the label loading in the framework) expects,
- the produced label values are the mathematically correct ones on small
  hand-checked graphs.

Known bugs are kept as strict xfail tests so they are documented and flip to
XPASS when fixed.
"""
from __future__ import annotations

import gzip
import pickle
import types

import networkx as nx
import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from simplegnn.datasets.utils.node_labeling import (
    BetweennessCentralityNodeLabeling,
    LabeledDegreeNodeLabeling,
    get_label_string,
    save_clique_labels,
    save_cycle_labels,
    save_degree_labels,
    save_in_circle_labels,
    save_labeled_degree_labels,
    save_subgraph_labels,
    save_wl_labels,
)
from simplegnn.datasets.utils.node_labeling_functions import weisfeiler_lehman_node_labeling
from simplegnn.models.ShareGNN.preprocessing.properties import (
    write_distance_edge_properties,
    write_distance_properties,
)


# --------------------------------------------------------------------- helpers
def stub(graphs, name="stub"):
    return types.SimpleNamespace(name=name, nx_graphs=graphs)


def load_label_column(file):
    _, _, node_labels = torch.load(file, weights_only=False)
    return node_labels[:, 0].tolist()


def per_graph(labels_flat, graphs):
    out, offset = [], 0
    for g in graphs:
        out.append(labels_flat[offset:offset + g.number_of_nodes()])
        offset += g.number_of_nodes()
    return out


def same_partition(a, b):
    """Two label sequences induce the same grouping of positions."""
    return len(set(a)) == len(set(b)) == len(set(zip(a, b)))


def attach_primary_labels(graph, labels):
    for node, label in zip(graph.nodes(), labels):
        graph.nodes[node]["primary_node_labels"] = label
    return graph


def permute(graph, perm):
    """Isomorphic copy with nodes inserted in id order (0..n-1).

    Label positions follow graph.nodes() insertion order, so a fair
    isomorphism-invariance test must keep insertion order = node id, as
    to_networkx does for pipeline graphs.
    """
    out = nx.Graph()
    out.add_nodes_from(range(graph.number_of_nodes()))
    out.add_edges_from((perm[u], perm[v]) for u, v in graph.edges())
    return out


# ---------------------------------------------------------------------- degree
def test_degree_labels_are_correct_degrees(tmp_path):
    g = nx.star_graph(3)  # center degree 3, leaves degree 1
    file = save_degree_labels(stub([g]), label_path=tmp_path)
    assert load_label_column(file) == [3, 1, 1, 1]


def test_degree_labels_match_bincount_on_random_graphs(tmp_path):
    rng = np.random.default_rng(0)
    graphs = [nx.fast_gnp_random_graph(20, 0.2, seed=int(rng.integers(1 << 31))) for _ in range(20)]
    file = save_degree_labels(stub(graphs), label_path=tmp_path)
    expected = [int(d) for g in graphs for _, d in sorted(g.degree())]
    assert load_label_column(file) == expected


# -------------------------------------------------------------------------- WL
def test_wl_labels_are_isomorphism_invariant():
    g = nx.fast_gnp_random_graph(20, 0.2, seed=7)
    perm = {i: (i * 7 + 3) % 20 for i in range(20)}
    g_perm = permute(g, perm)
    labels, _, _ = weisfeiler_lehman_node_labeling([g, g_perm], depth=3)
    for v in g.nodes():
        assert labels[0][v] == labels[1][perm[v]]


def test_wl_labels_do_not_depend_on_other_graphs_in_dataset():
    g1 = nx.path_graph(6)
    g2 = nx.star_graph(5)
    labels_pair, _, _ = weisfeiler_lehman_node_labeling([g1, g2], depth=2)
    labels_alone, _, _ = weisfeiler_lehman_node_labeling([g1], depth=2)
    assert same_partition(labels_pair[0], labels_alone[0])


def test_wl_labels_refine_degree_partition():
    g = nx.lollipop_graph(4, 3)
    labels, _, _ = weisfeiler_lehman_node_labeling([g], depth=3)
    degree_of_label = {}
    for v in g.nodes():
        label = labels[0][v]
        assert degree_of_label.setdefault(label, g.degree(v)) == g.degree(v)


def test_wl_filename_matches_get_label_string(tmp_path):
    file = save_wl_labels(stub([nx.path_graph(4)]), depth=3, label_path=tmp_path)
    assert file.name == f"stub_labels_{get_label_string({'label_type': 'wl', 'depth': 3})}.pt"


# -------------------------------------------------------------- labeled degree
def test_save_labeled_degree_labels_is_permutation_invariant(tmp_path):
    g = attach_primary_labels(nx.star_graph(2), [1, 2, 3])
    g_swapped = nx.Graph()
    g_swapped.add_node(0, primary_node_labels=1)
    g_swapped.add_node(1, primary_node_labels=3)
    g_swapped.add_node(2, primary_node_labels=2)
    g_swapped.add_edges_from([(0, 2), (0, 1)])  # same star, neighbors swapped
    file = save_labeled_degree_labels(stub([g, g_swapped]), label_path=tmp_path)
    labels = per_graph(load_label_column(file), [g, g_swapped])
    assert labels[0][0] == labels[1][0]          # both centers: (1, {2, 3})
    assert set(labels[0][1:]) == set(labels[1][1:])


def test_save_labeled_degree_labels_no_cross_graph_contamination(tmp_path):
    g1 = attach_primary_labels(nx.path_graph(3), [1, 1, 1])
    g2 = attach_primary_labels(nx.star_graph(2), [2, 2, 2])
    file_pair = save_labeled_degree_labels(stub([g1, g2], name="pair"), label_path=tmp_path)
    file_alone = save_labeled_degree_labels(stub([g1], name="alone"), label_path=tmp_path)
    labels_pair = per_graph(load_label_column(file_pair), [g1, g2])[0]
    labels_alone = load_label_column(file_alone)
    assert same_partition(labels_pair, labels_alone)


def test_labeled_degree_class_no_cross_graph_contamination():
    g1 = attach_primary_labels(nx.path_graph(3), [1, 1, 1])
    g2 = attach_primary_labels(nx.star_graph(2), [2, 2, 2])
    labels_pair = LabeledDegreeNodeLabeling(stub([g1, g2])).generate()
    labels_alone = LabeledDegreeNodeLabeling(stub([g1])).generate()
    assert same_partition(labels_pair[0], labels_alone[0])


# ---------------------------------------------------------------------- cycles
def test_cycle_labels_separate_cycle_from_path_nodes(tmp_path):
    cycle, path = nx.cycle_graph(5), nx.path_graph(5)
    file = save_cycle_labels(stub([cycle, path] * 5), max_cycle_length=6,
                             cycle_type="induced", label_path=tmp_path)
    labels = per_graph(load_label_column(file), [cycle, path] * 5)
    assert len(set(labels[0])) == 1              # all C5 nodes alike
    assert len(set(labels[1])) == 1              # all path nodes alike (no cycle)
    assert labels[0][0] != labels[1][0]


def test_cycle_labels_are_isomorphism_invariant(tmp_path):
    g = nx.lollipop_graph(4, 2)                  # K4 with a 2-tail
    perm = {i: (i * 5 + 2) % 6 for i in range(6)}
    g_perm = permute(g, perm)
    file = save_cycle_labels(stub([g, g_perm] * 5), max_cycle_length=6,
                             cycle_type="simple", label_path=tmp_path)
    labels = per_graph(load_label_column(file), [g, g_perm] * 5)
    for v in g.nodes():
        assert labels[0][v] == labels[1][perm[v]]


def test_save_cycle_labels_works_for_small_datasets(tmp_path):
    save_cycle_labels(stub([nx.cycle_graph(5)] * 5), max_cycle_length=6, label_path=tmp_path)


def test_cycle_filename_without_max_matches_get_label_string(tmp_path):
    file = save_cycle_labels(stub([nx.cycle_graph(5)] * 10), max_cycle_length=None,
                             label_path=tmp_path)
    expected = get_label_string({"label_type": "simple_cycles"})
    assert file.name == f"stub_labels_{expected}.pt"


def test_cycle_labels_do_not_depend_on_edge_numbering(tmp_path):
    # both graphs: node 0 is the hub of one triangle and one square
    ga = nx.Graph()
    ga.add_edges_from([(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 5), (5, 0)])
    gb = nx.Graph()
    gb.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (4, 5), (5, 0)])
    graphs = [ga, gb] + [nx.path_graph(3) for _ in range(8)]
    file = save_cycle_labels(stub(graphs), max_cycle_length=6, label_path=tmp_path)
    labels = per_graph(load_label_column(file), graphs)
    assert labels[0][0] == labels[1][0]


# -------------------------------------------------------------------- in-circle
def test_in_circle_labels_flag_cycle_membership(tmp_path):
    g = nx.cycle_graph(5)
    g.add_edge(0, 5)                             # tail node 5 outside the cycle
    file = save_in_circle_labels(stub([g]), length_bound=6, label_path=tmp_path)
    assert load_label_column(file) == [1, 1, 1, 1, 1, 0]


# ---------------------------------------------------------------------- cliques
def test_clique_labels_separate_triangle_from_isolated(tmp_path):
    g = nx.complete_graph(3)
    g.add_node(3)
    file = save_clique_labels(stub([g]), max_clique=6, label_path=tmp_path)
    labels = load_label_column(file)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] != labels[0]


def test_clique_labels_do_not_depend_on_node_numbering(tmp_path):
    import itertools
    base = [(0, 1), (1, 2), (2, 0), (0, 3)]      # triangle + pendant edge at hub 0
    graphs, hubs = [], []
    for perm in itertools.permutations(range(4)):
        g = nx.Graph()
        g.add_nodes_from(range(4))
        g.add_edges_from((perm[u], perm[v]) for u, v in base)
        graphs.append(g)
        hubs.append(perm[0])
    file = save_clique_labels(stub(graphs), max_clique=6, label_path=tmp_path)
    labels = per_graph(load_label_column(file), graphs)
    assert len({labels[i][hubs[i]] for i in range(len(graphs))}) == 1


# --------------------------------------------------------------------- subgraph
def test_subgraph_labels_mark_pattern_nodes(tmp_path):
    g = nx.complete_graph(3)                     # one triangle
    g.add_edge(2, 3)                             # plus a pendant node
    file = save_subgraph_labels(stub([g]), subgraphs=[nx.complete_graph(3)],
                                subgraph_id=0, label_path=tmp_path)
    labels = load_label_column(file)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] != labels[0]


# ------------------------------------------------------------------ betweenness
def test_betweenness_labels_order_star_center_above_leaves():
    g = nx.star_graph(4)
    labels = BetweennessCentralityNodeLabeling(stub([g]), num_bins=2).generate()
    center, leaves = labels[0][0], labels[0][1:]
    assert len(set(leaves)) == 1
    assert center > leaves[0]


def test_betweenness_labels_constant_on_vertex_transitive_graph():
    labels = BetweennessCentralityNodeLabeling(stub([nx.cycle_graph(6)]), num_bins=4).generate()
    assert len(set(labels[0])) == 1


# ------------------------------------------------------------------- wl_labeled
def test_wl_labeled_primary_base_filename_matches_get_label_string():
    label_dict = {"label_type": "wl_labeled", "depth": 3,
                  "base_labels": {"label_type": "primary"}}
    # save_wl_labeled_labels names the file 'wl_labeled_3' for a primary base
    assert get_label_string(label_dict) == "wl_labeled_3"


# ---------------------------------------------------------- distance properties
def test_write_distance_properties_pairs_and_slices(tmp_path):
    # graph 0: path 0-1-2 (distances 0,1,2), graph 1: two isolated nodes (only 0)
    g1, g2 = nx.path_graph(3), nx.empty_graph(2)
    graph_data = types.SimpleNamespace(
        name="stub", nx_graphs=[g1, g2],
        slices={"x": torch.tensor([0, 3, 5])},
    )
    write_distance_properties(graph_data, out_path=tmp_path)
    with gzip.open(tmp_path / "stub_properties_distances.pt", "rb") as f:
        valid, pairs, slices = pickle.loads(f.read())

    assert valid == {0, 1, 2}
    as_set = lambda t: set(map(tuple, t.tolist()))
    assert as_set(pairs[0]) == {(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)}
    assert as_set(pairs[1]) == {(0, 1), (1, 0), (1, 2), (2, 1)}
    assert as_set(pairs[2]) == {(0, 2), (2, 0)}
    # slices are cumulative pair counts per graph: length num_graphs + 1
    assert slices[0].tolist() == [0, 3, 5]
    assert slices[1].tolist() == [0, 4, 4]
    assert slices[2].tolist() == [0, 2, 2]


# ---------------------------------------------- edge-label distance properties
def edge_labeled(edges_with_labels, num_nodes):
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    for u, v, label in edges_with_labels:
        g.add_edge(u, v, primary_edge_labels=label)
    return g


def load_edge_property_keys(tmp_path, name):
    with gzip.open(tmp_path / f"{name}_properties_edge_label_distances.pt", "rb") as f:
        valid, pairs, _ = pickle.loads(f.read())
    return valid, pairs


def test_edge_label_distance_keys_on_labeled_path(tmp_path):
    # path 0-1-2, edge labels 0 and 1: keys are (distance, #paths, label counts)
    g = edge_labeled([(0, 1, 0), (1, 2, 1)], 3)
    graph_data = types.SimpleNamespace(name="p3", nx_graphs=[g],
                                       slices={"x": torch.tensor([0, 3])})
    write_distance_edge_properties(graph_data, out_path=tmp_path)
    valid, pairs = load_edge_property_keys(tmp_path, "p3")
    assert valid == {(1, 1, (1,)), (1, 1, (0, 1)), (2, 1, (1, 1))}
    assert all(len(v) == 2 for v in pairs.values())     # each unordered pair twice


def test_edge_label_distance_aggregates_all_shortest_paths(tmp_path):
    # C4, all edge labels 0: opposite corners have 2 shortest paths of length 2,
    # so label 0 occurs 4 times over all shortest paths
    g = edge_labeled([(0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 0, 0)], 4)
    graph_data = types.SimpleNamespace(name="c4", nx_graphs=[g],
                                       slices={"x": torch.tensor([0, 4])})
    write_distance_edge_properties(graph_data, out_path=tmp_path)
    valid, pairs = load_edge_property_keys(tmp_path, "c4")
    assert valid == {(1, 1, (1,)), (2, 2, (4,))}
    assert len(pairs[(2, 2, (4,))]) == 4                # both diagonals, both directions


def test_edge_label_distance_rejects_or_handles_negative_labels(tmp_path):
    # negative labels cannot be encoded in the positional label-count tuple,
    # so they are rejected with a clear error instead of an IndexError
    g = edge_labeled([(0, 1, -1)], 2)
    graph_data = types.SimpleNamespace(name="neg", nx_graphs=[g],
                                       slices={"x": torch.tensor([0, 2])})
    with pytest.raises(ValueError, match="non-negative"):
        write_distance_edge_properties(graph_data, out_path=tmp_path)
