from __future__ import annotations

import types

import networkx as nx
import numpy as np
import pytest


torch = pytest.importorskip("torch")

from simplegnn.datasets.utils import label_hashing
from simplegnn.datasets.utils.label_hashing import (
    HASH_SCHEMA_VERSION,
    RESERVED_CAPPED,
    RESERVED_INVALID,
    hash_vocabulary,
    stable_hash,
)
from simplegnn.datasets.utils.NodeLabels import NodeLabels
from simplegnn.datasets.utils.node_labeling import (
    combine_node_labels,
    load_labels,
    relabel_node_labels,
    save_degree_labels,
    save_index_labels,
    save_labels_to_file,
    save_trivial_labels,
)
from simplegnn.datasets.utils.node_labeling_functions import weisfeiler_lehman_node_labeling


def stub(graphs, name="stub"):
    return types.SimpleNamespace(name=name, nx_graphs=graphs)


def same_partition(a, b):
    return len(set(a)) == len(set(b)) == len(set(zip(a, b)))


# ------------------------------------------------------------- stable_hash A0
def test_stable_hash_pinned_vectors():
    # pinned int64 vectors: a refactor that silently changes the encoding
    # (and thereby invalidates every stored vocabulary) must fail here
    assert stable_hash(HASH_SCHEMA_VERSION, "degree", (), 3) == -1836586706441606698
    assert stable_hash(HASH_SCHEMA_VERSION, "degree", (), 1) == 7990179534348851488
    assert stable_hash(HASH_SCHEMA_VERSION, "wl", (2,), "init") == 4883674280670285552
    assert stable_hash(HASH_SCHEMA_VERSION, "trivial", (), 0) == -2753004365921176839
    assert stable_hash(HASH_SCHEMA_VERSION, "combined", (), (5, 7)) == 4366841019705982721
    assert stable_hash(HASH_SCHEMA_VERSION, "induced_cycles", (None, 6), "[(5, 1)]") == 23912062271163866
    assert stable_hash(HASH_SCHEMA_VERSION, "closed_walks", (2, 4), (2, 0, 6)) == 8570924683683167378
    assert stable_hash(HASH_SCHEMA_VERSION, "primary", (), 6) == 3007455640912296806


def test_stable_hash_discriminates_types_and_structure():
    assert stable_hash("a") != stable_hash(("a",))
    assert stable_hash(0) != stable_hash(False)
    assert stable_hash(1) != stable_hash(True)
    assert stable_hash(None) != stable_hash(0)
    assert stable_hash((1, 2), 3) != stable_hash(1, (2, 3))
    assert stable_hash("ab", "c") != stable_hash("a", "bc")
    # numpy ints hash like python ints
    assert stable_hash(np.int64(42)) == stable_hash(42)


def test_stable_hash_is_int64():
    for parts in [("x",), (12345678901234,), ((1, "a", None),)]:
        value = stable_hash(*parts)
        assert -2 ** 63 <= value < 2 ** 63


def test_hash_vocabulary_commits_to_kind_and_params():
    same_signature = {0: "sig"}
    a = hash_vocabulary(same_signature, "wl", (3,))
    b = hash_vocabulary(same_signature, "wl", (4,))
    c = hash_vocabulary(same_signature, "cliques", (3,))
    assert a[0] != b[0]
    assert a[0] != c[0]


def test_hash_vocabulary_detects_collisions(monkeypatch):
    monkeypatch.setattr(label_hashing, "stable_hash", lambda *parts: 12345)
    with pytest.raises(ValueError, match="collision"):
        hash_vocabulary({0: "first", 1: "second"}, "kind", ())


def test_hash_vocabulary_rejects_reserved_slots(monkeypatch):
    monkeypatch.setattr(label_hashing, "stable_hash", lambda *parts: RESERVED_CAPPED)
    with pytest.raises(ValueError, match="reserved"):
        hash_vocabulary({0: "sig"}, "kind", ())


# --------------------------------------------------- v2 file format + relabel
def test_hashes_follow_labels_through_frequency_sorting(tmp_path):
    # original ids 0,1,2 with frequencies 1,3,2 -> freq-sorted ids 2,0,1
    labels = torch.tensor([0, 1, 1, 1, 2, 2])
    hashes = np.array([100, 101, 102], dtype=np.int64)
    file = tmp_path / "labels.pt"
    save_labels_to_file(file, "db", "x", labels, max_labels=None, label_hashes=hashes,
                        hash_meta={"canonical": True, "kind": "k", "params": ()})

    loaded = load_labels(file)

    for node, original_id in enumerate([0, 1, 1, 1, 2, 2]):
        new_id = int(loaded.node_labels[node])
        assert int(loaded.label_hashes[new_id]) == 100 + original_id
    assert loaded.has_canonical_hashes
    assert loaded.hash_meta["capped"] is False
    assert loaded.hash_meta["schema"] == HASH_SCHEMA_VERSION


def test_invalid_labels_map_to_reserved_invalid(tmp_path):
    labels = torch.tensor([0, 0, 1, -1])
    hashes = np.array([100, 101], dtype=np.int64)
    file = tmp_path / "labels.pt"
    save_labels_to_file(file, "db", "x", labels, max_labels=None, label_hashes=hashes,
                        hash_meta={"canonical": True, "kind": "k", "params": ()})

    loaded = load_labels(file)

    assert int(loaded.node_labels[3]) == -1
    # the invalid pseudo-label occupies the last frequency-sorted slot
    assert int(loaded.label_hashes[-1]) == RESERVED_INVALID
    assert int(loaded.label_hashes[int(loaded.node_labels[0])]) == 100


def test_capping_collapses_merged_ids_to_reserved_capped(tmp_path):
    labels = torch.tensor([0, 0, 0, 1, 1, 2, 3])
    hashes = np.array([100, 101, 102, 103], dtype=np.int64)
    file = tmp_path / "labels.pt"
    save_labels_to_file(file, "db", "x", labels, max_labels=2, label_hashes=hashes,
                        hash_meta={"canonical": True, "kind": "k", "params": ()})

    loaded = load_labels(file)

    assert loaded.label_hashes.shape[0] == 2
    assert int(loaded.label_hashes[0]) == 100
    assert int(loaded.label_hashes[1]) == RESERVED_CAPPED
    assert loaded.hash_meta["capped"] is True


def test_v1_tuple_files_still_load(tmp_path):
    file = tmp_path / "legacy.pt"
    torch.save(("db", "legacy", relabel_node_labels(torch.tensor([0, 1, 0]), None)), file)

    loaded = load_labels(file)

    assert loaded.dataset_name == "db"
    assert loaded.label_name == "legacy"
    assert loaded.label_hashes is None
    assert loaded.hash_meta is None
    assert not loaded.has_canonical_hashes
    assert loaded.node_labels.tolist() == [0, 1, 0]


def test_v2_without_hashes_records_meta_only(tmp_path):
    file = tmp_path / "labels.pt"
    save_labels_to_file(file, "db", "x", torch.tensor([0, 1]), max_labels=None,
                        hash_meta={"canonical": False, "kind": "index", "params": ()})

    loaded = load_labels(file)

    assert loaded.label_hashes is None
    assert loaded.hash_meta["canonical"] is False
    assert not loaded.has_canonical_hashes


# ---------------------------------------------------------------- combined A1
def test_combined_label_hashes():
    hashes_a = torch.tensor([100, 101], dtype=torch.int64)
    hashes_b = torch.tensor([200, 201], dtype=torch.int64)
    meta = {"schema": HASH_SCHEMA_VERSION, "canonical": True, "kind": "k", "params": (), "capped": False}
    a = NodeLabels("db", "a", torch.tensor([[0, 0], [1, 1], [0, 0], [-1, -1]]), hashes_a, dict(meta))
    b = NodeLabels("db", "b", torch.tensor([[1, 1], [0, 0], [1, 1], [-1, -1]]), hashes_b, dict(meta))

    combined = combine_node_labels([a, b])

    assert combined.label_hashes is not None
    assert combined.has_canonical_hashes
    # node 0 carries pair (0, 1); node 1 carries pair (1, 0)
    pair_01 = stable_hash(HASH_SCHEMA_VERSION, "combined", (), (100, 201))
    pair_10 = stable_hash(HASH_SCHEMA_VERSION, "combined", (), (101, 200))
    assert int(combined.label_hashes[int(combined.node_labels[0])]) == pair_01
    assert int(combined.label_hashes[int(combined.node_labels[1])]) == pair_10
    assert int(combined.node_labels[3]) == -1
    assert RESERVED_INVALID in combined.label_hashes.tolist()


def test_combined_labels_without_hashes_stay_hashless():
    a = NodeLabels("db", "a", torch.tensor([[0, 0], [1, 1]]))
    b = NodeLabels("db", "b", torch.tensor([[1, 1], [0, 0]]))

    combined = combine_node_labels([a, b])

    assert combined.label_hashes is None
    assert not combined.has_canonical_hashes


def test_combined_canonical_only_if_both_canonical():
    hashes = torch.tensor([100], dtype=torch.int64)
    canonical_meta = {"schema": HASH_SCHEMA_VERSION, "canonical": True, "kind": "k", "params": (), "capped": False}
    relative_meta = {"schema": HASH_SCHEMA_VERSION, "canonical": False, "kind": "k", "params": (), "capped": False}
    a = NodeLabels("db", "a", torch.tensor([[0, 0]]), hashes, canonical_meta)
    b = NodeLabels("db", "b", torch.tensor([[0, 0]]), hashes, relative_meta)

    combined = combine_node_labels([a, b])

    assert combined.label_hashes is not None
    assert not combined.has_canonical_hashes


# ------------------------------------------------------------ WL canonical A3
def test_wl_hashes_partition_matches_nx_subgraph_hashes():
    graphs = [nx.cycle_graph(5), nx.path_graph(4), nx.lollipop_graph(4, 2)]
    labels, _, _, signatures = weisfeiler_lehman_node_labeling(graphs, depth=3, return_hashes=True)

    node_hashes = [signatures[label] for graph_labels in labels for label in graph_labels]
    union = nx.disjoint_union_all(graphs)
    nx_hashes = nx.weisfeiler_lehman_subgraph_hashes(union, iterations=3)
    nx_final = [nx_hashes[node][-1] for node in union.nodes()]

    assert same_partition(node_hashes, nx_final)


def test_wl_hashes_transfer_between_disjoint_datasets():
    shared = nx.lollipop_graph(4, 2)
    dataset_a = [nx.cycle_graph(5), nx.path_graph(4), shared]
    dataset_b = [shared, nx.star_graph(6)]

    labels_a, _, _, sig_a = weisfeiler_lehman_node_labeling(dataset_a, depth=3, return_hashes=True)
    labels_b, _, _, sig_b = weisfeiler_lehman_node_labeling(dataset_b, depth=3, return_hashes=True)

    hashes_a = [sig_a[label] for label in labels_a[2]]
    hashes_b = [sig_b[label] for label in labels_b[0]]
    assert hashes_a == hashes_b


def test_wl_hashes_do_not_stall_after_early_stabilization():
    # a cycle stabilizes after one round; deeper depth params must still
    # produce depth-specific hashes (the hash chain keeps iterating)
    graphs = [nx.cycle_graph(5)]
    _, _, _, sig_2 = weisfeiler_lehman_node_labeling(graphs, depth=2, return_hashes=True)
    _, _, _, sig_4 = weisfeiler_lehman_node_labeling(graphs, depth=4, return_hashes=True)

    assert set(sig_2.values()) != set(sig_4.values())

    # and the same depth computed on differently-composed datasets agrees
    labels_pair, _, _, sig_pair = weisfeiler_lehman_node_labeling([nx.cycle_graph(5), nx.path_graph(7)], depth=4, return_hashes=True)
    assert {sig_pair[label] for label in labels_pair[0]} == set(sig_4.values())


def test_wl_labeled_hashes_transfer_across_base_compactions():
    # the same structural role must hash identically even when the two
    # datasets compacted their base labels to different (permuted) ids
    graph = nx.path_graph(3)
    hash_x = stable_hash(HASH_SCHEMA_VERSION, "primary", (), 1)
    hash_y = stable_hash(HASH_SCHEMA_VERSION, "primary", (), 2)
    meta = {"schema": HASH_SCHEMA_VERSION, "canonical": True, "kind": "primary", "params": (), "capped": False}

    base_a = NodeLabels("a", "base", torch.tensor([[0, 0], [1, 1], [0, 0]]),
                        torch.tensor([hash_x, hash_y], dtype=torch.int64), dict(meta))
    base_b = NodeLabels("b", "base", torch.tensor([[1, 1], [0, 0], [1, 1]]),
                        torch.tensor([hash_y, hash_x], dtype=torch.int64), dict(meta))

    _, _, _, sig_a = weisfeiler_lehman_node_labeling([graph], depth=2, labeled=True,
                                                     base_labels={"labels": base_a}, return_hashes=True)
    _, _, _, sig_b = weisfeiler_lehman_node_labeling([graph.copy()], depth=2, labeled=True,
                                                     base_labels={"labels": base_b}, return_hashes=True)

    assert set(sig_a.values()) == set(sig_b.values())


def test_wl_labeled_without_base_hashes_returns_none():
    base = NodeLabels("a", "base", torch.tensor([[0, 0], [1, 1], [0, 0]]))
    labels, unique, db_unique, signatures = weisfeiler_lehman_node_labeling(
        [nx.path_graph(3)], depth=2, labeled=True, base_labels={"labels": base}, return_hashes=True)

    assert signatures is None
    assert len(labels[0]) == 3


def test_wl_labeling_without_return_hashes_keeps_old_signature():
    result = weisfeiler_lehman_node_labeling([nx.path_graph(3)], depth=2)
    assert len(result) == 3


# ------------------------------------------------------------------ producers
def test_save_degree_labels_writes_canonical_hashes(tmp_path):
    file = save_degree_labels(stub([nx.star_graph(3)]), label_path=tmp_path)

    loaded = load_labels(file)

    assert loaded.has_canonical_hashes
    assert loaded.hash_meta["kind"] == "degree"
    center, leaf = int(loaded.node_labels[0]), int(loaded.node_labels[1])
    assert int(loaded.label_hashes[center]) == stable_hash(HASH_SCHEMA_VERSION, "degree", (), 3)
    assert int(loaded.label_hashes[leaf]) == stable_hash(HASH_SCHEMA_VERSION, "degree", (), 1)


def test_save_trivial_labels_writes_canonical_hashes(tmp_path):
    graph_data = types.SimpleNamespace(name="stub", data=types.SimpleNamespace(x=torch.zeros(4, 1)))
    file = save_trivial_labels(graph_data, label_path=tmp_path)

    loaded = load_labels(file)

    assert loaded.has_canonical_hashes
    assert int(loaded.label_hashes[0]) == stable_hash(HASH_SCHEMA_VERSION, "trivial", (), 0)


def test_save_index_labels_excluded_from_hashing(tmp_path):
    file = save_index_labels(stub([nx.path_graph(3)]), label_path=tmp_path)

    loaded = load_labels(file)

    assert loaded.label_hashes is None
    assert loaded.hash_meta["canonical"] is False
    assert not loaded.has_canonical_hashes


def test_degree_hashes_transfer_between_disjoint_datasets(tmp_path):
    file_a = save_degree_labels(stub([nx.star_graph(3), nx.path_graph(5)], name="a"), label_path=tmp_path)
    file_b = save_degree_labels(stub([nx.cycle_graph(4), nx.star_graph(3)], name="b"), label_path=tmp_path)

    loaded_a = load_labels(file_a)
    loaded_b = load_labels(file_b)

    # star centers (degree 3) hash identically although the compacted ids differ
    hash_a = int(loaded_a.label_hashes[int(loaded_a.node_labels[0])])
    hash_b = int(loaded_b.label_hashes[int(loaded_b.node_labels[4])])
    assert hash_a == hash_b == stable_hash(HASH_SCHEMA_VERSION, "degree", (), 3)
