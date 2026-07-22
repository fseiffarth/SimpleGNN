"""Tests for the hash-keyed transfer engine (spec 18 B0-B3).

Unit level: the vectorized hash join (full/partial/disjoint overlap, reserved
hash exclusion), the B0 overlap measurement on synthetic vocabularies, and the
sidecar loader error paths.

Integration level (real MUTAG via the shared fixtures): slot->key export of
the invariant layers checked against a brute-force re-derivation, the sidecar
save/load roundtrip, and apply_transfer end-to-end (same-dataset full match,
head reinit, on_missing: zero, lazy slot-key reconstruction, freezing).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simplegnn.datasets.utils.NodeLabels import NodeLabels
from simplegnn.datasets.utils.label_hashing import (
    HASH_SCHEMA_VERSION,
    RESERVED_CAPPED,
    RESERVED_INVALID,
)
from simplegnn.framework.utils.transfer import (
    apply_transfer,
    apply_transfer_strategy,
    hash_join,
    load_transfer_sidecar,
    measure_label_overlap,
    measure_pair_overlap,
    resolve_source_checkpoint,
    sidecar_path_for,
)


# --------------------------------------------------------------------- helpers
def make_node_labels(name, ids, hashes, canonical=True):
    """NodeLabels with node ids `ids` (col 0 == col 1) and id->hash array."""
    ids_t = torch.tensor(ids, dtype=torch.int64)
    return NodeLabels(
        name, "test_labels", torch.stack([ids_t, ids_t], dim=1),
        label_hashes=torch.tensor(hashes, dtype=torch.int64),
        hash_meta={"schema": HASH_SCHEMA_VERSION, "canonical": canonical,
                   "kind": "test", "params": (), "capped": False},
    )


# -------------------------------------------------------------------- hash_join
def test_hash_join_full_overlap():
    src = (torch.tensor([1, 2, 3]), torch.tensor([10, 20, 30]))
    tgt = (torch.tensor([3, 1, 2]), torch.tensor([30, 10, 20]))
    src_rows, tgt_rows = hash_join(src, tgt)
    assert tgt_rows.tolist() == [0, 1, 2]
    assert src_rows.tolist() == [2, 0, 1]


def test_hash_join_partial_and_disjoint_overlap():
    src = (torch.tensor([1, 2]), torch.tensor([10, 20]))
    partial = (torch.tensor([2, 5]), torch.tensor([20, 50]))
    src_rows, tgt_rows = hash_join(src, partial)
    assert tgt_rows.tolist() == [0]
    assert src_rows.tolist() == [1]

    disjoint = (torch.tensor([7, 8]), torch.tensor([70, 80]))
    src_rows, tgt_rows = hash_join(src, disjoint)
    assert src_rows.numel() == 0 and tgt_rows.numel() == 0


def test_hash_join_excludes_reserved_hashes():
    # identical rows on both sides, but every row contains a reserved value in
    # one column -> no match may ever be produced
    src = (torch.tensor([RESERVED_INVALID, 5, RESERVED_CAPPED]),
           torch.tensor([1, RESERVED_CAPPED, 2]))
    tgt = (torch.tensor([RESERVED_INVALID, 5, RESERVED_CAPPED]),
           torch.tensor([1, RESERVED_CAPPED, 2]))
    src_rows, tgt_rows = hash_join(src, tgt)
    assert src_rows.numel() == 0 and tgt_rows.numel() == 0

    # the same non-reserved row still matches
    src = (torch.tensor([RESERVED_INVALID, 5]), torch.tensor([1, 6]))
    tgt = (torch.tensor([5, RESERVED_CAPPED]), torch.tensor([6, 2]))
    src_rows, tgt_rows = hash_join(src, tgt)
    assert tgt_rows.tolist() == [0]
    assert src_rows.tolist() == [1]


def test_hash_join_single_column():
    src_rows, tgt_rows = hash_join((torch.tensor([10, 20, 30]),),
                                   (torch.tensor([30, 40]),))
    assert tgt_rows.tolist() == [0]
    assert src_rows.tolist() == [2]


# ------------------------------------------------------------------ B0: overlap
def test_measure_label_overlap_counts_unique_and_occurrences():
    source = make_node_labels("SRC", [0, 0, 1, 2], [10, 20, 30])
    target = make_node_labels("TGT", [0, 1, 1, 1], [10, 99])
    report = measure_label_overlap(source, target)
    assert report.source_unique == 3
    assert report.target_unique == 2
    assert report.shared_unique == 1
    assert report.unique_overlap == pytest.approx(0.5)
    # target node hashes: [10, 99, 99, 99] -> only the first is covered
    assert report.target_occurrences == 4
    assert report.covered_occurrences == 1
    assert report.weighted_coverage == pytest.approx(0.25)
    assert report.source_canonical and report.target_canonical


def test_measure_label_overlap_reserved_slots_never_match():
    source = make_node_labels("SRC", [0, 1], [10, RESERVED_CAPPED])
    target = make_node_labels("TGT", [0, 1, -1], [10, RESERVED_CAPPED])
    report = measure_label_overlap(source, target)
    # the capped bucket and the -1 node are excluded from vocab and coverage
    assert report.source_unique == 1 and report.target_unique == 1
    assert report.shared_unique == 1
    assert report.covered_occurrences == 1
    assert report.target_reserved_occurrences == 2  # capped node + invalid node


def test_measure_label_overlap_requires_hash_vocabulary():
    ids = torch.tensor([[0, 0], [1, 1]], dtype=torch.int64)
    v1 = NodeLabels("OLD", "test_labels", ids)
    target = make_node_labels("TGT", [0], [10])
    with pytest.raises(ValueError, match="preprocessing"):
        measure_label_overlap(v1, target)


def test_measure_pair_overlap_triple_mass():
    source = make_node_labels("SRC", [0, 1, 2], [10, 20, 30])
    target = make_node_labels("TGT", [0, 1], [20, 10])
    source_props = {1: torch.tensor([[0, 1], [1, 0], [1, 2]])}
    # target triples at value 1: (20,10) twice and (10,20) once
    target_props = {1: torch.tensor([[0, 1], [0, 1], [1, 0]]),
                    2: torch.tensor([[0, 1]])}  # value 2 absent in source -> ignored
    report = measure_pair_overlap(source, source_props, target, target_props)
    assert report.kind == "pairs"
    # source triples: (10,20), (20,10), (20,30); target unique: (20,10), (10,20)
    assert report.target_unique == 2
    assert report.shared_unique == 2
    assert report.target_occurrences == 3
    assert report.covered_occurrences == 3
    assert report.weighted_coverage == pytest.approx(1.0)


def test_measure_pair_overlap_disjoint_and_reserved():
    source = make_node_labels("SRC", [0, 1], [10, 20])
    target = make_node_labels("TGT", [0, 1, 2], [77, 88, RESERVED_INVALID])
    source_props = {1: torch.tensor([[0, 1]])}
    target_props = {1: torch.tensor([[0, 1], [1, 2]])}
    report = measure_pair_overlap(source, source_props, target, target_props)
    assert report.covered_occurrences == 0
    assert report.target_occurrences == 2
    assert report.target_reserved_occurrences == 1


# ------------------------------------------------------- sidecar + checkpoints
def test_load_transfer_sidecar_missing_file_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="save_transfer_keys"):
        load_transfer_sidecar(tmp_path / "model_x.keys.pt")


def test_sidecar_path_for():
    path = sidecar_path_for(Path("/a/b/model_Configuration_000000_run_0_val_step_0.pt"))
    assert path.name == "model_Configuration_000000_run_0_val_step_0.keys.pt"


def test_resolve_source_checkpoint(tmp_path):
    models = tmp_path / "DB" / "Models"
    models.mkdir(parents=True)
    grid = models / "model_Configuration_000000_run_0_val_step_0.pt"
    grid.touch()
    (models / "model_Configuration_000000_run_0_val_step_0.keys.pt").touch()
    assert resolve_source_checkpoint(tmp_path, "DB", "best") == grid

    best = models / "model_Best_Configuration_000000_run_0_val_step_0.pt"
    best.touch()
    assert resolve_source_checkpoint(tmp_path, "DB", "best") == best
    assert resolve_source_checkpoint(
        tmp_path, "DB", {"config_id": 0, "run_id": 0, "validation_id": 0}) == best

    with pytest.raises(FileNotFoundError):
        resolve_source_checkpoint(tmp_path, "DB", {"config_id": 7})
    with pytest.raises(FileNotFoundError):
        resolve_source_checkpoint(tmp_path, "MISSING", "best")
    with pytest.raises(ValueError, match="select"):
        resolve_source_checkpoint(tmp_path, "DB", "bogus")


def _write_run_csv(results_dir, run_id, accuracies, losses):
    """Minimal per-epoch result CSV in the framework's ';'-separated format."""
    path = results_dir / f"DB_Best_Configuration_000000_Results_run_id_{run_id}_validation_step_0.csv"
    rows = ["Epoch;ValidationAccuracy;ValidationLoss"]
    rows += [f"{epoch};{accuracy};{loss}"
             for epoch, (accuracy, loss) in enumerate(zip(accuracies, losses))]
    path.write_text("\n".join(rows) + "\n")


def test_resolve_source_checkpoint_best_validation(tmp_path):
    models = tmp_path / "DB" / "Models"
    results = tmp_path / "DB" / "Results"
    models.mkdir(parents=True)
    results.mkdir(parents=True)
    for run_id in range(3):
        (models / f"model_Best_Configuration_000000_run_{run_id}_val_step_0.pt").touch()

    # regression-style run: accuracy stays 0, the loss decides -> run 1 wins
    _write_run_csv(results, 0, [0.0, 0.0], [0.5, 0.4])
    _write_run_csv(results, 1, [0.0, 0.0], [0.5, 0.2])
    _write_run_csv(results, 2, [0.0, 0.0], [0.9, 0.7])
    assert resolve_source_checkpoint(tmp_path, "DB", "best_validation").name == \
        "model_Best_Configuration_000000_run_1_val_step_0.pt"
    # plain 'best' stays positional (run 0), which is what it always did
    assert resolve_source_checkpoint(tmp_path, "DB", "best").name == \
        "model_Best_Configuration_000000_run_0_val_step_0.pt"

    # classification-style run: accuracy varies and decides -> run 2 wins
    _write_run_csv(results, 0, [50.0, 60.0], [0.5, 0.4])
    _write_run_csv(results, 1, [50.0, 61.0], [0.5, 0.2])
    _write_run_csv(results, 2, [50.0, 72.0], [0.9, 0.7])
    assert resolve_source_checkpoint(tmp_path, "DB", "best_validation").name == \
        "model_Best_Configuration_000000_run_2_val_step_0.pt"


def test_resolve_source_checkpoint_best_validation_without_results(tmp_path):
    models = tmp_path / "DB" / "Models"
    models.mkdir(parents=True)
    (models / "model_Best_Configuration_000000_run_0_val_step_0.pt").touch()
    with pytest.raises(FileNotFoundError, match="best_validation"):
        resolve_source_checkpoint(tmp_path, "DB", "best_validation")


def test_apply_transfer_rejects_bad_enums():
    module = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match="match"):
        apply_transfer(module, {}, {"layers": {}}, {"invariant_transfer": {"match": "bogus"}})
    with pytest.raises(ValueError, match="on_missing"):
        apply_transfer(module, {}, {"layers": {}}, {"invariant_transfer": {"on_missing": "bogus"}})
    with pytest.raises(ValueError, match="reinit"):
        apply_transfer(module, {}, {"layers": {}}, {"head": {"reinit": "bogus"}})
    with pytest.raises(ValueError, match="strategy"):
        apply_transfer_strategy(module, {"strategy": "bogus"}, apply_transfer(module, {}, {"layers": {}}, {}))


# --------------------------------------------------- integration on real MUTAG
@pytest.fixture
def mutag_net(share_gnn_setup):
    from simplegnn.models.model import GraphModel

    graph_data, para = share_gnn_setup
    net = GraphModel(graph_data=graph_data, para=para, seed=42, device="cpu")
    return graph_data, para, net


def _conv(net):
    return next(l for l in net.net_layers if "Message Passing" in l.name)


def _pool(net):
    return next(l for l in net.net_layers if "Aggregation" in l.name)


@pytest.mark.integration
def test_conv_slot_keys_match_brute_force(mutag_net):
    """Exported (src_hash, tgt_hash) slot keys must equal an independent
    re-derivation from the raw label/property tensors."""
    graph_data, para, net = mutag_net
    conv = _conv(net)
    export = conv.export_weight_keys()
    assert export["layer_type"] == "invariant_based_convolution"

    total_slots = 0
    for head in export["heads"]:
        source_nl = graph_data.node_labels[head["source_label"]]
        target_nl = graph_data.node_labels[head["target_label"]]
        prop = graph_data.properties[head["property"]]
        for key in head["keys"]:
            real_key = next(k for k in prop.properties if str(k) == key["property_key"])
            pairs = prop.properties[real_key]
            s_ids = source_nl.node_labels[pairs[:, 0]].cpu()
            t_ids = target_nl.node_labels[pairs[:, 1]].cpu()
            valid = (s_ids >= 0) & (t_ids >= 0)
            pair_set = sorted(set(zip(s_ids[valid].tolist(), t_ids[valid].tolist())))
            assert key["num_weights"] == len(pair_set)
            expected_src = [int(source_nl.label_hashes[s]) for s, _ in pair_set]
            expected_tgt = [int(target_nl.label_hashes[t]) for _, t in pair_set]
            assert key["src_hash"].tolist() == expected_src
            assert key["tgt_hash"].tolist() == expected_tgt
            total_slots += key["num_weights"] * head["num_replicas"]
    # the key blocks must tile Param_W exactly
    assert total_slots == export["param_w_size"] == conv.Param_W.numel()

    # bias slots: hash of slot k must equal the hash of any node mapped to k
    for bias_head in export["bias"]:
        bias_nl = graph_data.node_labels[bias_head["bias_label"]]
        slot = conv._bias_slot[bias_head["bias_label"]]
        inverse = getattr(conv, f"_bias_idx_{slot}").cpu().long()
        node_hashes = bias_nl.label_hashes[bias_nl.node_labels.cpu().long()]
        assert torch.equal(bias_head["bias_hash"][inverse], node_hashes)


@pytest.mark.integration
def test_pooling_slot_keys_match_per_node(mutag_net):
    graph_data, para, net = mutag_net
    pool = _pool(net)
    export = pool.export_weight_keys()
    assert export["layer_type"] == "invariant_based_aggregation"
    for head in export["heads"]:
        nl = graph_data.node_labels[head["label"]]
        inverse = getattr(pool, f"_agg_idx_{head['head_id']}").cpu().long()
        node_hashes = nl.label_hashes[nl.node_labels.cpu().long()]
        assert torch.equal(head["label_hash"][inverse], node_hashes)
        assert head["n_labels"] == nl.num_unique_node_labels
        assert int(head["counts"].sum()) == inverse.numel()


@pytest.mark.integration
def test_lazy_slot_key_reconstruction_matches_eager(mutag_net):
    """_ensure_slot_keys must produce identical keys with and without the
    material retained during _build_distributions, including the full
    recompute fallback for old cache formats (uniques not stored)."""
    graph_data, para, net = mutag_net
    conv = _conv(net)
    eager = conv.export_weight_keys()

    # drop the retained material -> reconstruction from the indices cache
    conv._slot_keys = None
    conv._slot_key_uniques = {}
    from_cache = conv.export_weight_keys()

    # simulate pre-B1 caches: loader raises -> full recompute from the labels
    conv._slot_keys = None
    conv._slot_key_uniques = {}

    def _raise(*args, **kwargs):
        raise FileNotFoundError("simulated old cache")

    conv._load_cached_indices = _raise
    conv._save_cached_indices = lambda *args, **kwargs: None  # keep shared caches untouched
    recomputed = conv.export_weight_keys()

    for other in (from_cache, recomputed):
        for head_a, head_b in zip(eager["heads"], other["heads"]):
            for key_a, key_b in zip(head_a["keys"], head_b["keys"]):
                assert key_a["param_offset"] == key_b["param_offset"]
                assert key_a["num_weights"] == key_b["num_weights"]
                assert torch.equal(key_a["src_hash"], key_b["src_hash"])
                assert torch.equal(key_a["tgt_hash"], key_b["tgt_hash"])
                assert torch.equal(key_a["counts"], key_b["counts"])


@pytest.mark.integration
def test_sidecar_save_load_roundtrip(mutag_net, tmp_path):
    graph_data, para, net = mutag_net
    payload = net.export_transfer_keys()
    assert payload["schema"] == 1
    assert payload["layers"], "invariant layers must contribute keys"
    assert set(payload["summary"]) == set(payload["layers"])

    checkpoint = tmp_path / "model_Configuration_000000_run_0_val_step_0.pt"
    torch.save(net.state_dict(), checkpoint)
    torch.save(payload, sidecar_path_for(checkpoint))

    loaded = load_transfer_sidecar(sidecar_path_for(checkpoint))
    assert set(loaded["layers"]) == set(payload["layers"])
    for prefix, keys in payload["layers"].items():
        loaded_keys = loaded["layers"][prefix]
        assert loaded_keys["layer_type"] == keys["layer_type"]
        for head_a, head_b in zip(keys["heads"], loaded_keys["heads"]):
            for field in ("src_hash", "tgt_hash", "label_hash"):
                if field in head_a:
                    assert torch.equal(head_a[field], head_b[field])
            for key_a, key_b in zip(head_a.get("keys", []), head_b.get("keys", [])):
                assert torch.equal(key_a["src_hash"], key_b["src_hash"])
                assert torch.equal(key_a["tgt_hash"], key_b["tgt_hash"])


@pytest.mark.integration
def test_apply_transfer_same_dataset_full_match(share_gnn_setup):
    """Same dataset, independently seeded nets: every invariant slot and every
    standard layer must transfer bit-exactly, while the head keeps its fresh
    initialization."""
    from simplegnn.models.model import GraphModel

    graph_data, para = share_gnn_setup
    net_a = GraphModel(graph_data=graph_data, para=para, seed=1, device="cpu")
    with torch.no_grad():
        for param in net_a.parameters():
            torch.nn.init.normal_(param, std=0.5)
    source_sd = {k: v.clone() for k, v in net_a.state_dict().items()}
    source_keys = net_a.export_transfer_keys()

    net_b = GraphModel(graph_data=graph_data, para=para, seed=2, device="cpu")
    linear_ids = [i for i, l in enumerate(net_b.net_layers)
                  if type(l).__name__ == "LinearLayer"]
    head_id = linear_ids[-1]
    head_before = {k: v.clone() for k, v in net_b.net_layers[head_id].state_dict().items()}

    # MUTAG primary labels carry hashes but are flagged non-canonical
    cfg = {"invariant_transfer": {"allow_non_canonical": True}}
    report = apply_transfer(net_b, source_sd, source_keys, cfg)

    conv_a, conv_b = _conv(net_a), _conv(net_b)
    torch.testing.assert_close(conv_b.Param_W, conv_a.Param_W, rtol=0, atol=0)
    torch.testing.assert_close(conv_b.Param_b, conv_a.Param_b, rtol=0, atol=0)
    pool_a, pool_b = _pool(net_a), _pool(net_b)
    torch.testing.assert_close(pool_b.Param_W, pool_a.Param_W, rtol=0, atol=0)
    torch.testing.assert_close(pool_b.Param_b, pool_a.Param_b, rtol=0, atol=0)
    # first linear copied, head kept fresh
    first_linear = net_b.net_layers[linear_ids[0]]
    for name, value in first_linear.state_dict().items():
        torch.testing.assert_close(
            value, net_a.net_layers[linear_ids[0]].state_dict()[name], rtol=0, atol=0)
    for name, value in net_b.net_layers[head_id].state_dict().items():
        assert torch.equal(value, head_before[name]), f"head param {name} was overwritten"

    by_layer = {entry["layer"]: entry for entry in report.layers}
    for prefix in (f"net_layers.{i}" for i, l in enumerate(net_b.net_layers)
                   if hasattr(l, "export_weight_keys")):
        assert by_layer[prefix]["matched"] == by_layer[prefix]["total"] > 0
    assert by_layer[f"net_layers.{head_id}"]["action"] == "head_reinit"


@pytest.mark.integration
def test_apply_transfer_disjoint_source_on_missing_zero(share_gnn_setup):
    """A source whose hashes share nothing with the target must transfer no
    invariant slot; on_missing: zero zeroes the fresh init instead."""
    from simplegnn.models.model import GraphModel

    graph_data, para = share_gnn_setup
    net_a = GraphModel(graph_data=graph_data, para=para, seed=1, device="cpu")
    with torch.no_grad():
        for param in net_a.parameters():
            torch.nn.init.normal_(param, std=0.5)
    source_sd = {k: v.clone() for k, v in net_a.state_dict().items()}
    source_keys = net_a.export_transfer_keys()
    # shift every hash so no key can match (values stay non-reserved)
    for keys in source_keys["layers"].values():
        for head in keys["heads"]:
            for key in head.get("keys", []):
                key["src_hash"] = key["src_hash"] + 1
                key["tgt_hash"] = key["tgt_hash"] + 1
            if "label_hash" in head:
                head["label_hash"] = head["label_hash"] + 1
        for bias_head in keys.get("bias", []):
            bias_head["bias_hash"] = bias_head["bias_hash"] + 1

    net_b = GraphModel(graph_data=graph_data, para=para, seed=2, device="cpu")
    with torch.no_grad():
        for param in net_b.parameters():
            torch.nn.init.normal_(param, std=0.5)
    cfg = {"invariant_transfer": {"allow_non_canonical": True, "on_missing": "zero",
                                  "min_overlap_warn": 0.1}}
    report = apply_transfer(net_b, source_sd, source_keys, cfg)

    conv_b = _conv(net_b)
    assert torch.count_nonzero(conv_b.Param_W) == 0
    assert torch.count_nonzero(conv_b.Param_b) == 0
    assert any("min_overlap_warn" in w for w in report.warnings)
    # pooling Param_b is config-shaped and transfers positionally even here
    torch.testing.assert_close(_pool(net_b).Param_b, _pool(net_a).Param_b, rtol=0, atol=0)


@pytest.mark.integration
def test_non_canonical_heads_skipped_by_default(share_gnn_setup):
    """MUTAG primary labels are non-canonical: without allow_non_canonical the
    invariant tables must keep their fresh initialization."""
    from simplegnn.models.model import GraphModel

    graph_data, para = share_gnn_setup
    net_a = GraphModel(graph_data=graph_data, para=para, seed=1, device="cpu")
    with torch.no_grad():
        for param in net_a.parameters():
            torch.nn.init.normal_(param, std=0.5)
    source_sd = {k: v.clone() for k, v in net_a.state_dict().items()}
    source_keys = net_a.export_transfer_keys()

    net_b = GraphModel(graph_data=graph_data, para=para, seed=2, device="cpu")
    conv_before = _conv(net_b).Param_W.detach().clone()
    report = apply_transfer(net_b, source_sd, source_keys, {})
    assert torch.equal(_conv(net_b).Param_W, conv_before)
    assert any("not canonical" in w for w in report.warnings)


@pytest.mark.integration
def test_transfer_strategy_linear_probe_and_freeze(share_gnn_setup):
    from simplegnn.models.model import GraphModel

    graph_data, para = share_gnn_setup
    net_a = GraphModel(graph_data=graph_data, para=para, seed=1, device="cpu")
    source_sd = {k: v.clone() for k, v in net_a.state_dict().items()}
    source_keys = net_a.export_transfer_keys()
    net_b = GraphModel(graph_data=graph_data, para=para, seed=2, device="cpu")

    cfg = {"strategy": "linear_probe",
           "invariant_transfer": {"allow_non_canonical": True}}
    report = apply_transfer(net_b, source_sd, source_keys, cfg)
    frozen = apply_transfer_strategy(net_b, cfg, report)
    assert frozen
    assert not _conv(net_b).Param_W.requires_grad
    linear_ids = [i for i, l in enumerate(net_b.net_layers)
                  if type(l).__name__ == "LinearLayer"]
    # the re-initialized head stays trainable
    for param in net_b.net_layers[linear_ids[-1]].parameters():
        assert param.requires_grad
    assert report.frozen_parameters == frozen

    # explicit freeze globs on a fresh net
    net_c = GraphModel(graph_data=graph_data, para=para, seed=3, device="cpu")
    cfg = {"freeze": [f"net_layers.{linear_ids[0]}"],
           "invariant_transfer": {"allow_non_canonical": True}}
    report = apply_transfer(net_c, source_sd, source_keys, cfg)
    apply_transfer_strategy(net_c, cfg, report)
    for param in net_c.net_layers[linear_ids[0]].parameters():
        assert not param.requires_grad
    assert _conv(net_c).Param_W.requires_grad


@pytest.mark.integration
def test_apply_transfer_random_init_freezes_backbone_untouched(share_gnn_setup):
    """random_init: true must leave every backbone layer at its own fresh
    init (no weight copied from the source), while strategy: linear_probe
    still freezes the whole backbone and leaves only the reinitialized head
    trainable -- the untrained-backbone baseline."""
    from simplegnn.models.model import GraphModel

    graph_data, para = share_gnn_setup
    net_a = GraphModel(graph_data=graph_data, para=para, seed=1, device="cpu")
    with torch.no_grad():
        for param in net_a.parameters():
            torch.nn.init.normal_(param, std=0.5)
    source_sd = {k: v.clone() for k, v in net_a.state_dict().items()}
    source_keys = net_a.export_transfer_keys()

    net_b = GraphModel(graph_data=graph_data, para=para, seed=2, device="cpu")
    conv_before = _conv(net_b).Param_W.detach().clone()
    pool_before = _pool(net_b).Param_W.detach().clone()
    linear_ids = [i for i, l in enumerate(net_b.net_layers)
                  if type(l).__name__ == "LinearLayer"]
    first_linear_before = {k: v.clone()
                           for k, v in net_b.net_layers[linear_ids[0]].state_dict().items()}

    cfg = {"strategy": "linear_probe", "random_init": True,
           "invariant_transfer": {"allow_non_canonical": True}}
    report = apply_transfer(net_b, source_sd, source_keys, cfg)
    frozen = apply_transfer_strategy(net_b, cfg, report)

    # nothing was copied from net_a: every layer kept its own fresh init
    torch.testing.assert_close(_conv(net_b).Param_W, conv_before, rtol=0, atol=0)
    torch.testing.assert_close(_pool(net_b).Param_W, pool_before, rtol=0, atol=0)
    for name, value in net_b.net_layers[linear_ids[0]].state_dict().items():
        torch.testing.assert_close(value, first_linear_before[name], rtol=0, atol=0)
    assert all(entry["matched"] == 0 for entry in report.layers)
    assert all(entry["action"] in ("random_init", "head_reinit") for entry in report.layers)

    # random_init still freezes the whole backbone, only the head trains
    assert frozen
    assert not _conv(net_b).Param_W.requires_grad
    assert not _pool(net_b).Param_W.requires_grad
    for param in net_b.net_layers[linear_ids[-1]].parameters():
        assert param.requires_grad


def test_non_canonical_property_keys_gate_head():
    from simplegnn.framework.utils.transfer import TransferReport, _head_usable

    head = {'head_id': 0, 'has_hashes': True, 'canonical': True,
            'property': 'edge_label_distances_cutoff_3', 'property_canonical': False}
    report = TransferReport()
    assert not _head_usable(head, False, 'net_layers.0', report, 'Target')
    assert any('dataset-relative keys' in w for w in report.warnings)
    # allow_non_canonical opts back in
    assert _head_usable(head, True, 'net_layers.0', report, 'Target')

    # sidecars written before the flag existed derive it from the description
    legacy = {'head_id': 0, 'has_hashes': True, 'canonical': True,
              'property': 'edge_label_distances'}
    assert not _head_usable(legacy, False, 'net_layers.0', TransferReport(), 'Source')
    plain = {'head_id': 0, 'has_hashes': True, 'canonical': True,
             'property': 'distances'}
    assert _head_usable(plain, False, 'net_layers.0', TransferReport(), 'Source')
    # bias/pooling heads carry no property entry and are unaffected
    bias = {'head_id': 0, 'has_hashes': True, 'canonical': True, 'bias_label': 'wl_0'}
    assert _head_usable(bias, False, 'net_layers.0', TransferReport(), 'Source')
