"""Tests for spec 18 B4-B6: static/runtime transfer-config validation and the
CI-fast cross-dataset transfer integration test.

Unit level: the `transfer:` block schema checks of configuration_checks.py
(good config passes, unknown keys and bad enum values raise, the
non-constant-input-features warning) and the run-start existence checks
(missing sidecar, legacy v1 target label files).

Integration level: two synthetic datasets sampled INDEPENDENTLY from one
generator (custom_benchmarks snowflakes) are preprocessed separately; a model
pretrained on A is transferred to a freshly built model on B. Asserts a high
matched fraction for the canonical WL vocabulary (the case dataset-relative
label ids cannot deliver), bit-equality of every matched weight before any
finetune step, and that the finetune run itself trains through the
`transfer:` block wire-up in initialize_model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simplegnn.datasets.utils.NodeLabels import NodeLabels
from simplegnn.datasets.utils.label_hashing import HASH_SCHEMA_VERSION, RESERVED_INVALID
from simplegnn.framework.utils.configuration_checks import (
    check_hyperparameter_configuration_file,
    check_transfer_configuration,
    check_transfer_runtime_requirements,
)


def valid_transfer_config():
    return {
        "source": {"results_path": "results/pretrain/", "dataset": "NCI1", "select": "best"},
        "strategy": "finetune",
        "head": {"reinit": "always"},
        "invariant_transfer": {"match": "hashes", "on_missing": "reinit",
                               "allow_non_canonical": False, "min_overlap_warn": 0.10},
        "freeze": [],
    }


# ------------------------------------------------- B4: static schema validation
def test_valid_transfer_config_passes():
    check_transfer_configuration(valid_transfer_config())


def test_minimal_transfer_config_passes():
    # everything except source.results_path/dataset is optional
    check_transfer_configuration({"source": {"results_path": "r/", "dataset": "DB"}})


def test_transfer_config_list_is_validated_per_entry():
    check_transfer_configuration([valid_transfer_config(), valid_transfer_config()])
    with pytest.raises(ValueError, match="strategy"):
        check_transfer_configuration([valid_transfer_config(),
                                      {**valid_transfer_config(), "strategy": "bogus"}])


@pytest.mark.parametrize("patch, match", [
    ({"bogus_key": 1}, r"Unknown key\(s\) \['bogus_key'\]"),
    ({"strategy": "fine_tune"}, "transfer.strategy"),
    ({"head": {"reinit": "sometimes"}}, "head.reinit"),
    ({"head": {"bogus": 1}}, "transfer.head"),
    ({"invariant_transfer": {"match": "ids"}}, "invariant_transfer.match"),
    ({"invariant_transfer": {"on_missing": "raise"}}, "invariant_transfer.on_missing"),
    ({"invariant_transfer": {"allow_non_canonical": "yes"}}, "allow_non_canonical"),
    ({"invariant_transfer": {"min_overlap_warn": 1.5}}, "min_overlap_warn"),
    ({"invariant_transfer": {"min_overlap_warn": "high"}}, "min_overlap_warn"),
    ({"invariant_transfer": {"bogus": 1}}, "invariant_transfer"),
    ({"freeze": "net_layers.0*"}, "freeze"),
    ({"freeze": [0]}, "freeze"),
    ({"source": {"results_path": "r/"}}, "source.dataset"),
    ({"source": {"dataset": "DB"}}, "source.results_path"),
    ({"source": {"results_path": "r/", "dataset": "DB", "bogus": 1}}, "transfer.source"),
    ({"source": {"results_path": "r/", "dataset": "DB", "select": "latest"}}, "select"),
    ({"source": {"results_path": "r/", "dataset": "DB",
                 "select": {"config_id": "zero"}}}, "select.config_id"),
    ({"source": {"results_path": "r/", "dataset": "DB",
                 "select": {"bogus": 0}}}, "source.select"),
    ({"source": None}, "transfer.source"),
    ({"random_init": "yes"}, "transfer.random_init"),
])
def test_bad_transfer_config_raises(patch, match):
    cfg = valid_transfer_config()
    cfg.update(patch)
    with pytest.raises(ValueError, match=match):
        check_transfer_configuration(cfg)


def test_transfer_config_must_be_mapping():
    with pytest.raises(ValueError, match="mapping"):
        check_transfer_configuration("finetune")


def test_transfer_config_random_init_passes():
    cfg = valid_transfer_config()
    cfg["strategy"] = "linear_probe"
    cfg["random_init"] = True
    check_transfer_configuration(cfg)


def test_hyperparameter_check_validates_transfer_block(minimal_hyper_config):
    cfg = dict(minimal_hyper_config)
    cfg["input_features"] = {"name": "constant", "value": 1.0}
    cfg["transfer"] = valid_transfer_config()
    check_hyperparameter_configuration_file(cfg)

    cfg["transfer"] = {**valid_transfer_config(), "strategy": "bogus"}
    with pytest.raises(ValueError, match="strategy"):
        check_hyperparameter_configuration_file(cfg)


def test_hyperparameter_check_without_transfer_ignores_new_checks(minimal_hyper_config):
    # zero behavior change when no transfer block is present
    cfg = dict(minimal_hyper_config)
    check_hyperparameter_configuration_file(cfg)


def test_transfer_with_non_constant_input_features_warns(minimal_hyper_config, capsys):
    cfg = dict(minimal_hyper_config)
    cfg["input_features"] = {"name": "node_labels", "transformation": "one_hot"}
    cfg["transfer"] = valid_transfer_config()
    check_hyperparameter_configuration_file(cfg)
    assert "dataset-dependent widths" in capsys.readouterr().out

    cfg["input_features"] = [{"name": "constant", "value": 1.0}]
    check_hyperparameter_configuration_file(cfg)
    assert "dataset-dependent widths" not in capsys.readouterr().out


# ------------------------------------------- B4: run-start existence checks
def _fake_source_run(tmp_path, dataset="SRC", with_sidecar=True):
    models = tmp_path / dataset / "Models"
    models.mkdir(parents=True)
    checkpoint = models / "model_Configuration_000000_run_0_val_step_0.pt"
    checkpoint.touch()
    if with_sidecar:
        checkpoint.with_suffix(".keys.pt").touch()
    return checkpoint


def _transfer_config_for(tmp_path, dataset="SRC", **invariant):
    cfg = {"source": {"results_path": str(tmp_path), "dataset": dataset}}
    if invariant:
        cfg["invariant_transfer"] = invariant
    return cfg


def test_runtime_requirements_resolve_checkpoint(tmp_path):
    checkpoint = _fake_source_run(tmp_path)
    assert check_transfer_runtime_requirements(_transfer_config_for(tmp_path)) == checkpoint


def test_runtime_requirements_missing_sidecar_raises(tmp_path):
    _fake_source_run(tmp_path, with_sidecar=False)
    with pytest.raises(FileNotFoundError, match="save_transfer_keys"):
        check_transfer_runtime_requirements(_transfer_config_for(tmp_path))


def test_runtime_requirements_match_none_needs_no_sidecar(tmp_path):
    checkpoint = _fake_source_run(tmp_path, with_sidecar=False)
    cfg = _transfer_config_for(tmp_path, match="none")
    assert check_transfer_runtime_requirements(cfg) == checkpoint


def test_runtime_requirements_v1_target_label_file_raises(tmp_path):
    # a loaded target label file without a hash vocabulary (legacy v1) must
    # fail with the delete-labels-and-rerun-preprocessing() migration hint
    _fake_source_run(tmp_path)
    ids = torch.tensor([[0, 0], [1, 1]], dtype=torch.int64)
    graph_data = SimpleNamespace(node_labels={"wl_2": NodeLabels("OLD", "wl_2", ids)})
    with pytest.raises(ValueError, match="preprocessing"):
        check_transfer_runtime_requirements(_transfer_config_for(tmp_path), graph_data)

    # v2 label files with hashes pass
    graph_data = SimpleNamespace(node_labels={"wl_2": NodeLabels(
        "NEW", "wl_2", ids, label_hashes=torch.tensor([5, 6], dtype=torch.int64),
        hash_meta={"schema": HASH_SCHEMA_VERSION, "canonical": True,
                   "kind": "wl", "params": (2,), "capped": False})})
    check_transfer_runtime_requirements(_transfer_config_for(tmp_path), graph_data)


def test_runtime_requirements_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="pretraining"):
        check_transfer_runtime_requirements(_transfer_config_for(tmp_path))


# ---------------------------------------- B6: cross-dataset integration (CI-fast)
TEST_MODELS_YML = """
models:
  - - { layer_type: invariant_based_convolution,
        bias: True,
        activation: torch.nn.Identity(),
        heads: [
          { num: 1,
            bias: True,
            labels: {
              head: { label_type: wl, depth: 2 },
              tail: { label_type: wl, depth: 2 },
              bias: { label_type: wl, depth: 0 },
            },
            properties: { name: distances, values: [ 1, 2 ] },
          },
        ],
    }
    - { layer_type: linear, out_features: 8, bias: True, activation: torch.nn.LeakyReLU() }
    - { layer_type: layer_norm }
    - { layer_type: invariant_based_aggregation,
        activation: torch.nn.Identity(),
        bias: True,
        heads: [
          { num: 2, bias: True,
            labels: { label_type: closed_walks, min_walk_length: 2, max_walk_length: 4 } },
        ],
    }
    - { layer_type: layer_norm }
    - { layer_type: linear, out_features: 4, bias: True, activation: torch.nn.Identity() }
    - { layer_type: reshape }
"""

TEST_PARAMETERS = {
    "device": "cpu",
    "mode": "experiments",
    "precision": "double",
    "optimizer": ["Adam"],
    "loss": ["CrossEntropyLoss"],
    "batch_size": [8],
    "learning_rate": [0.01],
    "epochs": [1],
    "early_stopping": {"enabled": False, "patience": 25},
    "training_data_sampling": {"type": "default"},
    "input_features": {"name": "constant", "value": 1.0},
    "save_best_model": True,
    "save_transfer_keys": True,
}

NUM_GRAPHS = 32  # snowflakes: (largest - smallest + 1) * flakes_per_size


def _write_synthetic_experiment(root: Path, name: str, seed: int, models_yml: Path,
                                extra_params: dict = None) -> Path:
    """Main config for one independently generated + preprocessed snowflakes
    dataset (all paths private to `root/name`)."""
    base = root / name
    for sub in ("data", "labels", "properties", "results"):
        (base / sub).mkdir(parents=True)

    split_file = base / f"{name}_splits.json"
    split_file.write_text(json.dumps([{
        "test": list(range(0, 6)),
        "model_selection": [{"train": list(range(6, 26)),
                             "validation": list(range(26, NUM_GRAPHS))}],
    }]))

    params_file = base / "parameters.yml"
    params = dict(TEST_PARAMETERS)
    params.update(extra_params or {})
    params_file.write_text(yaml.safe_dump(params, sort_keys=False))

    main_file = base / "main.yml"
    main_file.write_text(yaml.safe_dump({"datasets": [{
        "name": name,
        "source": "generate_from_function",
        "generate_function": "snowflakes",
        "generate_function_args": {"smallest_snowflake": 3, "largest_snowflake": 6,
                                   "flakes_per_size": 8, "seed": seed,
                                   "generation_type": "binary"},
        "task": "graph_classification",
        "paths": {
            "data": str(base / "data"),
            "labels": str(base / "labels"),
            "properties": str(base / "properties"),
            "results": str(base / "results"),
            "models": str(models_yml),
            "hyperparameters": str(params_file),
            "splits": str(split_file),
        },
    }]}, sort_keys=False))
    return main_file


def _graph_data_and_para(experiment):
    """Mirror of conftest.share_gnn_setup_factory for an already preprocessed
    FrameworkMain (build a real GraphModel without the grid-search loop)."""
    from simplegnn.framework.core import preprocess_graph_data
    from simplegnn.framework.run_configuration import get_run_configs
    from simplegnn.framework.utils.parameters import Parameters
    from simplegnn.framework.utils.preprocessing import load_preprocessed_data_and_parameters

    dataset_key = next(iter(experiment.network_configurations))
    configuration = experiment.network_configurations[dataset_key][0]
    graph_data = preprocess_graph_data(configuration)
    run_config = get_run_configs(configuration)[0]
    para = Parameters()
    load_preprocessed_data_and_parameters(
        config_id=0, run_id=0, validation_id=0, validation_folds=1,
        graph_data=graph_data, run_config=run_config, para=para)
    return graph_data, para


def _layer_exports(net):
    """{state_dict_prefix: (layer, export_weight_keys())} of the invariant layers."""
    return {f"net_layers.{i}": (layer, layer.export_weight_keys())
            for i, layer in enumerate(net.net_layers)
            if hasattr(layer, "export_weight_keys")}


@pytest.mark.integration
def test_cross_dataset_hash_transfer_end_to_end(tmp_path):
    from simplegnn.framework.core import FrameworkMain
    from simplegnn.framework.utils.transfer import (
        apply_transfer, load_transfer_sidecar, measure_label_overlap,
        resolve_source_checkpoint, sidecar_path_for)
    from simplegnn.models.model import GraphModel

    models_yml = tmp_path / "models_ShareGNN.yml"
    models_yml.write_text(TEST_MODELS_YML)

    # ---- pretrain on dataset A (independent sample, seed 1) -----------------
    main_a = _write_synthetic_experiment(tmp_path, "SnowA", seed=11, models_yml=models_yml)
    exp_a = FrameworkMain(main_a)
    exp_a.preprocessing(num_threads=1)
    exp_a.run_configurations(num_threads=1)

    results_a = tmp_path / "SnowA" / "results"
    checkpoint = resolve_source_checkpoint(results_a, "SnowA", "best")
    sidecar = sidecar_path_for(checkpoint)
    assert checkpoint.is_file()
    assert sidecar.is_file(), "save_transfer_keys must produce the .keys.pt sidecar"

    # ---- dataset B: sampled from the SAME generator, preprocessed INDEPENDENTLY
    main_b = _write_synthetic_experiment(
        tmp_path, "SnowB", seed=22, models_yml=models_yml,
        extra_params={"transfer": {
            "source": {"results_path": str(results_a), "dataset": "SnowA", "select": "best"},
            "strategy": "finetune",
            "head": {"reinit": "always"},
            "invariant_transfer": {"match": "hashes", "on_missing": "reinit",
                                   "allow_non_canonical": False, "min_overlap_warn": 0.05},
        }})
    exp_b = FrameworkMain(main_b)
    exp_b.preprocessing(num_threads=1)

    # (a) canonical WL vocabularies of the two independent samples overlap highly
    overlap = measure_label_overlap(
        tmp_path / "SnowA" / "labels" / "SnowA" / "SnowA_labels_wl_2.pt",
        tmp_path / "SnowB" / "labels" / "SnowB" / "SnowB_labels_wl_2.pt")
    assert overlap.source_canonical and overlap.target_canonical
    assert overlap.unique_overlap > 0.8, f"WL unique overlap only {overlap.unique_overlap:.1%}"
    assert overlap.weighted_coverage > 0.9, f"WL coverage only {overlap.weighted_coverage:.1%}"

    # (b) transfer into a freshly built target net; matched slots bit-equal
    #     BEFORE any finetune step
    source_sd = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    source_keys = load_transfer_sidecar(sidecar)
    graph_data_b, para_b = _graph_data_and_para(exp_b)
    net_b = GraphModel(graph_data=graph_data_b, para=para_b, seed=7, device="cpu")
    report = apply_transfer(net_b, source_sd, source_keys, {})
    target_sd = net_b.state_dict()

    checked_slots = 0
    for prefix, (_, export) in _layer_exports(net_b).items():
        source_layer = source_keys["layers"][prefix]
        weight = target_sd[f"{prefix}.Param_W"]
        source_weight = source_sd[f"{prefix}.Param_W"]
        source_heads = {h["head_id"]: h for h in source_layer["heads"]}
        for head in export["heads"]:
            s_head = source_heads[head["head_id"]]
            assert head.get("canonical", False) and s_head.get("canonical", False)
            if export["layer_type"] == "invariant_based_convolution":
                for t_key in head["keys"]:
                    s_key = next(k for k in s_head["keys"]
                                 if k["property_key"] == t_key["property_key"])
                    s_rows = {(int(a), int(b)): i for i, (a, b) in
                              enumerate(zip(s_key["src_hash"], s_key["tgt_hash"]))
                              if int(a) != RESERVED_INVALID and int(b) != RESERVED_INVALID}
                    matched_rows = total_rows = 0
                    for i, (a, b) in enumerate(zip(t_key["src_hash"], t_key["tgt_hash"])):
                        if int(a) == RESERVED_INVALID or int(b) == RESERVED_INVALID:
                            continue
                        total_rows += 1
                        row = s_rows.get((int(a), int(b)))
                        if row is None:
                            continue
                        matched_rows += 1
                        assert torch.equal(
                            weight[t_key["param_offset"] + i],
                            source_weight[s_key["param_offset"] + row]), \
                            "matched invariant weight is not bit-equal after transfer"
                    checked_slots += matched_rows
                    # the crux of hash keying: independently preprocessed WL
                    # vocabularies must still line up almost completely
                    assert matched_rows / total_rows > 0.8, (
                        f"{prefix} {head['source_label']} @ {t_key['property_key']}: "
                        f"only {matched_rows}/{total_rows} slots matched")
            else:  # aggregation table: one hash per label row, `num` replicas
                s_rows = {int(h): i for i, h in enumerate(s_head["label_hash"])
                          if int(h) != RESERVED_INVALID}
                matched_rows = total_rows = 0
                for i, h in enumerate(head["label_hash"]):
                    if int(h) == RESERVED_INVALID:
                        continue
                    total_rows += 1
                    row = s_rows.get(int(h))
                    if row is None:
                        continue
                    matched_rows += 1
                    for replica in range(min(head["num_replicas"], s_head["num_replicas"])):
                        assert torch.equal(
                            weight[head["weight_base"] + replica * head["n_labels"] + i],
                            source_weight[s_head["weight_base"] + replica * s_head["n_labels"] + row])
                checked_slots += matched_rows
                assert matched_rows / total_rows > 0.8
    assert checked_slots > 0
    assert report.matched_parameters > 0

    # (c) the finetune run itself: initialize_model consumes the transfer:
    #     block (runtime checks + apply) and training runs through
    exp_b.run_configurations(num_threads=1)
    report_files = list((tmp_path / "SnowB" / "results" / "SnowB" / "TransferReports").glob("*.json"))
    assert report_files, "the applied transfer must persist a per-run report"
    persisted = json.loads(report_files[0].read_text())
    assert persisted["matched_parameters"] > 0
    results_files = list((tmp_path / "SnowB" / "results" / "SnowB" / "Results").glob("*.csv"))
    assert results_files, "the finetune run must write epoch results"
