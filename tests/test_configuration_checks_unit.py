from __future__ import annotations

from pathlib import Path

import pytest

from simplegnn.framework.utils.configuration_checks import (
    check_hyperparameter_configuration_file,
    check_main_configuration_file,
    check_model_configuration_file,
)


def test_check_main_configuration_file_sets_absolute_paths(minimal_dataset_config):
    cfg = {"datasets": [minimal_dataset_config]}

    check_main_configuration_file(cfg)

    paths = cfg["datasets"][0]["paths"]
    assert all(isinstance(paths[k], Path) and paths[k].is_absolute() for k in ["data", "results", "splits", "models", "hyperparameters"])


def test_check_main_configuration_file_missing_name_raises(minimal_dataset_config):
    broken = dict(minimal_dataset_config)
    broken.pop("name")
    cfg = {"datasets": [broken]}

    with pytest.raises(ValueError, match="name of the dataset"):
        check_main_configuration_file(cfg)


def test_check_model_configuration_requires_props_for_invariant_layer(tmp_path):
    dataset_cfg = {
        "paths": {
            "data": tmp_path,
            "results": tmp_path,
            "splits": tmp_path,
            "models": tmp_path,
            "hyperparameters": tmp_path,
        }
    }
    model_cfg = {"models": [[{"layer_type": "invariant_based_convolution"}]]}

    with pytest.raises(FileNotFoundError, match="Properties path is missing"):
        check_model_configuration_file(dataset_cfg, model_cfg)


def test_check_model_configuration_non_invariant_sets_paths_to_none(tmp_path):
    dataset_cfg = {
        "paths": {
            "data": tmp_path,
            "results": tmp_path,
            "splits": tmp_path,
            "models": tmp_path,
            "hyperparameters": tmp_path,
        }
    }
    model_cfg = {"models": [[{"layer_type": "linear", "out_features": 2}]]}

    check_model_configuration_file(dataset_cfg, model_cfg)

    assert dataset_cfg["paths"]["properties"] is None
    assert dataset_cfg["paths"]["labels"] is None
    assert dataset_cfg["with_invariant_layers"] is False


def test_check_hyperparameter_configuration_sets_defaults(minimal_hyper_config):
    cfg = dict(minimal_hyper_config)

    check_hyperparameter_configuration_file(cfg)

    assert cfg["device"] == "cpu"
    assert cfg["precision"] == "double"
    assert cfg["mode"] == "experiments"
    assert cfg["early_stopping"]["enabled"] is False
    assert cfg["rule_occurrence_threshold"] == 1


def test_check_hyperparameter_configuration_missing_required_raises(minimal_hyper_config):
    cfg = dict(minimal_hyper_config)
    cfg.pop("optimizer")

    with pytest.raises(ValueError, match='key "optimizer"'):
        check_hyperparameter_configuration_file(cfg)
