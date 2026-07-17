from __future__ import annotations

import pytest

from simplegnn.framework.run_configuration import (
    generate_layer_options,
    get_run_configs,
    preprocess_network_architectures,
)


def test_generate_layer_options_cross_product_with_properties():
    layer_dict = {
        "layer_type": "invariant_based_aggregation",
        "labels": [
            {"label_type": "primary", "max_labels": [3, 5], "depth": [1, 2]}
        ],
        "properties": [{"name": "dist", "values": [1, 2]}],
        "bias": True,
    }

    options = generate_layer_options(layer_dict)
    # grid axes: max_labels [3, 5] x depth [1, 2] = 4. A property's `values`
    # list is the set of valid property values for ONE config (e.g. all
    # distances of a layer), not a grid axis; grids over properties use
    # multiple property dicts.
    assert len(options) == 4
    combos = {
        (o["channels"][0]["labels"]["max_labels"], o["channels"][0]["labels"]["depth"])
        for o in options
    }
    assert combos == {(3, 1), (3, 2), (5, 1), (5, 2)}
    for o in options:
        assert o["channels"][0]["properties"] == {"name": "dist", "values": [1, 2]}
        assert o["bias"] is True


def test_preprocess_network_architectures_raises_on_invalid_architecture():
    with pytest.raises(ValueError, match="not correctly defined"):
        preprocess_network_architectures([[{"layer_type": "linear"}]])


def test_get_run_configs_matches_parameter_product():
    exp_cfg = {
        "task": "graph_classification",
        "models": [[{"layer_type": "linear", "out_features": 8}]],
        "batch_size": [16, 32],
        "learning_rate": [0.1, 0.01],
        "epochs": [1],
        "dropout": [0.0, 0.5],
        "optimizer": ["Adam"],
        "weight_decay": [0.0],
        "loss": ["CrossEntropyLoss", "MSELoss"],
    }

    run_configs = get_run_configs(exp_cfg)

    assert len(run_configs) == 2 * 2 * 1 * 2 * 1 * 1 * 2
    assert run_configs[0].task == "graph_classification"
