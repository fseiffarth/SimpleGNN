from __future__ import annotations

import pytest

from simplegnn.models.layers.utils.layer_loader import (
    check_layer,
    check_layer_short_type,
    check_network_architectures,
    layer_from_yml,
)


def test_check_layer_linear_requires_out_features():
    ok, msg = check_layer(0, {"layer_type": "linear"})
    assert not ok
    assert "out_features" in msg


def test_check_layer_global_pooling_mode_validation():
    ok, msg = check_layer(0, {"layer_type": "global_pooling", "mode": "median"})
    assert not ok
    assert "mean, max or sum" in msg


def test_check_layer_short_type_accepts_compact_invariant_definition():
    ok, _ = check_layer_short_type(
        0,
        {
            "layer_type": "invariant_based_convolution",
            "bias": True,
            "heads": [1, 2],
            "labels": [{"label_type": "primary", "max_labels": [3, 5]}],
            "properties": [{"name": "distance", "values": [0, 1]}],
        },
    )
    assert ok


def test_layer_from_yml_expands_short_invariant_definition():
    layers_per_architecture = []
    network_architecture = [
        {
            "layer_type": "invariant_based_convolution",
            "bias": False,
            "heads": [1, 2],
            "labels": [{"label_type": "primary", "max_labels": [3, 5]}],
            "properties": [
                {"name": "distance", "values": [0, 1]},
                {"name": "distance", "values": [2, 3]},
            ],
        }
    ]

    layer_from_yml(0, network_architecture[0], layers_per_architecture, network_architecture)

    # combinations: heads(2) * label options(2) * properties(2)
    assert len(layers_per_architecture) == 1
    assert len(layers_per_architecture[0]) == 8


def test_check_network_architectures_detects_invalid_layer():
    ok = check_network_architectures(
        [[{"layer_type": "linear", "out_features": 4}], [{"layer_type": "linear"}]]
    )
    assert ok is False


def test_layer_from_yml_invalid_type_raises():
    with pytest.raises(ValueError, match="not supported"):
        layer_from_yml(0, {"layer_type": "bad_type"}, [], [{"layer_type": "bad_type"}])
