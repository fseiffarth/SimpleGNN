from __future__ import annotations

import pytest

from simplegnn.datasets.graph_dataset import get_graph_data, relabel_most_frequent, transform_data


def test_relabel_most_frequent_relabels_to_frequency_order():
    labels = type(
        "Labels",
        (),
        {
            "db_unique_node_labels": {0: 5, 1: 3, 2: 1},
            "node_labels": [[0, 1, 2], [1, 1, 0]],
            "unique_node_labels": [{}, {}],
            "num_unique_node_labels": 0,
        },
    )()

    relabel_most_frequent(labels, num_max_labels=2)

    assert labels.num_unique_node_labels == 2
    # least frequent mapped to overflow class 1
    assert labels.node_labels[0][-1] == 1


def test_transform_data_applies_single_expression():
    out = transform_data(2, {"transformation": "lambda input, factor=1: input * factor", "transformation_args": {"factor": 3}})
    assert out == 6


def test_transform_data_applies_multiple_expressions():
    out = transform_data(
        2,
        {
            "transformation": [
                "lambda input, add=0: input + add",
                "lambda input, mul=1: input * mul",
            ],
            "transformation_args": [{"add": 1}, {"mul": 4}],
        },
    )
    assert out == 12


def test_get_graph_data_invalid_format_raises(tmp_path):
    with pytest.raises(ValueError, match="not supported"):
        get_graph_data("MUTAG", tmp_path, graph_format="unknown")
