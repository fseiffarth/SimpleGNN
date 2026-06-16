from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from simplegnn.utils.path_conversions import config_paths_to_absolute, paths_to_absolute
from simplegnn.utils.utils import (
    balance_data,
    convert_to_list,
    convert_to_tuple,
    diff,
    get_accuracy,
    get_data_indices,
    get_train_test_list,
    get_train_validation_test_list,
    reshape_indices,
)


def test_config_paths_to_absolute_mutates_config(tmp_path: Path):
    cfg = {
        "paths": {
            "data": Path("data"),
            "results": Path("results"),
            "splits": Path("splits"),
            "labels": Path("labels"),
            "properties": Path("properties"),
        }
    }

    config_paths_to_absolute(cfg, tmp_path)

    assert cfg["paths"]["data"] == tmp_path / "data"
    assert cfg["paths"]["labels"] == tmp_path / "labels"


def test_paths_to_absolute_mutates_present_keys(tmp_path: Path):
    paths = {"data": Path("data"), "results": Path("results")}
    paths_to_absolute(paths, tmp_path)
    assert paths["data"] == tmp_path / "data"
    assert paths["results"] == tmp_path / "results"


def test_tuple_and_list_conversion_roundtrip():
    value = [1, [2, 3]]
    as_tuple = convert_to_tuple(value)
    back = convert_to_list(as_tuple)

    assert as_tuple == (1, (2, 3))
    assert back == [1, [2, 3]]


def test_diff_and_splitting_helpers_are_deterministic():
    assert diff([1, 2, 3], [2]) == [1, 3]

    train1, test1 = get_train_test_list(20, seed=7)
    train2, test2 = get_train_test_list(20, seed=7)
    assert train1 == train2
    assert test1 == test2


def test_train_val_test_balanced_path():
    folds = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]
    labels = [0, 1, 0, 1, 0, 1]

    train, val, test = get_train_validation_test_list(
        folds,
        validation_step=0,
        seed=1,
        balanced=True,
        graph_labels=labels,
        val_size=0.25,
    )

    assert len(test) == 2
    assert len(val) >= 1
    assert len(train) >= 3


def test_balance_data_and_get_accuracy_and_indices():
    balanced = balance_data([0, 1, 2], [0, 1, 1])
    assert len(balanced) >= 3

    out = torch.tensor([[0.1, 0.9], [0.9, 0.1]])
    labels = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    assert get_accuracy(out, labels, one_hot_encoding=True) == 1.0

    indices = get_data_indices(10, seed=42, kFold=5)
    assert len(indices) == 5
    assert sorted(np.concatenate(indices).tolist()) == list(range(10))


def test_reshape_indices_maps_positions():
    a = np.array([[10, 20], [30, 40]])
    b = np.array([[1, 2], [3, 4]])

    mapping = reshape_indices(a, b)

    assert mapping[(0, 0)] == (0, 0)
    assert mapping[(1, 1)] == (1, 1)
