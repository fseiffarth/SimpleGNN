from __future__ import annotations

import json
from pathlib import Path

import pytest

from simplegnn.framework.utils.preprocessing import Preprocessing, load_splits


def test_load_splits_valid_file(tmp_path: Path):
    split_file = tmp_path / "splits.json"
    split_file.write_text(
        json.dumps(
            [
                {"test": [0], "model_selection": [{"train": [1], "validation": [2]}]},
                {"test": [2], "model_selection": [{"train": [0], "validation": [1]}]},
            ]
        )
    )

    splits = load_splits(split_file)
    assert splits["test"][0] == [0]
    assert splits["train"][1] == [0]


def test_load_splits_overlapping_indices_raises(tmp_path: Path):
    split_file = tmp_path / "bad_splits.json"
    split_file.write_text(
        json.dumps(
            [
                {"test": [0], "model_selection": [{"train": [0], "validation": [2]}]},
            ]
        )
    )

    with pytest.raises(ValueError, match="Overlapping indices"):
        load_splits(split_file)


def test_create_split_file_requires_custom_split_function_when_disabled():
    prep = Preprocessing.__new__(Preprocessing)
    prep.db_name = "MUTAG"
    prep.graph_data = object()
    prep.experiment_configuration = {
        "with_splits": False,
        "paths": {"data": Path("/tmp/data"), "splits": Path("/tmp/splits")},
    }

    with pytest.raises(ValueError, match="split function"):
        prep.create_split_file()


def test_load_configuration_splits_uses_json_file_path(tmp_path: Path):
    split_file = tmp_path / "my_splits.json"
    split_file.write_text(
        json.dumps(
            [
                {"test": [0], "model_selection": [{"train": [1], "validation": [2]}]},
            ]
        )
    )

    prep = Preprocessing.__new__(Preprocessing)
    prep.db_name = "MUTAG"
    prep.experiment_configuration = {"paths": {"splits": split_file}}

    prep.load_configuration_splits()

    assert prep.experiment_configuration["splits"]["test"] == [[0]]
