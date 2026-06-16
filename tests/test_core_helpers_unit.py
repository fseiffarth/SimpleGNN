from __future__ import annotations

from pathlib import Path

import pytest

from simplegnn.framework.core import collect_paths, preprocess_graph_data


def test_collect_paths_requires_mandatory_paths():
    with pytest.raises(FileNotFoundError, match="Data path is missing"):
        collect_paths(
            main_configuration={"paths": {}},
            model_configuration={"models": []},
            dataset_configuration={"paths": {}},
        )


def test_collect_paths_requires_props_labels_for_invariant_layers():
    dataset_cfg = {"paths": {"data": "d", "results": "r", "splits": "s"}}
    model_cfg = {"models": [[{"layer_type": "invariant_based_convolution"}]]}

    with pytest.raises(FileNotFoundError, match="Properties path is missing"):
        collect_paths({}, model_cfg, dataset_cfg)


def test_collect_paths_appends_results_for_pretraining():
    dataset_cfg = {
        "paths": {"data": "d/", "results": "res/", "splits": "s/"},
        "pretraining_datasets": ["A", "B"],
    }

    out = collect_paths({}, {"models": []}, dataset_cfg)

    assert out["results"].endswith("pretraining_A_B/")


def test_preprocess_graph_data_delegates_to_get_graph_data(monkeypatch, tmp_path):
    calls = {}

    class FakeGraph:
        def __init__(self):
            self.moved_to = None

        def to(self, device):
            self.moved_to = device

    fake_graph = FakeGraph()

    def fake_get_graph_data(**kwargs):
        calls.update(kwargs)
        return fake_graph

    monkeypatch.setattr("simplegnn.framework.core.get_graph_data", fake_get_graph_data)

    cfg = {
        "name": "MUTAG",
        "paths": {"data": tmp_path},
        "task": "graph_classification",
        "format": "RuleGNNDataset",
        "precision": "float",
        "device": "cpu",
    }

    out = preprocess_graph_data(cfg)

    assert out is fake_graph
    assert calls["db_name"] == "MUTAG"
    assert calls["data_path"] == tmp_path
    assert fake_graph.moved_to == "cpu"
