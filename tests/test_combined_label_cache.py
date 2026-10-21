from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import simplegnn.models.ShareGNN.preprocessing.preprocessing as sharegnn_preprocessing
from simplegnn.datasets.utils.node_labeling import load_labels


def make_graph_data():
    primary = torch.tensor([0, 1, 1, 2, 0])
    return SimpleNamespace(name="TOY", node_labels={"primary": primary},
                           data=SimpleNamespace(x=torch.zeros((5, 1))))


def run_combined(tmp_path: Path, graph_data) -> Path:
    config = {"paths": {"labels": tmp_path}}
    layer = json.dumps({"label_type": ["primary", "trivial"]})
    return sharegnn_preprocessing.layer_to_labels(config, layer, graph_data)


def test_combined_labels_are_cached_and_rebuilt_when_component_changes(tmp_path: Path, monkeypatch):
    graph_data = make_graph_data()
    combine_calls = []
    original_combine = sharegnn_preprocessing.combine_node_labels

    def counting_combine(labels):
        combine_calls.append(1)
        return original_combine(labels)

    monkeypatch.setattr(sharegnn_preprocessing, "combine_node_labels", counting_combine)

    file_path = run_combined(tmp_path, graph_data)
    assert file_path == tmp_path / "TOY" / "TOY_labels_primary_trivial.pt"
    assert load_labels(file_path).label_name == "primary_trivial"
    assert len(combine_calls) == 1

    # existing combined file newer than its components -> skipped
    run_combined(tmp_path, graph_data)
    assert len(combine_calls) == 1

    # a component regenerated after the combined file -> rebuilt
    component = tmp_path / "TOY" / "TOY_labels_primary.pt"
    newer = file_path.stat().st_mtime + 10
    os.utime(component, (newer, newer))
    run_combined(tmp_path, graph_data)
    assert len(combine_calls) == 2
