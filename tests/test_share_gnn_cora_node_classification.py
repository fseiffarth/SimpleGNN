"""End-to-end smoke test: ShareGNN node classification on Cora.

Exercises the node-level task path of the framework (node splits, per-node
training/evaluation on a single graph) against the real Planetoid Cora
dataset (downloaded on first run, cached under data/Planetoid/ and tmp/
afterwards). Complements the MUTAG integration tests, which only cover
graph-level tasks.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "share_gnn_cora"
MODELS = FIXTURES / "models_ShareGNN.yml"
PARAMETERS = FIXTURES / "parameters.yml"


def _write_split_file(split_file: Path) -> None:
    """Split JSON from the standard Planetoid masks (140/500/1000 nodes)."""
    import torch
    from torch_geometric.datasets import Planetoid

    data = Planetoid(root="tmp/", name="Cora")[0]
    splits = [{
        "test": torch.where(data.test_mask)[0].tolist(),
        "model_selection": [{
            "train": torch.where(data.train_mask)[0].tolist(),
            "validation": torch.where(data.val_mask)[0].tolist(),
        }],
    }]
    split_file.write_text(json.dumps(splits))


@pytest.mark.integration
def test_share_gnn_node_classification_on_cora(tmp_path):
    from simplegnn.framework.core import FrameworkMain

    split_file = tmp_path / "Cora_splits.json"
    _write_split_file(split_file)

    config = {
        "datasets": [
            {
                "name": "Cora",
                "source": "Planetoid",
                "task": "node_classification",
                "validation_folds": 1,
                "paths": {
                    "data": str(ROOT / "data" / "Planetoid"),
                    "labels": str(ROOT / "data" / "Planetoid" / "labels"),
                    "properties": str(ROOT / "data" / "Planetoid" / "properties"),
                    "results": str(tmp_path / "results"),
                    "models": str(MODELS),
                    "hyperparameters": str(PARAMETERS),
                    "splits": str(split_file),
                },
            }
        ]
    }
    main_config_path = tmp_path / "main.yml"
    main_config_path.write_text(yaml.safe_dump(config))

    experiment = FrameworkMain(main_config_path)
    experiment.preprocessing(num_threads=1)
    experiment.run_configurations(num_threads=1)

    results_dir = tmp_path / "results" / "Cora" / "Results"
    csv_files = list(results_dir.glob("Cora_Configuration_*_Results_run_id_0_validation_step_0.csv"))
    assert csv_files, "expected a per-epoch results CSV to be written"

    with open(csv_files[0], newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    assert len(rows) == 3, "expected one row per epoch"
    for row in rows:
        # per-node accuracies must be valid percentages on train and validation
        assert 0.0 <= float(row["EpochAccuracy"]) <= 100.0
        assert 0.0 <= float(row["ValidationAccuracy"]) <= 100.0
        # 140 train / 500 validation / 1000 test nodes (standard Planetoid split)
        assert int(row["TrainingSize"]) == 140
        assert int(row["ValidationSize"]) == 500
        assert int(row["TestSize"]) == 1000

    # training must actually learn something: the loss must drop across epochs
    losses = [float(row[[c for c in row if c.startswith("EpochLoss")][0]]) for row in rows]
    assert losses[-1] < losses[0], f"training loss did not decrease: {losses}"

    models_dir = tmp_path / "results" / "Cora" / "Models"
    assert any(models_dir.glob("*.pt")), "expected a saved model checkpoint"
