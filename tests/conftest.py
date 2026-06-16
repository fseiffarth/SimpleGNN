from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def seed_all():
    random.seed(1337)
    np.random.seed(1337)
    try:
        import torch

        torch.manual_seed(1337)
    except Exception:
        pass


@pytest.fixture
def minimal_dataset_config(tmp_path: Path):
    data = tmp_path / "data"
    results = tmp_path / "results"
    splits = tmp_path / "splits"
    data.mkdir()
    results.mkdir()
    splits.mkdir()

    split_file = splits / "MUTAG_splits.json"
    split_file.write_text(
        """[
  {"test": [0], "model_selection": [{"train": [1], "validation": [2]}]},
  {"test": [2], "model_selection": [{"train": [0], "validation": [1]}]}
]
"""
    )

    return {
        "name": "MUTAG",
        "source": "TUDataset",
        "task": "graph_classification",
        "paths": {
            "data": data,
            "results": results,
            "splits": split_file,
            "models": tmp_path / "models.yml",
            "hyperparameters": tmp_path / "hparams.yml",
        },
    }


@pytest.fixture
def minimal_hyper_config():
    return {
        "input_features": {"name": "ones"},
        "batch_size": [8],
        "epochs": [2],
        "learning_rate": [0.01],
        "optimizer": ["Adam"],
        "loss": ["CrossEntropyLoss"],
    }


@pytest.fixture
def minimal_model_config():
    return {"models": [[{"layer_type": "linear", "out_features": 4}]]}
