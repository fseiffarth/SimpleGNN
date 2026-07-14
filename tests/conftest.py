from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

FIXTURES = Path(__file__).resolve().parent / "fixtures"
MUTAG_SPLITS = ROOT / "src/simplegnn/datasets/splits/standard/MUTAG_splits.json"


@pytest.fixture
def mutag_main_config(tmp_path: Path):
    """Write a MUTAG main.yml pointing at the given model/parameter fixtures.

    The real MUTAG dataset is used (downloaded to data/TUDatasets/ on first
    run and cached thereafter); only the results directory is per-test.
    """

    def _write(models: Path, hyperparameters: Path) -> Path:
        config = {
            "datasets": [
                {
                    "name": "MUTAG",
                    "source": "TUDataset",
                    "task": "graph_classification",
                    "paths": {
                        "data": str(ROOT / "data" / "TUDatasets"),
                        "labels": str(ROOT / "data" / "TUDatasets" / "labels"),
                        "properties": str(ROOT / "data" / "TUDatasets" / "properties"),
                        "results": str(tmp_path / "results"),
                        "models": str(models),
                        "hyperparameters": str(hyperparameters),
                        "splits": str(MUTAG_SPLITS),
                    },
                }
            ]
        }
        path = tmp_path / "main.yml"
        path.write_text(yaml.safe_dump(config))
        return path

    return _write


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
def share_gnn_setup(mutag_main_config):
    """Preprocess MUTAG and return (graph_data, para) for the ShareGNN fixture model.

    This mirrors what FrameworkMain.run_configuration does before handing off to
    ModelConfiguration, so tests can build a real GraphModel without going
    through the full grid search.
    """
    from simplegnn.framework.core import FrameworkMain, preprocess_graph_data
    from simplegnn.framework.run_configuration import get_run_configs
    from simplegnn.framework.utils.parameters import Parameters
    from simplegnn.framework.utils.preprocessing import (
        load_preprocessed_data_and_parameters,
    )

    share_gnn = FIXTURES / "share_gnn_mutag"
    main_config_path = mutag_main_config(
        models=share_gnn / "models_ShareGNN.yml",
        hyperparameters=share_gnn / "parameters.yml",
    )

    experiment = FrameworkMain(main_config_path)
    experiment.preprocessing(num_threads=1)

    dataset_key = next(iter(experiment.network_configurations))
    configuration = experiment.network_configurations[dataset_key][0]
    graph_data = preprocess_graph_data(configuration)
    run_config = get_run_configs(configuration)[0]

    para = Parameters()
    load_preprocessed_data_and_parameters(
        config_id=0,
        run_id=0,
        validation_id=0,
        validation_folds=run_config.config.get("validation_folds", 10),
        graph_data=graph_data,
        run_config=run_config,
        para=para,
    )
    return graph_data, para


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
