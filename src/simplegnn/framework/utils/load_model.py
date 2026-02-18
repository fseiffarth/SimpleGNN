"""Model loading utilities for trained SimpleGNN networks."""

from __future__ import annotations

import re
from pathlib import Path

import torch

from simplegnn.framework.run_configuration import get_run_configs
from simplegnn.framework.utils.parameters import Parameters
from simplegnn.framework.utils.preprocessing import load_preprocessed_data_and_parameters
from simplegnn.models.model import GraphModel

BEST_MODEL_NAME_RE = re.compile(
    r"^model_Best_Configuration_(\d+)_run_(\d+)_val_step_(\d+)\.pt$"
)


def _preprocess_graph_data(experiment_configuration: dict):
    # Lazy import avoids import-time cycles with framework.core.
    from simplegnn.framework.core import preprocess_graph_data

    return preprocess_graph_data(experiment_configuration)


def _get_validation_folds(experiment_configuration: dict) -> int:
    if "validation_folds" in experiment_configuration:
        return experiment_configuration["validation_folds"]
    splits = experiment_configuration.get("splits", {})
    train_splits = splits.get("train")
    if train_splits is None:
        return 10
    return len(train_splits)


def load_model_old(
    experiment_configuration: dict,
    db_name: str,
    config_id: int = 0,
    run_id: int = 0,
    validation_id: int = 0,
    best: bool = True,
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    """
    Legacy model loader kept for behavior comparison.

    This preserves the old path resolution strategy, including first-match
    globbing for best models.
    """
    graph_data = _preprocess_graph_data(experiment_configuration)
    run_configs = get_run_configs(experiment_configuration)
    path_to_models = experiment_configuration["paths"]["results"].joinpath(db_name).joinpath("Models")

    if best:
        if not path_to_models.exists():
            raise FileNotFoundError(f"Model directory {path_to_models} not found")
        curr_path = next(path_to_models.glob("*Best_Configuration*"))
        config_id = int(curr_path.name.split("_")[3])
        model_path = path_to_models.joinpath(
            f"model_Best_Configuration_{str(config_id).zfill(6)}_run_{run_id}_val_step_{validation_id}.pt"
        )
    else:
        if not path_to_models.exists():
            raise FileNotFoundError(f"Model directory {path_to_models} not found")
        model_path = path_to_models.joinpath(
            f"model_Configuration_{str(config_id).zfill(6)}_run_{run_id}_val_step_{validation_id}.pt"
        )

    run_config = run_configs[config_id]
    if not model_path.exists():
        raise FileNotFoundError(f"Model {model_path} not found")

    para = Parameters()
    load_preprocessed_data_and_parameters(
        config_id=config_id,
        run_id=run_id,
        validation_id=validation_id,
        graph_data=graph_data,
        run_config=run_config,
        para=para,
        validation_folds=_get_validation_folds(experiment_configuration),
    )
    para.set_file_index(size=6)
    seed = 42 + validation_id + para.n_val_runs * run_id
    net = GraphModel(graph_data=graph_data, para=para, seed=seed, device=device)
    net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    net.eval()
    return net


def load_model(
    experiment_configuration: dict,
    db_name: str,
    config_id: int = 0,
    run_id: int = 0,
    validation_id: int = 0,
    best: bool = True,
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    """Load a trained model with deterministic model-path resolution."""
    graph_data = _preprocess_graph_data(experiment_configuration)
    run_configs = get_run_configs(experiment_configuration)
    path_to_models = experiment_configuration["paths"]["results"].joinpath(db_name).joinpath("Models")

    if not path_to_models.exists():
        raise FileNotFoundError(f"Model directory {path_to_models} not found")

    if best:
        pattern = f"model_Best_Configuration_*_run_{run_id}_val_step_{validation_id}.pt"
        matches = sorted(path_to_models.glob(pattern), key=lambda p: p.name)
        if not matches:
            raise FileNotFoundError(
                f"No best model found in {path_to_models} for run_id={run_id}, validation_id={validation_id}"
            )
        if len(matches) > 1:
            candidates = ", ".join(path.name for path in matches)
            raise ValueError(
                "Multiple best models matched for "
                f"run_id={run_id}, validation_id={validation_id}: {candidates}"
            )

        model_path = matches[0]
        match = BEST_MODEL_NAME_RE.match(model_path.name)
        if match is None:
            raise ValueError(f"Malformed best model filename: {model_path.name}")
        parsed_config_id, parsed_run_id, parsed_validation_id = map(int, match.groups())
        if parsed_run_id != run_id or parsed_validation_id != validation_id:
            raise ValueError(
                f"Best model filename mismatch for {model_path.name}: "
                f"run_id={parsed_run_id}, validation_id={parsed_validation_id}"
            )
        config_id = parsed_config_id
    else:
        model_path = path_to_models.joinpath(
            f"model_Configuration_{str(config_id).zfill(6)}_run_{run_id}_val_step_{validation_id}.pt"
        )
        if not model_path.exists():
            raise FileNotFoundError(f"Model {model_path} not found")

    run_config = run_configs[config_id]
    para = Parameters()
    load_preprocessed_data_and_parameters(
        config_id=config_id,
        run_id=run_id,
        validation_id=validation_id,
        graph_data=graph_data,
        run_config=run_config,
        para=para,
        validation_folds=_get_validation_folds(experiment_configuration),
    )
    para.set_file_index(size=6)
    seed = 42 + validation_id + para.n_val_runs * run_id
    net = GraphModel(graph_data=graph_data, para=para, seed=seed, device=device)
    net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    return net
