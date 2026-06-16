from __future__ import annotations

from pathlib import Path

import pytest

from simplegnn.framework.utils import load_model as load_model_fn
from simplegnn.framework.utils import load_model_old as load_model_old_fn
from simplegnn.framework.utils.load_model import _get_validation_folds


def test_get_validation_folds_priority_rules():
    assert _get_validation_folds({"validation_folds": 4}) == 4
    assert _get_validation_folds({"splits": {"train": [[1], [2], [3]]}}) == 3
    assert _get_validation_folds({}) == 10


def test_load_model_missing_model_directory_raises(tmp_path):
    cfg = {"paths": {"results": tmp_path}, "models": [[{"layer_type": "linear", "out_features": 1}]], "name": "MUTAG"}

    with pytest.raises(FileNotFoundError, match="Model directory"):
        load_model_fn(cfg, "MUTAG")


def test_load_model_best_no_match_raises(tmp_path):
    models_dir = tmp_path / "MUTAG" / "Models"
    models_dir.mkdir(parents=True)
    cfg = {
        "paths": {"results": tmp_path},
        "models": [[{"layer_type": "linear", "out_features": 1}]],
        "name": "MUTAG",
    }

    with pytest.raises(FileNotFoundError, match="No best model"):
        load_model_fn(cfg, "MUTAG", best=True)


def test_load_model_best_multiple_matches_raises(tmp_path):
    models_dir = tmp_path / "MUTAG" / "Models"
    models_dir.mkdir(parents=True)
    (models_dir / "model_Best_Configuration_000001_run_0_val_step_0.pt").write_text("x")
    (models_dir / "model_Best_Configuration_000002_run_0_val_step_0.pt").write_text("x")
    cfg = {
        "paths": {"results": tmp_path},
        "models": [[{"layer_type": "linear", "out_features": 1}]],
        "name": "MUTAG",
    }

    with pytest.raises(ValueError, match="Multiple best models"):
        load_model_fn(cfg, "MUTAG", best=True)


def test_load_model_old_missing_model_dir_raises(tmp_path):
    cfg = {
        "paths": {"results": tmp_path},
        "models": [[{"layer_type": "linear", "out_features": 1}]],
        "name": "MUTAG",
    }

    with pytest.raises(FileNotFoundError, match="Model directory"):
        load_model_old_fn(cfg, "MUTAG")
