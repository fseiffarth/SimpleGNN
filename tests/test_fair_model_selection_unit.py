from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from simplegnn.framework.utils.evaluation import (
    fair_model_selection_evaluation,
    model_selection_evaluation,
)

DB_NAME = "TESTDB"
CONFIGS = [0, 1, 2]
FOLDS = list(range(10))
RUNS = [0, 1]
EPOCHS = [0, 1, 2]  # best epoch is always the last one (max validation accuracy)

# Header for a classification run (note the double space before the loss name,
# matching model_configuration.py — fair selection must locate it by substring).
HEADER = (
    "Dataset;Time;RunNumber;ValidationNumber;Seed;Epoch;TrainingSize;ValidationSize;"
    "TestSize;EpochLoss  (CrossEntropyLoss);EpochAccuracy;EpochTime;ValidationAccuracy;"
    "ValidationLoss;TestAccuracy;TestLoss"
)

TRAIN_SIZE = 80
VAL_SIZE = 10
TEST_SIZE = 10


def winning_config(fold: int) -> int:
    """The configuration that should be selected for a given outer fold."""
    return fold % 3


def winner_test_accuracy(fold: int, run: int) -> float:
    """Deterministic, distinct test accuracy for the winning (config, fold, run)."""
    return 0.70 + 0.01 * fold + 0.02 * run


def _write_run_csv(results_dir: Path, config: int, fold: int, run: int) -> None:
    is_winner = config == winning_config(fold)
    rows = [HEADER]
    for epoch in EPOCHS:
        # validation accuracy rises with epoch so the best epoch is the last one
        if is_winner:
            val_acc = [0.30, 0.60, 0.90][epoch]
            test_acc = winner_test_accuracy(fold, run) if epoch == EPOCHS[-1] else 0.10
        else:
            val_acc = [0.20, 0.35, 0.50][epoch]
            test_acc = 0.40 if epoch == EPOCHS[-1] else 0.10
        val_loss = [0.9, 0.6, 0.3][epoch]  # decreasing -> no tie ambiguity
        seed = 42 + fold + 2 * run
        rows.append(
            f"{DB_NAME};2024-01-01 00:00:00;{run};{fold};{seed};{epoch};"
            f"{TRAIN_SIZE};{VAL_SIZE};{TEST_SIZE};0.5;0.5;0.01;"
            f"{val_acc};{val_loss};{test_acc};0.5"
        )
    file_name = (
        f"{DB_NAME}_Configuration_{config:06d}_Results_run_id_{run}_validation_step_{fold}.csv"
    )
    (results_dir / file_name).write_text("\n".join(rows) + "\n")


@pytest.fixture()
def results_root(tmp_path: Path) -> Path:
    results_dir = tmp_path / DB_NAME / "Results"
    results_dir.mkdir(parents=True)
    for config in CONFIGS:
        for fold in FOLDS:
            for run in RUNS:
                _write_run_csv(results_dir, config, fold, run)
    return tmp_path


def _config(results_root: Path) -> dict:
    return {"paths": {"results": results_root}, "evaluation_type": "accuracy"}


def test_fair_selects_intended_config_per_fold(results_root: Path):
    n_folds = fair_model_selection_evaluation(
        db_name=DB_NAME, experiment_config=_config(results_root)
    )
    assert n_folds == len(FOLDS)

    summary = pd.read_csv(results_root / DB_NAME / "summary_fair.csv")
    summary = summary.sort_values("ValidationNumber").reset_index(drop=True)

    assert list(summary["ValidationNumber"]) == FOLDS
    for _, row in summary.iterrows():
        fold = int(row["ValidationNumber"])
        assert int(row["ConfigurationId"]) == winning_config(fold)
        assert int(row["N_runs"]) == len(RUNS)
        # per-fold test mean is the mean across the two runs of the winner
        expected = np.mean([winner_test_accuracy(fold, r) for r in RUNS])
        assert row["Test Accuracy Mean"] == pytest.approx(expected)


def test_fair_mean_matches_hand_computed(results_root: Path):
    fair_model_selection_evaluation(db_name=DB_NAME, experiment_config=_config(results_root))

    mean_df = pd.read_csv(results_root / DB_NAME / "summary_fair_mean.csv")
    assert mean_df.shape[0] == 1

    per_fold_means = [
        np.mean([winner_test_accuracy(f, r) for r in RUNS]) for f in FOLDS
    ]
    assert mean_df.loc[0, "Test Accuracy Mean"] == pytest.approx(np.mean(per_fold_means))
    assert mean_df.loc[0, "Test Accuracy Std"] == pytest.approx(np.std(per_fold_means))
    assert int(mean_df.loc[0, "N_folds"]) == len(FOLDS)

    selected_ids = [int(x) for x in str(mean_df.loc[0, "Selected Configuration Ids"]).split()]
    assert selected_ids == [winning_config(f) for f in FOLDS]


def test_get_best_per_fold_returns_map(results_root: Path):
    best_per_fold = fair_model_selection_evaluation(
        db_name=DB_NAME, experiment_config=_config(results_root), get_best_per_fold=True
    )
    assert best_per_fold == {f: winning_config(f) for f in FOLDS}
    # get_best_per_fold must not write the summary files
    assert not (results_root / DB_NAME / "summary_fair.csv").exists()


def test_global_model_selection_unchanged(results_root: Path):
    """Regression guard: the global path still works and is independent of the fair files."""
    config = _config(results_root)
    # fair path writes only summary_fair*.csv
    fair_model_selection_evaluation(db_name=DB_NAME, experiment_config=config)
    assert not (results_root / DB_NAME / "summary.csv").exists()

    # global path: write summary.csv, then identify the single global best config
    model_selection_evaluation(db_name=DB_NAME, experiment_config=config)
    assert (results_root / DB_NAME / "summary.csv").exists()

    best = model_selection_evaluation(
        db_name=DB_NAME, experiment_config=config, get_best_model=True
    )
    # config 0 wins 4 folds (0,3,6,9) -> highest mean validation accuracy globally
    assert int(best) == 0
