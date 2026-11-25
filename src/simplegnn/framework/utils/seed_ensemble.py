"""Seed ensembles from saved best-checkpoint predictions.

``save_best_predictions: True`` makes every training run write
``model_<config>_run_<r>_val_step_<k>.predictions.npz`` next to its best
checkpoint (see ``ModelConfiguration._write_best_predictions``). This module
averages those files over the runs of one configuration, which scores a seed
ensemble without loading a single model.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

PREDICTIONS_NAME_RE = re.compile(r"^model_(.+)_run_(\d+)_val_step_(\d+)\.predictions\.npz$")


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return float('nan')
    return float(roc_auc_score(labels, scores))


def seed_ensemble(models_dir, config_prefix: str = 'Best_Configuration',
                  validation_id: int = 0) -> dict:
    """
    Score the ensemble of all runs of one configuration.

    Parameters
    ----------
    models_dir : str or Path
        ``<results>/<dataset>/Models`` directory holding the ``.predictions.npz`` files.
    config_prefix : str
        Configuration name prefix, e.g. ``'Best_Configuration'`` (the rerun of the
        selected configuration) or ``'Configuration_000003'``.
    validation_id : int
        Validation fold whose runs are ensembled.

    Returns
    -------
    dict
        Per-run validation/test AUCs, their mean and std, and the validation/test
        AUC of the probability-averaged (``mean``) and rank-averaged (``rank``)
        ensembles.

    Raises
    ------
    FileNotFoundError
        If no prediction file matches.
    ValueError
        If the runs disagree on graph ids or labels.
    """
    models_dir = Path(models_dir)
    runs = []
    for path in sorted(models_dir.glob(f'model_{config_prefix}*.predictions.npz')):
        match = PREDICTIONS_NAME_RE.match(path.name)
        if match is None or int(match.group(3)) != validation_id:
            continue
        with np.load(path) as data:
            runs.append((int(match.group(2)), {key: data[key] for key in data.files}))
    if not runs:
        raise FileNotFoundError(
            f"No prediction files model_{config_prefix}*_val_step_{validation_id}.predictions.npz "
            f"in {models_dir}; run with save_best_predictions: True")
    runs.sort(key=lambda run: run[0])

    result = {'models_dir': str(models_dir), 'config_prefix': config_prefix,
              'validation_id': validation_id, 'num_runs': len(runs),
              'run_ids': [run_id for run_id, _ in runs],
              'best_epochs': [int(data['epoch']) for _, data in runs]}
    for split in ('validation', 'test'):
        reference = runs[0][1]
        for run_id, data in runs[1:]:
            if not (np.array_equal(data[f'{split}_ids'], reference[f'{split}_ids'])
                    and np.array_equal(data[f'{split}_labels'], reference[f'{split}_labels'])):
                raise ValueError(f"run {run_id} has different {split} graph ids or labels than run {runs[0][0]}")
        labels = reference[f'{split}_labels']
        scores = np.stack([data[f'{split}_scores'] for _, data in runs])
        run_aucs = [_auc(labels, run_scores) for run_scores in scores]
        result[f'{split}_auc_runs'] = run_aucs
        result[f'{split}_auc_mean'] = float(np.mean(run_aucs))
        result[f'{split}_auc_std'] = float(np.std(run_aucs))
        result[f'{split}_auc_ensemble_mean'] = _auc(labels, scores.mean(axis=0))
        result[f'{split}_auc_ensemble_rank'] = _auc(labels, np.mean([rankdata(s) for s in scores], axis=0))
    return result


def write_seed_ensemble(results_dir, db_name: str, config_prefix: str = 'Best_Configuration',
                        validation_id: int = 0) -> dict:
    """
    Run :func:`seed_ensemble` on ``<results_dir>/<db_name>/Models`` and save the
    result as ``<results_dir>/<db_name>/seed_ensemble.json``.
    """
    dataset_dir = Path(results_dir) / db_name
    result = seed_ensemble(dataset_dir / 'Models', config_prefix=config_prefix,
                           validation_id=validation_id)
    with open(dataset_dir / 'seed_ensemble.json', 'w') as f:
        json.dump(result, f, indent=2)
    return result
