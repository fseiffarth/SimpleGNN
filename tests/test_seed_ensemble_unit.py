"""Unit tests for best-checkpoint prediction files and the seed ensemble."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from sklearn.metrics import roc_auc_score

from simplegnn.framework.model_configuration import ModelConfiguration
from simplegnn.framework.utils.seed_ensemble import seed_ensemble, write_seed_ensemble

VALIDATION_IDS = np.array([10, 11, 12, 13])
VALIDATION_LABELS = np.array([0, 1, 0, 1])
TEST_IDS = np.array([20, 21, 22, 23, 24])
TEST_LABELS = np.array([0, 0, 1, 1, 0])


def _write_run(models_dir, run_id, validation_scores, test_scores, config='Best_Configuration_000000',
               test_ids=TEST_IDS):
    np.savez(models_dir / f'model_{config}_run_{run_id}_val_step_0.predictions.npz',
             epoch=run_id, validation_ids=VALIDATION_IDS, validation_scores=np.asarray(validation_scores),
             validation_labels=VALIDATION_LABELS, test_ids=test_ids, test_scores=np.asarray(test_scores),
             test_labels=TEST_LABELS)


def test_write_best_predictions_combines_validation_and_test_scores(tmp_path):
    config = ModelConfiguration.__new__(ModelConfiguration)
    config.test_data = TEST_IDS
    path = tmp_path / 'model_Best_Configuration_000000_run_0_val_step_0.predictions.npz'
    config._pending_best_predictions = {
        'path': path, 'epoch': 3, 'validation_ids': VALIDATION_IDS,
        'validation_scores': np.array([0.1, 0.9, 0.2, 0.8]), 'validation_labels': VALIDATION_LABELS,
    }
    outputs = torch.tensor([[2.0, 0.0], [0.0, 0.0], [0.0, 2.0], [0.0, 1.0], [1.0, 0.0]], dtype=torch.double)
    config._write_best_predictions(outputs, torch.as_tensor(TEST_LABELS))
    assert config._pending_best_predictions is None
    with np.load(path) as data:
        assert int(data['epoch']) == 3
        assert data['test_ids'].tolist() == TEST_IDS.tolist()
        assert data['test_scores'] == pytest.approx(torch.softmax(outputs, dim=1)[:, 1].numpy())
        assert data['validation_scores'].tolist() == [0.1, 0.9, 0.2, 0.8]


def test_seed_ensemble_averages_runs(tmp_path):
    test_runs = [[0.1, 0.6, 0.5, 0.9, 0.2], [0.3, 0.2, 0.4, 0.8, 0.7]]
    for run_id, test_scores in enumerate(test_runs):
        _write_run(tmp_path, run_id, [0.1, 0.9, 0.2, 0.8], test_scores)
    # another configuration in the same directory must not be picked up
    _write_run(tmp_path, 0, [0.9, 0.1, 0.9, 0.1], [1, 1, 0, 0, 1], config='Configuration_000000')

    result = seed_ensemble(tmp_path)
    run_aucs = [roc_auc_score(TEST_LABELS, scores) for scores in test_runs]
    assert result['num_runs'] == 2
    assert result['best_epochs'] == [0, 1]
    assert result['test_auc_runs'] == pytest.approx(run_aucs)
    assert result['test_auc_mean'] == pytest.approx(np.mean(run_aucs))
    assert result['test_auc_ensemble_mean'] == pytest.approx(
        roc_auc_score(TEST_LABELS, np.mean(test_runs, axis=0)))
    assert result['validation_auc_ensemble_mean'] == pytest.approx(1.0)
    assert 0.0 <= result['test_auc_ensemble_rank'] <= 1.0


def test_seed_ensemble_rejects_mismatched_graph_ids(tmp_path):
    _write_run(tmp_path, 0, [0.1, 0.9, 0.2, 0.8], [0.1, 0.2, 0.3, 0.4, 0.5])
    _write_run(tmp_path, 1, [0.1, 0.9, 0.2, 0.8], [0.1, 0.2, 0.3, 0.4, 0.5],
               test_ids=TEST_IDS[::-1])
    with pytest.raises(ValueError, match='different test graph ids'):
        seed_ensemble(tmp_path)


def test_seed_ensemble_without_predictions_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match='save_best_predictions'):
        seed_ensemble(tmp_path)


def test_write_seed_ensemble_saves_json(tmp_path):
    models_dir = tmp_path / 'ogbg-molhiv' / 'Models'
    models_dir.mkdir(parents=True)
    _write_run(models_dir, 0, [0.1, 0.9, 0.2, 0.8], [0.1, 0.6, 0.5, 0.9, 0.2])
    result = write_seed_ensemble(tmp_path, 'ogbg-molhiv')
    with open(tmp_path / 'ogbg-molhiv' / 'seed_ensemble.json') as f:
        assert json.load(f) == result
