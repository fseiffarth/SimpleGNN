"""Exercise early validation through real optimizer steps, CSVs and selection."""
from types import SimpleNamespace
import shutil

import numpy as np
import pandas as pd
import pytest
import torch

from simplegnn.framework.model_configuration import ModelConfiguration
from simplegnn.framework.utils.evaluation import model_selection_evaluation


class TinyGraphModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.5)
        self.linear = torch.nn.Linear(2, 2)
        self.training_flags = []

    def forward(self, batch_data, pos):
        self.training_flags.append(self.training)
        return self.linear(self.dropout(batch_data.x))


def training_configuration(tmp_path, interval=2, epochs=1):
    torch.manual_seed(123)
    settings = {
        'task': 'graph_classification', 'device': 'cpu', 'precision': 'float',
        'paths': {'results': tmp_path}, 'evaluation_metric': 'roc_auc',
        'evaluation_type': 'roc_auc', 'best_model': True,
    }
    if interval:
        settings['step_validation'] = {'interval': interval, 'epochs': epochs}
    para = SimpleNamespace(
        db='toy', config_id='Configuration_000000', validation_id=0,
        n_epochs=2, n_val_runs=1, learning_rate=0.05,
        print_results=False, save_weights=False, save_prediction_values=False,
        count=0, run_config=SimpleNamespace(
            config=settings, task='graph_classification', loss='CrossEntropyLoss',
            optimizer='SGD', weight_decay=0.0, batch_size=2),
    )
    graph_data = SimpleNamespace(y=torch.arange(18) % 2, num_classes=2)
    config = ModelConfiguration(
        0, 0, graph_data, (np.arange(10), np.arange(10, 14), np.arange(14, 18)),
        42, para)
    config.net = TinyGraphModel()
    config.initialize_model = lambda **kwargs: None
    config.collect_network_info = lambda: {}
    config.write_network_markdown = lambda *args: None
    config._save_transfer_sidecar = lambda path: None
    (tmp_path / 'toy' / 'Results').mkdir(parents=True)
    config._assemble_share_gnn_batch = lambda ids: (
        SimpleNamespace(x=torch.ones(len(ids), 2)), ids)
    config.validation_states = {}

    def evaluate(graph_ids):
        assert not config.net.training
        # Even random work inside validation must not affect later dropout.
        torch.rand(1)
        config.validation_states[config._optimizer_step] = {
            k: v.clone() for k, v in config.net.state_dict().items()}
        labels = graph_data.y[graph_ids]
        sign = 1 if config._optimizer_step == 2 else -1
        scores = sign * (labels.float() * 2 - 1)
        return labels, torch.stack((-scores, scores), dim=1)

    config.evaluate_graph_task = evaluate
    return config


@pytest.mark.parametrize('interval,steps', [(2, [2, 4, 5, 10]), (5, [5, 10])])
def test_step_checkpoints_respect_epoch_limit_and_avoid_duplicate_endpoints(
        tmp_path, interval, steps):
    config = training_configuration(tmp_path, interval)
    config.train_configuration()
    path = next((tmp_path / 'toy' / 'Results').glob('*.csv'))
    results = pd.read_csv(path, sep=';')
    assert results['OptimizerStep'].tolist() == steps
    assert results['EpochProgress'].tolist() == pytest.approx(
        [step / 5 for step in steps])
    assert all(config.net.training_flags)
    assert config.net.training
    assert config._csv_buffer == []


def test_early_best_model_and_summaries_use_the_same_optimizer_checkpoint(tmp_path):
    config = training_configuration(tmp_path)
    config.train_configuration()
    results_dir = tmp_path / 'toy' / 'Results'
    csv_path = next(results_dir.glob('*.csv'))
    results = pd.read_csv(csv_path, sep=';')
    assert results.iloc[0]['ValidationAUC'] == 1.0
    assert results.iloc[0]['TestAUC'] == 1.0
    assert pd.isna(results.iloc[1]['TestAUC'])
    assert config.best_epoch['epoch'] == 0
    saved = torch.load(
        tmp_path / 'toy' / 'Models' / 'model_Configuration_000000_run_0_val_step_0.pt',
        weights_only=True)
    for name, value in saved.items():
        assert torch.equal(value, config.validation_states[2][name])
    model_selection_evaluation('toy', experiment_config=config.para.run_config.config)
    summary = pd.read_csv(tmp_path / 'toy' / 'summary.csv').iloc[0]
    assert summary['Optimizer Step Mean'] == 2
    assert summary['Epoch Progress Mean'] == 0.4
    assert summary['Test AUC Mean'] == 1.0

    shutil.copy(csv_path, results_dir / csv_path.name.replace(
        'Configuration_', 'Best_Configuration_'))
    model_selection_evaluation(
        'toy', evaluate_best_model=True, experiment_config=config.para.run_config.config)
    summary_mean = pd.read_csv(tmp_path / 'toy' / 'summary_best_mean.csv').iloc[0]
    assert summary_mean['Optimizer Step Mean'] == 2
    assert summary_mean['Epoch Progress Mean'] == 0.4

    model_selection_evaluation(
        'toy', evaluate_validation_only=True,
        experiment_config=config.para.run_config.config)
    validation_summary = pd.read_csv(tmp_path / 'toy' / 'summary_sota.csv').iloc[0]
    assert validation_summary['OptimizerStep'] == 2


def test_extra_validation_preserves_training_rng_loss_and_optimizer_updates(tmp_path):
    enabled = training_configuration(tmp_path / 'enabled')
    enabled.train_configuration()
    disabled = training_configuration(tmp_path / 'disabled', interval=0)
    disabled.train_configuration()
    for name, value in enabled.net.state_dict().items():
        assert torch.equal(value, disabled.net.state_dict()[name])
    enabled_csv = pd.read_csv(next((tmp_path / 'enabled' / 'toy' / 'Results').glob('*.csv')), sep=';')
    disabled_csv = pd.read_csv(next((tmp_path / 'disabled' / 'toy' / 'Results').glob('*.csv')), sep=';')
    epoch_rows = enabled_csv[enabled_csv.EpochProgress.isin([1, 2])]
    assert epoch_rows['EpochLoss (CrossEntropyLoss)'].tolist() == pytest.approx(
        disabled_csv['EpochLoss (CrossEntropyLoss)'].tolist())
    assert len(disabled_csv) == 2
    assert 'OptimizerStep' not in disabled_csv


@pytest.mark.integration
def test_step_validation_with_real_share_gnn(share_gnn_setup):
    graph_data, para = share_gnn_setup
    para.run_config.config.update({
        'step_validation': {'interval': 2, 'epochs': 1},
        'evaluation_metric': 'roc_auc', 'best_model': True,
    })
    para.n_epochs = 1
    config = ModelConfiguration(
        0, 0, graph_data, (np.arange(24), np.arange(24, 32), np.arange(32, 40)),
        42, para)
    config.train_configuration()
    results_dir = config.results_path / para.db / 'Results'
    csv_path = next(results_dir.glob('*Results_run_id_0_validation_step_0.csv'))
    results = pd.read_csv(csv_path, sep=';')
    assert results['OptimizerStep'].tolist() == [2, 3]
    assert results['EpochProgress'].tolist() == pytest.approx([2 / 3, 1])
    assert results['ValidationAUC'].between(0, 1).all()
    assert (config.results_path / para.db / 'Models').is_dir()
