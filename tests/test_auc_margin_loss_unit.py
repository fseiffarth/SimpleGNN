"""Unit tests for ``CEAUCMarginLoss`` and its wiring in ``ModelConfiguration``.

The AUC-margin term is a training-only objective: validation/test loss and
ReduceLROnPlateau must keep scoring plain cross-entropy.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from simplegnn.framework.model_configuration import CEAUCMarginLoss, ModelConfiguration


def _bare_model_configuration(loss='CEAUCMarginLoss', auc_margin_loss=None, weighted_loss=False):
    config = ModelConfiguration.__new__(ModelConfiguration)
    run_config = {'task': 'graph_classification', 'weighted_loss': weighted_loss}
    if auc_margin_loss is not None:
        run_config['auc_margin_loss'] = auc_margin_loss
    config.para = SimpleNamespace(run_config=SimpleNamespace(loss=loss, config=run_config))
    config.graph_data = SimpleNamespace(y=torch.as_tensor([0, 0, 0, 1]), num_classes=2)
    config.training_data = np.asarray([0, 1, 2, 3])
    config.dtype = torch.double
    config.device = torch.device('cpu')
    config.class_weights = None
    config.set_loss_function()
    config.set_train_loss_function()
    return config


def _one_hot(labels):
    return torch.nn.functional.one_hot(torch.as_tensor(labels), num_classes=2).double()


def test_auc_term_matches_pairwise_square_loss():
    outputs = torch.tensor([[0.0, 2.0], [1.0, 0.0], [0.0, 0.5], [2.0, 0.0]], dtype=torch.double)
    labels = torch.tensor([1, 0, 1, 0])
    scores = torch.softmax(outputs, dim=1)[:, 1]
    positive, negative = scores[labels == 1], scores[labels == 0]
    pairs = [(1.0 - (p - n)) ** 2 for p in positive for n in negative]
    expected = torch.nn.functional.cross_entropy(outputs, labels) + 0.5 * torch.stack(pairs).mean()
    loss = CEAUCMarginLoss(auc_weight=0.5, margin=1.0)
    assert loss(outputs, labels).item() == pytest.approx(expected.item())
    # one-hot targets, as the graph-classification training loop passes them
    assert loss(outputs, _one_hot(labels)).item() == pytest.approx(expected.item())


def test_margin_clips_well_separated_pairs():
    outputs = torch.tensor([[-5.0, 5.0], [5.0, -5.0]], dtype=torch.double)
    labels = torch.tensor([1, 0])
    loss = CEAUCMarginLoss(margin=0.5)
    ce = torch.nn.functional.cross_entropy(outputs, labels)
    assert loss(outputs, labels).item() == pytest.approx(ce.item())


def test_single_class_batch_is_cross_entropy_only():
    outputs = torch.tensor([[1.0, 0.0], [0.3, 0.2]], dtype=torch.double, requires_grad=True)
    labels = torch.tensor([0, 0])
    loss = CEAUCMarginLoss()(outputs, labels)
    assert loss.item() == pytest.approx(torch.nn.functional.cross_entropy(outputs, labels).item())
    loss.backward()
    assert torch.isfinite(outputs.grad).all()


def test_train_criterion_is_auc_margin_and_evaluation_is_cross_entropy():
    config = _bare_model_configuration(auc_margin_loss={'weight': 2.0, 'margin': 0.7})
    assert isinstance(config.criterion, torch.nn.CrossEntropyLoss)
    assert isinstance(config.train_criterion, CEAUCMarginLoss)
    assert config.train_criterion.auc_weight == 2.0
    assert config.train_criterion.margin == 0.7


def test_auc_margin_defaults_follow_libauc():
    config = _bare_model_configuration()
    assert config.train_criterion.auc_weight == 1.0
    assert config.train_criterion.margin == 1.0


def test_weighted_auc_margin_weights_only_cross_entropy_term():
    config = _bare_model_configuration(weighted_loss=True)
    assert isinstance(config.train_criterion, CEAUCMarginLoss)
    assert config.train_criterion.cross_entropy.weight.tolist() == pytest.approx([0.25, 0.75])
    assert config.criterion.weight is None
