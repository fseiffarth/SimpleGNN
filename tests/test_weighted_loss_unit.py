"""Unit tests for the class-weighted training loss in
``simplegnn.framework.model_configuration``.

These pin down two weighted_loss bugs:
- class weights were computed per batch from ``torch.unique`` counts, so a
  binary batch with a single class broadcast its count into a flat 0.5/0.5
  weighting and a 3+ class batch missing a class crashed,
- the weighted loss replaced ``self.criterion``, so validation/test loss (and
  ReduceLROnPlateau) were scored with the last training batch's weights.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from simplegnn.framework.model_configuration import ModelConfiguration


def _bare_model_configuration(y, training_data, num_classes, task='graph_classification',
                              weighted_loss=True, loss='CrossEntropyLoss'):
    config = ModelConfiguration.__new__(ModelConfiguration)
    config.para = SimpleNamespace(
        run_config=SimpleNamespace(
            loss=loss,
            config={'task': task, 'weighted_loss': weighted_loss},
        ),
    )
    config.graph_data = SimpleNamespace(y=torch.as_tensor(y), num_classes=num_classes)
    config.training_data = np.asarray(training_data)
    config.dtype = torch.double
    config.device = torch.device('cpu')
    config.class_weights = None
    config.set_loss_function()
    return config


def test_class_weights_use_whole_training_split():
    # 1 positive in 4 training graphs; graph 4 is not in the training split
    config = _bare_model_configuration(y=[0, 0, 1, 0, 1], training_data=[0, 1, 2, 3], num_classes=2)
    weights = config.compute_class_weights()
    assert weights.tolist() == pytest.approx([0.25, 0.75])
    assert weights.dtype == torch.double


def test_class_weights_absent_class_keeps_its_index():
    # class 1 is missing: the old torch.unique counts broadcast to [n, n] (0.5/0.5)
    # for two classes and crashed for three
    config = _bare_model_configuration(y=[0, 2, 2, 2], training_data=[0, 1, 2, 3], num_classes=3)
    assert config.compute_class_weights().tolist() == pytest.approx([0.75, 1.0, 0.25])


def test_weighted_loss_leaves_evaluation_criterion_unweighted():
    config = _bare_model_configuration(y=[0, 0, 0, 1], training_data=[0, 1, 2, 3], num_classes=2)
    config.set_train_loss_function()
    assert config.criterion.weight is None
    assert config.train_criterion is not config.criterion
    assert config.train_criterion.weight.tolist() == pytest.approx([0.25, 0.75])


def test_weighted_loss_is_constant_across_batches():
    # a single-class batch must be weighted with the split weights, not 0.5/0.5
    config = _bare_model_configuration(y=[0, 0, 0, 1], training_data=[0, 1, 2, 3], num_classes=2)
    config.set_train_loss_function()
    outputs = torch.tensor([[0.3, -0.2], [1.0, 0.5]], dtype=torch.double)
    targets = torch.nn.functional.one_hot(torch.tensor([0, 0]), num_classes=2).double()
    expected = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.25, 0.75], dtype=torch.double))(outputs, targets)
    assert config.train_criterion(outputs, targets).item() == pytest.approx(expected.item())


def test_unweighted_train_criterion_is_evaluation_criterion():
    config = _bare_model_configuration(y=[0, 1], training_data=[0, 1], num_classes=2, weighted_loss=False)
    config.set_train_loss_function()
    assert config.train_criterion is config.criterion
    assert config.class_weights is None


def test_weighted_loss_rejects_regression():
    config = _bare_model_configuration(y=[0.5, 1.5], training_data=[0, 1], num_classes=1,
                                       task='graph_regression', loss='MAE')
    with pytest.raises(ValueError, match="weighted_loss"):
        config.set_train_loss_function()
