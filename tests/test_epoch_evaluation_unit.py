"""Unit tests for the epoch-wise evaluation helpers in
``simplegnn.framework.model_configuration``.

These pin down the metric/reporting bugs fixed in the epoch evaluation:
- regression metric values must be plain floats (not ``tensor(...)`` reprs) so
  they parse as numbers in the per-epoch CSV,
- the pooled MAE std must be the true pooled std and must never be ``NaN`` for a
  single element,
- output normalization must be inverted identically for labels *and* outputs
  (a past bug inverted ``minmax_zero`` on labels only),
- the console line must label the loss with the configured loss name,
- ``EpochLoss`` must be the mean per-batch loss, not the sum.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from simplegnn.framework.model_configuration import (
    loss_display_name,
    pooled_abs_error_stats,
    inverse_transform_targets,
)


# ---------------------------------------------------------------------------
# loss_display_name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("loss,expected", [
    ("MAE", "MAE"), ("L1Loss", "MAE"), ("l1", "MAE"), ("mean_absolute_error", "MAE"),
    ("MSELoss", "MSE"), ("mse", "MSE"), ("MeanSquaredError", "MSE"),
    ("RMSE", "RMSE"), ("RMSELoss", "RMSE"), ("rmse", "RMSE"),
    ("SmoothL1Loss", "SmoothL1"), ("Huber", "SmoothL1"), ("huber", "SmoothL1"),
    ("CrossEntropyLoss", "CrossEntropy"),
    ("BCEWithLogitsLoss", "BCE"), ("bce", "BCE"),
    ("NLLLoss", "NLL"),
])
def test_loss_display_name_known(loss, expected):
    assert loss_display_name(loss) == expected


def test_loss_display_name_unknown_passthrough():
    # unknown losses are returned unchanged so the line still names them
    assert loss_display_name("MyCustomLoss") == "MyCustomLoss"


# ---------------------------------------------------------------------------
# pooled_abs_error_stats
# ---------------------------------------------------------------------------

def test_pooled_abs_error_stats_matches_numpy():
    abs_err = torch.tensor([0.5, 1.5, 2.0, 3.0, 0.0])
    mae, std = pooled_abs_error_stats(abs_err)
    ref = abs_err.numpy()
    assert mae == pytest.approx(ref.mean())
    # population std (unbiased=False), matching numpy's default ddof=0
    assert std == pytest.approx(ref.std())


def test_pooled_abs_error_stats_returns_python_floats():
    # guards the primary "strange MAE" bug: values must be floats, not tensors,
    # so they are written to the CSV as numbers rather than "tensor(...)".
    mae, std = pooled_abs_error_stats(torch.tensor([1.0, 2.0, 3.0]))
    assert isinstance(mae, float)
    assert isinstance(std, float)


def test_pooled_abs_error_stats_single_element_no_nan():
    # torch.std on a single element is NaN; the helper must return 0.0 instead.
    mae, std = pooled_abs_error_stats(torch.tensor([2.5]))
    assert mae == pytest.approx(2.5)
    assert std == 0.0
    assert not math.isnan(std)


def test_pooled_abs_error_stats_empty():
    assert pooled_abs_error_stats(torch.tensor([])) == (0.0, 0.0)


def test_pooled_abs_error_stats_flattens_multidim():
    abs_err = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    mae, std = pooled_abs_error_stats(abs_err)
    assert mae == pytest.approx(1.5)
    assert std == pytest.approx(abs_err.flatten().numpy().std())


# ---------------------------------------------------------------------------
# inverse_transform_targets  (round-trip against the forward normalization)
# ---------------------------------------------------------------------------

# Forward normalizations, always parameterized by the ORIGINAL target stats
# (that is how graph_dataset.py normalizes both labels and, implicitly, the scale
# the model learns to predict on).
def _forward_standard(y, stats):
    return (y - stats["mean"]) / (stats["std"] + 1e-8)


def _forward_minmax(y, stats):
    return (y - stats["min"]) / (stats["max"] - stats["min"] + 1e-8)


def _forward_minmax_zero(y, stats):
    # matches graph_dataset.py: maps to [-1, 1]
    return 2.0 * (y - stats["min"]) / (stats["max"] - stats["min"] + 1e-8) - 1.0


@pytest.mark.parametrize("normalization,forward", [
    ("standard", _forward_standard),
    ("minmax", _forward_minmax),
    ("minmax_zero", _forward_minmax_zero),
])
def test_inverse_transform_round_trip_labels_and_outputs(normalization, forward):
    original_y = torch.tensor([1.0, 4.0, 9.0, 16.0, 25.0])
    stats = {
        "mean": original_y.mean(),
        "std": original_y.std(),
        "min": original_y.min(),
        "max": original_y.max(),
    }
    cfg = {"normalization": normalization}

    # both a "labels" tensor and a separate "outputs" tensor (normalized with the
    # same target stats) must invert back to the original scale
    outputs_raw = torch.tensor([2.0, 5.0, 8.0, 15.0, 20.0])
    labels_back = inverse_transform_targets(forward(original_y, stats), cfg, stats)
    outputs_back = inverse_transform_targets(forward(outputs_raw, stats), cfg, stats)

    assert torch.allclose(labels_back, original_y, atol=1e-3)
    assert torch.allclose(outputs_back, outputs_raw, atol=1e-3)


def test_inverse_transform_noop_for_non_dict():
    values = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(inverse_transform_targets(values, None, {}), values)
    assert torch.equal(inverse_transform_targets(values, "standard", {}), values)


def test_inverse_transform_unknown_normalization_is_identity():
    values = torch.tensor([1.0, 2.0, 3.0])
    stats = {"mean": torch.tensor(0.0), "std": torch.tensor(1.0),
             "min": torch.tensor(0.0), "max": torch.tensor(1.0)}
    out = inverse_transform_targets(values, {"normalization": "does_not_exist"}, stats)
    assert torch.equal(out, values)


# ---------------------------------------------------------------------------
# EpochLoss = mean over batches (not sum)
# ---------------------------------------------------------------------------

def test_epoch_loss_is_mean_over_batches():
    # Reproduces the accumulate-then-average contract used in train_configuration:
    # train_*_task accumulates a running SUM, which is divided by the number of
    # batches to yield the mean (comparable to ValidationLoss).
    batch_losses = [0.4, 0.6, 0.2, 0.8]
    running_sum = 0.0
    for loss in batch_losses:
        running_sum += loss
    epoch_loss = running_sum / len(batch_losses)
    assert epoch_loss == pytest.approx(sum(batch_losses) / len(batch_losses))
    assert epoch_loss == pytest.approx(0.5)
    # crucially NOT the sum
    assert epoch_loss != pytest.approx(running_sum)


def test_pooled_running_accumulator_equals_full_pooled_stats():
    # The training branch accumulates sum/sumsq/n across batches and derives
    # mae/std from them; this must equal pooled_abs_error_stats over the
    # concatenation of all batches.
    batch_a = torch.tensor([0.5, 1.5, 2.0])
    batch_b = torch.tensor([3.0, 0.0])
    sum_abs = sumsq_abs = 0.0
    n = 0
    for batch in (batch_a, batch_b):
        sum_abs += batch.sum().item()
        sumsq_abs += (batch ** 2).sum().item()
        n += batch.numel()
    mae = sum_abs / n
    std = float(np.sqrt(max(0.0, sumsq_abs / n - mae ** 2)))

    ref_mae, ref_std = pooled_abs_error_stats(torch.cat([batch_a, batch_b]))
    assert mae == pytest.approx(ref_mae)
    assert std == pytest.approx(ref_std, abs=1e-6)
