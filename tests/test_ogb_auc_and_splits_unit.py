"""Unit tests for the OGB experiment support added alongside ``experiments/ogb``.

These pin down the two things that make an OGB ``rocauc`` run meaningful:

- ``binary_roc_auc`` must rank by the positive-class *score*, not by the hard
  ``argmax`` prediction. Ranking by 0/1 labels collapses the ROC curve to a
  single operating point, which silently reports balanced accuracy where the
  OGB leaderboard reports AUC.
- ``simplegnn.utils.ogb_splits`` must emit the framework's split JSON format,
  and the shipped ``ogbg-*_splits.json`` files must be disjoint full partitions
  of their dataset (they are OGB's official scaffold splits, so a mangled file
  would silently produce unreproducible numbers).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sklearn_metrics = pytest.importorskip("sklearn.metrics")

from simplegnn.framework.model_configuration import binary_roc_auc
from simplegnn.utils import ogb_splits as ogb_splits_module

SPLIT_DIR = Path(__file__).resolve().parents[1] / "src" / "simplegnn" / "datasets" / "splits" / "fixed"


def test_binary_roc_auc_uses_scores_not_argmax():
    """A perfectly ranked but badly calibrated model still scores AUC 1.0.

    Every sample is predicted negative by ``argmax`` (class-0 logit always
    wins), yet the positive class is ranked strictly above the negative one.
    The argmax-based computation this replaced would score 0.5 here.
    """
    outputs = torch.tensor([
        [2.0, 1.0],   # label 0, positive-score margin -1.0
        [2.0, 1.5],   # label 0, margin -0.5
        [2.0, 1.8],   # label 1, margin -0.2
        [2.0, 1.9],   # label 1, margin -0.1
    ])
    labels = torch.tensor([0, 0, 1, 1])
    assert binary_roc_auc(outputs, labels) == pytest.approx(1.0)


def test_binary_roc_auc_matches_sklearn_on_softmax_scores():
    torch.manual_seed(0)
    outputs = torch.randn(64, 2)
    labels = (torch.rand(64) > 0.5).long()
    expected = sklearn_metrics.roc_auc_score(
        labels.numpy(), torch.softmax(outputs, dim=1)[:, 1].numpy()
    )
    assert binary_roc_auc(outputs, labels) == pytest.approx(expected)


def test_binary_roc_auc_accepts_one_hot_labels():
    torch.manual_seed(1)
    outputs = torch.randn(32, 2)
    labels = (torch.rand(32) > 0.5).long()
    one_hot = torch.nn.functional.one_hot(labels, num_classes=2)
    assert binary_roc_auc(outputs, one_hot) == pytest.approx(binary_roc_auc(outputs, labels))


def test_binary_roc_auc_single_class_returns_neutral():
    """``roc_auc_score`` raises when a batch holds one class; we return 0.5."""
    outputs = torch.randn(8, 2)
    assert binary_roc_auc(outputs, torch.zeros(8, dtype=torch.long)) == 0.5
    assert binary_roc_auc(outputs, torch.ones(8, dtype=torch.long)) == 0.5


def test_binary_roc_auc_single_logit_column():
    """A 1-column output is a raw logit; the sigmoid is monotone so AUC is unchanged."""
    logits = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]])
    labels = torch.tensor([0, 0, 1, 1])
    assert binary_roc_auc(logits, labels) == pytest.approx(1.0)


def test_supported_datasets_are_single_task_only():
    """Multi-task ogbg-mol* datasets must stay out: their targets are NaN-masked."""
    multi_task = {"ogbg-moltox21", "ogbg-molsider", "ogbg-molclintox",
                  "ogbg-moltoxcast", "ogbg-molmuv", "ogbg-molpcba", "ogbg-molchembl"}
    assert not multi_task & set(ogb_splits_module.SUPPORTED_DATASETS)
    assert set(ogb_splits_module.SUPPORTED_DATASETS) == (
        set(ogb_splits_module.SINGLE_TASK_CLASSIFICATION)
        | set(ogb_splits_module.SINGLE_TASK_REGRESSION)
    )


@pytest.mark.parametrize("db_name", ogb_splits_module.SUPPORTED_DATASETS)
def test_shipped_split_file_is_a_disjoint_full_partition(db_name):
    split_file = SPLIT_DIR / f"{db_name}_splits.json"
    if not split_file.is_file():
        pytest.skip(f"{split_file.name} not generated; run `python -m simplegnn.utils.ogb_splits --all`")

    folds = json.load(open(split_file))
    assert len(folds) == 1, "OGB ships exactly one official split per dataset"
    fold = folds[0]
    train = fold["model_selection"][0]["train"]
    validation = fold["model_selection"][0]["validation"]
    test = fold["test"]

    assert not set(train) & set(validation)
    assert not set(train) & set(test)
    assert not set(validation) & set(test)
    # the three parts must cover 0..N-1 exactly, with no gaps or duplicates
    assert sorted(train + validation + test) == list(range(len(train) + len(validation) + len(test)))
