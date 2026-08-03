"""Unit tests for the BREC expressiveness benchmark (specs/21).

BREC's RPC protocol is a statistical decision rule, not an averaged metric, so
the parts worth pinning down are the ones a silent mistake would corrupt
without any run failing:

- the **graph layout**: pairs are contiguous blocks with the two graphs
  interleaved (``A0 B0 A1 B1 ...``). Get this wrong and the T-squared test
  compares a graph against its own relabelings, which never separates anything.
- the **T-squared statistic**, including the pseudo-inverse (the covariance is
  singular whenever a model maps all relabelings to identical embeddings).
- the **decision rule**, whose second condition (the pair's statistic must
  differ from the isomorphic control's) is what stops a model that emits a huge
  statistic for everything from scoring 400/400.
- the placeholder **split file**, which must still be a valid disjoint
  partition of the dataset.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from simplegnn.datasets.graph_dataset_preprocessing import (BREC_NUM_IDS, BREC_NUM_PAIRS,
                                                            BREC_NUM_RELABEL, BREC_PARTS,
                                                            brec_graph_index, parse_brec_name)
from simplegnn.utils.brec_splits import build_brec_splits

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_main_brec():
    """Import ``experiments/brec/main_brec.py`` (not an installed module)."""
    path = REPO_ROOT / 'experiments' / 'brec' / 'main_brec.py'
    spec = importlib.util.spec_from_file_location('main_brec', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['main_brec'] = module
    spec.loader.exec_module(module)
    return module


main_brec = _load_main_brec()


# --- dataset naming and layout ------------------------------------------------

def test_parse_brec_name_official_and_reduced():
    assert parse_brec_name('BREC') == BREC_NUM_RELABEL
    assert parse_brec_name('brec') == BREC_NUM_RELABEL
    assert parse_brec_name('BREC-r4') == 4
    assert parse_brec_name('BREC-r32') == BREC_NUM_RELABEL


@pytest.mark.parametrize('name', ['BREC-r0', 'BREC-r33', 'BREC-r', 'BRECr4', 'ZINC', 'BREC-rx'])
def test_parse_brec_name_rejects_bad_names(name):
    with pytest.raises(ValueError):
        parse_brec_name(name)


def test_graph_index_layout_is_interleaved_blocks():
    # pair 0 occupies the first 2*R positions, with the two graphs alternating
    assert [brec_graph_index(0, r, s, 4) for r in range(4) for s in (0, 1)] == list(range(8))
    assert brec_graph_index(1, 0, 0, 4) == 8
    # the A-side of every relabeling is even, the B-side odd -> pred[0::2] /
    # pred[1::2] splits a contiguous block into the two graphs
    assert all(brec_graph_index(3, r, 0, 32) % 2 == 0 for r in range(32))
    assert all(brec_graph_index(3, r, 1, 32) % 2 == 1 for r in range(32))
    # official dataset size
    assert brec_graph_index(BREC_NUM_IDS - 1, BREC_NUM_RELABEL - 1, 1) == BREC_NUM_IDS * BREC_NUM_RELABEL * 2 - 1


def test_parts_cover_all_pairs_without_overlap():
    covered = [pair_id for _, start, end in BREC_PARTS for pair_id in range(start, end)]
    assert covered == list(range(BREC_NUM_PAIRS))


def test_reliability_control_id_offset():
    # the control for pair i is pair i + 400, so its graphs come from the
    # dataset's second half
    evaluator_ids = lambda pair_id, num_relabel: list(
        range(pair_id * 2 * num_relabel, (pair_id + 1) * 2 * num_relabel))
    assert evaluator_ids(BREC_NUM_PAIRS, 4)[0] == BREC_NUM_PAIRS * 8
    assert max(evaluator_ids(BREC_NUM_IDS - 1, 4)) == BREC_NUM_IDS * 8 - 1


# --- split file ---------------------------------------------------------------

@pytest.mark.parametrize('num_relabel', [1, 4, 32])
def test_build_brec_splits_is_a_disjoint_full_partition(num_relabel):
    folds = build_brec_splits(num_relabel)
    assert len(folds) == 1
    fold = folds[0]
    train = fold['model_selection'][0]['train']
    validation = fold['model_selection'][0]['validation']
    test = fold['test']
    num_graphs = BREC_NUM_IDS * 2 * num_relabel
    assert set(train) | set(validation) | set(test) == set(range(num_graphs))
    assert not set(train) & set(validation)
    assert not set(train) & set(test)
    assert not set(validation) & set(test)
    # validation/test hold exactly one pair id each
    assert len(validation) == 2 * num_relabel
    assert len(test) == 2 * num_relabel


def test_shipped_split_files_match_the_generator():
    import json
    for name in ('BREC', 'BREC-r4'):
        path = REPO_ROOT / 'src' / 'simplegnn' / 'datasets' / 'splits' / 'fixed' / f'{name}_splits.json'
        assert path.is_file(), f'missing split file {path}'
        with open(path) as f:
            shipped = json.load(f)
        assert shipped == build_brec_splits(parse_brec_name(name))


# --- T-squared statistic ------------------------------------------------------

def test_t2_statistic_is_zero_for_identical_embeddings():
    # a model that maps both graphs of a pair to the same embedding: D == 0, so
    # the covariance is singular and only the pseudo-inverse keeps this finite
    embeddings = torch.arange(8, dtype=torch.double).reshape(4, 2).repeat_interleave(2, dim=0)
    assert float(main_brec.t2_statistic(embeddings)) == pytest.approx(0.0)


def test_t2_statistic_matches_manual_computation():
    torch.manual_seed(0)
    num_relabel, output_dim = 6, 3
    embeddings = torch.randn(2 * num_relabel, output_dim, dtype=torch.double)
    X = embeddings[0::2].T
    Y = embeddings[1::2].T
    D = X - Y
    mean = D.mean(dim=1, keepdim=True)
    expected = mean.T @ torch.linalg.pinv(torch.cov(D)) @ mean
    assert float(main_brec.t2_statistic(embeddings)) == pytest.approx(float(expected))


def test_t2_statistic_clears_threshold_for_a_noisy_separation():
    # the realistic "distinguished" signal: the two graphs are separated, and
    # each relabeling differs slightly (floating-point noise)
    torch.manual_seed(1)
    num_relabel, output_dim = 32, 16
    embeddings = torch.empty(2 * num_relabel, output_dim, dtype=torch.double)
    embeddings[0::2] = torch.randn(num_relabel, output_dim, dtype=torch.double) * 0.01 + 1.0
    embeddings[1::2] = torch.randn(num_relabel, output_dim, dtype=torch.double) * 0.01 - 1.0
    assert float(main_brec.t2_statistic(embeddings)) > main_brec.THRESHOLD


def test_t2_statistic_collapses_to_zero_on_a_perfectly_consistent_separation():
    # A permutation-equivariant model in double precision can map every
    # relabeling to bit-identical embeddings. Then cov(D) == 0, pinv(0) == 0 and
    # the statistic is 0 however far apart the two graphs are -- a false
    # negative, not a failure to distinguish. The ridge recovers it.
    num_relabel, output_dim = 32, 16
    embeddings = torch.empty(2 * num_relabel, output_dim, dtype=torch.double)
    embeddings[0::2] = 1.0
    embeddings[1::2] = -1.0
    assert float(main_brec.t2_statistic(embeddings)) == pytest.approx(0.0)
    assert main_brec.is_degenerate(embeddings)
    assert float(main_brec.t2_statistic(embeddings, epsilon=main_brec.EPSILON_MATRIX)) > main_brec.THRESHOLD


def test_is_degenerate_false_when_embeddings_agree_or_vary():
    num_relabel, output_dim = 8, 4
    # identical embeddings for both graphs: not separated at all
    same = torch.ones(2 * num_relabel, output_dim, dtype=torch.double)
    assert not main_brec.is_degenerate(same)
    # separated with variation across relabelings: the normal case
    torch.manual_seed(2)
    varying = torch.empty(2 * num_relabel, output_dim, dtype=torch.double)
    varying[0::2] = torch.randn(num_relabel, output_dim, dtype=torch.double) + 1.0
    varying[1::2] = torch.randn(num_relabel, output_dim, dtype=torch.double) - 1.0
    assert not main_brec.is_degenerate(varying)


# --- decision rule ------------------------------------------------------------

def test_decide_requires_threshold_and_a_difference_from_the_control():
    t = lambda value: torch.tensor(value, dtype=torch.double)
    # clean win: above threshold, control below it
    assert main_brec.decide(t(50.0), t(0.1)) == (True, True)
    # below threshold -> not distinguished
    assert main_brec.decide(t(1.0), t(0.1)) == (False, True)
    # control also above threshold -> unreliable, but the pair still counts if
    # the two statistics differ (this is the reference implementation's rule)
    assert main_brec.decide(t(50.0), t(40.0)) == (True, False)
    # a model that emits the same large statistic for the isomorphic control
    # must not score the pair
    assert main_brec.decide(t(50.0), t(50.0)) == (False, False)


def test_decide_threshold_is_the_reference_value():
    assert main_brec.THRESHOLD == 5.0
    assert main_brec.EPOCH == 50
    assert main_brec.BATCH_SIZE == 16
    assert main_brec.MARGIN == 0.0
    assert main_brec.LEARNING_RATE == pytest.approx(1e-3)
    assert main_brec.WEIGHT_DECAY == pytest.approx(1e-5)


# --- pair selection -----------------------------------------------------------

def test_selected_pairs_defaults_to_the_whole_benchmark():
    selection = main_brec.selected_pairs(None, None)
    assert [name for name, _ in selection] == [name for name, _, _ in BREC_PARTS]
    assert sum(len(ids) for _, ids in selection) == BREC_NUM_PAIRS


def test_selected_pairs_restricts_parts_and_counts():
    selection = main_brec.selected_pairs('Basic,CFI', 3)
    assert selection == [('Basic', [0, 1, 2]), ('CFI', [260, 261, 262])]


def test_selected_pairs_rejects_unknown_part():
    import click
    with pytest.raises(click.BadParameter):
        main_brec.selected_pairs('Nonexistent', None)
