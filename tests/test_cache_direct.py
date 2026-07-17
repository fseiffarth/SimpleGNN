"""Weight-distribution caching in the ShareGNN invariant layers.

Building an invariant layer computes a weight distribution (which shared weight
each node pair indexes into) and caches it to disk under <data>/caches/<hash>.pt.
Rebuilding the same model must hit that cache and produce an identical
distribution -- a stale or mis-keyed cache would silently hand the layer the
wrong weight indices.

The original version of this file timed two builds and printed a speedup;
timing is too flaky to assert on in CI, so we assert on the cache artifact and
on the equality of the recomputed distributions instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CACHE_DIR = ROOT / "data" / "TUDatasets" / "caches"


def _invariant_layers(net):
    from simplegnn.models.ShareGNN.layers.inv_based_message_passing import (
        InvariantBasedMessagePassingLayer,
    )

    return [m for m in net.net_layers if isinstance(m, InvariantBasedMessagePassingLayer)]


@pytest.mark.integration
def test_invariant_layer_weight_distribution_is_cached_and_stable(share_gnn_setup):
    from simplegnn.models.model import GraphModel

    graph_data, para = share_gnn_setup

    first = GraphModel(graph_data=graph_data, para=para, seed=42, device="cpu")
    assert list(CACHE_DIR.glob("*.pt")), (
        "building the model should have written a weight-distribution cache"
    )

    # Rebuilding must reuse the cache and reproduce the same weight distribution.
    second = GraphModel(graph_data=graph_data, para=para, seed=42, device="cpu")

    first_layers = _invariant_layers(first)
    second_layers = _invariant_layers(second)
    assert first_layers, "fixture model should contain an invariant message-passing layer"
    assert len(first_layers) == len(second_layers)

    for first_layer, second_layer in zip(first_layers, second_layers):
        assert sum(first_layer.weight_num) == sum(second_layer.weight_num)
        for head_id in range(len(first_layer.layer.layer_heads)):
            assert torch.equal(
                getattr(first_layer, f"_pv_{head_id}"),
                getattr(second_layer, f"_pv_{head_id}"),
            )
