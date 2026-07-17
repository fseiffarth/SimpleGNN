"""Degree normalization of the invariant convolution.

The optional per-layer option ``degree_normalization: 'row' | 'symmetric'``
rescales each aggregation weight by the nonzero pattern counts of the head's
weight matrix ('row': mean aggregation, 'symmetric': GCN-style
1/sqrt(deg_i * deg_j)). These tests pin:

1. the semantics against a dense reference computed from the unnormalized
   weight matrix, and
2. the equivalence of all four execution paths (per-graph dense/sparse,
   batched dense/sparse), so the config switches stay interchangeable.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _build_net(graph_data, para, seed=42):
    from simplegnn.models.model import GraphModel

    return GraphModel(graph_data=graph_data, para=para, seed=seed, device="cpu")


def _batched_input(graph_data, positions):
    slices = graph_data.slices["x"]
    x = torch.cat([graph_data.x[int(slices[p]):int(slices[p + 1])] for p in positions])
    return SimpleNamespace(x=x)


def _set_degree_normalization(para, norm):
    from simplegnn.models.layers.utils.layer_types import LayerTypes

    for layer in para.layers:
        if layer.layer_type == LayerTypes.INVARIANT_BASED_CONVOLUTION.value:
            layer.layer_dict["degree_normalization"] = norm


def _first_conv(net):
    from simplegnn.models.ShareGNN.layers.inv_based_message_passing import (
        InvariantBasedMessagePassingLayer,
    )

    return next(l for l in net.net_layers if isinstance(l, InvariantBasedMessagePassingLayer))


@pytest.mark.integration
@pytest.mark.parametrize("norm", ["row", "symmetric"])
def test_degree_normalization_semantics(share_gnn_setup, norm):
    """With all shared weights set to 1, the normalized weight matrix must be
    the pattern adjacency scaled by 1/deg ('row') resp. 1/sqrt(deg_i * deg_j)
    ('symmetric'), where the degrees are the pattern counts per head."""
    graph_data, para = share_gnn_setup

    net_raw = _build_net(graph_data, para)
    conv_raw = _first_conv(net_raw)
    with torch.no_grad():
        conv_raw.Param_W.fill_(1.0)
    conv_raw.set_weights(0)
    W_raw = conv_raw.current_W.clone()  # (H, N, N), binary pattern

    _set_degree_normalization(para, norm)
    try:
        net_norm = _build_net(graph_data, para)
        conv_norm = _first_conv(net_norm)
        with torch.no_grad():
            conv_norm.Param_W.fill_(1.0)
        conv_norm.set_weights(0)
        W_norm = conv_norm.current_W

        row_counts = (W_raw != 0).sum(dim=-1, keepdim=True).clamp(min=1).to(W_raw.dtype)
        if norm == "row":
            expected = W_raw / row_counts
        else:
            col_counts = (W_raw != 0).sum(dim=-2, keepdim=True).clamp(min=1).to(W_raw.dtype)
            expected = W_raw / torch.sqrt(row_counts * col_counts)
        torch.testing.assert_close(W_norm, expected, rtol=1e-9, atol=1e-9)
    finally:
        _set_degree_normalization(para, None)


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["dense", "sparse"])
@pytest.mark.parametrize("norm", ["row", "symmetric"])
def test_degree_normalization_batched_matches_per_graph(share_gnn_setup, mode, norm):
    """All four execution paths (per-graph and batched, dense and sparse) must
    agree with degree normalization enabled."""
    graph_data, para = share_gnn_setup
    para.run_config.config["share_gnn_forward"] = {"batched": True, "mode": mode}
    _set_degree_normalization(para, norm)
    try:
        net = _build_net(graph_data, para)
        net.eval()

        positions = [3, 3, 0, 7, 5]
        with torch.no_grad():
            per_graph = torch.stack([net(graph_data[p], pos=p) for p in positions])
            batched = net(_batched_input(graph_data, positions), pos=positions)

        torch.testing.assert_close(batched, per_graph, rtol=1e-9, atol=1e-9)
    finally:
        _set_degree_normalization(para, None)


@pytest.mark.integration
def test_degree_normalization_gradients_flow(share_gnn_setup):
    """The normalization factors are constants: gradients must still reach the
    shared weights, and must differ from the unnormalized gradients."""
    graph_data, para = share_gnn_setup
    criterion = torch.nn.CrossEntropyLoss()
    positions = list(range(4))
    labels = torch.stack([graph_data[p].y for p in positions]).squeeze()

    def _grads(norm):
        _set_degree_normalization(para, norm)
        try:
            net = _build_net(graph_data, para)
            outputs = torch.stack([net(graph_data[p], pos=p) for p in positions])
            criterion(outputs, labels).backward()
            conv = _first_conv(net)
            assert conv.Param_W.grad is not None
            return conv.Param_W.grad.clone()
        finally:
            _set_degree_normalization(para, None)

    grad_norm = _grads("symmetric")
    grad_raw = _grads(None)
    assert grad_norm.abs().sum() > 0
    assert not torch.allclose(grad_norm, grad_raw)


def test_invalid_degree_normalization_rejected(share_gnn_setup):
    graph_data, para = share_gnn_setup
    _set_degree_normalization(para, "sym")
    try:
        with pytest.raises(ValueError, match="degree_normalization"):
            _build_net(graph_data, para)
    finally:
        _set_degree_normalization(para, None)
