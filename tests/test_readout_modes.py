"""Structured (non-flattening) ShareGNN readouts.

``invariant_based_aggregation`` folds its H heads into the feature dimension by
default, so the readout after it has to learn a free weight per
(head, feature, output) triple -- on ZINC that single layer holds ~80% of the
model's parameters (specs/09-zinc-readout-and-architecture-improvements.md).

With ``flatten: False`` the layer keeps the (H, F) structure of the graph
embedding, which makes two cheaper readouts reachable: a ``channel_wise``
linear (one F -> F' matrix per head) and a ``factorized`` linear (a CP
decomposition of the (H, F, F') tensor). These tests pin the parameter counts
and check that both readouts behave identically in the per-graph and the
batched forward.
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


def _layer(net, name_fragment):
    return next(l for l in net.net_layers if name_fragment in l.name)


@pytest.mark.integration
def test_unflattened_aggregation_keeps_head_axis(share_gnn_setup_factory):
    """flatten: False reports the heads as channels instead of folding them
    into out_features, and emits (H, 1, F) per graph / (B, H, F) batched."""
    graph_data, para = share_gnn_setup_factory("models_ShareGNN_channelwise_readout.yml")
    net = _build_net(graph_data, para)
    net.eval()

    aggregation = _layer(net, "Aggregation")
    assert aggregation.out_channels == 5  # 2 + 3 heads
    assert aggregation.out_features == 8  # unchanged feature dimension

    shapes = []
    handle = aggregation.register_forward_hook(lambda _m, _i, out: shapes.append(tuple(out.shape)))
    try:
        with torch.no_grad():
            net(graph_data[0], pos=0)
            net(_batched_input(graph_data, [0, 1, 2]), pos=[0, 1, 2])
    finally:
        handle.remove()

    assert shapes == [(5, 1, 8), (3, 5, 8)]


@pytest.mark.integration
@pytest.mark.parametrize(
    "models_file,readout_params",
    [
        # channel_wise: one (F=8 -> F'=4) matrix plus a bias per head (H=5)
        ("models_ShareGNN_channelwise_readout.yml", 5 * 8 * 4 + 5 * 4),
        # factorized (rank 3): A (H=5, R) + B (F=8, R) + C (R, F'=4) + bias F'
        ("models_ShareGNN_factorized_readout.yml", 3 * (5 + 8 + 4) + 4),
    ],
)
def test_readout_parameter_count(share_gnn_setup_factory, models_file, readout_params):
    """The structured readouts must cost exactly what their factorization says
    -- this is the whole point of the change, so it is worth pinning."""
    graph_data, para = share_gnn_setup_factory(models_file)
    net = _build_net(graph_data, para)

    readout = net.net_layers[4]  # conv, linear, layer_norm, aggregation, readout
    assert sum(p.numel() for p in readout.parameters()) == readout_params


@pytest.mark.integration
@pytest.mark.parametrize(
    "models_file",
    ["models_ShareGNN_channelwise_readout.yml", "models_ShareGNN_factorized_readout.yml"],
)
def test_structured_readout_batched_matches_per_graph(share_gnn_setup_factory, models_file):
    """The batched forward keeps the batch on the leading axis while the
    per-graph forward keeps a singleton node axis, so both readouts have two
    shape paths. They must agree."""
    graph_data, para = share_gnn_setup_factory(models_file)
    para.run_config.config["share_gnn_forward"] = {"batched": True}
    net = _build_net(graph_data, para)
    net.eval()

    positions = [3, 3, 0, 7, 3]
    with torch.no_grad():
        per_graph = torch.stack([net(graph_data[p], pos=p) for p in positions])
        batched = net(_batched_input(graph_data, positions), pos=positions)

    assert per_graph.shape == (len(positions), 2)
    assert batched.shape == per_graph.shape
    torch.testing.assert_close(batched, per_graph, rtol=1e-9, atol=1e-9)


@pytest.mark.integration
@pytest.mark.parametrize(
    "models_file",
    ["models_ShareGNN_channelwise_readout.yml", "models_ShareGNN_factorized_readout.yml"],
)
def test_structured_readout_gradients(share_gnn_setup_factory, models_file):
    """Every readout factor must receive a gradient -- a CP factor that is
    silently detached would train as a random projection."""
    graph_data, para = share_gnn_setup_factory(models_file)
    net = _build_net(graph_data, para)

    positions = list(range(8))
    labels = torch.stack([graph_data[p].y for p in positions]).squeeze()
    outputs = torch.stack([net(graph_data[p], pos=p) for p in positions])
    torch.nn.CrossEntropyLoss()(outputs, labels).backward()

    readout = net.net_layers[4]
    for name, param in readout.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.any(param.grad != 0), f"{name} received an all-zero gradient"
