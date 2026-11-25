"""``reduce: mean`` of ``invariant_based_aggregation``.

The mean divides each graph's label-weighted node sum by its node count before
the bias. The per-graph and the batched forward implement it separately, so
both paths are pinned against the sum.
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


def _batched_input(graph_data, positions):
    slices = graph_data.slices["x"]
    x = torch.cat([graph_data.x[int(slices[p]):int(slices[p + 1])] for p in positions])
    return SimpleNamespace(x=x)


def _aggregation_outputs(net, aggregation, graph_data, positions):
    outputs = []
    handle = aggregation.register_forward_hook(lambda _m, _i, out: outputs.append(out.detach().clone()))
    try:
        with torch.no_grad():
            for p in positions:
                net(graph_data[p], pos=p)
            net(_batched_input(graph_data, positions), pos=positions)
    finally:
        handle.remove()
    per_graph = torch.cat(outputs[:len(positions)])
    return per_graph, outputs[-1]


@pytest.mark.integration
def test_mean_pooling_divides_sum_by_node_count(share_gnn_setup_factory):
    from simplegnn.models.model import GraphModel

    graph_data, para = share_gnn_setup_factory("models_ShareGNN.yml")
    para.run_config.config["share_gnn_forward"] = {"batched": True}
    net = GraphModel(graph_data=graph_data, para=para, seed=42, device="cpu")
    net.eval()
    aggregation = next(l for l in net.net_layers if "Aggregation" in l.name)
    aggregation.activation = torch.nn.Identity()
    bias = aggregation.Param_b.detach().flatten() if aggregation.bias else 0.0

    positions = [3, 0, 7, 3]
    counts = torch.tensor([int(graph_data.slices["x"][p + 1] - graph_data.slices["x"][p]) for p in positions],
                          dtype=torch.double).view(-1, 1)
    assert aggregation.reduce == "sum"
    sum_graph, sum_batched = _aggregation_outputs(net, aggregation, graph_data, positions)
    aggregation.reduce = "mean"
    mean_graph, mean_batched = _aggregation_outputs(net, aggregation, graph_data, positions)

    expected = (sum_graph.double() - bias) / counts + bias
    torch.testing.assert_close(mean_graph.double(), expected, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(mean_batched, mean_graph, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(sum_batched, sum_graph, rtol=1e-9, atol=1e-9)

