"""Equivalence tests: batched vs per-graph ShareGNN forward on real MUTAG.

The batched forward (specs/07-batched-share-gnn-forward.md) processes all
graphs of a batch jointly via one block-diagonal sparse matmul per invariant
layer. These tests pin its contract: for the same weights it must produce the
same outputs and the same gradients as the classical per-graph forward, so the
two code paths remain interchangeable via the
``share_gnn_forward: {batched: true}`` config switch.
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


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["auto", "dense", "sparse"])
def test_batched_forward_matches_per_graph(share_gnn_setup, mode):
    """Both batched implementations -- the padded dense matmul and the
    block-diagonal sparse mm, selected by share_gnn_forward.mode -- must
    reproduce the per-graph forward."""
    graph_data, para = share_gnn_setup
    para.run_config.config["share_gnn_forward"] = {"batched": True, "mode": mode}
    net = _build_net(graph_data, para)
    net.eval()

    positions = list(range(16))
    with torch.no_grad():
        per_graph = torch.stack([net(graph_data[p], pos=p) for p in positions])
        batched = net(_batched_input(graph_data, positions), pos=positions)

    assert batched.shape == per_graph.shape
    torch.testing.assert_close(batched, per_graph, rtol=1e-9, atol=1e-9)


@pytest.mark.integration
def test_batched_forward_handles_duplicate_graph_ids(share_gnn_setup):
    """random/balanced sampling can repeat a graph inside one batch."""
    graph_data, para = share_gnn_setup
    net = _build_net(graph_data, para)
    net.eval()

    positions = [3, 3, 0, 7, 3]
    with torch.no_grad():
        per_graph = torch.stack([net(graph_data[p], pos=p) for p in positions])
        batched = net(_batched_input(graph_data, positions), pos=positions)

    torch.testing.assert_close(batched, per_graph, rtol=1e-9, atol=1e-9)


@pytest.mark.integration
def test_full_pipeline_with_batched_forward(tmp_path, mutag_main_config):
    """The config-driven path: share_gnn_forward: {batched: true} must run the
    complete train/validate/test pipeline through ModelConfiguration."""
    import csv

    from simplegnn.framework.core import FrameworkMain

    fixtures = Path(__file__).resolve().parent / "fixtures" / "share_gnn_mutag"
    main_config_path = mutag_main_config(
        models=fixtures / "models_ShareGNN.yml",
        hyperparameters=fixtures / "parameters_batched.yml",
    )

    experiment = FrameworkMain(main_config_path)
    experiment.preprocessing(num_threads=1)
    experiment.run_configurations(num_threads=-1)
    experiment.evaluate_results()
    experiment.run_best_configuration(num_threads=-1)
    experiment.evaluate_results(evaluate_best_model=True)

    summary_path = tmp_path / "results" / "MUTAG" / "summary_best_mean.csv"
    assert summary_path.is_file(), "expected summary_best_mean.csv to be written"
    with open(summary_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert 0.0 <= float(rows[0]["Validation Accuracy Mean"]) <= 100.0


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["auto", "dense", "sparse"])
def test_batched_backward_matches_per_graph(share_gnn_setup, mode):
    """Both paths must push identical gradients into the shared parameters."""
    graph_data, para = share_gnn_setup
    para.run_config.config["share_gnn_forward"] = {"batched": True, "mode": mode}
    positions = list(range(16))
    labels = torch.stack([graph_data[p].y for p in positions]).squeeze()
    criterion = torch.nn.CrossEntropyLoss()

    net_a = _build_net(graph_data, para)
    outputs_a = torch.stack([net_a(graph_data[p], pos=p) for p in positions])
    criterion(outputs_a, labels).backward()

    net_b = _build_net(graph_data, para)
    outputs_b = net_b(_batched_input(graph_data, positions), pos=positions)
    criterion(outputs_b, labels).backward()

    grads_a = {n: p.grad for n, p in net_a.named_parameters() if p.grad is not None}
    grads_b = {n: p.grad for n, p in net_b.named_parameters() if p.grad is not None}
    assert grads_a.keys() == grads_b.keys()
    assert grads_a, "expected at least one parameter to receive a gradient"
    for name in grads_a:
        torch.testing.assert_close(grads_b[name], grads_a[name], rtol=1e-8, atol=1e-10,
                                   msg=lambda m, name=name: f"gradient mismatch for {name}: {m}")
