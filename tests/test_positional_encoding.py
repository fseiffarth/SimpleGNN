"""Tests for the invariant-based positional encoding layer and the pre-norm
residual block on the invariant convolution (specs/15).

Runs on real MUTAG via the shared fixtures: the model config
(models_ShareGNN_pe.yml) puts an invariant_based_positional_encoding layer in
front of a convolution with ``residual: True, pre_layer_norm: True``.
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


@pytest.fixture
def pe_setup(share_gnn_setup_factory):
    return share_gnn_setup_factory("models_ShareGNN_pe.yml")


@pytest.fixture
def pe_net(pe_setup):
    from simplegnn.models.model import GraphModel

    graph_data, para = pe_setup
    net = GraphModel(graph_data=graph_data, para=para, seed=42, device="cpu")
    net.eval()
    return graph_data, net


def _batched_input(graph_data, positions):
    slices = graph_data.slices["x"]
    x = torch.cat([graph_data.x[int(slices[p]):int(slices[p + 1])] for p in positions])
    return SimpleNamespace(x=x)


def _pe_layer(net):
    return next(l for l in net.net_layers if "Positional Encoding" in l.name)


def _conv_layer(net):
    return next(l for l in net.net_layers if "Message Passing" in l.name)


@pytest.mark.integration
def test_pe_layer_builds_with_expected_dimensions(pe_net):
    graph_data, net = pe_net
    pe = _pe_layer(net)

    # heads: {num: 3, primary} + {num: 1 (default), wl}; a head contributes
    # num features per node (num learned entries per label value)
    assert pe.n_heads_per_label == [3, 1]
    assert pe.pe_dim == 3 + 1
    assert pe.out_features == graph_data.num_node_features + pe.pe_dim
    assert pe.out_channels == 1
    assert pe.Param_W.numel() == sum(
        num * n for num, n in zip(pe.n_heads_per_label, pe.n_node_labels)
    )


@pytest.mark.integration
def test_pe_forward_shape_and_input_passthrough(pe_net):
    graph_data, net = pe_net
    pe = _pe_layer(net)

    x = graph_data[0].x
    with torch.no_grad():
        out = pe(x, None, pos=0)

    assert out.shape == (x.shape[0], x.shape[1] + pe.pe_dim)
    assert torch.isfinite(out).all()
    # concatenate_input: True keeps the incoming features untouched
    torch.testing.assert_close(out[:, : x.shape[1]], x)


@pytest.mark.integration
def test_pe_embeddings_shared_per_invariant_id(pe_net):
    """Nodes with the same invariant ID must receive the identical embedding
    slice; nodes with different IDs (generically) different ones."""
    graph_data, net = pe_net
    pe = _pe_layer(net)

    x = graph_data[0].x
    num_nodes = x.shape[0]
    with torch.no_grad():
        emb = pe(x, None, pos=0)[:, x.shape[1]:]

    col = 0
    for head_id, width in enumerate(pe.n_heads_per_label):
        head_emb = emb[:, col : col + width]
        ids = getattr(pe, f"_pe_idx_{head_id}")[: num_nodes].long()
        for a in range(num_nodes):
            for b in range(a + 1, num_nodes):
                if ids[a] == ids[b]:
                    torch.testing.assert_close(head_emb[a], head_emb[b])
                else:
                    assert not torch.equal(head_emb[a], head_emb[b])
        col += width


@pytest.mark.integration
def test_full_net_batched_matches_per_graph(pe_setup):
    """The whole PE + pre-norm-residual-conv architecture must agree between
    the per-graph and the batched forward, including duplicate graph ids."""
    from simplegnn.models.model import GraphModel

    graph_data, para = pe_setup
    para.run_config.config["share_gnn_forward"] = {"batched": True}
    net = GraphModel(graph_data=graph_data, para=para, seed=42, device="cpu")
    net.eval()

    positions = [0, 5, 5, 3, 11]
    with torch.no_grad():
        per_graph = torch.stack([net(graph_data[p], pos=p) for p in positions])
        batched = net(_batched_input(graph_data, positions), pos=positions)

    assert batched.shape == per_graph.shape
    torch.testing.assert_close(batched, per_graph, rtol=1e-9, atol=1e-9)


@pytest.mark.integration
def test_conv_pre_norm_residual_matches_manual(pe_net):
    """out must equal activation(Conv(LayerNorm(x))) + repeat_interleave(x)."""
    graph_data, net = pe_net
    pe = _pe_layer(net)
    conv = _conv_layer(net)
    assert conv.residual and conv.pre_layer_norm

    x = graph_data[0].x
    with torch.no_grad():
        conv_in = pe(x, None, pos=0)
        out = conv(conv_in, None, pos=0)

        # recompute with the block disabled: plain conv on the normalized
        # input plus the feature-aligned skip
        conv.residual = False
        conv.pre_layer_norm = False
        try:
            manual = conv(conv.pre_norm(conv_in), None, pos=0) \
                + conv_in.repeat_interleave(conv.num_heads, dim=1)
        finally:
            conv.residual = True
            conv.pre_layer_norm = True

    assert out.shape == (conv_in.shape[0], conv_in.shape[1] * conv.num_heads)
    torch.testing.assert_close(out, manual, rtol=1e-12, atol=1e-12)


@pytest.mark.integration
def test_gradients_flow_through_pe_and_pre_norm(pe_net):
    graph_data, net = pe_net
    net.train()

    out = net(graph_data[0], pos=0)
    out.sum().backward()

    pe = _pe_layer(net)
    conv = _conv_layer(net)
    assert pe.Param_W.grad is not None
    assert pe.Param_W.grad.abs().sum() > 0
    assert conv.pre_norm.weight.grad is not None
    assert conv.Param_W.grad is not None


@pytest.mark.integration
def test_ogb_closed_walk_encoder_trains_with_l1(share_gnn_setup_factory):
    """Exercise the OGB architecture on small real graphs, including its
    closed-walk label generation, preserved inputs and regularized updates.
    """
    import numpy as np
    import yaml

    from simplegnn.framework.model_configuration import ModelConfiguration

    experiment_dir = ROOT / 'experiments' / 'ogb'
    graph_data, para = share_gnn_setup_factory(
        experiment_dir / 'models_ogb_classification.yml')
    settings = yaml.safe_load(
        (experiment_dir / 'parameters_ogb_classification.yml').read_text())
    for key in ('l1_regularization', 'weight_initialization',
                'rule_occurrence_threshold', 'evaluation_metric'):
        para.run_config.config[key] = settings[key]
    para.run_config.config['step_validation'] = {'interval': 2, 'epochs': 1}
    para.run_config.config['best_model'] = True
    para.learning_rate = settings['learning_rate'][0]
    para.run_config.optimizer = settings['optimizer'][0]
    para.run_config.weight_decay = settings['weight_decay'][0]
    para.n_epochs = 1
    config = ModelConfiguration(
        0, 0, graph_data, (np.arange(24), np.arange(24, 32), np.arange(32, 40)),
        42, para)
    config.train_configuration()
    net = config.net
    pe = _pe_layer(net)
    assert pe.pe_dim == 10
    assert _conv_layer(net).in_features == 16
    assert pe.Param_W.grad is not None
    assert pe.Param_W.grad.abs().sum() > 0
    assert torch.isfinite(pe.Param_W).all()

    net.eval()
    with torch.no_grad():
        x = graph_data[0].x
        encoded = pe(x, None, pos=0)
        torch.testing.assert_close(encoded[:, :x.shape[1]], x)
        positions = [0, 3, 3]
        batched = net(_batched_input(graph_data, positions), pos=positions)
        per_graph = torch.stack([net(graph_data[p], pos=p) for p in positions])
    assert batched.shape == (3, 2)
    assert torch.isfinite(batched).all()
    torch.testing.assert_close(batched, per_graph, rtol=1e-8, atol=1e-8)
