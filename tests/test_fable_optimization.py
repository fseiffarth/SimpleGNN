"""Tests for the invariant-layer optimizations from specs/08-fable-invariant-layer-optimization.md.

Covers:
- Parameter registration / device movement (the GPU frozen-weights bug, spec Phase 1a/1b).
- Fine-cache persistence of the invalid-pair flag (Phase 1c).
- Dense vs. sparse forward equivalence, outputs AND gradients (Phase 2).
- Sparse index structure invariants (coalesced per-graph blocks, Phase 2b).
- Coarse per-layer distribution cache round trip (Phase 3d).

The heavyweight fixtures reuse ``share_gnn_setup`` from conftest.py (real MUTAG
data, cached after the first run).
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


def _build_net(graph_data, para, seed=42, device="cpu"):
    from simplegnn.models.model import GraphModel

    return GraphModel(graph_data=graph_data, para=para, seed=seed, device=device)


def _mp_layers(net):
    from simplegnn.models.ShareGNN.layers.inv_based_message_passing import (
        InvariantBasedMessagePassingLayer,
    )

    return [m for m in net.modules() if isinstance(m, InvariantBasedMessagePassingLayer)]


def _forward_backward(net, graph_data, graph_ids):
    criterion = torch.nn.CrossEntropyLoss()
    outputs = torch.stack([net(graph_data[i], pos=i) for i in graph_ids])
    labels = torch.stack([graph_data[i].y for i in graph_ids]).squeeze()
    loss = criterion(outputs, labels)
    loss.backward()
    return outputs.detach()


@pytest.mark.integration
def test_invariant_parameters_are_registered(share_gnn_setup):
    """Param_W/Param_b must be real nn.Parameters visible to the optimizer.

    Regression test for the GPU bug: ``nn.Parameter(...).to(device)`` returns a
    plain tensor once the device changes, silently dropping the weights from
    net.parameters().
    """
    graph_data, para = share_gnn_setup
    net = _build_net(graph_data, para)

    param_names = {name for name, _ in net.named_parameters()}
    for layer_idx, module in enumerate(net.net_layers):
        if hasattr(module, "Param_W") and module.Param_W is not None:
            assert isinstance(module.Param_W, torch.nn.Parameter), (
                f"layer {layer_idx}: Param_W is {type(module.Param_W)}, not nn.Parameter"
            )
            assert any("Param_W" in n for n in param_names)
        if hasattr(module, "Param_b") and module.Param_b is not None:
            assert isinstance(module.Param_b, torch.nn.Parameter), (
                f"layer {layer_idx}: Param_b is {type(module.Param_b)}, not nn.Parameter"
            )


@pytest.mark.integration
def test_distribution_buffers_registered_and_not_persistent(share_gnn_setup):
    """weight/bias_distribution are buffers (moved by net.to) but not in state_dict."""
    graph_data, para = share_gnn_setup
    net = _build_net(graph_data, para)

    buffer_names = {name for name, _ in net.named_buffers()}
    assert any("weight_distribution" in n for n in buffer_names), (
        "weight_distribution should be a registered buffer"
    )
    state_keys = set(net.state_dict().keys())
    assert not any("weight_distribution" in k for k in state_keys), (
        "weight_distribution must not be persisted in state_dict"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.integration
def test_invariant_layer_weights_are_updated_on_gpu(share_gnn_setup):
    """GPU variant of the weight-update gate (T3 in spec 08)."""
    graph_data, para = share_gnn_setup
    net = _build_net(graph_data, para)

    cpu_param_count = sum(p.numel() for p in net.parameters())
    net.to("cuda")
    graph_data.to("cuda")
    assert sum(p.numel() for p in net.parameters()) == cpu_param_count

    mp_layers = _mp_layers(net)
    assert mp_layers
    for module in mp_layers:
        assert module.Param_W.is_cuda, "Param_W was not moved to CUDA"
        assert module.weight_distribution.is_cuda, "weight_distribution buffer not moved"

    weights_before = [m.Param_W.detach().clone() for m in mp_layers]
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    _forward_backward(net, graph_data, list(range(16)))

    for module in mp_layers:
        assert module.Param_W.grad is not None, "Param_W got no gradient on GPU"
        assert torch.any(module.Param_W.grad != 0), "Param_W gradient is all zeros on GPU"

    optimizer.step()
    for module, before in zip(mp_layers, weights_before):
        assert not torch.equal(module.Param_W.detach(), before), (
            "Param_W did not change after an optimizer step on GPU"
        )


@pytest.mark.integration
def test_fine_cache_persists_invalid_flag(share_gnn_setup, tmp_path):
    """The (indices, counts) cache must round-trip do_invalid_indices_exist;
    old-format caches without the flag must be treated as a miss."""
    graph_data, para = share_gnn_setup
    net = _build_net(graph_data, para)
    layer = _mp_layers(net)[0]

    indices = torch.tensor([0, 1, 2, 1], dtype=torch.int64)
    counts = torch.tensor([1, 2, 1], dtype=torch.int64)
    head = layer.layer.layer_heads[0]

    cache_file = tmp_path / "flag_true.pt"
    layer._save_cached_indices(cache_file, head, 1, indices, counts, True)
    loaded_indices, loaded_counts, flag = layer._load_cached_indices(cache_file, head, 1)
    assert torch.equal(loaded_indices, indices)
    assert torch.equal(loaded_counts, counts)
    assert flag is True

    # old cache format (no flag) -> must raise so callers treat it as a miss
    legacy_file = tmp_path / "legacy.pt"
    torch.save({"indices": indices, "counts": counts}, str(legacy_file))
    with pytest.raises(Exception):
        layer._load_cached_indices(legacy_file, head, 1)


@pytest.mark.integration
def test_dense_and_sparse_forward_are_equivalent(share_gnn_setup):
    """Outputs and Param_W gradients must match between forward modes (T6)."""
    graph_data, para = share_gnn_setup

    para.run_config.config["share_gnn_forward"] = {"mode": "sparse"}
    net_sparse = _build_net(graph_data, para, seed=42)
    para.run_config.config["share_gnn_forward"] = {"mode": "dense"}
    net_dense = _build_net(graph_data, para, seed=42)
    para.run_config.config.pop("share_gnn_forward")

    graph_ids = list(range(len(graph_data)))
    out_sparse = _forward_backward(net_sparse, graph_data, graph_ids)
    out_dense = _forward_backward(net_dense, graph_data, graph_ids)

    torch.testing.assert_close(out_sparse, out_dense, rtol=1e-5, atol=1e-6)
    for layer_sparse, layer_dense in zip(_mp_layers(net_sparse), _mp_layers(net_dense)):
        assert layer_sparse.forward_mode == "sparse"
        assert layer_dense.forward_mode == "dense"
        torch.testing.assert_close(
            layer_sparse.Param_W.grad, layer_dense.Param_W.grad, rtol=1e-5, atol=1e-6
        )


@pytest.mark.integration
def test_sparse_index_structures_are_coalesced_blocks(share_gnn_setup):
    """_fwd_indices must be strictly increasing per graph block under the
    (row * N + col) encoding — the precondition for is_coalesced=True — and the
    sparse matrix must reproduce the dense set_weights scatter (T8)."""
    graph_data, para = share_gnn_setup
    net = _build_net(graph_data, para)
    layer = _mp_layers(net)[0]

    for pos in range(min(len(graph_data), 20)):
        start, end = layer._fwd_slices[pos], layer._fwd_slices[pos + 1]
        num_nodes = layer._num_nodes_list[pos]
        block = layer._fwd_indices[:, start:end]
        if block.shape[1] == 0:
            continue
        linear = block[0] * num_nodes + block[1]
        assert torch.all(linear[1:] > linear[:-1]), (
            f"graph {pos}: _fwd_indices block is not strictly increasing (duplicates "
            f"or unsorted entries would make is_coalesced=True invalid)"
        )
        # sparse values scattered densely == dense set_weights result
        values = layer.Param_W[layer._fwd_param_idx[start:end]]
        sparse_dense = torch.sparse_coo_tensor(
            block, values, (layer.num_heads * num_nodes, num_nodes), is_coalesced=True
        ).to_dense().view(layer.num_heads, num_nodes, num_nodes)
        layer.set_weights(pos)
        torch.testing.assert_close(sparse_dense, layer.current_W)


@pytest.mark.integration
def test_layer_distribution_cache_round_trip(share_gnn_setup):
    """Coarse per-layer cache: cold build == warm load, and the warm model
    still trains (T9)."""
    graph_data, para = share_gnn_setup

    net_probe = _build_net(graph_data, para)
    layer_probe = _mp_layers(net_probe)[0]
    cache_path = layer_probe._get_layer_cache_path()

    # cold build (cache removed), then warm load
    cache_path.unlink(missing_ok=True)
    cache_path.with_suffix(".json").unlink(missing_ok=True)
    net_cold = _build_net(graph_data, para, seed=42)
    assert cache_path.exists(), "cold build should write the layer cache"
    net_warm = _build_net(graph_data, para, seed=42)

    for cold, warm in zip(_mp_layers(net_cold), _mp_layers(net_warm)):
        assert torch.equal(cold.weight_distribution, warm.weight_distribution)
        assert torch.equal(cold.weight_distribution_slices, warm.weight_distribution_slices)
        assert cold.weight_num == warm.weight_num
        assert cold.weight_offset == warm.weight_offset
        assert cold.bias_num == warm.bias_num
        assert cold.b_head_offset == warm.b_head_offset
        if cold.bias:
            assert torch.equal(cold.bias_distribution, warm.bias_distribution)
            assert torch.equal(cold.bias_distribution_slices, warm.bias_distribution_slices)

    # the warm-cache model must still train (invalid-bucket bug regression gate)
    mp_layers = _mp_layers(net_warm)
    weights_before = [m.Param_W.detach().clone() for m in mp_layers]
    optimizer = torch.optim.Adam(net_warm.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    _forward_backward(net_warm, graph_data, list(range(16)))
    for module in mp_layers:
        assert module.Param_W.grad is not None
        assert torch.any(module.Param_W.grad != 0)
    optimizer.step()
    for module, before in zip(mp_layers, weights_before):
        assert not torch.equal(module.Param_W.detach(), before)
