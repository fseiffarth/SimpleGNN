"""Regression tests for the invariant-layer load speedup (specs/25-invariant-layer-load-speedup.md).

Step 1 replaced the per-head `target_rows` recomputation and the multi-pass
int64 index remap with a per-build memo and a single int32 lookup table, and
loads the (indices, counts) cache with mmap + weights_only. None of this may
change the built structures: the reference below re-derives every `_pv_*`
vector with the original per-value algorithm straight from the cache files.
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


def _mp_layers(net):
    from simplegnn.models.ShareGNN.layers.inv_based_message_passing import (
        InvariantBasedMessagePassingLayer,
    )

    return [m for m in net.modules() if isinstance(m, InvariantBasedMessagePassingLayer)]


def _reference_pv(layer, threshold, upper_threshold):
    """The pre-spec-25 pv construction, one (head, property value) at a time."""
    x_slices = layer.graph_data.slices['x']
    weight_offset = [0]
    pvs = []
    for head_id, head in enumerate(layer.layer.layer_heads):
        prop = layer.graph_data.properties[layer.property_descriptions[head_id]]
        cons = prop.consolidated(x_slices)
        pv = torch.full((cons.total_rows,), -1, dtype=torch.int32)
        for property_key in prop.valid_values[(layer.layer_id, head_id)]:
            indices, counts, invalid, _, _ = layer._load_cached_indices(
                layer.get_cache_path(head, property_key), head, property_key)
            num_weights = len(counts) - (1 if invalid else 0)
            if threshold > 1 or invalid or upper_threshold is not None:
                if upper_threshold is not None:
                    valid = torch.where((counts >= threshold) & (counts <= upper_threshold))[0]
                else:
                    valid = torch.where(counts >= threshold)[0]
                mapping = torch.full((counts.size(0),), -1, dtype=torch.int64)
                mapping[valid] = torch.arange(valid.numel(), dtype=torch.int64)
                indices = mapping[indices.long()]
                num_weights = valid.numel()
            base_offset = weight_offset[-1]
            target = cons.target_rows(property_key)
            pv[target] = torch.where(indices >= 0, indices + base_offset, indices).to(torch.int32)
            for _ in range(layer.n_heads_per_label[head_id]):
                weight_offset.append(weight_offset[-1] + num_weights)
        pvs.append(pv)
    return pvs, weight_offset


@pytest.mark.integration
@pytest.mark.parametrize("threshold,upper_threshold", [(1, None), (2, None), (2, 50)])
def test_pv_matches_reference_construction(share_gnn_setup, threshold, upper_threshold):
    from simplegnn.models.model import GraphModel

    graph_data, para = share_gnn_setup
    config = para.run_config.config
    saved = {key: config.get(key) for key in ('rule_occurrence_threshold', 'rule_occurrence_upper_threshold')}
    config['rule_occurrence_threshold'] = threshold
    config['rule_occurrence_upper_threshold'] = upper_threshold
    try:
        net = GraphModel(graph_data=graph_data, para=para, seed=42, device='cpu')
        layers = _mp_layers(net)
        assert layers, "fixture model has no invariant message passing layer"
        for layer in layers:
            reference_pvs, reference_offsets = _reference_pv(layer, threshold, upper_threshold)
            for head_id, reference in enumerate(reference_pvs):
                built = getattr(layer, f'_pv_{head_id}')
                assert built.dtype == torch.int32
                assert torch.equal(built, reference), f"layer {layer.layer_id} head {head_id} pv differs"
            assert layer.weight_offset == reference_offsets
            assert set(layer._init_timing) >= {'cache_load', 'target_rows', 'scatter', 'finalize', 'total'}
    finally:
        for key, value in saved.items():
            if value is None:
                config.pop(key, None)
            else:
                config[key] = value


@pytest.mark.integration
def test_legacy_pickle_cache_still_loads(share_gnn_setup, tmp_path):
    """Caches written with the legacy (non-zip) serializer cannot be mmap'ed
    and must fall back to a full load instead of being treated as a miss."""
    from simplegnn.models.model import GraphModel

    graph_data, para = share_gnn_setup
    layer = _mp_layers(GraphModel(graph_data=graph_data, para=para, seed=42, device='cpu'))[0]
    head = layer.layer.layer_heads[0]
    indices = torch.tensor([0, 1, 2, 1], dtype=torch.int64)
    counts = torch.tensor([1, 2, 1], dtype=torch.int64)
    legacy_file = tmp_path / "legacy_format.pt"
    torch.save({'indices': indices, 'counts': counts, 'do_invalid_indices_exist': False},
               str(legacy_file), _use_new_zipfile_serialization=False)

    loaded_indices, loaded_counts, invalid, uniques, max_label = layer._load_cached_indices(legacy_file, head, 1)
    assert torch.equal(loaded_indices, indices)
    assert torch.equal(loaded_counts, counts)
    assert invalid is False
    assert uniques is None and max_label is None


@pytest.mark.parametrize("labels", [
    torch.tensor([5, -1, 3, 5, 0, 3, 7, -1], dtype=torch.int64),
    torch.randint(-1, 40, (5000,), generator=torch.Generator().manual_seed(0)),
    torch.tensor([2.5, -1.0, 2.5, 0.0]),
    torch.tensor([[1, 2], [0, 5], [1, 2], [-1, 0]]),
])
def test_label_inverse_matches_unique_dim0(labels):
    """The bias / aggregation / positional-encoding index vectors must not
    change: label_inverse is a faster drop-in for unique(dim=0)'s inverse."""
    from simplegnn.models.ShareGNN.utils import label_inverse

    _, expected = torch.unique(labels, dim=0, return_inverse=True, sorted=False)
    assert torch.equal(label_inverse(labels), expected)


def _reference_materialized_rows(layer):
    """The pre-spec-25 materialization: assemble graph chunks from the
    factored structures, then stable-argsort the rows by graph."""
    num_graphs = len(layer._num_nodes_list)
    columns, rows_per_graph = [[], [], [], []], []
    for start in range(0, num_graphs, 4096):
        positions = list(range(start, min(start + 4096, num_graphs)))
        *rows, graph_slot = layer._assemble_rows(positions)
        order = torch.argsort(graph_slot, stable=True)
        for column, values in zip(columns, rows):
            column.append(values[order].to(torch.int32))
        rows_per_graph.append(torch.bincount(graph_slot, minlength=len(positions)))
    slices = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cat(rows_per_graph).cumsum(dim=0)])
    return [torch.cat(column) for column in columns], slices


@pytest.mark.integration
@pytest.mark.parametrize("models_file", ["models_ShareGNN.yml", "models_ShareGNN_pe.yml"])
@pytest.mark.parametrize("chunk_graphs", [16384, 7])
def test_materialized_rows_match_reference(share_gnn_setup_factory, monkeypatch, models_file, chunk_graphs):
    """The direct-placement _materialize_rows must reproduce the old
    argsort-based layout row for row (order matters for float accumulation),
    including head replicas (num > 1 in the _pe fixture) and chunk borders."""
    from simplegnn.models.model import GraphModel
    from simplegnn.models.ShareGNN.layers.inv_based_message_passing import (
        InvariantBasedMessagePassingLayer,
    )

    monkeypatch.setattr(InvariantBasedMessagePassingLayer, '_MATERIALIZE_CHUNK_GRAPHS', chunk_graphs)
    graph_data, para = share_gnn_setup_factory(models_file)
    para.run_config.config['share_gnn_forward'] = {'precompute_rows': True}
    try:
        net = GraphModel(graph_data=graph_data, para=para, seed=42, device='cpu')
    finally:
        para.run_config.config.pop('share_gnn_forward')
    for layer in _mp_layers(net):
        assert layer._rows_materialized
        layer._rows_materialized = False  # make _assemble_rows use the factored path
        (heads, i_local, j_local, params), slices = _reference_materialized_rows(layer)
        layer._rows_materialized = True
        assert torch.equal(layer._rc_slices, slices)
        assert torch.equal(layer._rc_heads, heads)
        assert torch.equal(layer._rc_i, i_local)
        assert torch.equal(layer._rc_j, j_local)
        assert torch.equal(layer._rc_params, params)


def _materialized_columns(layer):
    return [getattr(layer, f'_rc_{name}') for name in ('slices', 'heads', 'i', 'j', 'params')]


@pytest.mark.integration
def test_deferred_rows_build_lazily_and_identically(share_gnn_setup_factory, monkeypatch):
    """Rows deferred to the target device (CUDA runs) must, when the net never
    reaches a GPU, be built on first use — identical to the eager build."""
    from simplegnn.models.model import GraphModel
    from simplegnn.models.ShareGNN.layers.inv_based_message_passing import (
        InvariantBasedMessagePassingLayer,
    )

    graph_data, para = share_gnn_setup_factory("models_ShareGNN_pe.yml")
    para.run_config.config['share_gnn_forward'] = {'precompute_rows': True}
    try:
        eager = _mp_layers(GraphModel(graph_data=graph_data, para=para, seed=42, device='cpu'))
        monkeypatch.setattr(InvariantBasedMessagePassingLayer, '_defer_rows_to_device', lambda self: True)
        deferred_net = GraphModel(graph_data=graph_data, para=para, seed=42, device='cpu')
    finally:
        para.run_config.config.pop('share_gnn_forward')
    deferred = _mp_layers(deferred_net)
    for layer in deferred:
        assert layer._rows_pending is not None and not layer._rows_materialized
    deferred_net.to('cpu')  # not a GPU: must stay deferred
    assert all(layer._rows_pending is not None for layer in deferred)

    for eager_layer, layer in zip(eager, deferred):
        layer._assemble_rows([0])
        assert layer._rows_pending is None and layer._rows_materialized
        for built, expected in zip(_materialized_columns(layer), _materialized_columns(eager_layer)):
            assert torch.equal(built, expected)


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA/ROCm GPU")
def test_rows_materialize_on_gpu_identically(share_gnn_setup_factory):
    """On a CUDA run the rows are built on the GPU during net.to(device) and
    must equal the CPU build."""
    from simplegnn.models.model import GraphModel

    graph_data, para = share_gnn_setup_factory("models_ShareGNN_pe.yml")
    para.run_config.config['share_gnn_forward'] = {'precompute_rows': True}
    try:
        cpu_layers = _mp_layers(GraphModel(graph_data=graph_data, para=para, seed=42, device='cpu'))
        gpu_net = GraphModel(graph_data=graph_data, para=para, seed=42, device='cuda')
    finally:
        para.run_config.config.pop('share_gnn_forward')
    gpu_layers = _mp_layers(gpu_net)
    assert all(layer._rows_pending is not None for layer in gpu_layers)
    gpu_net.to('cuda')
    for cpu_layer, gpu_layer in zip(cpu_layers, gpu_layers):
        assert gpu_layer._rows_pending is None and gpu_layer._rows_materialized
        assert gpu_layer._rc_heads.is_cuda
        for built, expected in zip(_materialized_columns(gpu_layer), _materialized_columns(cpu_layer)):
            assert torch.equal(built.cpu(), expected)
