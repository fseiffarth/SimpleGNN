"""Tests for the ShareGNN layer visualization (`draw()`) code.

The unit tests cover the shared helpers in graph_drawing.py; the integration
tests build a real GraphModel on MUTAG (same fixture as the other ShareGNN
integration tests) and smoke-draw both invariant layers onto an Agg canvas,
which used to crash on stale label indexing / the legacy pooling weight
format.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simplegnn.datasets.utils.graph_drawing import (
    GraphDrawing,
    compute_positions,
    filter_weight_bounds,
    load_positions,
    resolve_positions,
    save_positions,
)


class TestFilterWeightBounds:
    def test_none_filter_returns_input_unchanged(self):
        weights = np.array([1.0, -2.0, 3.0])
        assert np.array_equal(filter_weight_bounds(weights, None), weights)

    def test_absolute_keeps_top_and_bottom_values(self):
        weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        filtered = filter_weight_bounds(weights, {"absolute": 2})
        assert np.array_equal(filtered, np.array([1.0, 2.0, 0.0, 4.0, 5.0]))

    def test_absolute_larger_than_unique_count_keeps_everything_once(self):
        weights = np.array([1.0, 2.0, 3.0])
        filtered = filter_weight_bounds(weights, {"absolute": 10})
        # overlapping bounds must not double the weights (old code summed
        # the two np.where results)
        assert np.array_equal(filtered, weights)

    def test_percentage_zero_keeps_only_extremes(self):
        weights = np.linspace(0.0, 1.0, 10)
        filtered = filter_weight_bounds(weights, {"percentage": 0.01})
        # keep clamps to at least 1 unique value at each end
        assert filtered[0] == weights[0]
        assert filtered[-1] == weights[-1]
        assert np.count_nonzero(filtered[1:-1]) == 0

    def test_percentage_one_keeps_everything(self):
        weights = np.array([0.5, 1.5, 2.5])
        filtered = filter_weight_bounds(weights, {"percentage": 1.0})
        assert np.array_equal(filtered, weights)

    def test_all_equal_weights(self):
        weights = np.full(4, 2.0)
        filtered = filter_weight_bounds(weights, {"absolute": 1})
        assert np.array_equal(filtered, weights)

    def test_empty_weights(self):
        filtered = filter_weight_bounds(np.array([]), {"absolute": 3})
        assert filtered.size == 0

    def test_unknown_keys_raise(self):
        with pytest.raises(ValueError):
            filter_weight_bounds(np.array([1.0]), {"unknown": 1})


class TestPositions:
    def test_circle_layout_covers_ring(self):
        graph = nx.cycle_graph(6)
        pos = compute_positions(graph, "circle", root_node=0)
        assert set(pos) == set(graph.nodes())
        radii = [np.hypot(x, y) for x, y in pos.values()]
        assert np.allclose(radii, 400.0)

    def test_circle_layout_terminates_on_non_ring(self):
        # a path graph dead-ends the ring walk; the old loop never terminated
        graph = nx.path_graph(5)
        pos = compute_positions(graph, "circle", root_node=0)
        assert set(pos) == set(graph.nodes())

    def test_circle_layout_missing_root_falls_back(self):
        graph = nx.cycle_graph(4)
        pos = compute_positions(graph, "circle", root_node=None)
        assert set(pos) == set(graph.nodes())

    def test_kawai_layout_int_keys(self):
        graph = nx.cycle_graph(4)
        pos = compute_positions(graph, "kawai")
        assert all(isinstance(k, int) for k in pos)

    def test_save_load_roundtrip(self, tmp_path):
        pos = {0: (1.5, -2.0), 1: (0.0, 3.25)}
        pos_path = tmp_path / "pos.txt"
        save_positions(pos, pos_path)
        loaded = load_positions(pos_path)
        assert loaded == pos

    def test_load_missing_file_returns_none(self, tmp_path):
        assert load_positions(tmp_path / "absent.txt") is None
        assert load_positions("") is None

    def test_resolve_positions_caches_to_file(self, tmp_path):
        graph = nx.cycle_graph(4)
        pos_path = tmp_path / "pos.txt"
        pos = resolve_positions(graph, "kawai", pos_path=pos_path)
        assert pos_path.is_file()
        cached = resolve_positions(graph, "kawai", pos_path=pos_path)
        assert set(cached) == set(pos)


@pytest.fixture(scope="module")
def _module_agg_close():
    yield
    plt.close("all")


@pytest.mark.integration
class TestLayerDraw:
    @pytest.fixture()
    def net(self, share_gnn_setup):
        from simplegnn.models.model import GraphModel

        graph_data, para = share_gnn_setup
        return GraphModel(graph_data=graph_data, para=para, seed=0, device="cpu")

    @pytest.fixture()
    def drawings(self):
        return (
            GraphDrawing(node_size=40, edge_width=1, draw_type="kawai"),
            GraphDrawing(node_size=40, edge_width=1, weight_edge_width=2.5,
                         weight_arrow_size=10, draw_type="kawai"),
        )

    def _layers(self, net):
        from simplegnn.models.ShareGNN.layers.inv_based_message_passing import (
            InvariantBasedMessagePassingLayer,
        )
        from simplegnn.models.ShareGNN.layers.inv_based_pooling import (
            InvariantBasedAggregationLayer,
        )

        mp = next(l for l in net.net_layers
                  if isinstance(l, InvariantBasedMessagePassingLayer))
        agg = next(l for l in net.net_layers
                   if isinstance(l, InvariantBasedAggregationLayer))
        return mp, agg

    def test_message_passing_draw_graph_only(self, net, drawings):
        mp, _ = self._layers(net)
        fig, ax = plt.subplots()
        mp.draw(ax=ax, graph_id=0, graph_drawing=drawings, graph_only=True)
        assert ax.collections or ax.patches
        plt.close(fig)

    def test_message_passing_draw_weights(self, net, drawings):
        mp, _ = self._layers(net)
        for filter_weights in (None, {"absolute": 3}, {"percentage": 0.1}):
            fig, ax = plt.subplots()
            mp.draw(ax=ax, graph_id=0, graph_drawing=drawings,
                    filter_weights=filter_weights)
            assert ax.collections or ax.patches
            plt.close(fig)

    def test_message_passing_draw_every_head(self, net, drawings):
        mp, _ = self._layers(net)
        for head in range(mp.num_heads):
            fig, ax = plt.subplots()
            mp.draw(ax=ax, graph_id=1, graph_drawing=drawings, head=head)
            plt.close(fig)

    def test_message_passing_pos_path_roundtrip(self, net, drawings, tmp_path):
        mp, _ = self._layers(net)
        pos_path = tmp_path / "graph0_pos.txt"
        fig, ax = plt.subplots()
        mp.draw(ax=ax, graph_id=0, graph_drawing=drawings, graph_only=True,
                pos_path=pos_path)
        plt.close(fig)
        assert pos_path.is_file()
        # second call reuses the cached positions
        fig, ax = plt.subplots()
        mp.draw(ax=ax, graph_id=0, graph_drawing=drawings, pos_path=pos_path)
        plt.close(fig)

    def test_aggregation_draw(self, net, drawings):
        _, agg = self._layers(net)
        fig, ax = plt.subplots()
        agg.draw(ax=ax, graph_id=0, graph_drawing=drawings)
        assert ax.collections
        plt.close(fig)

    def test_aggregation_draw_graph_only(self, net, drawings):
        _, agg = self._layers(net)
        fig, ax = plt.subplots()
        agg.draw(ax=ax, graph_id=0, graph_drawing=drawings, graph_only=True)
        assert ax.collections
        plt.close(fig)
