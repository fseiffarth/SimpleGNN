"""Behavioral tests for all layers in src/simplegnn/models/layers/.

Each test verifies that a layer actually computes what its name promises,
by comparing against hand-computed reference values (not by re-calling the
same implementation). Tests marked "documents current behavior" pin down
existing quirks/bugs so that intentional fixes show up as test changes.
"""

import types

import pytest
import torch

from simplegnn.models.layers.nn_standard.activation import ActivationLayer
from simplegnn.models.layers.nn_standard.batch_normalization import BatchNormLayer
from simplegnn.models.layers.nn_standard.dropout import DropoutLayer
from simplegnn.models.layers.nn_standard.layer_normalization import LayerNormalization
from simplegnn.models.layers.nn_standard.linear import LinearLayer
from simplegnn.models.layers.nn_standard.reshape import Reshape
from simplegnn.models.layers.mpnn_classical.gcn_conv import GCNConv
from simplegnn.models.layers.mpnn_classical.gin_conv import GINConv
from simplegnn.models.layers.mpnn_classical.gat_conv import GATConv
from simplegnn.models.layers.mpnn_classical.gatv2_conv import GATv2Conv
from simplegnn.models.layers.mpnn_classical.sage_conv import SAGEConv
from simplegnn.models.layers.mpnn_classical.global_pooling import GlobalPooling


def make_args(in_features=4, out_features=4, **overrides):
    args = {
        "layer_id": 0,
        "name": "test-layer",
        "seed": 42,
        "dtype": torch.float32,
        "in_features": in_features,
        "out_features": out_features,
        "in_channels": 1,
        "out_channels": 1,
    }
    args.update(overrides)
    return args


def make_batch(edge_index=None, batch=None, edge_attributes=None):
    """Stand-in for GraphDataset: layers only touch these attributes."""
    return types.SimpleNamespace(
        edge_index=edge_index, batch=batch, edge_attributes=edge_attributes
    )


# Undirected path graph 0-1-2-3 (both edge directions listed).
PATH_EDGE_INDEX = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])


def path_adjacency(num_nodes=4, self_loops=False):
    adj = torch.zeros(num_nodes, num_nodes)
    adj[PATH_EDGE_INDEX[0], PATH_EDGE_INDEX[1]] = 1.0
    if self_loops:
        adj += torch.eye(num_nodes)
    return adj


# ---------------------------------------------------------------------------
# nn_standard layers
# ---------------------------------------------------------------------------


class TestActivationLayer:
    def test_applies_activation_function(self):
        layer = ActivationLayer(make_args(activation_function=torch.nn.ReLU()))
        x = torch.tensor([[-1.0, 2.0], [3.0, -4.0]])
        out = layer(x, make_batch())
        assert torch.equal(out, torch.relu(x))

    def test_default_is_identity(self):
        layer = ActivationLayer(make_args())
        x = torch.randn(5, 4)
        assert torch.equal(layer(x, make_batch()), x)

    def test_does_not_set_its_own_name(self):
        """Documents current behavior: unlike every other layer,
        ActivationLayer does not set layer_args['name'] before calling the
        base constructor, so it crashes unless the config supplies a name."""
        args = make_args()
        del args["name"]
        with pytest.raises(ValueError, match="name"):
            ActivationLayer(args)


class TestBatchNormLayer:
    def test_normalizes_features_over_batch(self):
        layer = BatchNormLayer(make_args(in_features=4, out_features=4))
        layer.train()
        x = torch.randn(64, 4) * 3.0 + 5.0
        out = layer(x, make_batch())
        assert torch.allclose(out.mean(dim=0), torch.zeros(4), atol=1e-5)
        assert torch.allclose(
            out.var(dim=0, unbiased=False), torch.ones(4), atol=1e-3
        )

    def test_eval_uses_running_stats(self):
        layer = BatchNormLayer(make_args(in_features=4, out_features=4))
        layer.eval()
        x = torch.randn(8, 4)
        out = layer(x, make_batch())
        # Fresh running stats are mean=0, var=1 -> eval output ~ input.
        assert torch.allclose(out, x, atol=1e-4)


class TestDropoutLayer:
    def test_training_zeros_and_rescales(self):
        torch.manual_seed(0)
        p = 0.5
        layer = DropoutLayer(make_args(p=p))
        layer.train()
        x = torch.ones(1000, 4)
        out = layer(x, make_batch())
        zero_fraction = (out == 0).float().mean().item()
        assert 0.4 < zero_fraction < 0.6
        nonzero = out[out != 0]
        assert torch.allclose(nonzero, torch.full_like(nonzero, 1.0 / (1.0 - p)))

    def test_eval_is_identity(self):
        layer = DropoutLayer(make_args(p=0.9))
        layer.eval()
        x = torch.randn(50, 4)
        assert torch.equal(layer(x, make_batch()), x)


class TestLayerNormalization:
    def test_2d_normalizes_each_node_row(self):
        layer = LayerNormalization(make_args(in_features=8, out_features=8))
        x = torch.randn(10, 8) * 4.0 + 2.0
        out = layer(x)
        assert torch.allclose(out.mean(dim=-1), torch.zeros(10), atol=1e-5)
        assert torch.allclose(
            out.var(dim=-1, unbiased=False), torch.ones(10), atol=1e-4
        )

    def test_3d_normalizes_each_node_row(self):
        """For (C, N, F) input the layer normalizes over the feature dim only
        (per node row), consistent with the 2D case -- it used to normalize
        over the last TWO dims jointly in the per-graph forward, making the
        batched and per-graph forwards compute different functions."""
        layer = LayerNormalization(make_args(in_features=8, out_features=8))
        x = torch.randn(3, 10, 8) * 4.0 + 2.0
        out = layer(x)
        assert torch.allclose(out.mean(dim=-1), torch.zeros(3, 10), atol=1e-5)
        assert torch.allclose(
            out.var(dim=-1, unbiased=False), torch.ones(3, 10), atol=1e-4
        )

    def test_has_learnable_affine_parameters(self):
        layer = LayerNormalization(make_args(in_features=8, out_features=8))
        params = dict(layer.named_parameters())
        assert params["layer_norm.weight"].shape == (8,)
        assert params["layer_norm.bias"].shape == (8,)
        assert torch.equal(params["layer_norm.weight"], torch.ones(8))
        assert torch.equal(params["layer_norm.bias"], torch.zeros(8))
        # Gradients flow to both parameters.
        out = layer(torch.randn(10, 8))
        out.sum().backward()
        assert params["layer_norm.weight"].grad is not None
        assert params["layer_norm.bias"].grad is not None

    def test_affine_parameters_scale_and_shift_output(self):
        layer = LayerNormalization(make_args(in_features=4, out_features=4))
        with torch.no_grad():
            layer.layer_norm.weight.fill_(2.0)
            layer.layer_norm.bias.fill_(3.0)
        x = torch.randn(6, 4)
        plain = torch.nn.functional.layer_norm(x, normalized_shape=[4])
        assert torch.allclose(layer(x), plain * 2.0 + 3.0, atol=1e-6)

    def test_elementwise_affine_false_has_no_parameters(self):
        layer = LayerNormalization(
            make_args(in_features=8, out_features=8, elementwise_affine=False)
        )
        assert list(layer.parameters()) == []

    def test_bias_false_has_weight_only(self):
        layer = LayerNormalization(make_args(in_features=8, out_features=8, bias=False))
        params = dict(layer.named_parameters())
        assert "layer_norm.weight" in params
        assert "layer_norm.bias" not in params or params["layer_norm.bias"] is None


class TestLinearLayer:
    def test_aggr_features_matches_matmul(self):
        layer = LinearLayer(make_args(in_features=4, out_features=3))
        x = torch.randn(6, 4)
        out = layer(x)
        expected = x @ layer.Param_W + layer.Param_b
        assert torch.allclose(out, expected)
        assert out.shape == (6, 3)

    def test_aggr_features_no_bias(self):
        layer = LinearLayer(make_args(in_features=4, out_features=3, bias=False))
        x = torch.randn(6, 4)
        assert torch.allclose(layer(x), x @ layer.Param_W)

    def test_aggr_channels_flattens_channels_then_projects(self):
        # 3D input (C, N, F) with per-channel in_features (as model.py passes
        # it) and num_heads = C: Param_W is (C*F, F').
        C, N, F = 2, 5, 3
        layer = LinearLayer(
            make_args(
                in_features=F, out_features=4, in_channels=C,
                num_heads=C, mode="aggr_channels",
            )
        )
        assert layer.Param_W.shape == (C * F, 4)
        x = torch.randn(C, N, F)
        out = layer(x)
        expected = x.permute(1, 0, 2).reshape(N, C * F) @ layer.Param_W + layer.Param_b
        assert torch.allclose(out, expected)
        assert out.shape == (N, 4)

    def test_aggr_channels_accepts_2d_input(self):
        # ZINC-style: the invariant message-passing layers emit (N, C*F) with
        # the channels already flattened, and report the full width as
        # out_features, so the following linear layer sees num_heads=1.
        layer = LinearLayer(make_args(in_features=6, out_features=4, mode="aggr_channels"))
        x = torch.randn(5, 6)
        out = layer(x)
        assert torch.allclose(out, x @ layer.Param_W + layer.Param_b)
        assert out.shape == (5, 4)

    def test_channel_wise_applies_separate_transform_per_channel(self):
        C, N, F = 3, 5, 4
        layer = LinearLayer(
            make_args(
                in_features=F, out_features=2, in_channels=C, out_channels=1,
                num_heads=C, mode="channel_wise",
            )
        )
        x = torch.randn(C, N, F)
        out = layer(x)
        expected = torch.einsum("cnf,cfo->cno", x, layer.Param_W) + layer.Param_b
        assert torch.allclose(out, expected)
        assert out.shape == (C, N, 2)
        # Channels really get different transforms.
        assert not torch.allclose(layer.Param_W[0], layer.Param_W[1])

    def test_channel_wise_has_per_channel_bias(self):
        # Param_b is (C, 1, F'): one bias per channel, broadcasting over nodes.
        C, N, F = 3, 5, 4
        layer = LinearLayer(
            make_args(
                in_features=F, out_features=2, in_channels=C, out_channels=C,
                num_heads=C, mode="channel_wise",
            )
        )
        assert layer.Param_b.shape == (C, 1, 2)
        x = torch.randn(C, N, F)
        out = layer(x)
        expected = torch.einsum("cnf,cfo->cno", x, layer.Param_W) + layer.Param_b
        assert torch.allclose(out, expected)
        # Channels really get different biases.
        assert not torch.allclose(layer.Param_b[0], layer.Param_b[1])


class TestReshape:
    def test_default_flattens_everything(self):
        layer = Reshape(make_args(in_features=4, in_channels=2))
        x = torch.randn(2, 3, 4)
        out = layer(x)
        assert out.shape == (24,)
        assert torch.equal(out, x.reshape(-1))

    def test_flatten_head_merges_channels_into_nodes(self):
        layer = Reshape(make_args(in_features=4, in_channels=2, shape="flatten_head"))
        x = torch.randn(2, 3, 4)
        out = layer(x)
        assert out.shape == (6, 4)
        assert torch.equal(out, x.reshape(-1, 4))

    def test_explicit_shape(self):
        layer = Reshape(make_args(in_features=4, shape=[3, 8]))
        x = torch.randn(2, 3, 4)
        out = layer(x)
        assert out.shape == (3, 8)


# ---------------------------------------------------------------------------
# Classical message-passing layers
# ---------------------------------------------------------------------------


class TestGCNConv:
    def test_matches_normalized_adjacency_formula(self):
        """GCN: out = D^-1/2 (A+I) D^-1/2 X W^T + b."""
        torch.manual_seed(0)
        layer = GCNConv(make_args(in_features=4, out_features=3))
        layer.eval()
        x = torch.randn(4, 4)
        out = layer(x, make_batch(edge_index=PATH_EDGE_INDEX))

        adj = path_adjacency(self_loops=True)
        deg_inv_sqrt = adj.sum(dim=1).pow(-0.5)
        norm_adj = deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]
        expected = norm_adj @ (x @ layer.layer.lin.weight.T) + layer.layer.bias
        assert torch.allclose(out, expected, atol=1e-5)

    def test_activation_is_applied(self):
        torch.manual_seed(0)
        args = make_args(in_features=4, out_features=3)
        plain = GCNConv(dict(args))
        with_act = GCNConv(dict(args, activation="torch.nn.ReLU()"))
        with_act.load_state_dict(plain.state_dict())
        plain.eval(), with_act.eval()
        x = torch.randn(4, 4)
        batch = make_batch(edge_index=PATH_EDGE_INDEX)
        assert torch.allclose(with_act(x, batch), torch.relu(plain(x, batch)))

    def test_residual_adds_input(self):
        torch.manual_seed(0)
        args = make_args(in_features=4, out_features=4)
        plain = GCNConv(dict(args))
        res = GCNConv(dict(args, residual=True))
        res.load_state_dict(plain.state_dict())
        plain.eval(), res.eval()
        x = torch.randn(4, 4)
        batch = make_batch(edge_index=PATH_EDGE_INDEX)
        assert torch.allclose(res(x, batch), plain(x, batch) + x)

    def test_dropout_inactive_in_eval(self):
        torch.manual_seed(0)
        layer = GCNConv(make_args(in_features=4, out_features=3, dropout=0.9))
        layer.eval()
        x = torch.randn(4, 4)
        batch = make_batch(edge_index=PATH_EDGE_INDEX)
        assert torch.allclose(layer(x, batch), layer(x, batch))

    def test_batch_norm_works_when_conv_changes_feature_dim(self):
        """Batch norm is applied to the conv output, so it must be sized by
        out_features (was in_features, which crashed for in != out)."""
        torch.manual_seed(0)
        layer = GCNConv(make_args(in_features=4, out_features=3, batch_norm=True))
        layer.train()
        x = torch.randn(32, 4)
        src = torch.arange(32)
        dst = (src + 1) % 32
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
        out = layer(x, make_batch(edge_index=edge_index))
        assert out.shape == (32, 3)
        assert torch.allclose(out.mean(dim=0), torch.zeros(3), atol=1e-5)

    def test_batch_norm_normalizes_output(self):
        torch.manual_seed(0)
        layer = GCNConv(make_args(in_features=4, out_features=4, batch_norm=True))
        layer.train()
        x = torch.randn(32, 4)
        # Ring over 32 nodes so every node has neighbors.
        src = torch.arange(32)
        dst = (src + 1) % 32
        edge_index = torch.stack(
            [torch.cat([src, dst]), torch.cat([dst, src])]
        )
        out = layer(x, make_batch(edge_index=edge_index))
        assert torch.allclose(out.mean(dim=0), torch.zeros(4), atol=1e-5)


class TestGINConv:
    def test_matches_sum_aggregation_formula(self):
        """GIN: out = MLP((1+eps) * x_i + sum_{j in N(i)} x_j)."""
        torch.manual_seed(0)
        layer = GINConv(make_args(in_features=4, out_features=4))
        layer.eval()
        x = torch.randn(4, 4)
        out = layer(x, make_batch(edge_index=PATH_EDGE_INDEX))

        adj = path_adjacency(self_loops=False)
        aggregated = (1.0 + 0.0) * x + adj @ x
        expected = layer.layer.nn(aggregated)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_supports_changing_feature_dimension(self):
        """The internal MLP maps in_features -> out_features, so GIN layers
        that change the feature dimension work."""
        torch.manual_seed(0)
        layer = GINConv(make_args(in_features=4, out_features=8))
        layer.eval()
        x = torch.randn(4, 4)
        out = layer(x, make_batch(edge_index=PATH_EDGE_INDEX))
        assert out.shape == (4, 8)

        adj = path_adjacency(self_loops=False)
        expected = layer.layer.nn((1.0 + 0.0) * x + adj @ x)
        assert torch.allclose(out, expected, atol=1e-5)


class TestSAGEConv:
    def test_matches_mean_aggregation_formula(self):
        """GraphSAGE (mean): out = lin_l(mean_{j in N(i)} x_j) + lin_r(x_i)."""
        torch.manual_seed(0)
        layer = SAGEConv(make_args(in_features=4, out_features=3))
        layer.eval()
        x = torch.randn(4, 4)
        out = layer(x, make_batch(edge_index=PATH_EDGE_INDEX))

        adj = path_adjacency(self_loops=False)
        mean_neigh = adj @ x / adj.sum(dim=1, keepdim=True)
        expected = layer.layer.lin_l(mean_neigh) + layer.layer.lin_r(x)
        assert torch.allclose(out, expected, atol=1e-5)


class TestGATConv:
    def test_constant_features_give_constant_output(self):
        """With identical node features, attention weights must be uniform
        and every node's output identical (attention is a convex combination)."""
        torch.manual_seed(0)
        layer = GATConv(make_args(in_features=4, out_features=3))
        layer.eval()
        x = torch.ones(4, 4) * torch.tensor([1.0, -2.0, 0.5, 3.0])
        out = layer(x, make_batch(edge_index=PATH_EDGE_INDEX))
        assert torch.allclose(out, out[0].expand_as(out), atol=1e-6)

    def test_wrapper_matches_inner_pyg_layer(self):
        torch.manual_seed(0)
        layer = GATConv(make_args(in_features=4, out_features=3))
        layer.eval()
        x = torch.randn(4, 4)
        out = layer(x, make_batch(edge_index=PATH_EDGE_INDEX))
        expected = layer.layer(x, PATH_EDGE_INDEX)
        assert torch.allclose(out, expected)

    def test_multi_head_concat_merges_to_out_features(self):
        torch.manual_seed(0)
        layer = GATConv(
            make_args(in_features=4, out_features=3, num_heads=2, concat=True)
        )
        layer.eval()
        out = layer(torch.randn(4, 4), make_batch(edge_index=PATH_EDGE_INDEX))
        assert out.shape == (4, 3)

    def test_batch_norm_without_merge_heads_normalizes_per_head(self):
        torch.manual_seed(0)
        layer = GATConv(
            make_args(
                in_features=4, out_features=3, num_heads=2,
                concat=True, merge_heads=False, batch_norm=True,
            )
        )
        layer.train()
        x = torch.randn(32, 4)
        src = torch.arange(32)
        dst = (src + 1) % 32
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
        out = layer(x, make_batch(edge_index=edge_index))
        assert out.shape == (32, 2 * 3)
        # Batch norm is applied per head on the (N*heads, out_features) view.
        per_head = out.view(-1, 3)
        assert torch.allclose(per_head.mean(dim=0), torch.zeros(3), atol=1e-5)


class TestGATv2Conv:
    def test_constant_features_give_constant_output(self):
        torch.manual_seed(0)
        layer = GATv2Conv(make_args(in_features=4, out_features=3))
        layer.eval()
        x = torch.ones(4, 4) * torch.tensor([1.0, -2.0, 0.5, 3.0])
        out = layer(x, make_batch(edge_index=PATH_EDGE_INDEX))
        assert torch.allclose(out, out[0].expand_as(out), atol=1e-6)

    def test_wrapper_matches_inner_pyg_layer(self):
        torch.manual_seed(0)
        layer = GATv2Conv(make_args(in_features=4, out_features=3))
        layer.eval()
        x = torch.randn(4, 4)
        out = layer(x, make_batch(edge_index=PATH_EDGE_INDEX))
        assert torch.allclose(out, layer.layer(x, PATH_EDGE_INDEX))

    def test_multi_head_via_num_heads(self):
        """GATv2Conv reads the head count from 'num_heads' (consistent with
        GATConv); 'heads' is reserved for ShareGNN-style head lists."""
        torch.manual_seed(0)
        layer = GATv2Conv(
            make_args(in_features=4, out_features=3, num_heads=2, concat=True)
        )
        assert layer.layer.heads == 2
        layer.eval()
        out = layer(torch.randn(4, 4), make_batch(edge_index=PATH_EDGE_INDEX))
        assert out.shape == (4, 3)  # merged back by linear_merge_heads


class TestGlobalPooling:
    def setup_method(self):
        self.x = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [10.0, 20.0], [30.0, 40.0]]
        )
        self.batch = torch.tensor([0, 0, 0, 1, 1])

    def _pool(self, mode):
        layer = GlobalPooling(make_args(in_features=2, out_features=2, mode=mode))
        return layer(self.x, make_batch(batch=self.batch))

    def test_mean_pooling(self):
        expected = torch.tensor([[3.0, 4.0], [20.0, 30.0]])
        assert torch.allclose(self._pool("mean"), expected)

    def test_sum_pooling(self):
        expected = torch.tensor([[9.0, 12.0], [40.0, 60.0]])
        assert torch.allclose(self._pool("sum"), expected)

    def test_max_pooling(self):
        expected = torch.tensor([[5.0, 6.0], [30.0, 40.0]])
        assert torch.allclose(self._pool("max"), expected)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unsupported pooling mode"):
            GlobalPooling(make_args(in_features=2, out_features=2, mode="min"))
