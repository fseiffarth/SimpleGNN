"""
Unit tests for the ``glu`` layer (``swiglu`` is an alias of it: the gate
activation is what distinguishes the GLU variants, and SiLU -- SwiGLU -- is
the default gate).

The layer is a gated feed-forward block that only touches the last (feature)
dimension, so the tests check the closed form against a manual reference for
every tensor shape used in the framework, plus the configuration handling
(hidden width, gate variants, bias, residual) and the loader/model wiring.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from simplegnn.models.layers.nn_standard.glu import GLULayer
from simplegnn.models.layers.utils.layer_loader import check_layer
from simplegnn.models.layers.utils.layer_types import LayerTypes


def make_args(in_features=4, out_features=4, **overrides):
    args = {
        "layer_id": 0,
        "seed": 42,
        "dtype": torch.float32,
        "in_features": in_features,
        "out_features": out_features,
        "in_channels": 1,
        "out_channels": 1,
    }
    args.update(overrides)
    return args


def reference(layer, x):
    gate = x @ layer.Param_W_g
    up = x @ layer.Param_W_u
    if layer.bias:
        gate = gate + layer.Param_b_g
        up = up + layer.Param_b_u
    out = (layer.gate_activation(gate) * up) @ layer.Param_W_d
    if layer.bias:
        out = out + layer.Param_b_d
    return out


class TestGLUForward:
    def test_matches_closed_form_on_node_level_input(self):
        layer = GLULayer(make_args(in_features=4, out_features=3))
        x = torch.randn(6, 4)
        out = layer(x)
        assert torch.allclose(out, reference(layer, x))
        assert out.shape == (6, 3)

    @pytest.mark.parametrize("shape", [(6, 4), (2, 6, 4), (3, 2, 4)])
    def test_preserves_leading_dimensions(self, shape):
        # (N, F) node-level, (C, N, F) multi-head and (B, C, F) batched
        # graph-level inputs all keep their leading dimensions
        layer = GLULayer(make_args(in_features=4, out_features=5))
        x = torch.randn(*shape)
        out = layer(x)
        assert out.shape == shape[:-1] + (5,)
        assert torch.allclose(out, reference(layer, x))

    def test_multi_head_input_is_applied_per_channel(self):
        # the same weights act on every channel slice independently
        layer = GLULayer(make_args(in_features=4, out_features=4, in_channels=3))
        x = torch.randn(3, 6, 4)
        out = layer(x)
        for c in range(3):
            assert torch.allclose(out[c], layer(x[c]), atol=1e-6)

    def test_no_bias_drops_all_three_bias_terms(self):
        layer = GLULayer(make_args(bias=False))
        assert not hasattr(layer, "Param_b_g")
        assert not hasattr(layer, "Param_b_u")
        assert not hasattr(layer, "Param_b_d")
        x = torch.randn(5, 4)
        gate = torch.nn.functional.silu(x @ layer.Param_W_g)
        assert torch.allclose(layer(x), (gate * (x @ layer.Param_W_u)) @ layer.Param_W_d)

    def test_activation_is_applied_to_the_output(self):
        layer = GLULayer(make_args(activation="torch.nn.ReLU()"))
        x = torch.randn(20, 4)
        assert torch.all(layer(x) >= 0)

    def test_residual_adds_the_input(self):
        plain = GLULayer(make_args(in_features=4, out_features=4))
        residual = GLULayer(make_args(in_features=4, out_features=4, residual=True))
        x = torch.randn(5, 4)
        assert torch.allclose(residual(x), plain(x) + x)

    def test_dropout_only_active_in_training_mode(self):
        layer = GLULayer(make_args(dropout=1.0))
        x = torch.randn(5, 4)
        layer.train()
        assert torch.all(layer(x) == 0)
        layer.eval()
        assert not torch.all(layer(x) == 0)

    def test_gradients_reach_all_projections(self):
        layer = GLULayer(make_args(in_features=4, out_features=3))
        out = layer(torch.randn(6, 4))
        out.sum().backward()
        for name in ("Param_W_g", "Param_W_u", "Param_W_d", "Param_b_g", "Param_b_u", "Param_b_d"):
            grad = getattr(layer, name).grad
            assert grad is not None and torch.any(grad != 0), f"no gradient for {name}"

    def test_double_precision_is_honoured(self):
        layer = GLULayer(make_args(dtype=torch.double))
        out = layer(torch.randn(5, 4, dtype=torch.double))
        assert out.dtype == torch.double


class TestGLUConfiguration:
    def test_hidden_features_defaults_to_expansion_times_in_features(self):
        layer = GLULayer(make_args(in_features=8))
        assert layer.hidden_features == 16
        assert layer.Param_W_g.shape == (8, 16)
        assert layer.Param_W_d.shape == (16, 4)

    def test_expansion_can_be_fractional(self):
        layer = GLULayer(make_args(in_features=6, expansion=8 / 3))
        assert layer.hidden_features == 16

    def test_explicit_hidden_features_wins_over_expansion(self):
        layer = GLULayer(make_args(in_features=8, hidden_features=5, expansion=4))
        assert layer.hidden_features == 5

    @pytest.mark.parametrize(
        "name,fn",
        [
            ("silu", torch.nn.functional.silu),
            ("gelu", torch.nn.functional.gelu),
            ("relu", torch.nn.functional.relu),
            ("sigmoid", torch.sigmoid),
            ("identity", lambda x: x),
        ],
    )
    def test_gate_activation_selects_the_gate(self, name, fn):
        layer = GLULayer(make_args(gate_activation=name, bias=False))
        x = torch.randn(5, 4)
        expected = (fn(x @ layer.Param_W_g) * (x @ layer.Param_W_u)) @ layer.Param_W_d
        assert torch.allclose(layer(x), expected)

    @pytest.mark.parametrize(
        "variant,gate",
        [
            ("swiglu", "silu"),
            ("geglu", "gelu"),
            ("reglu", "relu"),
            ("glu", "sigmoid"),
            ("bilinear", "identity"),
        ],
    )
    def test_named_variants_are_gate_presets(self, variant, gate):
        layer = GLULayer(make_args(variant=variant))
        assert layer.gate_activation_name == gate
        assert layer.gate_activation is GLULayer.GATE_ACTIVATIONS[gate]

    def test_default_variant_is_swiglu(self):
        assert GLULayer(make_args()).gate_activation_name == "silu"

    def test_gate_activation_overrides_variant(self):
        layer = GLULayer(make_args(variant="glu", gate_activation="gelu"))
        assert layer.gate_activation_name == "gelu"

    def test_unknown_gate_activation_raises(self):
        with pytest.raises(ValueError, match="gate activation"):
            GLULayer(make_args(gate_activation="swish_plus"))

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError, match="variant"):
            GLULayer(make_args(variant="megaglu"))

    def test_residual_with_mismatched_dimensions_raises(self):
        with pytest.raises(ValueError, match="residual requires"):
            GLULayer(make_args(in_features=4, out_features=3, residual=True))

    def test_non_positive_hidden_features_raises(self):
        with pytest.raises(ValueError, match="hidden_features must be positive"):
            GLULayer(make_args(hidden_features=0))

    def test_same_seed_gives_identical_weights(self):
        a = GLULayer(make_args())
        b = GLULayer(make_args())
        assert torch.equal(a.Param_W_g, b.Param_W_g)
        assert torch.equal(a.Param_W_d, b.Param_W_d)


class TestGLUWiring:
    def test_layer_types_are_registered(self):
        assert LayerTypes("glu") is LayerTypes.GLU
        assert LayerTypes("swiglu") is LayerTypes.SWIGLU

    @pytest.mark.parametrize("layer_type", ["glu", "swiglu"])
    def test_check_layer_accepts_a_minimal_definition(self, layer_type):
        ok, msg = check_layer(0, {"layer_type": layer_type})
        assert ok, msg

    def test_check_layer_rejects_unknown_gate_activation(self):
        ok, msg = check_layer(0, {"layer_type": "glu", "gate_activation": "swish_plus"})
        assert not ok
        assert "gate_activation" in msg

    def test_check_layer_rejects_unknown_variant(self):
        ok, msg = check_layer(0, {"layer_type": "glu", "variant": "megaglu"})
        assert not ok
        assert "variant" in msg

    @pytest.mark.parametrize("layer_type", ["glu", "swiglu"])
    def test_graph_model_builds_the_layer(self, layer_type):
        from simplegnn.models.model import GraphModel

        model = GraphModel.__new__(GraphModel)
        model.device = "cpu"
        model.seed = 1
        model.graph_data = type("GD", (), {"num_node_features": 4, "num_classes": 2})()
        model.net_layers = []
        model.precision = torch.float
        model.convolution_grad = True
        model.aggregation_grad = True
        model.para = type("P", (), {})()

        layer_spec = type(
            "L",
            (),
            {
                "layer_dict": {"layer_type": layer_type, "out_features": 3},
                "layer_id": 0,
                "layer_type": layer_type,
            },
        )()

        layer = GraphModel.get_model_layer(model, layer_spec)
        assert isinstance(layer, GLULayer)
        # the alias resolves to the same layer with the SiLU gate
        assert layer.gate_activation_name == "silu"
        # a feature-wise block neither creates nor removes channels
        assert layer.out_channels == 1
        assert layer(torch.randn(5, 4)).shape == (5, 3)


# ---------------------------------------------------------------------------
# Integration tier: real GraphModel on MUTAG
# ---------------------------------------------------------------------------

MODELS_FILE = "models_ShareGNN_glu.yml"


def _build_net(graph_data, para, seed=42):
    from simplegnn.models.model import GraphModel

    return GraphModel(graph_data=graph_data, para=para, seed=seed, device="cpu")


@pytest.mark.integration
def test_glu_in_share_gnn_forward(share_gnn_setup_factory):
    """A GLU block at node level (with residual) and at graph level after the
    reshape must both run inside a real ShareGNN network. The fixture also
    covers both layer_type spellings."""
    graph_data, para = share_gnn_setup_factory(MODELS_FILE)
    net = _build_net(graph_data, para)
    net.eval()

    blocks = [l for l in net.net_layers if isinstance(l, GLULayer)]
    assert len(blocks) == 2
    node_block, readout_block = blocks
    # node level: layer_type 'swiglu' alias, readout: 'glu' with variant geglu
    assert node_block.residual and node_block.hidden_features == 16
    assert node_block.gate_activation_name == "silu"
    assert readout_block.hidden_features == 16 and readout_block.out_features == 4
    assert readout_block.gate_activation_name == "gelu"

    with torch.no_grad():
        out = torch.stack([net(graph_data[p], pos=p) for p in range(4)])
    assert out.shape == (4, 2)
    assert torch.isfinite(out).all()


@pytest.mark.integration
def test_glu_gradients_in_share_gnn(share_gnn_setup_factory):
    """Every gate/up/down projection must receive a gradient -- a detached
    branch would silently train as a plain linear layer."""
    graph_data, para = share_gnn_setup_factory(MODELS_FILE)
    net = _build_net(graph_data, para)

    positions = list(range(8))
    labels = torch.stack([graph_data[p].y for p in positions]).squeeze()
    outputs = torch.stack([net(graph_data[p], pos=p) for p in positions])
    torch.nn.CrossEntropyLoss()(outputs, labels).backward()

    for block in (l for l in net.net_layers if isinstance(l, GLULayer)):
        for name, param in block.named_parameters():
            assert param.grad is not None, f"{name} received no gradient"
            assert torch.any(param.grad != 0), f"{name} received an all-zero gradient"
