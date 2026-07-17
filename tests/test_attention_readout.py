"""Attention-based ShareGNN readout.

The dense readout after ``invariant_based_aggregation`` learns a free weight
per (head, feature, output) triple -- on ZINC that single layer holds ~80% of
the model's parameters (specs/09-zinc-readout-and-architecture-improvements.md).
The ``attention_readout`` layer instead treats the H head embeddings of an
unflattened aggregation (``flatten: False``) as a set of tokens and pools them
with attention: gated attention pooling ('gated'), learned seed queries
('pma'), or a transformer encoder block followed by gated pooling
('transformer'). These tests pin the parameter counts of all three variants
and check that the readout behaves identically in the per-graph and the
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

from simplegnn.models.layers.nn_standard.attention_readout import AttentionReadoutLayer

VARIANTS = ["gated", "pma", "transformer"]

ATTENTION_FIXTURES = [
    "models_ShareGNN_attention_gated.yml",
    "models_ShareGNN_attention_pma.yml",
    "models_ShareGNN_attention_transformer.yml",
]


def make_args(in_features=8, out_features=4, in_channels=5, **overrides):
    args = {
        "layer_id": 0,
        "name": "test-layer",
        "seed": 42,
        "dtype": torch.float64,
        "in_features": in_features,
        "out_features": out_features,
        "in_channels": in_channels,
        "out_channels": in_channels,
        "attention_dim": 4,
        "num_seeds": 2,
        "num_attention_heads": 2,
        "ffn_dim": 16,
    }
    args.update(overrides)
    return args


def expected_param_count(variant, H, F, out, attention_dim, num_seeds,
                         ffn_dim, head_embeddings=True):
    """Closed-form parameter counts -- the cheapness of the readout is the
    whole point of the layer, so the formulas are worth pinning."""
    count = H * F if head_embeddings else 0
    mha = 3 * F * F + 3 * F + F * F + F  # in_proj W+b, out_proj W+b
    if variant in ("gated", "transformer"):
        count += 2 * F * attention_dim + attention_dim  # V, U, w
        count += F * out + out  # W, b
    if variant == "transformer":
        count += mha
        count += F * ffn_dim + ffn_dim + ffn_dim * F + F  # linear1, linear2
        count += 4 * F  # norm1, norm2
    if variant == "pma":
        count += num_seeds * F + mha
        count += num_seeds * F * out + out  # W, b
    return count


# ---------------------------------------------------------------------------
# Unit tier: direct instantiation, no dataset
# ---------------------------------------------------------------------------


class TestAttentionReadoutUnit:
    @pytest.mark.parametrize("variant", VARIANTS)
    @pytest.mark.parametrize("head_embeddings", [True, False])
    def test_output_shapes(self, variant, head_embeddings):
        layer = AttentionReadoutLayer(
            make_args(variant=variant, head_embeddings=head_embeddings)
        )
        layer.eval()
        assert layer.out_channels == 1
        per_graph = layer(torch.randn(5, 1, 8, dtype=torch.float64), pos=0)
        assert per_graph.shape == (1, 4)
        batched = layer(torch.randn(3, 5, 8, dtype=torch.float64), pos=[0, 1, 2])
        assert batched.shape == (3, 4)

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_batched_matches_per_graph(self, variant):
        layer = AttentionReadoutLayer(make_args(variant=variant))
        layer.eval()
        x = torch.randn(3, 5, 8, dtype=torch.float64)
        with torch.no_grad():
            batched = layer(x, pos=[0, 1, 2])
            per_graph = torch.cat(
                [layer(x[b].unsqueeze(1), pos=b) for b in range(3)]
            )
        torch.testing.assert_close(batched, per_graph, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_all_parameters_are_float64(self, variant):
        layer = AttentionReadoutLayer(make_args(variant=variant))
        for name, param in layer.named_parameters():
            assert param.dtype == torch.float64, f"{name} is {param.dtype}"
        out = layer(torch.randn(3, 5, 8, dtype=torch.float64), pos=[0, 1, 2])
        assert out.dtype == torch.float64

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_seed_determinism(self, variant):
        a = AttentionReadoutLayer(make_args(variant=variant))
        b = AttentionReadoutLayer(make_args(variant=variant))
        for (name, pa), (_, pb) in zip(
            a.state_dict().items(), b.state_dict().items()
        ):
            assert torch.equal(pa, pb), f"{name} differs between same-seed instances"
        x = torch.randn(3, 5, 8, dtype=torch.float64)
        a.eval(), b.eval()
        with torch.no_grad():
            torch.testing.assert_close(a(x, pos=[0, 1, 2]), b(x, pos=[0, 1, 2]))

    @pytest.mark.parametrize("variant", VARIANTS)
    @pytest.mark.parametrize("head_embeddings", [True, False])
    def test_parameter_count(self, variant, head_embeddings):
        layer = AttentionReadoutLayer(
            make_args(variant=variant, head_embeddings=head_embeddings)
        )
        expected = expected_param_count(
            variant, H=5, F=8, out=4, attention_dim=4, num_seeds=2,
            ffn_dim=16, head_embeddings=head_embeddings,
        )
        assert sum(p.numel() for p in layer.parameters()) == expected

    @pytest.mark.parametrize(
        "variant,expected",
        [
            ("gated", 36_964),
            ("pma", 64_600),
            ("transformer", 118_064),
        ],
    )
    def test_zinc_scale_parameter_count(self, variant, expected):
        """At ZINC scale (H=140, F=100, out=100) every variant must stay far
        below the 1,400,100 parameters of the dense readout it replaces."""
        layer = AttentionReadoutLayer(
            make_args(
                in_features=100, out_features=100, in_channels=140,
                variant=variant, attention_dim=64, num_seeds=1,
                num_attention_heads=4, ffn_dim=200,
            )
        )
        count = sum(p.numel() for p in layer.parameters())
        assert count == expected
        assert count < 150_000 < 1_400_100

    def test_head_embeddings_change_output(self):
        with_e = AttentionReadoutLayer(make_args(variant="gated"))
        without_e = AttentionReadoutLayer(
            make_args(variant="gated", head_embeddings=False)
        )
        # same seed -> the shared parameters are identical, so any output
        # difference comes from the embedding table alone
        x = torch.randn(3, 5, 8, dtype=torch.float64)
        with torch.no_grad():
            assert not torch.allclose(
                with_e(x, pos=[0, 1, 2]), without_e(x, pos=[0, 1, 2])
            )

    def test_invalid_variant_raises(self):
        with pytest.raises(ValueError, match="variant"):
            AttentionReadoutLayer(make_args(variant="linear"))

    def test_indivisible_attention_heads_raise(self):
        with pytest.raises(ValueError, match="divisible"):
            AttentionReadoutLayer(
                make_args(in_features=9, variant="pma", num_attention_heads=2)
            )

    def test_flattened_input_raises(self):
        layer = AttentionReadoutLayer(make_args(variant="gated"))
        with pytest.raises(ValueError, match="flatten"):
            layer(torch.randn(1, 40, dtype=torch.float64), pos=0)


# ---------------------------------------------------------------------------
# Integration tier: real GraphModel on MUTAG
# ---------------------------------------------------------------------------


def _build_net(graph_data, para, seed=42):
    from simplegnn.models.model import GraphModel

    return GraphModel(graph_data=graph_data, para=para, seed=seed, device="cpu")


def _batched_input(graph_data, positions):
    slices = graph_data.slices["x"]
    x = torch.cat([graph_data.x[int(slices[p]):int(slices[p + 1])] for p in positions])
    return SimpleNamespace(x=x)


@pytest.mark.integration
@pytest.mark.parametrize("models_file", ATTENTION_FIXTURES)
def test_attention_readout_batched_matches_per_graph(share_gnn_setup_factory, models_file):
    """The batched forward keeps the batch on the leading axis while the
    per-graph forward keeps a singleton node axis, so the readout has two
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
@pytest.mark.parametrize("models_file", ATTENTION_FIXTURES)
def test_attention_readout_gradients(share_gnn_setup_factory, models_file):
    """Every readout parameter must receive a gradient -- a head-embedding
    table or attention projection that is silently detached would train as a
    random projection."""
    graph_data, para = share_gnn_setup_factory(models_file)
    net = _build_net(graph_data, para)

    positions = list(range(8))
    labels = torch.stack([graph_data[p].y for p in positions]).squeeze()
    outputs = torch.stack([net(graph_data[p], pos=p) for p in positions])
    torch.nn.CrossEntropyLoss()(outputs, labels).backward()

    readout = next(l for l in net.net_layers if "Attention Readout" in l.name)
    for name, param in readout.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.any(param.grad != 0), f"{name} received an all-zero gradient"
