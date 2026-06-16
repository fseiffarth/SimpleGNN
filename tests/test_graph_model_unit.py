from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from simplegnn.models.model import GraphModel


class AddOneLayer:
    def __call__(self, x, batch_data, *args, **kwargs):
        return x + 1


class MulTwoLayer:
    def __call__(self, x, batch_data, *args, **kwargs):
        return x * 2


class BatchStub:
    def __init__(self, x):
        self.x = x


def build_model_stub(random_variation_bool=False, precision="double"):
    model = GraphModel.__new__(GraphModel)
    model.net_layers = [AddOneLayer(), MulTwoLayer()]
    model.random_variation_bool = random_variation_bool
    model.para = type(
        "P",
        (),
        {
            "run_config": type(
                "RC",
                (),
                {
                    "config": {
                        "precision": precision,
                        "input_features": {"random_variation": {"mean": 0.0, "std": 0.0}},
                    }
                },
            )()
        },
    )()
    return model


def test_forward_applies_layers_in_order_without_variation():
    model = build_model_stub(random_variation_bool=False)
    batch = BatchStub(torch.tensor([[1.0, 2.0]], dtype=torch.double))

    out = GraphModel.forward(model, batch)

    # ((x + 1) * 2)
    assert torch.equal(out, torch.tensor([[4.0, 6.0]], dtype=torch.double))


def test_forward_random_variation_path_works_with_zero_std():
    model = build_model_stub(random_variation_bool=True, precision="float")
    batch = BatchStub(torch.tensor([[1.0]], dtype=torch.float))

    out = GraphModel.forward(model, batch)

    assert out.dtype == torch.float
    assert torch.equal(out, torch.tensor([[4.0]], dtype=torch.float))


def test_get_model_layer_unknown_type_raises():
    model = GraphModel.__new__(GraphModel)
    model.device = "cpu"
    model.seed = 1
    model.graph_data = type("GD", (), {"num_node_features": 3, "num_classes": 2})()
    model.net_layers = []
    model.precision = torch.float
    model.convolution_grad = True
    model.aggregation_grad = True
    model.para = type("P", (), {})()

    fake_layer = type(
        "L",
        (),
        {
            "layer_dict": {},
            "layer_id": 0,
            "layer_type": "does_not_exist",
        },
    )()

    with pytest.raises(ValueError, match="not recognized"):
        GraphModel.get_model_layer(model, fake_layer)


def test_return_info_returns_class_type():
    model = GraphModel.__new__(GraphModel)
    assert GraphModel.return_info(model) is GraphModel
