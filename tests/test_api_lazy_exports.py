import importlib

import pytest


@pytest.mark.parametrize(
    "module_name, expected",
    [
        ("simplegnn", "FrameworkMain"),
        ("simplegnn", "GraphModel"),
        ("simplegnn.framework", "FrameworkMain"),
        ("simplegnn.models", "GraphModel"),
        ("simplegnn.framework.utils", "load_model"),
        ("simplegnn.framework.utils", "load_model_old"),
    ],
)
def test_lazy_exports_expose_expected_symbols(module_name, expected):
    module = importlib.import_module(module_name)
    assert expected in dir(module)


def test_lazy_exports_unknown_attribute_raises():
    module = importlib.import_module("simplegnn")
    with pytest.raises(AttributeError):
        getattr(module, "DOES_NOT_EXIST")
