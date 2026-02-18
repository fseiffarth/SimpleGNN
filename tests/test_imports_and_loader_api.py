import importlib
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_import_simplegnn_package():
    module = importlib.import_module("simplegnn")
    assert module.__version__


def test_framework_utils_exports_loaders():
    utils_module = importlib.import_module("simplegnn.framework.utils")
    assert "load_model" in utils_module.__all__
    assert "load_model_old" in utils_module.__all__


def test_frameworkmain_has_no_load_model_method():
    try:
        framework_module = importlib.import_module("simplegnn.framework.core")
    except ModuleNotFoundError as exc:
        pytest.skip(f"framework dependencies not available in this environment: {exc}")

    assert hasattr(framework_module, "FrameworkMain")
    assert not hasattr(framework_module.FrameworkMain, "load_model")
