"""SimpleGNN: A PyTorch-based Graph Neural Network experimentation framework."""

__version__ = "0.1.0"
__author__ = "Florian Seiffarth"
__license__ = "Apache-2.0"

__all__ = ["__version__", "FrameworkMain", "GraphModel"]


def __getattr__(name: str):
    if name == "FrameworkMain":
        from simplegnn.framework.core import FrameworkMain
        return FrameworkMain
    if name == "GraphModel":
        from simplegnn.models.model import GraphModel
        return GraphModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
