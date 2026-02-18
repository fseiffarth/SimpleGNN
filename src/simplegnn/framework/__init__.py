"""Framework for GNN training and evaluation."""

__all__ = ["FrameworkMain"]


def __getattr__(name: str):
    if name == "FrameworkMain":
        from simplegnn.framework.core import FrameworkMain
        return FrameworkMain
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
