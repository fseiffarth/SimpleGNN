"""GNN model implementations."""

__all__ = ["GraphModel"]


def __getattr__(name: str):
    if name == "GraphModel":
        from simplegnn.models.model import GraphModel
        return GraphModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
