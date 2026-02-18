"""Framework utility exports."""

__all__ = ["load_model", "load_model_old"]


def __getattr__(name: str):
    if name in {"load_model", "load_model_old"}:
        from simplegnn.framework.utils.load_model import load_model, load_model_old
        return {"load_model": load_model, "load_model_old": load_model_old}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
