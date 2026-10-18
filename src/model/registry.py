"""Model registration and automatic discovery within src.model."""

import importlib
import pkgutil

from torch import nn

_MODELS: dict[str, type[nn.Module]] = {}
_DISCOVERED = False


def register(cls=None, *, name: str | None = None):
    """Register a model with @register or @register(name="demo")."""
    def decorator(model_cls):
        model_name = name if name is not None else model_cls.__name__
        if model_name in _MODELS:
            raise ValueError(f"Model {model_name!r} is already registered.")
        _MODELS[model_name] = model_cls
        return model_cls

    return decorator if cls is None else decorator(cls)


def _discover_models() -> None:
    """Import model modules once so their registration decorators execute."""
    global _DISCOVERED
    if _DISCOVERED:
        return
    package = importlib.import_module(__package__)
    for module in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        if module.name != __name__:
            importlib.import_module(module.name)
    _DISCOVERED = True


def get_model(name: str) -> type[nn.Module]:
    """Return the registered model class, discovering modules on first use."""
    _discover_models()
    if name not in _MODELS:
        raise KeyError(f"Unknown model {name!r}. Available: {sorted(_MODELS)}")
    return _MODELS[name]


def build_model(name: str, **kwargs) -> nn.Module:
    """Create a registered model, passing keyword arguments to its constructor."""
    return get_model(name)(**kwargs)
