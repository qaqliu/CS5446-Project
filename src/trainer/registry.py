"""Trainer registration and automatic discovery within src.trainer."""

import importlib
import pkgutil

_TRAINERS: dict[str, type] = {}
_DISCOVERED = False


def register(cls=None, *, name: str | None = None):
    """Register a trainer with @register or @register(name="demo")."""
    def decorator(trainer_cls):
        trainer_name = name if name is not None else trainer_cls.__name__
        if trainer_name in _TRAINERS:
            raise ValueError(f"Trainer {trainer_name!r} is already registered.")
        _TRAINERS[trainer_name] = trainer_cls
        return trainer_cls

    return decorator if cls is None else decorator(cls)


def _discover_trainers() -> None:
    """Import trainer modules once so their registration decorators execute."""
    global _DISCOVERED
    if _DISCOVERED:
        return
    package = importlib.import_module(__package__)
    for module in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        if module.name != __name__:
            importlib.import_module(module.name)
    _DISCOVERED = True


def get_trainer(name: str) -> type:
    """Return the registered trainer class, discovering modules on first use."""
    _discover_trainers()
    if name not in _TRAINERS:
        raise KeyError(f"Unknown trainer {name!r}. Available: {sorted(_TRAINERS)}")
    return _TRAINERS[name]


def build_trainer(name: str, **kwargs):
    """Create a registered trainer, passing keyword arguments to its constructor."""
    return get_trainer(name)(**kwargs)
