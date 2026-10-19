"""Reward registration and automatic discovery within src.reward."""

import importlib
import pkgutil

_REWARDS: dict[str, type] = {}
_DISCOVERED = False


def register(cls=None, *, name: str | None = None):
    """Register a reward with @register or @register(name="demo")."""
    def decorator(reward_cls):
        reward_name = name if name is not None else reward_cls.__name__
        if reward_name in _REWARDS:
            raise ValueError(f"Reward {reward_name!r} is already registered.")
        _REWARDS[reward_name] = reward_cls
        return reward_cls

    return decorator if cls is None else decorator(cls)


def _discover_rewards() -> None:
    """Import reward modules once so their registration decorators execute."""
    global _DISCOVERED
    if _DISCOVERED:
        return
    package = importlib.import_module(__package__)
    for module in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        if module.name != __name__:
            importlib.import_module(module.name)
    _DISCOVERED = True


def get_reward(name: str) -> type:
    """Return the registered reward class, discovering modules on first use."""
    _discover_rewards()
    if name not in _REWARDS:
        raise KeyError(f"Unknown reward {name!r}. Available: {sorted(_REWARDS)}")
    return _REWARDS[name]


def build_reward(name: str, reward_config: dict | None = None):
    """Create a reward with the complete reward.config mapping (or None)."""
    return get_reward(name)(reward_config)
