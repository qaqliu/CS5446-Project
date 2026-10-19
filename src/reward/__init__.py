"""Training reward components and their registry."""

from .registry import build_reward, get_reward, register

__all__ = ["register", "get_reward", "build_reward"]
