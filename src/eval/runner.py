"""Play evaluation episodes and load policies from checkpoints.

A policy is any callable (env, observations, infos, masks, decisions) ->
{train_id: action}. Learned policies are loaded from checkpoints saved as
{"epoch", "config", "actor", ...} (the project trainers' format). Methods
with another format (e.g. value-based agents) provide a loader
`module:function(checkpoint_path) -> (policy_factory, config)` where
policy_factory(deterministic: bool) returns a policy.
"""

import importlib
from pathlib import Path
from typing import Callable

import torch

from src.eval.metrics import EpisodeTracker
from src.reward.deadlock import DeadlockChecker
from src.utils.action_mask import get_action_mask
from src.utils.agents import build_agents
from src.utils.environment import build_env, reset_env

PolicyFactory = Callable[[bool], Callable]


class ActorPolicy:
    """Run a project Actor (or any agent with the Actor call signature)."""

    def __init__(self, actor, deterministic: bool) -> None:
        self.actor = actor
        self.deterministic = deterministic

    @torch.no_grad()
    def __call__(self, env, observations, infos, masks, decisions):
        _, _, actions = self.actor(
            observations, infos, action_mask=masks, deterministic=self.deterministic,
        )
        return actions


def play(config: dict, policy) -> dict:
    """One episode on the scenario defined by `config`; returns episode metrics.

    deadlock_rate uses the team's DeadlockChecker (as in training logs);
    strict_deadlock_rate is the stricter cycle detector in src.eval.metrics.
    """
    env = build_env(config)
    observations, infos = reset_env(env, config)
    tracker = EpisodeTracker(env)
    checker = getattr(env.rewards, "deadlock_checker", None)
    own_checker = checker is None
    if own_checker:
        checker = DeadlockChecker()
        checker.reset(env)
    while True:
        masks = get_action_mask(env, infos)
        decisions = {h: int(m.sum()) > 1 for h, m in masks.items()}
        actions = policy(env, observations, infos, masks, decisions)
        observations, rewards, dones, infos = env.step(actions)
        if own_checker:
            checker.update_deadlocks()
        tracker.update(rewards)
        if dones["__all__"]:
            summary = tracker.summary()
            flagged = sum(bool(checker.is_deadlocked(a.handle)) for a in env.agents)
            summary["strict_deadlock_rate"] = summary.pop("deadlock_rate")
            summary["deadlock_rate"] = flagged / max(len(env.agents), 1)
            return summary


def load_actor_checkpoint(path: str | Path) -> tuple[PolicyFactory, dict]:
    """Default loader for checkpoints written by the project's policy trainers."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = dict(checkpoint["config"])
    config["trainer"] = dict(config.get("trainer", {}), device="cpu")
    actor, _ = build_agents(config)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    return (lambda deterministic: ActorPolicy(actor, deterministic)), config


def resolve_loader(spec: str | None):
    """`module:function` loader, or the default actor loader when None."""
    if not spec:
        return load_actor_checkpoint
    module, _, function = spec.partition(":")
    return getattr(importlib.import_module(module), function)


def checkpoint_epochs(run_dir: Path) -> list[Path]:
    """Numeric checkpoints `<epoch>.pt` of a run, in training order."""
    return sorted((p for p in run_dir.glob("*.pt") if p.stem.isdecimal()), key=lambda p: int(p.stem))


def checkpoint_env_steps(path: Path) -> int | None:
    """Environment steps recorded by scripts/train.py, when available."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    steps = checkpoint.get("env_steps")
    return int(steps) if steps is not None else None
