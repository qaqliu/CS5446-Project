"""Evaluate fixed-size independent policies on maps with more trains by reusing their networks in rotation.

    python -m scripts.evaluate RUN_DIR --loader src.eval.rotation:load_rotation

An independent actor has one network per training train (handles 0..K-1), so it cannot run a map with more trains.
This loader gives train h the network of training train h mod K. It is an evaluation-only stand-in for "the same
independent policies on a denser map"; results are labelled as rotation in reports.
"""

from pathlib import Path

import torch

from src.eval.runner import load_actor_checkpoint


class RotationPolicy:
    def __init__(self, actor, deterministic: bool) -> None:
        self.actor = actor
        self.deterministic = deterministic
        self.k = len(actor.model)

    @torch.no_grad()
    def __call__(self, env, observations, infos, masks, decisions):
        logits = {h: self.actor.model[str(h % self.k)](observations, infos, addtional_infos=None, agent_id=h)
                  for h in observations}
        return self.actor._get_action(logits, masks, deterministic=self.deterministic)


def load_rotation(path: str | Path):
    _, config = load_actor_checkpoint(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    from src.utils.agents import build_agents
    actor, _ = build_agents(dict(checkpoint["config"], trainer=dict(checkpoint["config"].get("trainer", {}), device="cpu")))
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    config = dict(config, actor=dict(config["actor"], type="independent_rotation"))   # not fixed-size for scripts.evaluate
    return (lambda deterministic: RotationPolicy(actor, deterministic)), config
