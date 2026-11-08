"""Checkpoint loaders for Q agents under the upstream shared evaluator.

The default Q loader is greedy in both evaluation modes. Repeated "sampled"
rows are deterministic replications, not independent seeds or epsilon trials.
The separate epsilon loader changes only test execution when explicitly selected.
"""

from pathlib import Path

import torch
from flatland.envs.rail_env import RailEnv

from src.agent import ShareNetworkValueAgent
from src.agent.value import ActionMasks, Actions, Infos, Observations
from src.eval.runner import PolicyFactory, load_actor_checkpoint
from src.utils.agents import build_value_agent
from src.utils.exploration import epsilon_probabilities


class GreedyValuePolicy:
    """Execute a local Q network using the evaluator's legal action masks."""

    def __init__(self, agent: ShareNetworkValueAgent) -> None:
        self.agent = agent.eval()
        self.epsilon = 0.0

    @torch.no_grad()
    def __call__(
        self,
        env: RailEnv,
        observations: Observations,
        infos: Infos,
        masks: ActionMasks,
        decisions: dict[int, bool],
    ) -> Actions:
        handles = list(observations)
        values = self.agent.model.forward_batch([observations[h] for h in handles])
        legal = torch.stack([masks[h] for h in handles]).to(values.device)
        actions = values.masked_fill(~legal, -torch.inf).argmax(-1).tolist()
        return dict(zip(handles, actions))


class EpsilonGreedyValuePolicy(GreedyValuePolicy):
    """Apply test-time exploration to frozen Q weights; never update parameters."""

    def __init__(self, agent: ShareNetworkValueAgent, epsilon: float = 0.05) -> None:
        super().__init__(agent)
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be between zero and one.")
        self.epsilon = epsilon

    @torch.no_grad()
    def __call__(
        self,
        env: RailEnv,
        observations: Observations,
        infos: Infos,
        masks: ActionMasks,
        decisions: dict[int, bool],
    ) -> Actions:
        if self.epsilon == 0:
            return super().__call__(env, observations, infos, masks, decisions)
        handles = list(observations)
        values = self.agent.model.forward_batch(
            [observations[h] for h in handles]
        ).cpu()
        legal = torch.stack([masks[h] for h in handles]).cpu()
        probabilities = epsilon_probabilities(values, legal, self.epsilon)
        actions = torch.multinomial(probabilities, 1).squeeze(-1).tolist()
        return dict(zip(handles, actions))


def load_checkpoint(path: str | Path) -> tuple[PolicyFactory, dict]:
    """Return the upstream policy factory for either a Q or PPO checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "value_agent" not in checkpoint:
        return load_actor_checkpoint(path)
    config = checkpoint["config"]
    config["trainer"] = dict(config["trainer"], device="cpu")
    agent = build_value_agent(config)
    agent.load_state_dict(checkpoint["value_agent"])
    agent.eval()
    return lambda deterministic: GreedyValuePolicy(agent), config


def load_epsilon_checkpoint(path: str | Path) -> tuple[PolicyFactory, dict]:
    """Q-only test loader: sampled mode uses epsilon=.05; deterministic is greedy.

    Use with scripts.evaluate --test-loader so validation still selects using
    the primary greedy loader. Report this auxiliary execution separately.
    """
    make_policy, config = load_checkpoint(path)
    policy = make_policy(True)
    if not isinstance(policy, GreedyValuePolicy):
        raise ValueError("The epsilon execution loader requires a Q checkpoint.")
    return lambda deterministic: EpsilonGreedyValuePolicy(
        policy.agent, epsilon=0.0 if deterministic else 0.05
    ), config
