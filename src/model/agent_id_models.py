"""Shared MLPs that also receive a one-hot train ID.

A shared network gives trains with identical observations (for example two
trains waiting at the same station) identical action distributions. Adding
the train ID lets one set of parameters break that symmetry while still
pooling every train's experience.
"""

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from .registry import register


class _AgentIdMLP(nn.Module):
    """Observation features concatenated with a one-hot ID of num_agents entries."""

    def __init__(
        self,
        output_dim: int,
        input_dim: int,
        num_agents: int,
        hidden_dims: Sequence[int],
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_agents = num_agents
        layers: list[nn.Module] = []
        previous_dim = input_dim + num_agents
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(previous_dim, hidden_dim), nn.ReLU()])
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def _features(self, observations: Sequence[Any], agent_ids: Tensor) -> Tensor:
        parameter = next(self.parameters())
        features = torch.stack([
            torch.zeros(self.input_dim, dtype=parameter.dtype)
            if observation is None else torch.as_tensor(
                observation, dtype=parameter.dtype, device="cpu",
            )
            for observation in observations
        ])
        one_hot = nn.functional.one_hot(agent_ids, self.num_agents).to(features)
        return torch.cat([features, one_hot], dim=-1).to(parameter.device)

    def forward(self, observations, current_infos, *, agent_id: int, **_) -> Tensor:
        ids = torch.tensor([agent_id])
        return self.network(self._features([observations[agent_id]], ids))[0]

    def forward_batch(self, observations: Sequence[Any]) -> Tensor:
        """Batched inputs from the shared wrappers, ordered time-major by train ID.

        ShareNetworkActor/ShareNetworkCritic flatten [time, agent] with agents
        in handle order 0..num_agents-1, so a sample's ID is its index modulo
        num_agents.
        """
        if len(observations) % self.num_agents:
            raise ValueError(
                "Batched inputs must hold all num_agents trains for each time step."
            )
        ids = torch.arange(len(observations)) % self.num_agents
        return self.network(self._features(observations, ids))


@register(name="mlp_policy_agent_id")
class MLPPolicyAgentId(_AgentIdMLP):
    """Five action logits from a flattened tree observation and a train ID."""

    def __init__(self, input_dim: int = 252, num_agents: int = 2,
                 hidden_dims: Sequence[int] = (64, 64)) -> None:
        super().__init__(5, input_dim, num_agents, hidden_dims)


@register(name="mlp_critic_agent_id")
class MLPCriticAgentId(_AgentIdMLP):
    """One state value from a flattened tree observation and a train ID."""

    def __init__(self, input_dim: int = 252, num_agents: int = 2,
                 hidden_dims: Sequence[int] = (64, 64)) -> None:
        super().__init__(1, input_dim, num_agents, hidden_dims)
