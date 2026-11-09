"""Centralized state-value model for MAPPO-style training.

The actor stays shared and decentralized (each train acts on its own
observation); only the critic, used during training, sees all trains. Each
train's value is computed from its own flattened tree observation together
with the mean over every train's observation, so the critic works for any
number of trains and the same weights can be tested at other sizes.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .registry import register


@register(name="mlp_critic_centralized")
class MLPCentralizedCritic(nn.Module):
    """Per-train values V(o_i, mean_j o_j) for use with CentralizedCritic (team_value: false)."""

    def __init__(self, input_dim: int = 252, hidden_dims: Sequence[int] = (64, 64)) -> None:
        super().__init__()
        self.input_dim = input_dim
        layers: list[nn.Module] = []
        previous_dim = 2 * input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(previous_dim, hidden_dim), nn.ReLU()])
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        observations: dict[int, Any],
        current_infos: dict[str, dict[int, Any]],
        *,
        additional_infos: dict[str, Any] | None = None,
        team_value: bool = False,
    ) -> dict[int, Tensor]:
        """Return {train: value of shape (1,)}, preserving gradients. Missing observations are zeros."""
        if team_value:
            raise ValueError("mlp_critic_centralized returns per-train values; set team_value: false.")
        agent_ids = sorted(observations)
        values = self.forward_batch([observations])[0]
        return {i: values[row] for row, i in enumerate(agent_ids)}

    def forward_batch(self, observations: Sequence[dict[int, Any]]) -> Tensor:
        """Values of shape [time, train, 1] for a list of per-step observation dicts (trains in ID order)."""
        parameter = next(self.parameters())
        agent_ids = sorted(observations[0])
        zeros = np.zeros(self.input_dim, dtype=np.float32)
        local = torch.from_numpy(np.stack([
            np.stack([zeros if step[i] is None else np.asarray(step[i], dtype=np.float32) for i in agent_ids])
            for step in observations
        ])).to(device=parameter.device, dtype=parameter.dtype)
        context = local.mean(dim=1, keepdim=True).expand_as(local)
        return self.network(torch.cat([local, context], dim=-1))
