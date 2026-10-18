"""Simple state-value models for flattened observations."""

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from .registry import register


# single agent critic model
@register(name="mlp_critic_demo")
class MLPCritic(nn.Module):
    """Estimate one agent's state value from its normalized flattened tree.

    Depth-2 observations contain 252 features. Set input_dim to the observation
    builder's output length when changing depth. hidden_dims specifies each
    hidden layer's width; an empty sequence gives a linear value model.
    Missing observations use a zero vector. Only the selected agent's
    observation is read; info dictionaries are accepted for Critic compatibility.
    """

    def __init__(
        self,
        input_dim: int = 252,
        hidden_dims: Sequence[int] = (64, 64),
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        layers: list[nn.Module] = []
        previous_dim = input_dim
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
        agent_id: int,
    ) -> Tensor:
        """Return a state value of shape (1,), preserving gradients."""
        parameter = next(self.parameters())
        observation = observations[agent_id]
        if observation is None:
            features = parameter.new_zeros(self.input_dim)
        else:
            features = torch.as_tensor(
                observation, dtype=parameter.dtype, device=parameter.device
            )
        return self.network(features)
