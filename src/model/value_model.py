"""Local Q networks for shared Flatland agents.

PQN's normalisation follows Gallici et al., Simplifying Deep Temporal
Difference Learning (ICLR 2025), https://arxiv.org/abs/2407.04811.
This is a PyTorch adaptation for vector observations, not the JAX benchmark.
"""

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from .registry import register


@register(name="mlp_q")
class MLPQ(nn.Module):
    """Map a local flattened tree to five unconstrained action values."""

    def __init__(
        self,
        input_dim: int = 1020,
        hidden_dims: Sequence[int] = (64, 64),
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or any(width <= 0 for width in hidden_dims):
            raise ValueError("Network dimensions must be positive.")
        self.input_dim = input_dim
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers.append(nn.Linear(previous, width))
            if layer_norm:
                layers.append(nn.LayerNorm(width))
            layers.append(nn.ReLU())
            previous = width
        layers.append(nn.Linear(previous, 5))
        self.network = nn.Sequential(*layers)

    def encode(self, observations: Sequence[Any]) -> Tensor:
        """Copy snapshots to CPU; missing observations have zero features."""
        return torch.stack(
            [
                torch.zeros(self.input_dim, dtype=torch.float32)
                if obs is None
                else torch.as_tensor(obs, dtype=torch.float32).clone()
                for obs in observations
            ]
        )

    def forward_features(self, features: Tensor) -> Tensor:
        parameter = next(self.parameters())
        return self.network(features.to(parameter))

    def forward_batch(self, observations: Sequence[Any]) -> Tensor:
        return self.forward_features(self.encode(observations))

    def forward(
        self,
        observations: dict,
        current_infos: dict,
        *,
        agent_id: int,
        addtional_infos: dict | None = None,
    ) -> Tensor:
        return self.forward_batch([observations[agent_id]])[0]
