"""Simple policy models for flattened observations."""

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from .registry import register


# single agent policy model
@register(name="mlp_policy_demo")
class MLPPolicy(nn.Module):
    """Produce five action logits from one agent's normalized flattened tree.

    Depth-2 observations contain 252 features. Set input_dim to the observation
    builder's output length when changing depth. hidden_dims specifies each
    hidden layer's width; an empty sequence gives a linear policy.
    Missing observations use a zero vector. Only the selected agent's
    observation is read; info dictionaries are accepted for Actor compatibility.
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
        layers.append(nn.Linear(previous_dim, 5))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        observations: dict[int, Any],
        current_infos: dict[str, dict[int, Any]],
        *,
        addtional_infos: dict[str, Any] | None = None,
        agent_id: int,
    ) -> Tensor:
        """Return unnormalized logits of shape (5,), preserving gradients."""
        parameter = next(self.parameters())
        observation = observations[agent_id]
        if observation is None:
            features = parameter.new_zeros(self.input_dim)
        else:
            features = torch.as_tensor(
                observation, dtype=parameter.dtype, device=parameter.device
            )
        return self.network(features)


    def forward_batch(self, observations: Sequence[Any]) -> Tensor:
        """Evaluate local observations together, with the same None -> zero rule.

        This opt-in path is for the stateless Linear/ReLU demo network. Stack
        inputs on CPU first so CUDA receives one batch instead of many tiny copies.
        """
        parameter = next(self.parameters())
        features = torch.stack([
            torch.zeros(self.input_dim, dtype=parameter.dtype)
            if observation is None else torch.as_tensor(
                observation, dtype=parameter.dtype, device="cpu",
            )
            for observation in observations
        ]).to(parameter.device)
        return self.network(features)

# team policy model

@register(name="communication_mlp_policy")
class CommunicationMLPPolicy(nn.Module):
    """Inject learned multi-head communication into a local MLP policy.

    Encode every train, project features to messages, attend to other messages,
    then concatenate the communication result with the query train's features.
    Mask rows/columns follow list(observations); True permits communication.
    No train-count-dependent parameters, dropout, or normalization layers.
    """

    def __init__(
        self,
        input_dim: int = 1020,
        hidden_dims: Sequence[int] = (256, 128),
        communication_dim: int = 32,
        num_heads: int = 4,
        head_hidden_dims: Sequence[int] = (128,),
    ) -> None:
        super().__init__()
        if communication_dim <= 0 or num_heads <= 0 or communication_dim % num_heads:
            raise ValueError("communication_dim must be positive and divisible by num_heads.")
        self.input_dim = input_dim
        layers: list[nn.Module] = []
        feature_dim = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(feature_dim, width), nn.ReLU()])
            feature_dim = width
        self.encoder = nn.Sequential(*layers)
        self.message = nn.Sequential(nn.Linear(feature_dim, communication_dim), nn.ReLU())
        self.attention = nn.MultiheadAttention(
            communication_dim, num_heads, dropout=0.0, batch_first=True,
        )
        layers = []
        previous_dim = feature_dim + communication_dim
        for width in head_hidden_dims:
            layers.extend([nn.Linear(previous_dim, width), nn.ReLU()])
            previous_dim = width
        layers.append(nn.Linear(previous_dim, 5))
        self.head = nn.Sequential(*layers)

    def forward(
        self,
        observations: dict[int, Any],
        attention_mask: Tensor | None = None,
        agent_id: int | None = None,
    ) -> Tensor | dict[int, Tensor]:
        """Return one train's [5] logits, or a dict for all trains.

        All observations provide keys/values, even when selecting one query.
        Missing observations become zeros. Queries with no allowed key receive
        zero communication features; finished queries may still read active keys.
        """
        agent_ids = list(observations)
        if agent_id is not None and agent_id not in observations:
            raise KeyError(f"Unknown agent_id: {agent_id}")
        n = len(agent_ids)
        if n == 0:
            return {}
        parameter = next(self.parameters())
        inputs = torch.stack([
            parameter.new_zeros(self.input_dim) if observations[i] is None else
            torch.as_tensor(observations[i], dtype=parameter.dtype, device=parameter.device)
            for i in agent_ids
        ])
        if attention_mask is None:
            attention_mask = torch.ones(n, n, dtype=torch.bool, device=parameter.device)
        elif attention_mask.shape != (n, n) or attention_mask.dtype != torch.bool:
            raise ValueError("attention_mask must be a boolean [num_agents, num_agents] tensor.")
        else:
            attention_mask = attention_mask.to(parameter.device)

        # Step 1: Each train encodes its observation and emits a compact message.
        features = self.encoder(inputs)
        messages = self.message(features)
        selected = slice(None)
        if agent_id is not None:
            index = agent_ids.index(agent_id)
            selected = slice(index, index + 1)

        # Step 2: Attend across trains, never across unrelated time steps.
        allowed = attention_mask[selected]
        no_keys = ~allowed.any(dim=-1)
        # A temporary key prevents all-masked softmax NaNs; its output is discarded.
        safe_allowed = allowed.clone()
        safe_allowed[no_keys, 0] = True
        communication, _ = self.attention(
            messages[selected].unsqueeze(0), messages.unsqueeze(0), messages.unsqueeze(0),
            attn_mask=~safe_allowed, need_weights=False,
        )
        communication = communication.squeeze(0).masked_fill(no_keys.unsqueeze(-1), 0.0)

        # Step 3: Preserve the local feature path and inject communication beside it.
        logits = self.head(torch.cat([features[selected], communication], dim=-1))
        if agent_id is not None:
            return logits[0]
        return {i: logits[row] for row, i in enumerate(agent_ids)}


    def forward_batch(
        self,
        observations: Sequence[dict[int, Any]],
        attention_mask: Tensor | None = None,
        agent_id: int | None = None,
    ) -> Tensor | dict[int, Tensor]:
        """Parallel time steps [T, N, D]; attention stays within each time step.

        Observation IDs/order must match across time. Masks are [T, N, N], or
        one [N, N] mask shared across time. Return [T, 5] per selected agent.
        Network parameters and the rollout-time forward path remain unchanged.
        """
        if not observations:
            raise ValueError("forward_batch requires at least one time step.")
        agent_ids = list(observations[0])
        if any(list(obs) != agent_ids for obs in observations):
            raise ValueError("Batch observation IDs and order must match across time.")
        if agent_id is not None and agent_id not in observations[0]:
            raise KeyError(f"Unknown agent_id: {agent_id}")
        batch_size, n = len(observations), len(agent_ids)
        if n == 0:
            return {}
        parameter = next(self.parameters())
        # Stack rollout data before transferring, rather than copying one train at a time.
        inputs = torch.stack([
            torch.zeros(self.input_dim, dtype=parameter.dtype) if obs[i] is None else
            torch.as_tensor(obs[i], dtype=parameter.dtype, device="cpu")
            for obs in observations for i in agent_ids
        ]).reshape(batch_size, n, self.input_dim).to(parameter.device)
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, n, n, dtype=torch.bool, device=parameter.device)
        else:
            if attention_mask.dtype != torch.bool or attention_mask.shape not in (
                (n, n), (batch_size, n, n),
            ):
                raise ValueError("Batch attention_mask must be boolean [T, N, N] or [N, N].")
            attention_mask = attention_mask.to(parameter.device)
            if attention_mask.ndim == 2:
                attention_mask = attention_mask.unsqueeze(0).expand(batch_size, -1, -1)

        features = self.encoder(inputs)
        messages = self.message(features)
        selected = slice(None)
        if agent_id is not None:
            index = agent_ids.index(agent_id)
            selected = slice(index, index + 1)
        allowed = attention_mask[:, selected, :]
        no_keys = ~allowed.any(dim=-1)
        safe_allowed = allowed.clone()
        safe_allowed[..., 0] |= no_keys
        # MultiheadAttention expects one mask per (time, head); time is a batch axis.
        communication, _ = self.attention(
            messages[:, selected], messages, messages,
            attn_mask=(~safe_allowed).repeat_interleave(self.attention.num_heads, dim=0),
            need_weights=False,
        )
        communication = communication.masked_fill(no_keys.unsqueeze(-1), 0.0)
        logits = self.head(torch.cat([features[:, selected], communication], dim=-1))
        if agent_id is not None:
            return logits[:, 0]
        return {i: logits[:, row] for row, i in enumerate(agent_ids)}
