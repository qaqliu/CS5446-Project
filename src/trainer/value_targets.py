"""Masked Q(lambda) targets for local PQN training."""

import torch


def masked_max(
    q_values: torch.Tensor, masks: torch.Tensor, dones: torch.Tensor
) -> torch.Tensor:
    """Never bootstrap a terminal; reject missing legal nonterminal actions."""
    if ((~masks.any(-1)) & ~dones).any():
        raise ValueError("Nonterminal next states must have a legal action.")
    return q_values.masked_fill(~masks, -torch.inf).amax(-1).masked_fill(dones, 0.0)


@torch.no_grad()
def q_lambda_returns(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    valid: torch.Tensor,
    next_values: torch.Tensor,
    gamma: float,
    trace_lambda: float,
) -> torch.Tensor:
    """Time-first targets, stopping at arrivals/horizons but not chunk edges.

    A finite timetable horizon is terminal, matching this project's PPO.
    Padding after arrival is excluded. Each world/train column is independent.
    """
    targets = torch.zeros_like(rewards)
    carry = next_values[-1]
    for step in reversed(range(len(rewards))):
        bootstrap = (1 - trace_lambda) * next_values[step] + trace_lambda * carry
        carry = rewards[step] + gamma * bootstrap.masked_fill(dones[step], 0.0)
        carry = carry.masked_fill(~valid[step], 0.0)
        targets[step] = carry
    return targets
