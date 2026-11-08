"""Comparable behaviour statistics on genuine action-choice opportunities."""

import torch


def epsilon_probabilities(
    q_values: torch.Tensor, masks: torch.Tensor, epsilon: float
) -> torch.Tensor:
    """Return the actual masked epsilon-greedy behaviour distribution."""
    greedy = q_values.masked_fill(~masks, -torch.inf).argmax(-1)
    probabilities = masks.float() * (epsilon / masks.sum(-1, keepdim=True))
    probabilities.scatter_add_(
        -1,
        greedy.unsqueeze(-1),
        torch.full_like(greedy.unsqueeze(-1), 1 - epsilon, dtype=torch.float32),
    )
    return probabilities


class ExplorationStats:
    """Ignore completed agents and forced actions; aggregate by decision count."""

    def __init__(self) -> None:
        self.decisions = 0
        self.entropy_sum = self.non_greedy = self.stops = 0.0

    def update(
        self,
        probabilities: torch.Tensor,
        masks: torch.Tensor,
        actions: torch.Tensor,
        valid: torch.Tensor,
        greedy_actions: torch.Tensor | None = None,
    ) -> None:
        probabilities = probabilities.detach().cpu()
        masks, actions, valid = masks.cpu(), actions.cpu(), valid.cpu()
        choices = masks.sum(-1)
        selected = valid & (choices > 1)
        self.decisions += int(selected.sum())
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
        self.entropy_sum += float(
            (entropy / choices.clamp_min(2).log())[selected].sum()
        )
        greedy = (
            probabilities.argmax(-1)
            if greedy_actions is None
            else greedy_actions.detach().cpu()
        )
        self.non_greedy += float((actions != greedy)[selected].sum())
        self.stops += float((actions == 4)[selected].sum())

    def summary(self) -> dict:
        count = self.decisions
        return {
            "decision_count": count,
            "normalized_entropy": self.entropy_sum / count if count else None,
            "non_greedy_rate": self.non_greedy / count if count else None,
            "stop_rate": self.stops / count if count else None,
        }
