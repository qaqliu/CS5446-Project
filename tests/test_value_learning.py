"""Numerical regressions for PQN bootstrapping and episode boundaries."""

import pytest
import torch

from src.agent import ShareNetworkValueAgent
from src.model import build_model
from src.trainer import PQNTrainer, get_trainer
from src.trainer.value_targets import masked_max, q_lambda_returns
from src.utils.exploration import ExplorationStats, epsilon_probabilities


def test_masked_bootstrap_ignores_terminal_nan_and_rejects_empty_live_mask():
    values = torch.tensor([[7.0, 100.0, 1.0, 0.0, 0.0], [float("nan")] * 5])
    masks = torch.tensor([[True, False, True, False, False], [False] * 5])
    torch.testing.assert_close(
        masked_max(values, masks, torch.tensor([False, True])), torch.tensor([7.0, 0.0])
    )
    with pytest.raises(ValueError, match="legal action"):
        masked_max(values, masks, torch.tensor([False, False]))


@pytest.mark.parametrize(
    "trace_lambda, expected",
    [
        (0.0, [[5.5, 60.0], [2.0, 120.0], [0.0, 34.0]]),
        (1.0, [[2.0, 28.5], [2.0, 37.0], [0.0, 34.0]]),
    ],
)
def test_q_lambda_stops_at_terminal_and_padding_but_bootstraps_chunk_edge(
    trace_lambda, expected
):
    rewards = torch.tensor([[1.0, 10.0], [2.0, 20.0], [99.0, 30.0]])
    dones = torch.tensor([[False, False], [True, False], [True, False]])
    valid = torch.tensor([[True, True], [True, True], [False, True]])
    next_values = torch.tensor([[9.0, 100.0], [1000.0, 200.0], [1000.0, 8.0]])
    result = q_lambda_returns(rewards, dones, valid, next_values, 0.5, trace_lambda)
    torch.testing.assert_close(result, torch.tensor(expected))
    assert not result.requires_grad


def test_q_lambda_worlds_and_trains_do_not_leak_across_episode_reset():
    rewards = torch.tensor([[[1.0, 2.0]], [[10.0, 20.0]]])
    # The first world episode ends at t=0; t=1 is a freshly reset episode.
    dones = torch.tensor([[[True, True]], [[False, False]]])
    result = q_lambda_returns(
        rewards, dones, torch.ones_like(dones), torch.full_like(rewards, 8.0), 0.5, 0.65
    )
    torch.testing.assert_close(result, torch.tensor([[[1.0, 2.0]], [[14.0, 24.0]]]))


def test_registered_q_model_and_masked_behaviour_distribution():
    model = build_model("mlp_q", input_dim=2, hidden_dims=[4], layer_norm=True)
    agent = ShareNetworkValueAgent(model)
    observations = {0: [1.0, 2.0], 1: None}
    masks = {
        0: torch.tensor([False, True, False, False, False]),
        1: torch.tensor([True, False, False, False, False]),
    }
    _, values, actions = agent(observations, {}, action_mask=masks, epsilon=1)
    assert actions == {0: 1, 1: 0}
    assert values[0].requires_grad
    assert get_trainer("pqn") is PQNTrainer
    q = torch.tensor([[0.0, 9.0, 3.0, 2.0, 1.0]])
    mask = torch.tensor([[True, False, True, False, False]])
    torch.testing.assert_close(
        epsilon_probabilities(q, mask, 0.2), torch.tensor([[0.1, 0.0, 0.9, 0.0, 0.0]])
    )


def test_exploration_ignores_forced_completed_steps_and_uses_q_greedy_at_epsilon_one():
    stats = ExplorationStats()
    masks = torch.tensor(
        [
            [True, False, False, False, True],
            [True, False, False, False, False],
            [True, False, False, False, True],
        ]
    )
    probabilities = masks.float() / masks.sum(-1, keepdim=True)
    stats.update(
        probabilities,
        masks,
        torch.tensor([4, 0, 0]),
        torch.tensor([True, True, False]),
        greedy_actions=torch.tensor([4, 0, 4]),
    )
    assert stats.summary() == dict(
        decision_count=1, normalized_entropy=1.0, non_greedy_rate=0.0, stop_rate=1.0
    )
