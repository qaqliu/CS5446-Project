"""Tests for the non-learning baselines."""

import warnings

import pytest
import yaml
from pathlib import Path

from src.baseline.planning import PrioritizedPlanningPolicy
from src.baseline.policies import DepartureGatedShortestPathPolicy, RandomPolicy, ShortestPathPolicy
from src.eval.protocol import protocol_config
from src.eval.runner import play
from src.utils.action_mask import get_action_mask
from src.utils.environment import build_env, reset_env

warnings.filterwarnings("ignore", module="flatland")
BASE = yaml.safe_load((Path(__file__).resolve().parent.parent
                       / "config/policy/single_map/ppo_demo_no_intermediate_stops.yaml").read_text())


@pytest.mark.parametrize("policy", [RandomPolicy(0), ShortestPathPolicy(),
                                    DepartureGatedShortestPathPolicy(2), PrioritizedPlanningPolicy()])
def test_policies_choose_only_allowed_actions(policy):
    config = protocol_config(BASE, trains=5, map_seed=1000)
    env = build_env(config)
    observations, infos = reset_env(env, config)
    while True:
        masks = get_action_mask(env, infos)
        decisions = {h: int(m.sum()) > 1 for h, m in masks.items()}
        actions = policy(env, observations, infos, masks, decisions)
        assert all(masks[h][a] for h, a in actions.items())
        observations, _, dones, infos = env.step(actions)
        if dones["__all__"]:
            break


def test_departure_cap_limits_trains_on_map():
    config = protocol_config(BASE, trains=10, map_seed=1001)
    env = build_env(config)
    observations, infos = reset_env(env, config)
    policy = DepartureGatedShortestPathPolicy(max_active=2)
    while True:
        masks = get_action_mask(env, infos)
        decisions = {h: int(m.sum()) > 1 for h, m in masks.items()}
        observations, _, dones, infos = env.step(policy(env, observations, infos, masks, decisions))
        assert sum(a.state.is_on_map_state() for a in env.agents) <= 2
        if dones["__all__"]:
            break


def test_prioritized_planning_completes_small_maps():
    for seed in (1000, 1001):
        summary = play(protocol_config(BASE, trains=5, map_seed=seed), PrioritizedPlanningPolicy())
        assert summary["completion_rate"] == 1.0
        assert summary["deadlock_rate"] == 0.0
