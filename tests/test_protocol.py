"""Tests for the shared evaluation protocol."""

import warnings

import numpy as np

from src.eval.metrics import gini
from src.eval.protocol import (
    RESERVED_SEEDS, TEST_SEEDS, VALIDATION_SEEDS, check_training_config,
    competition_map, protocol_config, training_sizes,
)
from src.utils.environment import build_env, reset_env

warnings.filterwarnings("ignore", module="flatland")

BASE = {
    "random_map": True, "map_seed": 7, "width": 30, "height": 30, "max_num_cities": 3,
    "grid_mode": True, "max_rails_between_cities": 3, "max_rail_pairs_in_city": 3,
    "max_episode_steps": 99, "number_of_agents": 6, "speed_ratio_map": {1.0: 0.5, 0.5: 0.5},
    "intermediate_stops": {"enabled": True, "line_length": 3},
    "observation": {"type": "tree", "max_depth": 2, "observation_radius": 2, "predictor": None},
    "reward": {"type": "custom_training_reward", "config": {}},
    "trainer": {"type": "ppo", "training_seed": 0, "device": "cuda"},
}


def test_competition_map_matches_paper_sizes():
    assert [competition_map(n)["width"] for n in (5, 10, 20, 50, 100)] == [25, 29, 32, 40, 50]
    assert competition_map(5)["max_episode_steps"] == 420
    assert competition_map(50)["max_num_cities"] == 7


def test_protocol_enforces_task_settings_and_keeps_observation():
    config = protocol_config(BASE, trains=10, map_seed=1003)
    assert (config["width"], config["number_of_agents"], config["map_seed"]) == (29, 10, 1003)
    assert config["random_map"] is False
    assert config["speed_ratio_map"] == {1.0: 1.0}
    assert config["intermediate_stops"]["enabled"] is False
    assert config["max_episode_steps"] == competition_map(10)["max_episode_steps"]
    assert config["reward"]["type"] == "champion"
    assert config["trainer"]["device"] == "cpu"
    assert config["observation"] == BASE["observation"]
    assert BASE["number_of_agents"] == 6, "the input config must not be modified"


def test_seed_sets_are_disjoint_and_leaks_are_flagged():
    assert not set(VALIDATION_SEEDS) & set(TEST_SEEDS)
    assert check_training_config(dict(BASE, random_map=True)) == []
    assert check_training_config(dict(BASE, random_map=False, map_seed=1000))
    assert check_training_config(dict(BASE, training_maps={"pool": [5, 2003]}))
    assert 1000 in RESERVED_SEEDS and 2019 in RESERVED_SEEDS


def test_training_sizes_default_and_multi_size():
    assert training_sizes(BASE) == (6,)
    assert training_sizes(dict(BASE, training_maps={"sizes": [5, 10]})) == (5, 10)


def test_same_seed_reproduces_test_scenario():
    grids = []
    for _ in range(2):
        config = protocol_config(BASE, trains=5, map_seed=1000)
        env = build_env(config)
        reset_env(env, config)
        grids.append((env.rail.grid.copy(), [a.initial_configuration for a in env.agents]))
    assert np.array_equal(grids[0][0], grids[1][0]) and grids[0][1] == grids[1][1]


def test_gini():
    assert gini(np.array([0.0, 0.0, 0.0, 4.0])) == 0.75
    assert gini(np.array([3.0, 3.0])) == 0.0
