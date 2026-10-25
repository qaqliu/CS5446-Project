"""Tests for training-map strategies in scripts/train.py."""

import random
import warnings
from pathlib import Path

import yaml

from scripts.train import make_reset, pool_seeds, prepare_config
from src.eval.protocol import RESERVED_SEEDS, check_training_config
from src.utils.environment import build_env

warnings.filterwarnings("ignore", module="flatland")
RECIPES = Path(__file__).resolve().parent.parent / "config/protocol"


def load(name: str, seed: int = 42) -> dict:
    return prepare_config(yaml.safe_load((RECIPES / f"{name}.yaml").read_text()), seed, None)


def scenario(env):
    return env.rail.grid.tobytes(), tuple(a.initial_configuration for a in env.agents)


def test_all_recipes_are_leak_free():
    for path in RECIPES.glob("*.yaml"):
        config = prepare_config(yaml.safe_load(path.read_text()), 42, None)
        assert check_training_config(config) == [], path.name


def test_pool_seeds_are_disjoint_from_held_out_and_across_seeds():
    assert not set(pool_seeds(100, 42)) & RESERVED_SEEDS
    assert not set(pool_seeds(100, 42)) & set(pool_seeds(100, 43))


def test_pool_mode_only_draws_pool_maps():
    config = load("shared_ppo_pool20")
    config["training_maps"]["pool"] = config["training_maps"]["pool"][:3]
    env, reset = build_env(config), make_reset(config)
    seen = set()
    for _ in range(12):
        reset(env, config)
        seen.add(scenario(env))
    assert len(seen) <= 3


def test_random_and_fixed_modes():
    random_cfg, fixed_cfg = load("shared_ppo_random_maps"), load("shared_ppo_fixed_map")
    for config, expected in ((random_cfg, 4), (fixed_cfg, 1)):
        env, reset = build_env(config), make_reset(config)
        seen = set()
        for _ in range(4):
            reset(env, config)
            seen.add(scenario(env))
        assert len(seen) == expected


def test_multi_size_changes_train_count_and_map():
    config = load("comm_ppo_multi_size")
    env, reset = build_env(config), make_reset(config)
    sizes = set()
    for _ in range(10):
        reset(env, config)
        sizes.add((len(env.agents), env.width))
        assert env.agent_positions.shape == (env.height, env.width)
    assert sizes <= {(5, 25), (10, 29), (20, 32)} and len(sizes) > 1
