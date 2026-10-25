"""Train a policy with a configurable training-map strategy.

    python -m scripts.train --config config/protocol/<recipe>.yaml [--seed 42] [--out DIR]

Runs the project trainer selected by the YAML (unchanged), adding three
things around it without touching shared code:

1. Training-map strategy, from an optional `training_maps` block:
       training_maps:
         generator: competition   # map size/cities/t_max from the train count (else YAML map fields)
         mode: random             # fixed | random | pool
         pool_size: 20            # pool mode: maps drawn from this many seeds
         sizes: [5, 10, 20]       # optional: new train count each episode (episode-based `ppo` only)
   Pool seeds are 100000 + 1000 * training_seed + i, disjoint from held-out seeds.
2. Environment steps are counted and saved in every checkpoint (`env_steps`),
   so methods can be compared at equal interaction budgets.
3. Training refuses to start if it would use validation or test map seeds.

Add your own strategy by extending `make_reset` (it decides which map each
episode uses); everything else, including evaluation, stays the same.
"""

import argparse
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator

import src.trainer.policy_trainer as policy_trainer
from src.eval.protocol import check_training_config, competition_map
from src.trainer import build_trainer
from src.utils.agents import build_agents
from src.utils.environment import _make_timetable, build_env, reset_env
from src.utils.seed import set_random_seed

POOL_SEED_BASE = 100_000


def pool_seeds(size: int, training_seed: int) -> list[int]:
    return [POOL_SEED_BASE + 1000 * training_seed + i for i in range(size)]


def prepare_config(config: dict, seed: int | None, out: Path | None) -> dict:
    """Resolve the training_maps block into concrete config fields."""
    if seed is not None:
        config["trainer"]["training_seed"] = seed
    training_seed = config["trainer"]["training_seed"]
    maps = config.get("training_maps") or {}
    if maps.get("generator") == "competition":
        config.update(competition_map(config["number_of_agents"]))
    mode = maps.get("mode")
    if mode in ("random", "pool"):
        config["random_map"] = mode == "random"
        config.setdefault("map_seed", training_seed)
        if config["map_seed"] is None:
            config["map_seed"] = training_seed
    elif mode == "fixed":
        config["random_map"] = False
    if mode == "pool":
        maps["pool"] = pool_seeds(int(maps["pool_size"]), training_seed)
    if maps.get("sizes") and config["trainer"]["type"] != "ppo":
        raise ValueError("training_maps.sizes needs the episode-based `ppo` trainer.")
    config["training_maps"] = maps or None
    if out is not None:
        config["save_dir"] = str(out)
        config["visualize_dir"] = str(out / "visualize")
    return config


def _resize(env: RailEnv, config: dict, trains: int) -> None:
    """Switch the environment to the competition map for `trains` before a reset."""
    params = competition_map(trains)
    env.width, env.height, env.number_of_agents = params["width"], params["height"], trains
    env.rail_generator = sparse_rail_generator(
        max_num_cities=params["max_num_cities"], grid_mode=params["grid_mode"],
        max_rails_between_cities=params["max_rails_between_cities"],
        max_rail_pairs_in_city=params["max_rail_pairs_in_city"],
    )
    env.timetable_generator = partial(_make_timetable, max_episode_steps=params["max_episode_steps"])


def make_reset(config: dict):
    """Return the reset function the trainer will call at every episode start."""
    maps = config.get("training_maps") or {}
    rng = random.Random(config["trainer"]["training_seed"])
    pool, sizes = maps.get("pool"), maps.get("sizes")

    def reset(env, cfg):
        if sizes:
            _resize(env, cfg, rng.choice(sizes))
        if pool:
            result = env.reset(regenerate_rail=True, regenerate_schedule=True, random_seed=rng.choice(pool))
        elif sizes and not cfg["random_map"]:
            result = env.reset(regenerate_rail=True, regenerate_schedule=True, random_seed=cfg["map_seed"])
        elif sizes:
            result = env.reset(regenerate_rail=True, regenerate_schedule=True)
        else:
            return reset_env(env, cfg)
        if sizes:
            # Flatland allocates the occupancy grid only in __init__; resize it with the map.
            env.agent_positions = np.full((env.height, env.width), -1, dtype=int)
            env._update_agent_positions_map(ignore_old_positions=False)
        return result
    return reset


def count_env_steps(env: RailEnv) -> dict:
    counter = {"steps": 0}
    step = env.step

    def counted(actions):
        counter["steps"] += 1
        return step(actions)
    env.step = counted
    return counter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, help="training seed (overrides the YAML)")
    parser.add_argument("--out", type=Path, help="checkpoint directory (overrides save_dir)")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = prepare_config(config, args.seed, args.out)
    problems = check_training_config(config)
    if problems:
        raise SystemExit("; ".join(problems))
    if config.get("save_dir"):
        Path(config["save_dir"]).expanduser().mkdir(parents=True, exist_ok=True)
        (Path(config["save_dir"]).expanduser() / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    env = build_env(config)
    counter = count_env_steps(env)
    policy_trainer.reset_env = make_reset(config)
    save = torch.save

    def save_with_steps(obj, *a, **k):
        if isinstance(obj, dict) and "actor" in obj and "config" in obj:
            obj = dict(obj, env_steps=counter["steps"])
        return save(obj, *a, **k)
    policy_trainer.torch.save = save_with_steps

    set_random_seed(config["trainer"]["training_seed"])
    actor, critic = build_agents(config)
    trainer = build_trainer(config["trainer"]["type"], env=env, actor=actor, critic=critic, config=config)
    print(f"Training {config['trainer']['type']} | maps {config.get('training_maps')} | "
          f"save_dir {config['save_dir']}", flush=True)
    trainer.train()
    print(f"Training finished after {counter['steps']} environment steps.", flush=True)


if __name__ == "__main__":
    main()
