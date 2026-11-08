"""Exact-budget, clean-seed PPO/PQN experiments on the upstream 4.3 protocol.

Training evaluates only validation maps. Testing is a separate, explicitly invoked
upstream scripts.evaluate command after all models/settings are frozen.
"""

import argparse
import hashlib
import json
import subprocess
from copy import deepcopy
from importlib.metadata import version
from math import ceil
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import yaml

from scripts.evaluate import episode
from src.eval.protocol import (
    RESERVED_SEEDS,
    VALIDATION_SEEDS,
    check_training_config,
    competition_map,
)
from src.trainer import build_trainer
from src.trainer.exact_ppo import ExactPPO
from src.utils.agents import build_value_agent
from src.utils.environment import build_env
from src.utils.memory import memory_usage
from src.utils.seed import preserve_rng, set_random_seed

LOADER = "src.eval.value:load_checkpoint"
UPSTREAM_BASE_COMMIT = "b254928e16500e7000eccd308642385637be592a"


def source_provenance() -> dict:
    """Record the actual checkout and source bytes, including uncommitted work."""
    root = Path(__file__).resolve().parents[1]
    files = sorted(
        p for folder in ("src", "scripts") for p in (root / folder).rglob("*.py")
    )
    source_hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return dict(commit=commit, dirty=dirty, sha256=source_hashes)


def prepare_config(config: dict, seed: int, steps: int, device: str) -> dict:
    """Use the same ordered, disjoint map pool for both learning algorithms."""
    if type(steps) is not int or steps <= 0 or type(seed) is not int or seed < 0:
        raise ValueError(
            "steps must be positive and seed must be nonnegative integers."
        )
    config = deepcopy(config)
    if config.get("init_from"):
        raise ValueError("This matched-budget driver requires fresh initialization.")
    config.get("wandb", {}).pop("api_key", None)
    config.update(competition_map(int(config["number_of_agents"])))
    maps = config["training_maps"]
    if maps.get("mode") != "pool" or maps.get("sizes"):
        raise ValueError(
            "This matched comparison requires one size and an explicit pool."
        )
    pool = [100_000 + 1000 * seed + i for i in range(int(maps["pool_size"]))]
    if set(pool) & RESERVED_SEEDS:
        raise ValueError(
            "The entire training pool must be disjoint from held-out seeds."
        )
    maps["pool"] = pool
    config.update(
        random_map=True, map_seed=pool[0], train_map_seeds=pool, total_env_steps=steps
    )
    config["trainer"].update(training_seed=seed, device=device)
    # Schedulers must not depend on the number of diagnostic checkpoints.
    if config["trainer"]["type"] in ("ppo", "ppo_demo"):
        config["total_epochs"] = ceil(steps / config["rollout_steps"])
        if (
            config["lr_schedule"] != "constant"
            or config["entropy_schedule"] != "constant"
        ):
            raise ValueError("The episode driver requires constant schedules.")
    problems = check_training_config(config)
    if problems:
        raise ValueError("; ".join(problems))
    return config


def validate(path: Path, seeds: list[int]) -> list[dict]:
    config = torch.load(path, map_location="cpu", weights_only=True)["config"]
    with preserve_rng():
        return [
            episode(
                dict(
                    checkpoint=str(path),
                    loader=LOADER,
                    run=path.parent.name,
                    epoch=int(path.stem),
                    env_steps=int(path.stem),
                    split="validation",
                    selection="curve",
                    mode="sampled",
                    episode=0,
                    trains=config["number_of_agents"],
                    map_seed=seed,
                )
            )
            for seed in seeds
        ]


def run(config: dict, out: Path, interval: int, validation_seeds: list[int]) -> dict:
    if version("flatland-rl") != "4.3.0":
        raise RuntimeError("This experiment is frozen to flatland-rl==4.3.0.")
    if out.exists():
        raise FileExistsError(f"Refusing to mix/overwrite a run: {out}")
    if not validation_seeds or not set(validation_seeds) <= set(VALIDATION_SEEDS):
        raise ValueError("During training only upstream validation seeds may be read.")
    worlds = config["trainer"].get("num_envs", 1)
    if interval <= 0 or interval % worlds or config["total_env_steps"] % worlds:
        raise ValueError("Budget/checkpoint interval must be divisible by world count.")
    out.mkdir(parents=True)
    config = deepcopy(config)
    config["save_dir"] = str(out)
    config["visualize_dir"] = str(out / "visualize")
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    set_random_seed(config["trainer"]["training_seed"])
    method = config["trainer"]["type"]
    experiment = (
        ExactPPO(config)
        if method in ("ppo", "ppo_demo")
        else build_trainer(
            method,
            env=build_env(config),
            value_agent=build_value_agent(config),
            config=config,
        )
    )
    meta = dict(
        upstream_base_commit=UPSTREAM_BASE_COMMIT,
        source=source_provenance(),
        config_sha256=hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
        flatland=version("flatland-rl"),
        torch=version("torch"),
        method=method,
        protocol="team-flatland43",
        fresh_initialization=True,
        prior_training_steps=0,
        q_evaluation="greedy in both mode labels"
        if method not in ("ppo", "ppo_demo")
        else None,
        total_env_steps=config["total_env_steps"],
        validation_seeds=validation_seeds,
        test_read_during_training=False,
        threads=torch.get_num_threads(),
    )
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    training_seconds = validation_seconds = 0.0
    for target in (
        [0]
        + list(range(interval, config["total_env_steps"], interval))
        + [config["total_env_steps"]]
    ):
        if target:
            start = perf_counter()
            metrics = experiment.train_steps(target - experiment.env_steps)
            training_seconds += perf_counter() - start
            metrics.update(training_seconds=training_seconds, **memory_usage())
            with (out / "training.jsonl").open("a") as file:
                file.write(json.dumps(metrics, allow_nan=False) + "\n")
            print(json.dumps(metrics, allow_nan=False), flush=True)
        path = out / f"{target}.pt"
        experiment.save_checkpoint(path)
        start = perf_counter()
        rows = validate(path, validation_seeds)
        validation_seconds += perf_counter() - start
        with (out / "validation.jsonl").open("a") as file:
            for row in rows:
                file.write(json.dumps(row, allow_nan=True) + "\n")
        print(
            f"validation steps={target} completion={np.mean([r['completion_rate'] for r in rows]):.4f} "
            f"score={np.mean([r['competition_score'] for r in rows]):.4f}",
            flush=True,
        )
    summary = dict(
        meta,
        completed_env_steps=experiment.env_steps,
        training_seconds=training_seconds,
        validation_seconds=validation_seconds,
        gradient_updates=experiment.gradient_updates,
    )
    (out / "complete.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--steps", type=int, help="override the configured interaction budget"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        help="override checkpoint_interval (joint environment ticks)",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--smoke-validation",
        action="store_true",
        help="two validation maps; pilot only",
    )
    args = parser.parse_args()
    recipe = yaml.safe_load(args.config.read_text())
    steps = recipe["total_env_steps"] if args.steps is None else args.steps
    interval = (
        recipe["checkpoint_interval"]
        if args.checkpoint_interval is None
        else args.checkpoint_interval
    )
    if args.threads <= 0 or steps <= 0:
        parser.error("threads and steps must be positive")
    torch.set_num_threads(args.threads)
    config = prepare_config(recipe, args.seed, steps, args.device)
    seeds = list(VALIDATION_SEEDS[:2] if args.smoke_validation else VALIDATION_SEEDS)
    run(config, args.out, interval, seeds)


if __name__ == "__main__":
    main()
