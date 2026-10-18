"""Policy training entry point."""

import argparse
from copy import deepcopy
from pathlib import Path
from pprint import pprint

import yaml

from src.trainer import build_trainer
from src.utils.agents import build_agents
from src.utils.environment import build_env
from src.utils.seed import set_random_seed


def load_config(config_path: str | Path) -> dict:
    """Load the complete YAML configuration into a dictionary."""
    with Path(config_path).expanduser().open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping.")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a policy from a YAML config.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config/policy/single_map/ppo_demo_no_intermediate_stops.yaml",
        help="Path to the YAML config (default: the single-map PPO demo).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    printed_config = deepcopy(config)
    if "api_key" in printed_config.get("wandb", {}):
        printed_config["wandb"]["api_key"] = "<redacted>"
    pprint(printed_config, sort_dicts=False)
    env = build_env(config)
    # Seed model initialization before passing the models to the trainer.
    set_random_seed(config["trainer"]["training_seed"])
    actor, critic = build_agents(config)
    trainer = build_trainer(
        config["trainer"]["type"],
        env=env, actor=actor, critic=critic, config=config,
    )
    print(f"Environment: {env.width}x{env.height}, {config['number_of_agents']} trains")
    print(actor)
    print(critic)
    print(f"Trainer: {type(trainer).__name__} | device: {config['trainer']['device']}", flush=True)
    trainer.train()
    print("Training finished.")


if __name__ == "__main__":
    main()
