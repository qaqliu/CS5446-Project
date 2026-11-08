"""Regressions for clean splits, episode fragmentation and exact budgets."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from scripts.train_value import prepare_config
from src.eval.protocol import RESERVED_SEEDS, check_training_config, protocol_config
from src.eval.value import load_checkpoint
from src.trainer import build_trainer
from src.trainer.exact_ppo import ExactPPO
from src.utils.agents import build_value_agent
from src.utils.environment import build_env, reset_env
from src.utils.seed import preserve_rng, set_random_seed

ROOT = Path(__file__).resolve().parents[1]


def small_config(method, steps=32):
    config = yaml.safe_load(
        (ROOT / f"config/protocol/value_{method}_scale20.yaml").read_text()
    )
    config["number_of_agents"] = 5
    config.update(rollout_steps=4, update_epochs=1, batch_size=16)
    if method == "pqn":
        config["value_agent"]["model"]["hidden_dims"] = [8]
        config["trainer"]["num_envs"] = 2
    else:
        config["actor"]["model"]["hidden_dims"] = [8]
        config["critic"]["model"]["hidden_dims"] = [8]
    return prepare_config(config, seed=42, steps=steps, device="cpu")


def test_split_guard_checks_actual_pool_and_eval_removes_it():
    config = small_config("pqn")
    assert not set(config["train_map_seeds"]) & RESERVED_SEEDS
    bad = deepcopy(config)
    bad["train_map_seeds"].append(1003)
    assert check_training_config(bad)
    evaluation = protocol_config(config, trains=5, map_seed=2000)
    assert "train_map_seeds" not in evaluation and "training_maps" not in evaluation
    env = build_env(evaluation)
    reset_env(env, evaluation)
    assert env.current_scenario_seed == 2000


@pytest.mark.parametrize("method", ["ppo", "pqn"])
def test_budget_training_and_upstream_loader_roundtrip(method, tmp_path):
    torch.set_num_threads(1)
    set_random_seed(42)
    config = small_config(method)
    if method == "ppo":
        experiment = ExactPPO(config)
        agent = experiment.actor
    else:
        agent = build_value_agent(config)
        experiment = build_trainer(
            method, env=build_env(config), value_agent=agent, config=config
        )
    before = deepcopy(agent.state_dict())
    first = experiment.train_steps(12)
    assert first["env_steps"] == 12
    if method == "ppo":
        assert first["gradient_updates"] == 0
        assert first["pending_episode_steps"] == 12
    metrics = experiment.train_steps(20)
    assert metrics["env_steps"] == 32
    assert metrics["agent_transitions"] == 32 * 5
    assert metrics["gradient_updates"] > 0
    assert any(not torch.equal(before[k], v) for k, v in agent.state_dict().items())
    with pytest.raises(ValueError, match="budget"):
        experiment.train_steps(2)
    path = tmp_path / "32.pt"
    experiment.save_checkpoint(path)
    factory, saved_config = load_checkpoint(path)
    assert saved_config["trainer"]["device"] == "cpu"
    env_config = protocol_config(config, trains=5, map_seed=2000)
    env = build_env(env_config)
    obs, infos = reset_env(env, env_config)
    from src.utils.action_mask import get_action_mask

    masks = get_action_mask(env, infos)
    actions = factory(True)(env, obs, infos, masks, {})
    assert set(actions) == set(obs)
    assert all(bool(masks[h][a]) for h, a in actions.items())


def test_checkpoint_fragmentation_preserves_episode_ppo_weights():
    config = small_config("ppo")
    set_random_seed(42)
    whole = ExactPPO(config)
    whole.train_steps(32)
    expected_actor = deepcopy(whole.actor.state_dict())
    expected_critic = deepcopy(whole.critic.state_dict())
    set_random_seed(42)
    split = ExactPPO(config)
    split.train_steps(12)
    with preserve_rng():
        # Validation consumes RNG but must not change future training actions.
        torch.rand(50)
    split.train_steps(20)
    for key, value in expected_actor.items():
        torch.testing.assert_close(value, split.actor.state_dict()[key], rtol=0, atol=0)
    for key, value in expected_critic.items():
        torch.testing.assert_close(
            value, split.critic.state_dict()[key], rtol=0, atol=0
        )
    assert split.optimizer_steps == whole.optimizer_steps


def test_upstream_predictor_patch_remains_available():
    from src.utils import flatland_patches

    assert flatland_patches is not None


def test_matched_recipes_align_task_reward_observation_and_seed_pool():
    ppo, pqn = (small_config(m) for m in ("ppo", "pqn"))
    for key in ("observation", "reward", "train_map_seeds", "speed_ratio_map"):
        assert ppo[key] == pqn[key]
    assert ppo["observation"]["observation_radius"] == 20
    assert ppo["reward"]["config"]["deadlock_penalty"] == -0.5
    assert ppo["total_env_steps"] == pqn["total_env_steps"] == 32


def test_fresh_driver_rejects_warm_starts_and_removes_credentials():
    config = small_config("pqn")
    config["init_from"] = "earlier.pt"
    with pytest.raises(ValueError, match="fresh initialization"):
        prepare_config(config, 42, 32, "cpu")
    del config["init_from"]
    config["wandb"]["api_key"] = "test-only-placeholder"
    prepared = prepare_config(config, 42, 32, "cpu")
    assert "api_key" not in prepared["wandb"]
    assert config["wandb"]["api_key"] == "test-only-placeholder"


def test_upstream_curriculum_selection_still_includes_parent_sizes():
    from src.eval.protocol import training_sizes

    assert training_sizes(
        dict(
            number_of_agents=20,
            training_maps={
                "curriculum_sizes": [5, 10],
                "sizes": [20],
            },
        )
    ) == (5, 10, 20)


def test_memory_diagnostics_without_platform_resource_module(monkeypatch):
    from src.utils import memory

    monkeypatch.setattr(memory, "resource", None)
    assert memory.memory_usage()["peak_rss_mib"] is None


def test_unseeded_upstream_random_maps_remain_supported():
    config = small_config("pqn")
    config.pop("train_map_seeds")
    config.update(map_seed=None, random_map=True)
    env = build_env(config)
    observations, _ = reset_env(env, config)
    assert len(observations) == config["number_of_agents"]
    assert env.current_scenario_seed is None
    assert env.scenario_counts[None] == 1


def test_training_entry_uses_the_configured_checkpoint_schedule(tmp_path, monkeypatch):
    from scripts import train_value

    cfg = small_config("pqn")
    cfg["checkpoint_interval"] = 8
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(
        "sys.argv",
        ["train_value", "--config", str(recipe), "--out", str(tmp_path / "run")],
    )
    captured = []
    monkeypatch.setattr(
        train_value,
        "run",
        lambda config, out, interval, seeds: captured.append((config, interval)),
    )
    train_value.main()
    assert captured[0][1] == 8
    assert captured[0][0]["total_env_steps"] == 32
