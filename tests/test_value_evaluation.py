"""Regression coverage for frozen Q execution and validation-only selection."""

from copy import deepcopy

import pandas as pd
import pytest
import torch

from scripts import evaluate
from src.agent import ShareNetworkValueAgent
from src.eval.value import (
    EpsilonGreedyValuePolicy,
    GreedyValuePolicy,
    load_epsilon_checkpoint,
)
from src.model import build_model


def test_zero_epsilon_preserves_greedy_actions_rng_and_weights():
    torch.manual_seed(42)
    agent = ShareNetworkValueAgent(
        build_model("mlp_q", input_dim=2, hidden_dims=[4], layer_norm=True)
    )
    observations = {0: [1.0, 2.0], 1: None}
    masks = {
        0: torch.tensor([True, False, True, False, False]),
        1: torch.tensor([True, False, False, False, False]),
    }
    before = deepcopy(agent.state_dict())
    state = torch.get_rng_state().clone()
    expected = GreedyValuePolicy(agent)(None, observations, {}, masks, {})
    assert (
        EpsilonGreedyValuePolicy(agent, 0)(None, observations, {}, masks, {})
        == expected
    )
    assert torch.equal(state, torch.get_rng_state())
    exploratory = EpsilonGreedyValuePolicy(agent, 0.05)
    torch.manual_seed(9)
    first = exploratory(None, observations, {}, masks, {})
    torch.manual_seed(9)
    assert exploratory(None, observations, {}, masks, {}) == first
    assert all(masks[h][a] for h, a in first.items())
    assert all(torch.equal(before[k], v) for k, v in agent.state_dict().items())
    with pytest.raises(ValueError, match="epsilon"):
        EpsilonGreedyValuePolicy(agent, 1.1)


def test_epsilon_loader_keeps_deterministic_mode_greedy(tmp_path):
    config = dict(
        trainer={"device": "cpu"},
        value_agent=dict(
            type="shared",
            model=dict(
                name="mlp_q",
                input_dim=2,
                hidden_dims=[4],
                layer_norm=True,
            ),
        ),
    )
    agent = ShareNetworkValueAgent(build_model(**config["value_agent"]["model"]))
    path = tmp_path / "32.pt"
    torch.save(dict(config=config, value_agent=agent.state_dict(), env_steps=32), path)
    factory, _ = load_epsilon_checkpoint(path)
    assert factory(True).epsilon == 0 and factory(False).epsilon == 0.05


class SerialPool:
    """Run task routing synchronously; this test does not simulate trains."""

    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def map(self, function, tasks, **kwargs):
        return list(map(function, tasks))


def test_test_only_epsilon_loader_cannot_change_validation_selection(
    tmp_path, monkeypatch
):
    run = tmp_path / "pqn_seed42"
    run.mkdir()
    for step in (0, 32):
        torch.save(dict(env_steps=step), run / f"{step}.pt")
    config = dict(number_of_agents=20, random_map=True, train_map_seeds=[142000])
    monkeypatch.setattr(evaluate, "ProcessPoolExecutor", SerialPool)
    monkeypatch.setattr(
        evaluate, "resolve_loader", lambda _: lambda path: (None, config)
    )
    observed = []

    def score(task):
        observed.append(task)
        # Greedy validation prefers the initial checkpoint. A changed validation
        # loader would instead prefer the last checkpoint, exposing leakage.
        prefer_initial = task["loader"] == "primary"
        value = 0.9 if ((task["epoch"] == 0) == prefer_initial) else 0.1
        return {
            k: v
            for k, v in dict(
                task, competition_score=value, completion_rate=value
            ).items()
            if k not in ("checkpoint",)
        }

    monkeypatch.setattr(evaluate, "episode", score)
    evaluate.evaluate_runs(
        [run],
        (20,),
        "primary",
        jobs=1,
        validation_seeds=(2000,),
        test_seeds=(9300000,),
        sampled_episodes=1,
        require_env_steps=32,
        test_loader="auxiliary",
        output_name="fresh_epsilon.csv",
    )
    rows = pd.read_csv(run / "fresh_epsilon.csv")
    assert not (run / "evaluation.csv").exists()
    primary = rows[(rows["split"] == "test") & (rows["selection"] == "best_val")]
    assert set(primary["epoch"]) == {0}
    assert all(t["loader"] == "primary" for t in observed if t["split"] == "validation")
    assert all(t["loader"] == "auxiliary" for t in observed if t["split"] == "test")
    assert set(primary["map_seed"]) == {9300000}
    with pytest.raises(ValueError, match="overlap the training pool"):
        evaluate.evaluate_runs([run], (20,), "primary", jobs=1, test_seeds=(142000,))


def test_custom_test_range_cannot_overlap_validation():
    with pytest.raises(ValueError, match="disjoint splits"):
        evaluate.evaluate_runs([], (20,), None, jobs=1, test_seeds=(2000,))


def test_upstream_report_renders_matched_value_recipes():
    from scripts import html_report

    frame = pd.DataFrame(
        [
            dict(
                method=method,
                run=f"{method}_seed{seed}",
                trains=20,
                split="test",
                selection="best_val",
                mode="sampled",
                env_steps=3000000,
                epoch=3000000,
                **{metric: 0.2 for metric in html_report.METRICS},
            )
            for method in ("value_ppo_scale20", "value_pqn_scale20")
            for seed in (42, 43, 44)
        ]
    )
    rendered = html_report.markdown_report(frame, html_report.test_table(frame))
    assert "Value-based (Task 4)" in rendered
    assert "Matched PQN, scale20, 3M" in rendered
    assert "Matched PPO, scale20, 3M" in rendered
