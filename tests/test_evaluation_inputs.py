"""Reject input combinations that could silently invalidate the final comparison."""

from pathlib import Path

import pytest
import torch

from scripts.evaluate import evaluate_runs


def test_same_run_names_across_rewards_are_rejected_before_evaluation():
    runs = [Path("/reward_minus5/ppo_seed42"), Path("/reward_minus05/ppo_seed42")]
    with pytest.raises(ValueError, match="directory names must be unique"):
        evaluate_runs(runs, (20,), None, jobs=1)


def test_required_budget_reads_checkpoint_counter_not_its_filename(tmp_path):
    run = tmp_path / "minus5_ppo_seed42"
    run.mkdir()
    torch.save({"env_steps": 2_900_000}, run / "3000000.pt")
    with pytest.raises(ValueError, match="2900000 environment steps; required 3000000"):
        evaluate_runs([run], (20,), None, jobs=1, require_env_steps=3_000_000)
