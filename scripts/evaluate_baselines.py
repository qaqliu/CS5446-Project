"""Evaluate the non-learning baselines under the shared protocol.

    python -m scripts.evaluate_baselines --out results/baselines [--include-100] [--jobs 8]

Baselines: random, shortest path, shortest path with a departure cap (cap
chosen per train count on validation maps by competition score), and
prioritized planning. They play the same test maps as scripts.evaluate and
write rows in the same format to <out>/evaluation.csv (one subfolder per
baseline is not needed: the `run` column names the baseline).
"""

import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import yaml

from src.baseline.planning import PrioritizedPlanningPolicy
from src.baseline.policies import DepartureGatedShortestPathPolicy, RandomPolicy, ShortestPathPolicy
from src.eval.protocol import (
    OPTIONAL_TEST_SIZES, SAMPLED_EPISODES, TEST_SEEDS, TEST_SIZES, VALIDATION_SEEDS, protocol_config,
)
from src.eval.runner import play

REPO = Path(__file__).resolve().parent.parent
# Baselines ignore observations; any project config supplies the env builder fields.
BASE_CONFIG = REPO / "config/policy/single_map/ppo_demo_no_intermediate_stops.yaml"
CAPS = (1, 2, 4, 8, 10**6)
METRICS = (
    "completion_rate", "competition_score", "deadlock_rate", "strict_deadlock_rate",
    "stuck_rate", "departure_rate", "delay_gini", "makespan", "mean_travel_time",
)


def make(name: str, cap: int | None, episode: int):
    if name == "random":
        return RandomPolicy(seed=episode)
    if name == "shortest_path":
        return ShortestPathPolicy()
    if name == "departure_cap":
        return DepartureGatedShortestPathPolicy(max_active=cap)
    if name == "prioritized_planning":
        return PrioritizedPlanningPolicy()
    raise ValueError(name)


def episode(task: dict) -> dict:
    warnings.filterwarnings("ignore")
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    config = protocol_config(base, task["trains"], task["map_seed"])
    summary = play(config, make(task["run"], task.get("cap"), task["map_seed"] * 1000 + task["episode"]))
    return {**task, **{m: summary.get(m) for m in METRICS}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("results/baselines"))
    parser.add_argument("--include-100", action="store_true")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    sizes = TEST_SIZES + (OPTIONAL_TEST_SIZES if args.include_100 else ())
    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    common = dict(epoch=None, env_steps=None, selection="baseline")
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        tuning = [dict(common, run="departure_cap", cap=cap, split="validation", mode="deterministic",
                       episode=0, trains=n, map_seed=s)
                  for n in sizes for cap in CAPS for s in VALIDATION_SEEDS]
        tuned = pd.DataFrame(pool.map(episode, tuning, chunksize=4))
        best_cap = tuned.groupby(["trains", "cap"])["competition_score"].mean().unstack().idxmax(axis=1).to_dict()
        print("departure cap chosen on validation:", best_cap, flush=True)
        tasks = []
        for n in sizes:
            for s in TEST_SEEDS:
                # Random is stochastic (several episodes); the others are deterministic.
                for k in range(SAMPLED_EPISODES):
                    tasks.append(dict(common, run="random", split="test", mode="sampled", episode=k, trains=n, map_seed=s))
                for name in ("shortest_path", "departure_cap", "prioritized_planning"):
                    tasks.append(dict(common, run=name, cap=best_cap[n] if name == "departure_cap" else None,
                                      split="test", mode="deterministic", episode=0, trains=n, map_seed=s))
        test = pd.DataFrame(pool.map(episode, tasks, chunksize=4))
    pd.concat([tuned, test]).to_csv(out / "evaluation.csv", index=False)
    print(test.groupby(["trains", "run"])[["completion_rate", "competition_score", "deadlock_rate"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
