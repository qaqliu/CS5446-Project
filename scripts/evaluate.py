"""Evaluate training runs under the shared protocol.

    python -m scripts.evaluate RUN_DIR [RUN_DIR ...] [--include-100] [--jobs 8]

A RUN_DIR holds numeric checkpoints `<epoch>.pt`. Directories without
checkpoints are searched one level down, so a sweep output folder works too.

1. Validation: every checkpoint plays one sampled episode on each validation
   map at the run's training size(s). The best checkpoint by mean competition
   score is selected; larger sizes are never used for selection.
2. Test: the selected and the final checkpoint play SAMPLED_EPISODES sampled
   episodes plus one deterministic episode on every test map at every test
   size. Policies tied to a fixed train count (independent networks) are
   tested only at their training size.

Writes <run>/evaluation.csv. Use scripts.compare to build tables and charts.
"""

import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import pandas as pd
import torch

from src.eval.protocol import (
    OPTIONAL_TEST_SIZES, SAMPLED_EPISODES, TEST_SEEDS, TEST_SIZES, VALIDATION_SEEDS,
    check_training_config, protocol_config, training_sizes,
)
from src.eval.runner import checkpoint_env_steps, checkpoint_epochs, play, resolve_loader

FIXED_SIZE_ACTORS = ("independent", "communication_independent")
METRICS = (
    "completion_rate", "competition_score", "deadlock_rate", "strict_deadlock_rate",
    "stuck_rate", "departure_rate", "delay_gini", "makespan", "mean_travel_time",
)


@lru_cache(maxsize=4)
def _load(path: str, loader: str | None):
    return resolve_loader(loader)(path)


def episode(task: dict) -> dict:
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    make_policy, config = _load(task["checkpoint"], task["loader"])
    torch.manual_seed(task["map_seed"] * 1000 + task["episode"])
    summary = play(protocol_config(config, task["trains"], task["map_seed"]),
                   make_policy(task["mode"] == "deterministic"))
    keep = {k: v for k, v in task.items() if k not in ("checkpoint", "loader")}
    return {**keep, **{m: summary.get(m) for m in METRICS}}


def run_dirs(paths: list[Path]) -> list[Path]:
    found = []
    for path in paths:
        if checkpoint_epochs(path):
            found.append(path)
        else:
            found += [d for d in sorted(path.iterdir()) if d.is_dir() and checkpoint_epochs(d)]
    return found


def evaluate_runs(runs: list[Path], sizes: tuple[int, ...], loader: str | None, jobs: int,
                  validation_seeds=VALIDATION_SEEDS, test_seeds=TEST_SEEDS,
                  sampled_episodes: int = SAMPLED_EPISODES) -> None:
    """Overrides of the seed sets / episode count are for smoke tests only."""
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        plans = {}
        validation_tasks = []
        for run in runs:
            checkpoints = checkpoint_epochs(run)
            _, config = resolve_loader(loader)(str(checkpoints[-1]))
            for problem in check_training_config(config):
                print(f"WARNING {run.name}: {problem}", flush=True)
            train_sizes = training_sizes(config)
            actor_config = config.get("actor", {})
            # One network per train, or a one-hot train-ID input sized to the training count.
            fixed = (actor_config.get("type") in FIXED_SIZE_ACTORS
                     or "num_agents" in actor_config.get("model", {}))
            plans[run] = dict(checkpoints=checkpoints, train_sizes=train_sizes,
                              test_sizes=train_sizes if fixed else sizes)
            for path in checkpoints:
                base = dict(run=run.name, checkpoint=str(path), loader=loader, epoch=int(path.stem),
                            env_steps=checkpoint_env_steps(path), split="validation", selection="curve",
                            mode="sampled", episode=0)
                validation_tasks += [dict(base, trains=n, map_seed=s) for n in train_sizes for s in validation_seeds]
        validation = pd.DataFrame(pool.map(episode, validation_tasks, chunksize=8))

        test_tasks = []
        for run, plan in plans.items():
            scores = validation[validation["run"] == run.name].groupby("epoch")["competition_score"].mean()
            chosen = {"best_val": int(scores.idxmax()), "final": int(plan["checkpoints"][-1].stem)}
            for selection, epoch in chosen.items():
                path = run / f"{epoch}.pt"
                for n in plan["test_sizes"]:
                    for s in test_seeds:
                        for k in range(sampled_episodes + 1):
                            test_tasks.append(dict(
                                run=run.name, checkpoint=str(path), loader=loader, epoch=epoch,
                                env_steps=checkpoint_env_steps(path), split="test", selection=selection,
                                mode="deterministic" if k == sampled_episodes else "sampled",
                                episode=k, trains=n, map_seed=s))
        test = pd.DataFrame(pool.map(episode, test_tasks, chunksize=8))

    for run in plans:
        rows = pd.concat([validation[validation["run"] == run.name], test[test["run"] == run.name]])
        rows.to_csv(run / "evaluation.csv", index=False)
        best = rows[(rows["split"] == "test") & (rows["selection"] == "best_val") & (rows["mode"] == "sampled")]
        line = ", ".join(f"{n} trains {g['completion_rate'].mean():.2f}/{g['competition_score'].mean():.2f}"
                         for n, g in best.groupby("trains"))
        print(f"{run.name}: best epoch {int(best['epoch'].iloc[0])} | test completion/score: {line}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--loader", help="module:function for non-standard checkpoints")
    parser.add_argument("--include-100", action="store_true", help="also test at 100 trains (slow)")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    sizes = TEST_SIZES + (OPTIONAL_TEST_SIZES if args.include_100 else ())
    runs = run_dirs([p.expanduser() for p in args.runs])
    if not runs:
        raise SystemExit("no run directories with numeric checkpoints found")
    evaluate_runs(runs, sizes, args.loader, args.jobs)


if __name__ == "__main__":
    main()
