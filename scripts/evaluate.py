"""Evaluate training runs under the shared protocol.

    python -m scripts.evaluate RUN_DIR [RUN_DIR ...] [--include-100] [--jobs 8]
    python -m scripts.evaluate RUN_DIR [RUN_DIR ...] --smoke [--jobs 4]
    python -m scripts.evaluate RUN_DIR [RUN_DIR ...] --all-checkpoints [--jobs 4]

A RUN_DIR holds numeric checkpoints `<epoch>.pt`. Directories without
checkpoints are searched one level down, so a sweep output folder works too.

1. Validation: every checkpoint plays one sampled episode on each validation
   map at the run's training size(s). The best checkpoint by mean competition
   score is selected; larger sizes are never used for selection.
2. Test: the selected and the final checkpoint play SAMPLED_EPISODES sampled
   episodes plus one deterministic episode on every test map at every test
   size. --all-checkpoints tests every numbered checkpoint and prints each
   epoch's results. Policies tied to a fixed train count (independent networks)
   are tested only at their training size.

Writes <run>/evaluation.csv. --smoke uses only the latest checkpoint, two
validation maps, two test maps, 5/10 trains, and one sampled plus one
deterministic test episode; it writes <run>/evaluation_smoke.csv instead.
Use scripts.html_report to build tables and charts from full evaluations.
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


def train_counts(value: str) -> tuple[int, ...]:
    """Parse a comma-separated list of positive test train counts."""
    try:
        counts = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--trains needs comma-separated positive integers") from exc
    if not counts or any(count < 1 for count in counts) or len(set(counts)) != len(counts):
        raise argparse.ArgumentTypeError("--trains needs distinct positive integers")
    return counts


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
                  sampled_episodes: int = SAMPLED_EPISODES, latest_only: bool = False,
                  output_name: str = "evaluation.csv", all_checkpoints: bool = False) -> None:
    """Overrides of protocol settings are for smoke tests only."""
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        plans = {}
        validation_tasks = []
        for run in runs:
            checkpoints = checkpoint_epochs(run)
            if latest_only:
                checkpoints = checkpoints[-1:]
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
        chosen_by_run = {}
        for run, plan in plans.items():
            scores = validation[validation["run"] == run.name].groupby("epoch")["competition_score"].mean()
            chosen = {"best_val": int(scores.idxmax()), "final": int(plan["checkpoints"][-1].stem)}
            chosen_by_run[run] = chosen
            epochs = (("all_checkpoints", int(path.stem)) for path in plan["checkpoints"]) if all_checkpoints else chosen.items()
            for selection, epoch in epochs:
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
        run_test = test[test["run"] == run.name]
        if all_checkpoints:
            # Keep the usual report labels without replaying selected episodes.
            aliases = []
            for selection, epoch in chosen_by_run[run].items():
                alias = run_test[run_test["epoch"] == epoch].copy()
                alias["selection"] = selection
                aliases.append(alias)
            run_test = pd.concat([run_test, *aliases], ignore_index=True)
        rows = pd.concat([validation[validation["run"] == run.name], run_test])
        rows.to_csv(run / output_name, index=False)
        if all_checkpoints:
            for epoch, group in run_test[(run_test["selection"] == "all_checkpoints") &
                                         (run_test["mode"] == "sampled")].groupby("epoch", sort=True):
                line = ", ".join(f"{n} trains {g['completion_rate'].mean():.2f}/{g['competition_score'].mean():.2f}"
                                 for n, g in group.groupby("trains"))
                print(f"{run.name}: epoch {epoch} | test completion/score: {line}", flush=True)
        best = rows[(rows["split"] == "test") & (rows["selection"] == "best_val") & (rows["mode"] == "sampled")]
        line = ", ".join(f"{n} trains {g['completion_rate'].mean():.2f}/{g['competition_score'].mean():.2f}"
                         for n, g in best.groupby("trains"))
        print(f"{run.name}: best epoch {int(best['epoch'].iloc[0])} | test completion/score: {line}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--loader", help="module:function for non-standard checkpoints")
    parser.add_argument("--include-100", action="store_true", help="also test at 100 trains (slow)")
    parser.add_argument("--trains", type=train_counts, metavar="N[,N...]",
                        help="override test train counts, for example --trains 5,10,20")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--smoke", action="store_true",
                        help="quick pipeline check: latest checkpoint (unless --all-checkpoints), fewer maps/episodes/sizes; writes evaluation_smoke.csv")
    parser.add_argument("--all-checkpoints", action="store_true",
                        help="test and print every numbered checkpoint, not just the best and final epochs")
    args = parser.parse_args()
    if args.smoke and args.include_100:
        parser.error("--smoke cannot be combined with --include-100")
    if args.trains and args.include_100:
        parser.error("--trains cannot be combined with --include-100; include 100 in --trains")
    sizes = args.trains or TEST_SIZES + (OPTIONAL_TEST_SIZES if args.include_100 else ())
    runs = run_dirs([p.expanduser() for p in args.runs])
    if not runs:
        raise SystemExit("no run directories with numeric checkpoints found")
    if args.smoke:
        evaluate_runs(runs, args.trains or (5, 10), args.loader, args.jobs,
                      validation_seeds=VALIDATION_SEEDS[:2], test_seeds=TEST_SEEDS[:2],
                      sampled_episodes=1, latest_only=not args.all_checkpoints,
                      output_name="evaluation_smoke.csv", all_checkpoints=args.all_checkpoints)
    else:
        evaluate_runs(runs, sizes, args.loader, args.jobs, all_checkpoints=args.all_checkpoints)


if __name__ == "__main__":
    main()
