"""Robustness test: held-out maps with random train breakdowns (malfunctions).

    python -m scripts.evaluate_robustness RUN_DIR [RUN_DIR ...] --baselines results/baselines \
        [--sizes 5 10 20] [--intervals 400 200 100] [--jobs 4] --out results/robustness

The protocol itself has no malfunctions; this re-plays the test maps with
breakdowns that start on average once every `interval` steps per train and
last 20-50 steps (the NeurIPS 2020 competition's duration range). Every
method sees the same breakdown random numbers on a given map and episode
(3 episodes per map). Methods: prioritized planning, shortest path,
shortest path + departure cap (cap as chosen on validation), each RL run's
validation-selected checkpoint (sampled policy), and its departure-cap hybrid
when scripts.evaluate_hybrid has scored it. Writes <out>/evaluation.csv with a
malfunction_interval column.
"""

import argparse
import re
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import pandas as pd
import torch
import yaml

from scripts.evaluate_baselines import BASE_CONFIG, METRICS, make
from src.baseline.policies import DepartureCap
from src.eval.protocol import SAMPLED_EPISODES, TEST_SEEDS, protocol_config
from src.eval.runner import load_actor_checkpoint, play

DURATION = (20, 50)


@lru_cache(maxsize=4)
def _load(path: str):
    return load_actor_checkpoint(path)


def episode(task: dict) -> dict:
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    seed = task["map_seed"] * 1000 + task["episode"]
    if task.get("checkpoint"):
        make_policy, config = _load(task["checkpoint"])
        torch.manual_seed(seed)
        policy = make_policy(False)
        if task.get("cap"):
            policy = DepartureCap(policy, max_active=int(task["cap"]))
    else:
        config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
        policy = make(task["run"], task.get("cap"), seed)
    config = protocol_config(config, task["trains"], task["map_seed"])
    config["malfunction"] = dict(interval=task["malfunction_interval"], min_duration=DURATION[0],
                                 max_duration=DURATION[1], seed=seed)
    summary = play(config, policy)
    return {**{k: v for k, v in task.items() if k != "checkpoint"}, **{m: summary.get(m) for m in METRICS}}


def chosen_caps(frame: pd.DataFrame, run: str) -> dict[int, int]:
    test = frame[(frame["run"] == run) & (frame["split"] == "test") & frame["cap"].notna()]
    return {int(n): int(g["cap"].iloc[0]) for n, g in test.groupby("trains")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", type=Path, nargs="*")
    parser.add_argument("--baselines", type=Path, required=True, help="folder with the baselines' evaluation.csv")
    parser.add_argument("--sizes", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--intervals", type=int, nargs="+", default=[400, 200, 100])
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("results/robustness"))
    args = parser.parse_args()
    # (run name, checkpoint or None, {trains: cap} or None, sizes)
    base = pd.read_csv(args.baselines.expanduser() / "evaluation.csv")
    methods = [("prioritized_planning", None, None, args.sizes), ("shortest_path", None, None, args.sizes),
               ("departure_cap", None, chosen_caps(base, "departure_cap"), args.sizes)]
    for run in [r.expanduser() for r in args.runs]:
        frame = pd.read_csv(run / "evaluation.csv")
        epoch = int(frame[(frame["split"] == "test") & (frame["selection"] == "best_val")]["epoch"].iloc[0])
        checkpoint = str(run / f"{epoch}.pt")
        tested = set(frame[frame["split"] == "test"]["trains"])  # fixed-size models are tested at one size only
        methods.append((run.name, checkpoint, None, [n for n in args.sizes if n in tested]))
        hybrid = run / "with_departure_cap" / "evaluation.csv"
        if hybrid.is_file():
            frame = pd.read_csv(hybrid)
            name = frame["run"].iloc[0]
            methods.append((name, checkpoint, chosen_caps(frame, name), [n for n in args.sizes if n in tested]))
    tasks = [dict(run=name, checkpoint=checkpoint, cap=caps[n] if caps else None, malfunction_interval=interval,
                  split="test", selection="robustness", trains=n, map_seed=s, episode=k,
                  mode="sampled" if checkpoint else "deterministic")
             for name, checkpoint, caps, sizes in methods for n in sizes
             for interval in args.intervals for s in TEST_SEEDS for k in range(SAMPLED_EPISODES)]
    print(f"{len(tasks)} episodes", flush=True)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        rows = pd.DataFrame(pool.map(episode, tasks, chunksize=2))
    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out / "evaluation.csv", index=False)
    rows["method"] = rows["run"].map(lambda r: re.sub(r"_seed\d+$", "", r))
    print(rows.groupby(["trains", "malfunction_interval", "method"])["completion_rate"].mean().unstack(1).round(2).to_string())


if __name__ == "__main__":
    main()
