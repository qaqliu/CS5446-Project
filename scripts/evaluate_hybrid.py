"""Evaluate learned policies combined with a departure cap (hybrid).

    python -m scripts.evaluate_hybrid RUN_DIR [RUN_DIR ...] [--jobs 3]

For each run that scripts.evaluate already scored, take its validation-
selected checkpoint, wrap the policy with src.baseline.policies.DepartureCap
(the learned policy routes and stops; the rule limits trains on the map),
choose the cap per train count on validation maps by competition score, and
play the test maps (sampled + deterministic, as scripts.evaluate). Writes
<run>/with_departure_cap/evaluation.csv with run name `<recipe>_with_cap_seedN`,
which scripts.html_report reports as its own method.
"""

import argparse
import re
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import pandas as pd
import torch

from scripts.evaluate import METRICS, run_dirs
from src.baseline.policies import DepartureCap
from src.eval.protocol import OPTIONAL_TEST_SIZES, SAMPLED_EPISODES, TEST_SEEDS, TEST_SIZES, VALIDATION_SEEDS, protocol_config
from src.eval.runner import load_actor_checkpoint, play

CAPS = (2, 4, 8, 16, 10**6)


@lru_cache(maxsize=2)
def _load(path: str):
    return load_actor_checkpoint(path)


def episode(task: dict) -> dict:
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    make_policy, config = _load(task["checkpoint"])
    torch.manual_seed(task["map_seed"] * 1000 + task["episode"])
    policy = DepartureCap(make_policy(task["mode"] == "deterministic"), max_active=task["cap"])
    summary = play(protocol_config(config, task["trains"], task["map_seed"]), policy)
    return {**{k: v for k, v in task.items() if k != "checkpoint"}, **{m: summary.get(m) for m in METRICS}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--include-100", action="store_true", help="also test at 100 trains (slow)")
    args = parser.parse_args()
    sizes = TEST_SIZES + (OPTIONAL_TEST_SIZES if args.include_100 else ())
    runs = [r for r in run_dirs([p.expanduser() for p in args.runs]) if (r / "evaluation.csv").is_file()]
    plans = {}
    for run in runs:
        frame = pd.read_csv(run / "evaluation.csv")
        epoch = int(frame[(frame["split"] == "test") & (frame["selection"] == "best_val")]["epoch"].iloc[0])
        base = re.sub(r"_seed(\d+)$", r"_with_cap_seed\1", run.name)
        plans[run] = dict(checkpoint=str(run / f"{epoch}.pt"), epoch=epoch, name=base)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        tuning = [dict(run=p["name"], checkpoint=p["checkpoint"], epoch=p["epoch"], env_steps=None,
                       split="validation", selection="cap_tuning", mode="sampled", episode=0,
                       trains=n, map_seed=s, cap=cap)
                  for p in plans.values() for n in sizes for cap in CAPS for s in VALIDATION_SEEDS]
        tuned = pd.DataFrame(pool.map(episode, tuning, chunksize=4))
        best = tuned.groupby(["run", "trains", "cap"])["competition_score"].mean().reset_index()
        best = best.loc[best.groupby(["run", "trains"])["competition_score"].idxmax()]
        chosen = {(r.run, r.trains): int(r.cap) for r in best.itertuples()}
        print("caps chosen on validation:", chosen, flush=True)
        tests = []
        for p in plans.values():
            for n in sizes:
                for s in TEST_SEEDS:
                    for k in range(SAMPLED_EPISODES + 1):
                        tests.append(dict(run=p["name"], checkpoint=p["checkpoint"], epoch=p["epoch"], env_steps=None,
                                          split="test", selection="best_val", episode=k, trains=n, map_seed=s,
                                          mode="deterministic" if k == SAMPLED_EPISODES else "sampled",
                                          cap=chosen[(p["name"], n)]))
        test = pd.DataFrame(pool.map(episode, tests, chunksize=4))
    for run, p in plans.items():
        out = run / "with_departure_cap"
        out.mkdir(exist_ok=True)
        rows = pd.concat([tuned[tuned["run"] == p["name"]], test[test["run"] == p["name"]]])
        rows.to_csv(out / "evaluation.csv", index=False)
        sampled = test[(test["run"] == p["name"]) & (test["mode"] == "sampled")]
        print(p["name"], ", ".join(f"{n} trains {g['completion_rate'].mean():.2f}/{g['competition_score'].mean():.2f}"
                                   for n, g in sampled.groupby("trains")), flush=True)


if __name__ == "__main__":
    main()
