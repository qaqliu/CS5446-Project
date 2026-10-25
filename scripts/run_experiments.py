"""Train several recipes x seeds in parallel, then evaluate them.

    python -m scripts.run_experiments --configs config/protocol/shared_ppo_random_maps.yaml \
        config/protocol/comm_ppo_random_maps.yaml --seeds 42 43 44 --out results/runs --jobs 6

Each (recipe, seed) trains in <out>/<recipe>_seed<seed> with scripts.train
(one CPU thread each; logs in train.log). Finished runs are skipped. With
--evaluate (default), all runs are scored with scripts.evaluate afterwards.
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def train(config: Path, seed: int, out: Path) -> str:
    run = out / f"{config.stem}_seed{seed}"
    if (run / "train.done").exists():
        return f"skip {run.name}"
    run.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1", PYTHONWARNINGS="ignore")
    with (run / "train.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            [sys.executable, "-m", "scripts.train", "--config", str(config), "--seed", str(seed), "--out", str(run)],
            cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        return f"FAILED {run.name} (see train.log)"
    (run / "train.done").touch()
    return f"done {run.name}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", type=Path, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--out", type=Path, default=Path("results/runs"))
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--no-evaluate", action="store_true")
    parser.add_argument("--include-100", action="store_true")
    args = parser.parse_args()
    out = args.out.expanduser().resolve()
    tasks = [(c.resolve(), s) for c in args.configs for s in args.seeds]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for message in pool.map(lambda t: train(t[0], t[1], out), tasks):
            print(message, flush=True)
    if not args.no_evaluate:
        command = [sys.executable, "-m", "scripts.evaluate", str(out), "--jobs", str(args.jobs)]
        if args.include_100:
            command.append("--include-100")
        subprocess.run(command, cwd=REPO, check=True)


if __name__ == "__main__":
    main()
