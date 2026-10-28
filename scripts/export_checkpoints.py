"""Collect evaluated runs into a shareable checkpoint folder with a manifest.

    python -m scripts.export_checkpoints results/runs ... --out checkpoints

For every run directory that scripts.evaluate scored, copy
    <out>/<method>/seed<N>/best.pt        validation-selected checkpoint (the reported one)
    <out>/<method>/seed<N>/final.pt       last checkpoint (use it as init_from to continue)
    <out>/<method>/seed<N>/config.yaml    resolved training config
    <out>/<method>/seed<N>/evaluation.csv per-episode validation/test results
Hybrid evaluations (<run>/with_departure_cap) get only evaluation.csv: they
reuse the base method's best.pt. MANIFEST.md lists every method with its
recipe, seeds, selected epochs, budget and held-out results, using the same
names and labels as scripts.html_report; a copy goes to docs/checkpoints.md.
"""

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd

from scripts.html_report import HYBRIDS, RL_GROUPS, label
from src.eval.runner import checkpoint_epochs

RECIPES = Path("config/protocol")
# Runs that were stopped and resumed from their last checkpoint; schedules were continued, results unaffected.
RESUMED = {"shared_ppo_15M_lr_decay": 2450, "shared_ppo_15M_curriculum10": 1960, "shared_ppo_13M_curriculum20": 586}


def split_name(run: str) -> tuple[str, int]:
    match = re.fullmatch(r"(.+)_seed(\d+)", run)
    return match.group(1), int(match.group(2))


def held_out(frame: pd.DataFrame) -> dict[int, tuple[float, float]]:
    test = frame[(frame["split"] == "test") & (frame["selection"] == "best_val") & (frame["mode"] == "sampled")]
    return {int(n): (g["completion_rate"].mean(), g["competition_score"].mean()) for n, g in test.groupby("trains")}


def export_run(run: Path, out: Path) -> dict:
    frame = pd.read_csv(run / "evaluation.csv")
    method, seed = split_name(frame["run"].iloc[0])
    target = out / method / f"seed{seed}"
    target.mkdir(parents=True, exist_ok=True)
    frame["run"] = f"{method}_seed{seed}"
    frame.to_csv(target / "evaluation.csv", index=False)
    best = int(frame[(frame["split"] == "test") & (frame["selection"] == "best_val")]["epoch"].iloc[0])
    final = checkpoint_epochs(run)[-1]
    shutil.copy2(run / f"{best}.pt", target / "best.pt")
    shutil.copy2(final, target / "final.pt")
    if (run / "config.yaml").is_file():
        shutil.copy2(run / "config.yaml", target / "config.yaml")
    curve = frame[frame["selection"] == "curve"]
    steps = curve[curve["epoch"] == best]["env_steps"]
    return dict(method=method, seed=seed, best_epoch=best, final_epoch=int(final.stem),
                env_steps=int(steps.iloc[0]) if len(steps) and pd.notna(steps.iloc[0]) else None,
                total_steps=curve["env_steps"].max() if curve["env_steps"].notna().any() else None,
                results=held_out(frame), hybrid=False)


def export_hybrid(run: Path, out: Path) -> dict:
    frame = pd.read_csv(run / "with_departure_cap" / "evaluation.csv")
    method, seed = split_name(frame["run"].iloc[0])
    target = out / method / f"seed{seed}"
    target.mkdir(parents=True, exist_ok=True)
    frame["run"] = f"{method}_seed{seed}"
    frame.to_csv(target / "evaluation.csv", index=False)
    tuning = frame[frame["selection"] == "cap_tuning"].groupby(["trains", "cap"])["competition_score"].mean().reset_index()
    caps = tuning.loc[tuning.groupby("trains")["competition_score"].idxmax()]
    return dict(method=method, seed=seed, best_epoch=int(frame["epoch"].iloc[0]), final_epoch=None, env_steps=None,
                total_steps=None,
                caps={int(r.trains): int(r.cap) for r in caps.itertuples()}, results=held_out(frame), hybrid=True)


def manifest(rows: list[dict]) -> str:
    sizes = sorted({n for r in rows for n in r["results"]})
    lines = ["# Checkpoints", "",
             "Names match `config/protocol/<method>.yaml` and the results report. Put this folder at the repo root as "
             "`checkpoints/` (git-ignored).", "",
             "- `best.pt`: checkpoint selected on validation maps; all reported numbers use it.",
             "- `final.pt`: last checkpoint; use it as `init_from` to continue training.",
             "- `evaluation.csv`: every validation/test episode behind the reported numbers.",
             "- Load a policy in code: `src.eval.runner.load_actor_checkpoint(\"checkpoints/<method>/seed42/best.pt\")`.", "",
             "Held-out completion at each train count (sampled policy, mean over 3 seeds). "
             "Hybrids reuse the base method's `best.pt` with a departure cap chosen per size on validation.", "",
             "| Method | Folder | Recipe | Seeds | Selected epoch per seed | Env steps (selected / trained) | "
             + " | ".join(f"{n} trains" for n in sizes) + " |",
             "|---|---|---|---|---|---|" + "---|" * len(sizes)]
    frame = pd.DataFrame(rows)
    order = [m for _, recipes in RL_GROUPS for m in recipes] + list(HYBRIDS)
    order += sorted(set(frame["method"]) - set(order))
    for method in [m for m in order if m in set(frame["method"])]:
        group = frame[frame["method"] == method]
        group = group.sort_values("seed")
        recipe = RECIPES / f"{method}.yaml"
        base = method.removesuffix("_with_cap")
        recipe_cell = (f"`{recipe}`" if recipe.is_file() else
                       f"`{RECIPES / base}.yaml` + cap" if group["hybrid"].iloc[0] else "–")
        epochs = ", ".join(str(e) for e in group["best_epoch"])
        steps, total = group["env_steps"].dropna(), group["total_steps"].dropna()
        steps_cell = (f"{steps.mean() / 1e6:.1f}M / {total.mean() / 1e6:.1f}M" if len(steps) else
                      "same as base" if group["hybrid"].iloc[0] else "–")
        cells = []
        for n in sizes:
            values = [r[n][0] for r in group["results"] if n in r]
            cells.append(f"{sum(values) / len(values):.0%}" if values else "–")
        lines.append(f"| {label(method)} | `{method}/` | {recipe_cell} | {len(group)} | {epochs} | {steps_cell} | "
                     + " | ".join(cells) + " |")
    notes = [f"- `{m}`: stopped at epoch {e} (machine ran out of memory) and resumed from that checkpoint with the LR/entropy "
             f"schedules continued; checkpoints after it are renumbered to follow on." for m, e in RESUMED.items()
             if m in set(frame["method"])]
    hybrids = frame[frame["hybrid"]]
    caps = [f"- `{r.method}` seed {r.seed}: caps {r.caps}" for r in hybrids.itertuples()]
    return "\n".join(lines + ["", "## Notes", ""] + notes + ["", "## Departure caps chosen on validation", ""] + caps) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folders", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("checkpoints"))
    parser.add_argument("--docs", type=Path, default=Path("docs/checkpoints.md"),
                        help="also write the manifest here (committed, so the list is browsable without downloading)")
    args = parser.parse_args()
    rows = []
    for folder in args.folders:
        for csv in sorted(folder.expanduser().glob("*/evaluation.csv")):
            rows.append(export_run(csv.parent, args.out))
            if (csv.parent / "with_departure_cap" / "evaluation.csv").is_file():
                rows.append(export_hybrid(csv.parent, args.out))
    text = manifest(rows)
    (args.out / "MANIFEST.md").write_text(text, encoding="utf-8")
    if args.docs:
        args.docs.parent.mkdir(parents=True, exist_ok=True)
        args.docs.write_text(text, encoding="utf-8")
    print(f"exported {len(rows)} runs to {args.out}")


if __name__ == "__main__":
    main()
