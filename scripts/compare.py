"""Build comparison tables and charts from protocol evaluations.

    python -m scripts.compare results/runs results/baselines --out results/report

Reads every evaluation.csv under the given folders (from scripts.evaluate and
scripts.evaluate_baselines). A method is a run name without its `_seedN`
suffix. Learned methods use the validation-selected checkpoint and sampled
episodes (deterministic shown separately); each run is averaged over test
maps first, then mean and std are taken across training seeds.
Writes summary.md, summary.csv and comparison.png.
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = ["completion_rate", "competition_score", "deadlock_rate", "delay_gini", "makespan"]
# Reference categorical palette, fixed order (light mode).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def load(folders: list[Path]) -> pd.DataFrame:
    frames = []
    for folder in folders:
        for path in sorted(folder.expanduser().rglob("evaluation.csv")):
            frame = pd.read_csv(path)
            frames.append(frame[frame["split"] == "test"])
    data = pd.concat(frames, ignore_index=True)
    data["method"] = data["run"].map(lambda r: re.sub(r"_seed\d+$", "", r))
    return data


def summarize(data: pd.DataFrame, mode: str) -> pd.DataFrame:
    keep = data[data["selection"].isin(["best_val", "baseline"])]
    learned = keep["selection"] == "best_val"
    keep = keep[~learned | (keep["mode"] == mode)]
    per_run = keep.groupby(["method", "trains", "run"])[METRICS].mean().reset_index()
    table = per_run.groupby(["method", "trains"])[METRICS].agg(["mean", "std"])
    table["seeds"] = per_run.groupby(["method", "trains"])["run"].nunique()
    return table


def markdown(table: pd.DataFrame, title: str) -> str:
    lines = [f"## {title}", ""]
    for trains in sorted(table.index.get_level_values("trains").unique()):
        lines += [f"### {trains} trains", "",
                  "| Method | Seeds | Completion | Competition score | Deadlock | Delay Gini | Makespan |",
                  "|---|---|---|---|---|---|---|"]
        part = table.xs(trains, level="trains").sort_values(("competition_score", "mean"), ascending=False)
        for method, row in part.iterrows():
            cells = []
            for m in METRICS:
                mean, std = row[(m, "mean")], row[(m, "std")]
                fmt = "{:.0f}" if m == "makespan" else "{:.2f}"
                cells.append(fmt.format(mean) + ("" if pd.isna(std) else " ± " + fmt.format(std)))
            lines.append(f"| {method} | {int(row['seeds'].iloc[0])} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def chart(table: pd.DataFrame, path: Path, methods: list[str] | None) -> None:
    methods = methods or list(table.index.get_level_values("method").unique())
    methods = methods[:len(PALETTE)]
    trains = sorted(table.index.get_level_values("trains").unique())
    xs = list(range(len(trains)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=SURFACE)
    for ax, (metric, title) in zip(axes, [("completion_rate", "Completion rate"),
                                           ("competition_score", "Competition score")]):
        for color, method in zip(PALETTE, methods):
            part = table.xs(method, level="method").reindex(trains)
            mean, std = part[(metric, "mean")].values, part[(metric, "std")].values
            ax.plot(xs, mean, color=color, linewidth=2, marker="o", markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=method, zorder=3)
            if not np.all(np.isnan(std)):
                ax.fill_between(xs, mean - np.nan_to_num(std), mean + np.nan_to_num(std),
                                color=color, alpha=0.15, linewidth=0)
        ax.set_xticks(xs, [str(n) for n in trains])
        ax.set_xlabel("Number of trains (competition map formula)", color=MUTED)
        ax.set_ylim(-0.03, 1.05)
        ax.set_facecolor(SURFACE)
        ax.set_title(title, color=TEXT, fontsize=11, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, labelcolor=TEXT, loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Held-out test maps (mean ± std over training seeds)", color=TEXT, fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folders", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("results/report"))
    parser.add_argument("--methods", nargs="*", help="methods to plot, in palette order (default: all)")
    args = parser.parse_args()
    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    data = load(args.folders)
    sampled, deterministic = summarize(data, "sampled"), summarize(data, "deterministic")
    sampled.to_csv(out / "summary.csv")
    text = ["# Protocol comparison", "",
            "Held-out test maps (seeds 1000-1019). Learned methods: validation-selected checkpoint.", "",
            markdown(sampled, "Sampled policy (headline)"), markdown(deterministic, "Deterministic policy")]
    (out / "summary.md").write_text("\n".join(text), encoding="utf-8")
    chart(sampled, out / "comparison.png", args.methods)
    print((out / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
