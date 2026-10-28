"""Build the results report (self-contained HTML, or plain markdown) from protocol evaluations.

    python -m scripts.html_report results/runs results/baselines ... --out results/report.html
    python -m scripts.html_report results/runs results/baselines ... --out docs/report/README.md

Reads every evaluation.csv under the given folders (scripts.evaluate,
scripts.evaluate_baselines, scripts.evaluate_hybrid). A method is its run name
without `_seedN`, which is also its recipe in config/protocol/ and its
checkpoint folder; each recipe's `report:` block gives its label, group and the
settings that differ from the reference recipe. `<recipe>_with_cap` is that
recipe's policy with a departure cap. The HTML has charts and tables; the
markdown has tables only.
"""

import argparse
import base64
import html
import io
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
METRICS = ["completion_rate", "competition_score", "deadlock_rate", "delay_gini", "makespan"]
RECIPE_DIR = Path(__file__).resolve().parent.parent / "config" / "protocol"
# Baselines have no recipe: label, kind, what it does, tuning.
BASELINES = {
    "prioritized_planning": ("Prioritized planning (centralized reference)", "Centralized planner",
                             "Plans trains one at a time with A* in space-time, reserving cells so later trains avoid "
                             "earlier ones. Sees the whole map and all trains (upper reference).", "None"),
    "departure_cap": ("Shortest path + departure cap", "Heuristic",
                      "Shortest path, but at most k trains on the map at once; others wait at their start.",
                      "k tuned per size on validation"),
    "shortest_path": ("Shortest path", "Heuristic", "Every train follows its own shortest path, no coordination.", "None"),
    "random": ("Random", "Heuristic", "Uniform random valid action.", "None"),
}
# RL recipe table: one column per design dimension; a recipe's report.changes override the reference values.
DIMENSIONS = ["Network", "Trainer", "Training maps", "Train sizes", "LR schedule", "Entropy", "Reward"]
REFERENCE = "shared_ppo_random_maps"
_REF = dict(zip(DIMENSIONS, ["Shared MLP 64-64", "ppo_demo, lr 3e‑4", "New random map each episode", "5",
                             "Constant", "0.01", "Champion + stuck"]))
GROUPS = ["Reference", "Training budget and schedule", "Training-map strategy", "Reward",
          "Network (Task 1: shared vs independent)", "Communication (teammate recipe)"]
GROUP_SIZES = {"Training budget and schedule": [5, 10, 20], "Training-map strategy": [5, 10, 20], "Reward": [5, 10, 20],
               "Communication (teammate recipe)": [5, 10, 20]}
GROUP_QUESTIONS = {
    "Training budget and schedule": "Does more training help, and does decaying the learning rate or continuing on larger maps (curriculum) help more?",
    "Training-map strategy": "Is it better to train on one map, a fixed pool, or a new random map every episode?",
    "Reward": "Does the extra stuck penalty matter?",
    "Network (Task 1: shared vs independent)": "One network shared by all trains, shared with a train ID, or one network per train?",
    "Communication (teammate recipe)": "Does attention-based communication between trains help, and does grouped entropy or multi-size training help it? "
                                       "(Different trainer and learning rate from the reference, so compare the communication rows with each other first.)",
}
MAIN = ["prioritized_planning", "departure_cap", "shared_ppo_15M_curriculum10_with_cap", "shared_ppo_random_maps_10M_with_cap",
        "shared_ppo_15M_curriculum10", "comm_ppo_multi_size", "shortest_path", "random"]
FAIRNESS = ["prioritized_planning", "departure_cap", "shared_ppo_15M_curriculum10_with_cap", "shared_ppo_15M_curriculum10",
            "shortest_path"]
CURVES = ["shared_ppo_random_maps", "shared_ppo_random_maps_10M", "shared_ppo_15M_lr_decay", "shared_ppo_15M_curriculum10",
          "shared_ppo_13M_curriculum20", "independent_ppo_random_maps"]


def _recipes() -> dict[str, dict]:
    return {p.stem: (yaml.safe_load(p.read_text()) or {}).get("report") or {} for p in sorted(RECIPE_DIR.glob("*.yaml"))}


RECIPES = _recipes()
RL_GROUPS = [(group, {name: meta.get("changes") or {} for name, meta in
                      sorted(RECIPES.items(), key=lambda item: item[1].get("order", 0)) if meta.get("group") == group})
             for group in GROUPS]
HYBRIDS = {f"{name}_with_cap": name for name in RECIPES}
BUDGET_NOTE = {name: meta["budget"] for name, meta in RECIPES.items() if meta.get("budget")}


def label(method: str) -> str:
    if method in BASELINES:
        return BASELINES[method][0]
    if method.endswith("_with_cap"):
        return label(method.removesuffix("_with_cap")) + " + departure cap"
    return RECIPES.get(method, {}).get("label", method)


def named(method: str) -> str:
    """Label plus the code name (recipe / checkpoint folder) for tables."""
    return label(method) if method in BASELINES else f"{label(method)} (`{method}`)"


def load(folders: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for folder in folders for p in sorted(folder.expanduser().rglob("evaluation.csv"))]
    data = pd.concat(frames, ignore_index=True)
    data["method"] = data["run"].map(lambda r: re.sub(r"_seed\d+$", "", r))
    return data


def test_table(data: pd.DataFrame, mode: str = "sampled") -> pd.DataFrame:
    test = data[(data["split"] == "test") & data["selection"].isin(["best_val", "baseline"])]
    test = test[(test["selection"] == "baseline") | (test["mode"] == mode)]
    per_run = test.groupby(["method", "trains", "run"])[METRICS].mean().reset_index()
    table = per_run.groupby(["method", "trains"])[METRICS].agg(["mean", "std"])
    table["seeds"] = per_run.groupby(["method", "trains"])["run"].nunique()
    return table


def _style(ax, title: str, ylabel: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)


def _png(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode()


def lines_chart(table: pd.DataFrame, methods: list[str], title: str) -> str | None:
    methods = [m for m in methods if m in table.index.get_level_values("method")][:len(PALETTE)]
    if not methods:
        return None
    trains = sorted(table.index.get_level_values("trains").unique())
    xs = list(range(len(trains)))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor=SURFACE)
    for ax, (metric, name) in zip(axes, [("completion_rate", "Completion rate"),
                                          ("competition_score", "Competition score (rewards fast arrival)")]):
        for color, method in zip(PALETTE, methods):
            part = table.xs(method, level="method").reindex(trains)
            mean, std = part[(metric, "mean")].values, part[(metric, "std")].values
            mask = ~np.isnan(mean)
            x = np.array(xs)[mask]
            ax.plot(x, mean[mask], color=color, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE,
                    markeredgewidth=2, label=label(method), zorder=3)
            if not np.all(np.isnan(std[mask])):
                s = np.nan_to_num(std[mask])
                ax.fill_between(x, mean[mask] - s, mean[mask] + s, color=color, alpha=0.13, linewidth=0)
        ax.set_xticks(xs, [str(n) for n in trains])
        ax.set_xlabel("Number of trains (competition map formula)", color=MUTED)
        ax.set_ylim(-0.03, 1.05)
        if metric == "completion_rate":
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        _style(ax, name)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, frameon=False, fontsize=9, labelcolor=TEXT, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02 - 0.035 * ((len(methods) + 1) // 2)))
    fig.suptitle(title, color=TEXT, fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout()
    return _png(fig)


def grouped_chart(table: pd.DataFrame, methods: list[str], panels: list[tuple[str, str]], title: str | None = None,
                  percent: tuple[str, ...] = ("completion_rate", "deadlock_rate")) -> str | None:
    """One group of bars per train count, one bar per method (error bar = std over seeds)."""
    methods = [m for m in methods if m in table.index.get_level_values("method")][:len(PALETTE)]
    if not methods:
        return None
    trains = sorted(table.index.get_level_values("trains").unique())
    width = 0.8 / len(methods)
    fig, axes = plt.subplots(len(panels), 1, figsize=(11.5, 3.6 * len(panels)), facecolor=SURFACE, squeeze=False)
    for ax, (metric, name) in zip(axes[:, 0], panels):
        for i, (color, method) in enumerate(zip(PALETTE, methods)):
            part = table.xs(method, level="method").reindex(trains)
            mean, std = part[(metric, "mean")].values, part[(metric, "std")].values
            x = np.arange(len(trains)) - 0.4 + width * (i + 0.5)
            mask = ~np.isnan(mean)
            ax.bar(x[mask], mean[mask], width * 0.92, color=color, label=label(method), zorder=3,
                   yerr=np.where(np.isnan(std[mask]), 0, std[mask]), error_kw=dict(ecolor=MUTED, lw=0.8, capsize=2))
            for xi, m, sd in zip(x[mask], mean[mask], np.nan_to_num(std[mask])):
                if len(methods) <= 8:
                    ax.text(xi, m + sd + 0.015, f"{m:.0%}" if metric in percent else f"{m:.2f}", ha="center",
                            fontsize=6.5, color=MUTED, rotation=90, va="bottom")
        ax.set_xticks(range(len(trains)), [f"{n} trains" for n in trains])
        ax.set_ylim(0, 1.15)
        if metric in percent:
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        _style(ax, name)
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_, frameon=False, fontsize=9, labelcolor=TEXT, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.025 - 0.035 * ((len(methods) + 1) // 2) / len(panels)))
    if title:
        fig.suptitle(title, color=TEXT, fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout()
    return _png(fig)


def bars_chart(table: pd.DataFrame, methods: list[str], trains: list[int], title: str) -> str | None:
    methods = [m for m in methods if m in table.index.get_level_values("method")]
    if not methods:
        return None
    fig, ax = plt.subplots(figsize=(11.5, 0.42 * len(methods) * len(trains) + 1.2), facecolor=SURFACE)
    y, ticks, names = 0, [], []
    for i, n in enumerate(trains):
        for method in methods:
            if (method, n) not in table.index:
                continue
            row = table.loc[(method, n)]
            mean, std = row[("completion_rate", "mean")], row[("completion_rate", "std")]
            ax.barh(y, mean, xerr=None if np.isnan(std) else std, color=PALETTE[i % len(PALETTE)], height=0.7,
                    error_kw=dict(ecolor=MUTED, lw=1, capsize=3))
            ax.text(mean + np.nan_to_num(std) + 0.012, y, f"{mean:.0%}", va="center", fontsize=8.5, color=TEXT)
            ticks.append(y)
            names.append(label(method) if len(trains) == 1 else f"{label(method)} · {n} trains")
            y += 1
        y += 0.6
    ax.set_yticks(ticks, names, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    _style(ax, title)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _png(fig)


def curves_chart(data: pd.DataFrame, methods: list[str], title: str) -> str | None:
    val = data[(data["split"] == "validation") & (data["selection"] == "curve") & data["method"].isin(methods)]
    val = val[val["env_steps"].notna() & (val["trains"] == 5)]
    if val.empty:
        return None
    fig, ax = plt.subplots(figsize=(11.5, 4.2), facecolor=SURFACE)
    for color, method in zip(PALETTE, [m for m in methods if m in set(val["method"])]):
        part = val[val["method"] == method]
        per_run = part.groupby(["run", "env_steps"])["competition_score"].mean().reset_index()
        curve = per_run.groupby("env_steps")["competition_score"].agg(["mean", "min", "max"])
        steps = curve.index.values / 1e6
        ax.plot(steps, curve["mean"], color=color, linewidth=2, label=label(method))
        ax.fill_between(steps, curve["min"], curve["max"], color=color, alpha=0.13, linewidth=0)
    ax.set_xlabel("Environment steps (millions, including any initial checkpoint)", color=MUTED)
    ax.set_ylim(0, 1)
    _style(ax, title, "Validation competition score")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="lower right")
    fig.tight_layout()
    return _png(fig)


def setup_section(data: pd.DataFrame) -> str:
    """Task, baselines, hybrids, RL recipes (dimension grid) and the evaluation protocol."""
    from src.eval.protocol import SAMPLED_EPISODES, TEST_SEEDS, VALIDATION_SEEDS, competition_map
    present = set(data["method"])
    steps = data[data["selection"] == "curve"].groupby("method")["env_steps"].max()

    def budget(method: str) -> str:
        value = steps.get(method, np.nan)
        return f"{value / 1e6:.1f}M" if not np.isnan(value) else BUDGET_NOTE.get(method, "–")

    base = "".join(f"<tr><td>{html.escape(label(m))}</td><td>{k}</td><td>{html.escape(d)}</td><td>{t}</td></tr>"
                   for m, (_, k, d, t) in BASELINES.items() if m in present)
    hybrid = "".join(f"<tr><td>{html.escape(label(m))}<br><code class='key'>{m}</code></td><td>{html.escape(label(src))} (<code>best.pt</code>)</td>"
                     "<td>Departure cap, k tuned per size on validation</td></tr>"
                     for m, src in HYBRIDS.items() if m in present)
    head = "".join(f"<th>{d}</th>" for d in DIMENSIONS)
    rows = []
    for group, recipes in RL_GROUPS:
        recipes = {m: c for m, c in recipes.items() if m in present}
        if not recipes:
            continue
        rows.append(f"<tr class='group'><td colspan='{len(DIMENSIONS) + 2}'>{html.escape(group)}</td></tr>")
        for method, changes in recipes.items():
            spec = {**_REF, **changes}
            cells = "".join(f"<td class='diff'>{html.escape(spec[d])}</td>" if spec[d] != _REF[d]
                            else f"<td>{html.escape(spec[d])}</td>" for d in DIMENSIONS)
            step_cell = budget(method)
            cls = " class='diff'" if method != REFERENCE and step_cell != budget(REFERENCE) else ""
            rows.append(f"<tr><td>{html.escape(label(method))}<br><code class='key'>{method}</code></td>{cells}"
                        f"<td{cls}>{step_cell}</td></tr>")
    episodes = SAMPLED_EPISODES + 1
    sizes = "".join(
        f"<tr><td>{n}</td><td>{p['width']}×{p['height']}</td><td>{p['max_num_cities']}</td><td>{p['max_episode_steps']}</td>"
        f"<td>{len(VALIDATION_SEEDS)} (only if trained at this size)</td><td>{len(TEST_SEEDS)}</td>"
        f"<td>{len(TEST_SEEDS) * episodes} ({len(TEST_SEEDS) * SAMPLED_EPISODES} sampled + {len(TEST_SEEDS)} deterministic)</td></tr>"
        for n, p in ((n, competition_map(n)) for n in (5, 10, 20, 50, 100)))
    return f"""<h2>Setup</h2>
<h3>Task</h3>
<p>Flatland 4.3: all trains share one rail network and must reach their targets before the time limit. Unit speed,
no malfunctions, no intermediate stations (origin to target only); trains may stop and wait anywhere. 5 actions, invalid ones masked.
All learned methods see the same depth-3 tree observation (1020 features per train). One PPO epoch = 1024 env steps.</p>

<h3>Baselines (no learning)</h3>
<table><tr><th>Method</th><th>Kind</th><th>What it does</th><th>Tuning</th></tr>{base}</table>

<h3>Hybrids (RL + rule)</h3>
<p class='lead'>The learned policy chooses every action; the rule only holds a train at its start while k trains are
already on the map.</p>
<table><tr><th>Method</th><th>Learned policy</th><th>Rule</th></tr>{hybrid}</table>

<h3>RL recipes</h3>
<p class='lead'>Each row is the reference recipe except the <span class='diffspan'>highlighted</span> cells, so each
group answers one question. The code name under each recipe is its YAML in <code>config/protocol/</code> and its checkpoint folder. Champion reward = +0.01 per step of progress, −5 deadlock, +10 arrival; "+ stuck" adds
−0.01 per step a train is stuck. The 15M and curriculum rows continue training from the 10M model. 3 training seeds
(42–44) per recipe.</p>
<table class='grid'><tr><th>Recipe</th>{head}<th>Env steps</th></tr>{''.join(rows)}</table>

<h3>Evaluation protocol</h3>
<ul>
<li><b>Maps</b> come from the NeurIPS 2020 Flatland competition generator; size, cities and time limit grow with the train count.</li>
<li><b>Validation</b> (map seeds {VALIDATION_SEEDS[0]}–{VALIDATION_SEEDS[-1]}) is only used to pick each run's checkpoint
(highest competition score at the sizes it trained on) and to tune departure caps.</li>
<li><b>Test</b> (map seeds {TEST_SEEDS[0]}–{TEST_SEEDS[-1]}): different maps, never seen in training or selection. RL trained
only at 5 trains is tested zero-shot at the larger sizes.</li>
<li><b>Per test map:</b> {SAMPLED_EPISODES} episodes with the sampled policy (headline numbers) + 1 deterministic (argmax)
episode. Deterministic baselines play each test map once; random plays it {SAMPLED_EPISODES} times.</li>
<li><b>Reported:</b> mean over the test episodes for each training seed, then mean ± std over the 3 seeds.</li>
</ul>
<table><tr><th>Trains</th><th>Map</th><th>Cities</th><th>Time limit (steps)</th><th>Validation maps</th><th>Test maps</th>
<th>Test episodes per seed (RL)</th></tr>{sizes}</table>
<h3>Metrics</h3>
<ul>
<li><b>Completion</b>: share of trains that reach their target.</li>
<li><b>Competition score</b>: 1 − mean(arrival time, or the time limit if not arrived) / time limit. The official
ranking metric; it rewards arriving fast.</li>
<li><b>Deadlock rate</b>: share of trains ever deadlocked. <b>Delay Gini</b>: how unequal delays are across trains
(0 = equal). <b>Makespan</b>: step of the last arrival.</li>
</ul>
"""


def html_table(table: pd.DataFrame, methods: list[str] | None = None) -> str:
    parts = []
    for n in sorted(table.index.get_level_values("trains").unique()):
        part = table.xs(n, level="trains")
        if methods:
            part = part[part.index.isin(methods)]
        part = part.sort_values(("competition_score", "mean"), ascending=False)
        rows = []
        for method, row in part.iterrows():
            cells = []
            for m in METRICS:
                mean, std = row[(m, "mean")], row[(m, "std")]
                if np.isnan(mean):
                    cells.append("–")
                    continue
                text = f"{mean:.0f}" if m == "makespan" else (f"{mean:.0%}" if m in ("completion_rate", "deadlock_rate") else f"{mean:.2f}")
                if not np.isnan(std):
                    text += f" <span class='sd'>± {std:.0f}</span>" if m == "makespan" else (
                        f" <span class='sd'>± {std:.0%}</span>" if m in ("completion_rate", "deadlock_rate") else f" <span class='sd'>± {std:.2f}</span>")
                cells.append(text)
            key = "" if method in BASELINES else f"<br><code class='key'>{method}</code>"
            rows.append(f"<tr><td>{html.escape(label(method))}{key}</td><td>{int(row['seeds'].iloc[0])}</td>"
                        + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
        parts.append(f"<h4>{n} trains</h4><table><tr><th>Method</th><th>Seeds</th><th>Completion</th>"
                     "<th>Competition score</th><th>Deadlock</th><th>Delay Gini</th><th>Makespan</th></tr>"
                     + "".join(rows) + "</table>")
    return "".join(parts)


CSS = """
body { font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif; color: #0b0b0b; background: #fcfcfb;
       max-width: 1080px; margin: 32px auto; padding: 0 20px; line-height: 1.5; font-size: 15px; }
h1 { font-size: 26px; margin-bottom: 4px; } h2 { font-size: 19px; margin-top: 36px; border-bottom: 1px solid #e4e3df; padding-bottom: 4px; }
h3 { font-size: 16px; margin: 20px 0 6px; } ul { padding-left: 20px; } h4 { margin: 18px 0 6px; color: #52514e; } .lead { color: #52514e; }
table { border-collapse: collapse; width: 100%; font-size: 13px; margin-bottom: 10px; }
th, td { border-bottom: 1px solid #e4e3df; padding: 5px 8px; text-align: left; } th { color: #52514e; font-weight: 600; }
.sd { color: #8a8984; font-size: 12px; } img { width: 100%; margin: 8px 0 4px; }
.note { background: #f3f2ee; border-radius: 6px; padding: 10px 14px; font-size: 14px; }
.diff, .diffspan { background: #fff1c2; } tr.group td { background: #f3f2ee; font-weight: 600; color: #52514e; }
table.grid td { font-size: 12.5px; } code.key { font-size: 11px; color: #8a8984; } table.grid td:first-child { min-width: 210px; }
details { margin: 10px 0; } summary { cursor: pointer; color: #2a78d6; }
"""


def _pct(mean: float, std: float = np.nan) -> str:
    if np.isnan(mean):
        return "–"
    return f"{mean:.0%}" if np.isnan(std) else f"{mean * 100:.0f} ± {std * 100:.0f}%"


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c).replace("|", "/") for c in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def _cell(table: pd.DataFrame, method: str, trains: int, metric: str, std: bool = False) -> str:
    if (method, trains) not in table.index:
        return "–"
    row = table.loc[(method, trains)]
    mean, sd = row[(metric, "mean")], row[(metric, "std")]
    if metric in ("completion_rate", "deadlock_rate"):
        return _pct(mean, sd if std else np.nan)
    if np.isnan(mean):
        return "–"
    return f"{mean:.2f}" + (f" ± {sd:.2f}" if std and not np.isnan(sd) else "")


def markdown_report(data: pd.DataFrame, table: pd.DataFrame) -> str:
    """Plain GitHub markdown (tables only, no HTML or images) with the same content as the HTML report."""
    from src.eval.protocol import SAMPLED_EPISODES, TEST_SEEDS, VALIDATION_SEEDS, competition_map
    present = set(data["method"])
    trains = sorted(table.index.get_level_values("trains").unique())
    steps = data[data["selection"] == "curve"].groupby("method")["env_steps"].max()

    def budget(method: str) -> str:
        value = steps.get(method, np.nan)
        return f"{value / 1e6:.1f}M" if not np.isnan(value) else BUDGET_NOTE.get(method, "–")

    out = ["# Multi-agent RL for railway coordination: results", "",
           "Generated by `python -m scripts.html_report <result folders> --out docs/report/README.md`. "
           "Protocol: [technical_guide/evaluation/1.protocol.md](../../technical_guide/evaluation/1.protocol.md); "
           "checkpoints: [docs/checkpoints.md](../checkpoints.md). All results are on held-out test maps; "
           "learned methods are the mean over 3 training seeds.", "",
           "## Setup", "", "### Task", "",
           "Flatland 4.3: all trains share one rail network and must reach their targets before the time limit. Unit speed, "
           "no malfunctions, no intermediate stations (origin to target only); trains may stop and wait anywhere. "
           "5 actions, invalid ones masked. All learned methods see the same depth-3 tree observation (1020 features per "
           "train). One PPO epoch = 1024 env steps.", "",
           "### Baselines (no learning)", "",
           _md_table(["Method", "Kind", "What it does", "Tuning"],
                     [[label(m), k, d, t] for m, (_, k, d, t) in BASELINES.items() if m in present]),
           "### Hybrids (RL + rule)", "",
           "The learned policy chooses every action; the rule only holds a train at its start while k trains are already "
           "on the map.", "",
           _md_table(["Method", "Folder / recipe", "Learned policy", "Rule"],
                     [[label(m), f"`{m}`", f"{label(src)} (`{src}/best.pt`)", "Departure cap, k tuned per size on validation"]
                      for m, src in HYBRIDS.items() if m in present]),
           "### RL recipes", "",
           f"Each row is the reference recipe (`{REFERENCE}`) except the **bold** cells, so each group answers one "
           "question. The folder / recipe name is the YAML in `config/protocol/` and the checkpoint folder. Champion "
           "reward = +0.01 per step of progress, −5 deadlock, +10 arrival; \"+ stuck\" adds −0.01 per step a train is "
           "stuck. The 15M and curriculum rows continue training from the 10M model.", ""]
    rows = []
    for group, recipes in RL_GROUPS:
        recipes = {m: c for m, c in recipes.items() if m in present}
        if not recipes:
            continue
        rows.append([f"**{group}**"] + [""] * (len(DIMENSIONS) + 2))
        for method, changes in recipes.items():
            spec = {**_REF, **changes}
            cells = [f"**{spec[d]}**" if spec[d] != _REF[d] else spec[d] for d in DIMENSIONS]
            b = budget(method)
            rows.append([label(method), f"`{method}`"] + cells
                        + [f"**{b}**" if method != REFERENCE and b != budget(REFERENCE) else b])
    out += [_md_table(["Recipe", "Folder / recipe"] + DIMENSIONS + ["Env steps"], rows),
            "### Evaluation protocol", "",
            "- **Maps**: NeurIPS 2020 Flatland competition generator; size, cities and time limit grow with the train count.",
            f"- **Validation** (map seeds {VALIDATION_SEEDS[0]}–{VALIDATION_SEEDS[-1]}): only used to pick each run's "
            "checkpoint (highest competition score at the sizes it trained on) and to tune departure caps.",
            f"- **Test** (map seeds {TEST_SEEDS[0]}–{TEST_SEEDS[-1]}): different maps, never seen in training or selection. "
            "RL trained only at 5 trains is tested zero-shot at larger sizes.",
            f"- **Per test map**: {SAMPLED_EPISODES} sampled-policy episodes (reported here) + 1 deterministic episode. "
            f"Deterministic baselines play each map once; random plays it {SAMPLED_EPISODES} times.", "",
            _md_table(["Trains", "Map", "Cities", "Time limit (steps)", "Validation maps", "Test maps",
                       "Test episodes per seed (RL)"],
                      [[n, f"{p['width']}×{p['height']}", p["max_num_cities"], p["max_episode_steps"],
                        f"{len(VALIDATION_SEEDS)} (only if trained at this size)", len(TEST_SEEDS),
                        f"{len(TEST_SEEDS) * (SAMPLED_EPISODES + 1)} ({len(TEST_SEEDS) * SAMPLED_EPISODES} sampled + "
                        f"{len(TEST_SEEDS)} deterministic)"]
                       for n, p in ((n, competition_map(n)) for n in (5, 10, 20, 50, 100))]),
            "### Metrics", "",
            "- **Completion**: share of trains that reach their target.",
            "- **Competition score**: 1 − mean(arrival time, or the time limit if not arrived) / time limit. The official "
            "ranking metric; it rewards arriving fast.",
            "- **Deadlock rate**: share of trains ever deadlocked. **Delay Gini**: how unequal delays are across trains "
            "(0 = equal).", ""]

    def methods_in(names):
        return [m for m in names if m in table.index.get_level_values("method")]

    out += ["## 1. Main comparison", "",
            "Completion / competition score at each train count. RL was trained at 5 trains (or the sizes in its recipe) "
            "and is tested zero-shot on larger maps. Prioritized planning is a centralized reference with full information.", "",
            _md_table(["Method"] + [f"{n} trains" for n in trains],
                      [[named(m)] + [f"{_cell(table, m, n, 'completion_rate')} / {_cell(table, m, n, 'competition_score')}"
                                     if (m, n) in table.index else "–" for n in trains] for m in methods_in(MAIN)])]
    number = 2
    for group, recipes in RL_GROUPS[1:]:
        methods = methods_in([REFERENCE] + list(recipes))
        if len(methods) < 2:
            continue
        sizes = GROUP_SIZES.get(group, [5])
        out += [f"## {number}. {group}", "", GROUP_QUESTIONS.get(group, ""), "",
                "Held-out completion (mean ± std over 3 seeds); only the bold settings in the RL recipes table differ "
                "from the reference.", "",
                _md_table(["Recipe"] + [f"{n} trains" for n in sizes],
                          [[named(m)] + [_cell(table, m, n, "completion_rate", std=True) for n in sizes] for m in methods])]
        number += 1
    fair = methods_in(FAIRNESS)
    out += [f"## {number}. Fairness and safety (Responsible AI)", "",
            "Delay Gini (lower = delay spread more evenly) / deadlock rate (share of trains ever deadlocked).", "",
            _md_table(["Method"] + [f"{n} trains" for n in trains],
                      [[named(m)] + [f"{_cell(table, m, n, 'delay_gini')} / {_cell(table, m, n, 'deadlock_rate')}"
                                     if (m, n) in table.index else "–" for n in trains] for m in fair]),
            f"## {number + 1}. Full tables", "",
            "Sampled policy, mean ± std over seeds. Deterministic-policy results are in each run's `evaluation.csv` and "
            "the HTML report.", ""]
    for n in trains:
        part = table.xs(n, level="trains").sort_values(("competition_score", "mean"), ascending=False)
        out += [f"### {n} trains", "",
                _md_table(["Method", "Seeds", "Completion", "Competition score", "Deadlock", "Delay Gini", "Makespan"],
                          [[named(m), int(row["seeds"].iloc[0]), _cell(table, m, n, "completion_rate", True),
                            _cell(table, m, n, "competition_score", True), _cell(table, m, n, "deadlock_rate", True),
                            _cell(table, m, n, "delay_gini", True),
                            "–" if np.isnan(row[("makespan", "mean")]) else f"{row[('makespan', 'mean')]:.0f}"]
                           for m, row in part.iterrows()])]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folders", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("results/report.html"),
                        help="report file; a .md path writes plain GitHub markdown (tables only)")
    parser.add_argument("--updates", type=Path, help="HTML fragment shown first (team news)")
    args = parser.parse_args()
    data = load(args.folders)
    table = test_table(data)
    deterministic = test_table(data, "deterministic")
    sections = []
    if args.updates and args.updates.expanduser().is_file():
        sections.append("<h2>Latest updates</h2>" + args.updates.expanduser().read_text(encoding="utf-8"))
    sections.append(setup_section(data))

    out = args.out.expanduser()
    if out.suffix == ".md":
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown_report(data, table), encoding="utf-8")
        print(f"wrote {out}")
        return

    def figure(png: str | None, caption: str) -> str:
        if png is None:
            return ""
        return f"<img src='data:image/png;base64,{png}'><p class='lead'>{caption}</p>"

    sections.append("<h2>1. Main comparison</h2>" + figure(
        lines_chart(table, MAIN, "Held-out test maps (20 unseen maps per size; learned methods: mean ± std over seeds)"),
        "RL methods were trained at 5 trains (or the sizes listed) and are tested zero-shot on larger maps. "
        "Prioritized planning is a centralized reference with full information."))
    sections.append("<h2>2. Learning curves</h2>" + figure(
        curves_chart(data, CURVES, "Validation score at 5 trains during training (band = seed min–max)"),
        "Validation maps are separate from test maps; each run's best validation checkpoint is the one tested."))
    number = 3
    for group, recipes in RL_GROUPS[1:]:
        methods = [REFERENCE] + list(recipes)
        sizes = GROUP_SIZES.get(group, [5])
        png = bars_chart(table, methods, sizes, f"Held-out completion: {group.lower()}")
        if png is None:
            continue
        sections.append(f"<h2>{number}. {html.escape(group)}</h2><p class='lead'>{html.escape(GROUP_QUESTIONS.get(group, ''))}</p>"
                        + figure(png, "Compared with the reference recipe; only the highlighted settings in the RL recipes "
                                      "table differ. Error bars: std over 3 training seeds."))
        number += 1
    sections.append(f"<h2>{number}. Fairness and safety (Responsible AI)</h2>" + figure(
        grouped_chart(table, FAIRNESS,
                      [("delay_gini", "Delay inequality (Gini, lower = fairer)"), ("deadlock_rate", "Deadlock rate (safety)")]),
        "Delay Gini measures how unevenly delay is spread across trains; deadlock rate is the share of trains ever deadlocked."))
    sections.append(f"<h2>{number + 1}. Full tables</h2><details open><summary>Sampled policy (headline)</summary>"
                    + html_table(table) + "</details><details><summary>Deterministic policy</summary>"
                    + html_table(deterministic) + "</details>")
    page = f"""<!doctype html><html><head><meta charset='utf-8'><title>CS5446 Flatland results</title><style>{CSS}</style></head><body>
<h1>Multi-agent RL for railway coordination: results</h1>
<p class='lead'>Flatland 4.3, NeurIPS 2020 competition map generator, held-out test maps (seeds 1000–1019), checkpoint
selection on separate validation maps (seeds 2000–2019), 3 sampled + 1 deterministic episode per map. No malfunctions,
unit speed, competition episode limit.</p>
<div class='note'>Completion = share of trains that reach their destination. Competition score = 1 − mean arrival time /
time limit (non-arriving trains count as the limit), so faster coordination scores higher.</div>
{''.join(sections)}
</body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
