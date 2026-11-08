"""Verify and regenerate published evaluation tables from frozen records.

This is a data audit, not training or inference. It never imports archive code,
extracts paths, or unpickles checkpoints. Standard task limits come from the
shared evaluation protocol; scores are checked independently from arrival times.
"""

import csv
import hashlib
import io
import json
import math
import statistics
import tarfile
from collections import defaultdict
from pathlib import Path

from src.eval.protocol import competition_map

PRIMARY_METRICS = (
    "completion_rate",
    "competition_score",
    "departure_rate",
    "deadlock_rate",
    "strict_deadlock_rate",
    "stuck_rate",
)
METRICS = PRIMARY_METRICS + (
    "mean_travel_time",
    "makespan",
    "delay_gini",
    "mean_delay_all",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_verified_bundle(path: Path, expected_sha256: str) -> dict[str, bytes]:
    """Read a checksum-pinned archive in memory; reject extra/changed members."""
    _require(
        hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256,
        "Raw bundle SHA-256 does not match the published provenance.",
    )
    files = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            name = Path(member.name)
            _require(
                member.isfile() and not name.is_absolute() and ".." not in name.parts,
                "Bundle must contain only relative regular files.",
            )
            _require(member.name not in files, "Duplicate archive member.")
            files[member.name] = archive.extractfile(member).read()
    backup = json.loads(files["backup_manifest.json"])
    _require(
        set(files) == set(backup["files"]) | {"backup_manifest.json"},
        "Bundle member coverage changed.",
    )
    for name, digest in backup["files"].items():
        _require(
            hashlib.sha256(files[name]).hexdigest() == digest,
            f"Bundle member checksum changed: {name}",
        )
    return files


def _cases(method: str) -> list[tuple]:
    if method == "ppo":
        return [("sampled", None, r) for r in range(3)] + [("greedy", None, 3)]
    _require(method == "pqn", "Unexpected method in frozen evaluation.")
    return [("greedy", 0.0, 0)] + [("epsilon_005", 0.05, r) for r in range(3)]


def verify_episode_scores(row: dict) -> int:
    """Check completion and score directly from per-train arrival records."""
    n = row["trains"]
    horizon = competition_map(n)["max_episode_steps"]
    records, metrics = row["train_records"], row["metrics"]
    _require(
        len(records) == n and {r["handle"] for r in records} == set(range(n)),
        "Missing or duplicate per-train records.",
    )
    _require(
        metrics["num_trains"] == n and metrics["max_episode_steps"] == horizon,
        "Episode task size/horizon changed.",
    )
    _require(0 < metrics["episode_steps"] <= horizon, "Invalid episode length.")
    arrivals = [r["arrival_step"] for r in records if r["arrival_step"] is not None]
    _require(all(0 <= a <= horizon for a in arrivals), "Invalid recorded arrival.")
    completion = len(arrivals) / n
    score = sum(horizon - a for a in arrivals) / (n * horizon)
    for key, value in (("completion_rate", completion), ("competition_score", score)):
        _require(
            math.isclose(value, metrics[key], rel_tol=0, abs_tol=1e-12),
            f"Recorded {key} disagrees with train arrivals.",
        )
    _require(sum(r["interventions"] for r in records) == 0, "Unexpected intervention.")
    return n


def _finite_mean(values: list) -> float | None:
    values = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.mean(values) if values else None


def summarize_episodes(rows: list[dict], seeds: list[int]) -> list[dict]:
    """Average repeats within map, maps within model, then training seeds."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[
            row["selection"], row["trains"], row["method"], row["execution"]
        ].append(row)
    groups = []
    for (role, trains, method, execution), part in sorted(grouped.items()):
        per_seed = []
        for seed in seeds:
            by_map = defaultdict(list)
            for row in part:
                if row["training_seed"] == seed:
                    by_map[row["map_seed"]].append(row)
            item = dict(seed=seed)
            for metric in METRICS:
                item[metric] = _finite_mean(
                    [
                        _finite_mean([r["metrics"][metric] for r in repeats])
                        for repeats in by_map.values()
                    ]
                )
            per_seed.append(item)
        aggregate = {}
        for metric in METRICS:
            values = [s[metric] for s in per_seed if s[metric] is not None]
            aggregate[metric] = dict(
                mean=_finite_mean(values),
                sd=statistics.stdev(values) if len(values) > 1 else None,
                valid_training_seeds=len(values),
            )
        groups.append(
            dict(
                selection=role,
                trains=trains,
                method=method,
                execution=execution,
                primary=execution == ("sampled" if method == "ppo" else "greedy"),
                episodes=len(part),
                per_seed=per_seed,
                metrics=aggregate,
            )
        )
    return groups


def verify_recorded_results(
    files: dict[str, bytes], provenance: dict
) -> tuple[dict, list[dict]]:
    """Audit all frozen identities and reproduce each native cohort summary."""

    def load(name: str) -> dict:
        return json.loads(files[name])

    manifest, backup = load("manifest.json"), load("backup_manifest.json")
    manifest_sha = hashlib.sha256(files["manifest.json"]).hexdigest()
    _require(
        manifest_sha == provenance["evaluation_manifest_sha256"],
        "Frozen evaluation manifest changed.",
    )
    _require(manifest["training_updates"] == 0, "Unexpected test-time updates.")
    _require(
        hashlib.sha256(files["training/scale_full_manifest.json"]).hexdigest()
        == provenance["training_manifest_sha256"],
        "Training manifest changed.",
    )
    _require(
        hashlib.sha256(files["training/frozen_models.json"]).hexdigest()
        == manifest["training_model_audit_sha256"],
        "Model freeze audit changed.",
    )
    models, seeds = manifest["models"], provenance["training_seeds"]
    _require(
        {(m["method"], m["seed"]) for m in models}
        == {(m, s) for m in ("ppo", "pqn") for s in seeds}
        and len(models) == 6,
        "Frozen model coverage changed.",
    )
    frozen = load("training/frozen_models.json")
    _require(
        frozen["formal_env_steps"] == 18_000_000 and frozen["test_episodes"] == 0,
        "Incomplete training/model freeze.",
    )
    for model in models:
        _require(model["training_budget"] == 3_000_000, "Different training budget.")
        prior = next(
            m
            for m in frozen["models"]
            if (m["method"], m["training_seed"]) == (model["method"], model["seed"])
        )
        _require(
            set(model["selections"]) == {"best_val", "final"},
            "Checkpoint roles changed.",
        )
        for role, ref in model["selections"].items():
            _require(
                (ref["path"], ref["steps"], ref["sha256"])
                == (
                    prior[role]["path"],
                    prior[role]["env_steps"],
                    prior[role]["sha256"],
                ),
                "Frozen checkpoint selection changed.",
            )
            copied = backup["weight_paths"][ref["path"]]
            _require(
                hashlib.sha256(files[copied]).hexdigest() == ref["sha256"],
                "Frozen checkpoint weights changed.",
            )
    counts, csv_rows, train_records, parity = {}, [], 0, 0
    for stage, tasks, count, sizes, map_seeds in (
        ("pilot", 12, 144, (5, 20, 50), [2000]),
        ("standard", 96, 3840, (5, 10, 20, 50), provenance["standard_test_seeds"]),
        ("confirmation", 96, 3840, (5, 10, 20, 50), provenance["confirmation_seeds"]),
    ):
        rows, fingerprints, identities = [], defaultdict(set), set()
        expected = {
            (m["run"], role, n, s, e, eps, r)
            for m in models
            for role in ("final", "best_val")
            for n in sizes
            for s in map_seeds
            for e, eps, r in _cases(m["method"])
        }
        stage_files = {n for n in files if n.startswith(stage + "/")}
        _require(
            stage_files
            == {
                f"{stage}/task_{i}/{name}"
                for i in range(tasks)
                for name in ("complete.json", "episodes.jsonl")
            },
            "Missing or unexpected evaluation shards.",
        )
        for index in range(tasks):
            prefix = f"{stage}/task_{index}"
            done = load(prefix + "/complete.json")
            _require(
                done["manifest_sha256"] == manifest_sha
                and done["training_updates"] == 0,
                "Shard provenance or update count changed.",
            )
            task = done["task"]
            model = models[task["model"]]
            ref = model["selections"][task["selection"]]
            _require(
                done["checkpoint_sha256"] == ref["sha256"], "Shard checkpoint changed."
            )
            part = [
                json.loads(line)
                for line in files[prefix + "/episodes.jsonl"].splitlines()
            ]
            _require(
                len(part) == done["episodes"]
                and done["test_episodes"] == (0 if stage == "pilot" else len(part)),
                "Shard episode count changed.",
            )
            for row in part:
                identity = (
                    row["run"],
                    row["selection"],
                    row["trains"],
                    row["map_seed"],
                    row["execution"],
                    row["epsilon"],
                    row["repeat"],
                )
                _require(
                    identity in expected and identity not in identities,
                    "Duplicate or unexpected episode.",
                )
                identities.add(identity)
                _require(
                    (row["run"], row["method"], row["training_seed"], row["selection"])
                    == (
                        model["run"],
                        model["method"],
                        model["seed"],
                        task["selection"],
                    ),
                    "Episode model identity changed.",
                )
                _require(
                    row["stage"] == stage
                    and row["split"] == ("validation" if stage == "pilot" else "test"),
                    "Episode split changed.",
                )
                _require(
                    row["checkpoint_sha256"] == ref["sha256"]
                    and row["checkpoint_steps"] == ref["steps"],
                    "Episode checkpoint changed.",
                )
                _require(
                    row["action_seed"] == row["map_seed"] * 1000 + row["repeat"],
                    "Action seed changed.",
                )
                _require(
                    row["counts"].get("interventions", 0) == 0,
                    "Unexpected intervention.",
                )
                _require(
                    all(
                        math.isfinite(row["metrics"][m]) and 0 <= row["metrics"][m] <= 1
                        for m in PRIMARY_METRICS
                    ),
                    "Invalid primary metric.",
                )
                train_records += verify_episode_scores(row)
                fingerprints[row["trains"], row["map_seed"]].add(
                    row["scenario_fingerprint"]
                )
                canonical = row["trains"] == 20 and (
                    (
                        model["method"] == "ppo"
                        and row["execution"] == "sampled"
                        and row["repeat"] == 0
                    )
                    or (model["method"] == "pqn" and row["execution"] == "greedy")
                )
                if stage == "pilot" and canonical:
                    previous = next(
                        r
                        for r in ref["validation_rows"]
                        if r["map_seed"] == row["map_seed"]
                    )
                    _require(
                        row["upstream_validation_parity"] is True
                        and all(
                            math.isclose(
                                row["metrics"][k], previous[k], rel_tol=0, abs_tol=1e-12
                            )
                            for k in PRIMARY_METRICS
                        ),
                        "Pilot validation parity failed.",
                    )
                    parity += 1
            rows.extend(part)
        _require(
            identities == expected and len(rows) == count,
            "Incomplete evaluation coverage.",
        )
        _require(
            len(fingerprints) == len(sizes) * len(map_seeds)
            and all(len(v) == 1 for v in fingerprints.values()),
            "Physical scenarios differ across methods.",
        )
        groups = summarize_episodes(rows, seeds)
        native = load(f"native_{stage}_summary.json")
        _require(
            native["groups"] == groups
            and native["episodes"] == count
            and native["training_updates"] == 0
            and native["manifest_sha256"] == manifest_sha,
            f"Recomputed {stage} summary disagrees with native audit.",
        )
        counts[stage] = count
        if stage != "pilot":
            for g in groups:
                item = dict(
                    stage=stage,
                    **{
                        k: g[k]
                        for k in (
                            "selection",
                            "trains",
                            "method",
                            "execution",
                            "primary",
                            "episodes",
                        )
                    },
                )
                for metric, values in g["metrics"].items():
                    item.update({f"{metric}_{k}": v for k, v in values.items()})
                csv_rows.append(item)
    _require(parity == 12, "Incomplete pilot parity checks.")
    audit = dict(
        episodes=counts,
        train_records_checked=train_records,
        canonical_parity_checks=parity,
        result_groups=len(csv_rows),
        native_summaries_reproduced=3,
        model_updates=0,
        training_rerun=False,
        policy_inference_rerun=False,
        evaluation_manifest_sha256=manifest_sha,
    )
    return audit, csv_rows


def summary_csv(rows: list[dict]) -> str:
    """Serialize regenerated aggregate tables with repository LF line endings."""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()
