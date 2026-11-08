"""Integrity and independent aggregation checks for published result reproduction."""

import hashlib
import io
import json
import tarfile

import pytest

from src.eval.value_results import (
    METRICS,
    read_verified_bundle,
    summarize_episodes,
    verify_episode_scores,
)


def bundle(tmp_path, name="row.json", extra=False):
    contents = b"{}"
    backup = json.dumps(
        {"files": {name: hashlib.sha256(contents).hexdigest()}}
    ).encode()
    path = tmp_path / "bundle.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        files = [(name, contents), ("backup_manifest.json", backup)]
        if extra:
            files.append(("unexpected.json", contents))
        for key, value in files:
            member = tarfile.TarInfo(key)
            member.size = len(value)
            tar.addfile(member, io.BytesIO(value))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_bundle_checksum_is_required_before_reading_records(tmp_path):
    path, digest = bundle(tmp_path)
    assert read_verified_bundle(path, digest)["row.json"] == b"{}"
    with pytest.raises(ValueError, match="SHA-256"):
        read_verified_bundle(path, "wrong")


@pytest.mark.parametrize("name,extra", [("../outside.json", False), ("row.json", True)])
def test_archive_cannot_escape_or_add_untracked_members(tmp_path, name, extra):
    path, digest = bundle(tmp_path, name, extra)
    with pytest.raises(ValueError):
        read_verified_bundle(path, digest)
    assert not (tmp_path.parent / "outside.json").exists()


def arrival_row():
    return dict(
        trains=5,
        metrics=dict(
            num_trains=5,
            max_episode_steps=420,
            episode_steps=420,
            completion_rate=0.4,
            competition_score=27 / 70,
        ),
        train_records=[
            dict(handle=i, arrival_step=a, interventions=0)
            for i, a in enumerate((10, 20, None, None, None))
        ],
    )


def test_scores_include_unfinished_trains_at_the_full_horizon():
    row = arrival_row()
    assert verify_episode_scores(row) == 5
    row["metrics"]["competition_score"] = 0.9
    with pytest.raises(ValueError, match="arrival"):
        verify_episode_scores(row)


@pytest.mark.parametrize("kind", ["duplicate", "negative_arrival", "intervention"])
def test_invalid_train_records_are_rejected(kind):
    row = arrival_row()
    if kind == "duplicate":
        row["train_records"][1]["handle"] = 0
    elif kind == "negative_arrival":
        row["train_records"][0]["arrival_step"] = -1
    else:
        row["train_records"][0]["interventions"] = 1
    with pytest.raises(ValueError):
        verify_episode_scores(row)


def test_repeats_are_averaged_within_map_before_seed_uncertainty():
    rows = []
    for seed, counts in (
        (42, [(1.0, 3), (0.0, 1)]),
        (43, [(1.0, 1), (0.0, 1)]),
        (44, [(0.0, 1), (0.0, 1)]),
    ):
        for map_seed, (completion, repeats) in enumerate(counts):
            for _ in range(repeats):
                metrics = {m: completion for m in METRICS}
                metrics["mean_travel_time"] = 2.0 if completion else None
                rows.append(
                    dict(
                        selection="best_val",
                        trains=20,
                        method="ppo",
                        execution="sampled",
                        training_seed=seed,
                        map_seed=map_seed,
                        metrics=metrics,
                    )
                )
    result = summarize_episodes(rows, [42, 43, 44])[0]
    assert [s["completion_rate"] for s in result["per_seed"]] == [0.5, 0.5, 0.0]
    assert result["metrics"]["completion_rate"]["mean"] == pytest.approx(1 / 3)
    assert result["metrics"]["mean_travel_time"]["valid_training_seeds"] == 2
