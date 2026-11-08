"""Regenerate published held-out evaluation tables from the pinned raw bundle."""

import argparse
import hashlib
import json
from pathlib import Path

from src.eval.value_results import (
    read_verified_bundle,
    summary_csv,
    verify_recorded_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="frozen raw .tar.gz, kept outside Git",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="new directory for regenerated CSV/audit",
    )
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output directory already exists")
    report = Path(__file__).resolve().parents[1] / "docs/report/value_based"
    provenance = json.loads((report / "provenance.json").read_text())
    expected = (report / "summary.csv").read_bytes()
    if hashlib.sha256(expected).hexdigest() != provenance["summary_sha256"]:
        raise ValueError("Published summary checksum changed.")
    files = read_verified_bundle(args.bundle, provenance["raw_bundle_sha256"])
    audit, rows = verify_recorded_results(files, provenance)
    regenerated = summary_csv(rows).encode()
    if regenerated != expected:
        raise ValueError("Recomputed result table differs from the published CSV.")
    audit.update(
        published_summary_byte_identical=True,
        raw_bundle_sha256=provenance["raw_bundle_sha256"],
        summary_sha256=provenance["summary_sha256"],
    )
    args.out.mkdir(parents=True)
    (args.out / "summary.csv").write_bytes(regenerated)
    (args.out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
