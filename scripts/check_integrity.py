#!/usr/bin/env python3
"""Verify all committed reference/result table hashes without requiring raw data."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    count = 0
    for manifest, directory in [
        ("reference_tables.json", "reference_tables"),
        ("verified_tables.json", "verified_tables"),
    ]:
        records = json.loads((ROOT / "paper/manifests" / manifest).read_text())
        for row in records:
            path = ROOT / "paper" / directory / row["file"]
            if (
                path.stat().st_size != row["bytes"]
                or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]
            ):
                raise RuntimeError(f"file drift: {directory}/{row['file']}")
            count += 1
    print(
        f"{count} table hashes verified. This is file integrity, not raw-data reproduction."
    )


if __name__ == "__main__":
    main()
