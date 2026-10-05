#!/usr/bin/env python3
"""Retrieve GEO count archives and metadata pinned by the reproduction manifest."""

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--include-counts", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "paper/manifests/geo_inputs.json").read_text())
    receipt = []
    for item in manifest:
        if item["file"].endswith("_RAW.tar") and not a.include_counts:
            continue
        target = a.output / item["file"]
        if target.exists():
            if digest(target) != item["sha256"]:
                raise RuntimeError(f"existing file differs: {target}")
        else:
            tmp = target.with_suffix(target.suffix + ".part")
            with (
                urllib.request.urlopen(item["url"], timeout=120) as response,
                tmp.open("wb") as out,
            ):
                shutil.copyfileobj(response, out, 8 * 1024 * 1024)
            if digest(tmp) != item["sha256"]:
                raise RuntimeError(f"download differs: {target}")
            tmp.replace(target)
        receipt.append(item)
        print(target.name, "verified", flush=True)
    (a.output / "geo_download_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
