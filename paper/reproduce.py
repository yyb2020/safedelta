#!/usr/bin/env python3
"""Run audited public-matrix workflows and compare newly computed plot tables.

No workflow reads reference_tables while generating its outputs. Model training
branches and other unaudited manuscript experiments are not implicitly included.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from verify import verify

HERE = Path(__file__).resolve().parent


def sha256(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--workflow",
        choices=["external", "sciplex", "tahoe", "liver", "weights", "all"],
        default="all",
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    out = a.output.resolve()
    if out.exists() and any(out.iterdir()):
        p.error("use a new empty output directory; no stale-result reuse")
    out.mkdir(parents=True, exist_ok=True)
    plan = []
    inputs = []
    if a.workflow in ["external", "all"]:
        directory = Path(config["external_dir"]).resolve()
        names = ["sciplex3", "KaggleCrossCell", "McFarland", "Haber"]
        plan.append(
            [
                "external_criterion.py",
                "--data-dir",
                str(directory),
                "--out-dir",
                str(out),
                "--datasets",
                *names,
            ]
        )
        inputs.extend(directory / (n + ".h5ad") for n in names)
    if a.workflow in ["sciplex", "all"]:
        raw = Path(config["sciplex_input"]).resolve()
        inputs.append(raw)
        plan.append(["sciplex_depth.py", "--input", str(raw), "--out-dir", str(out)])
    if a.workflow in ["tahoe", "all"]:
        key = (
            "tahoe_plate2_input"
            if "tahoe_plate2_input" in config
            else "tahoe_plate2_cache"
        )
        raw = Path(config[key]).resolve()
        if key.endswith("cache"):
            inputs.extend(x for x in raw.rglob("*") if x.is_file())
        else:
            inputs.append(raw)
        plan.append(
            [
                "tahoe_raw.py",
                "--plate",
                "2",
                "--input" if key.endswith("input") else "--cache",
                str(raw),
                "--output",
                str(out),
            ]
        )
        raw = Path(config["tahoe_plate7_input"]).resolve()
        inputs.append(raw)
        plan.extend(
            [
                [
                    "tahoe_raw.py",
                    "--plate",
                    "7",
                    "--input",
                    str(raw),
                    "--output",
                    str(out),
                ],
                [
                    "tahoe_analysis.py",
                    "--data-dir",
                    str(out),
                    "--output",
                    str(out),
                    "--ladder",
                ],
                ["derive_tables.py", "--input", str(out)],
            ]
        )
    if a.workflow in ["liver", "all"]:
        bulk = Path(config["liver_bulk_tar"]).resolve()
        spatial = Path(config["liver_spatial_tar"]).resolve()
        metadata = Path(config["liver_metadata_dir"]).resolve()
        inputs.extend([bulk, spatial, metadata / "GSE253493_series_matrix.txt.gz"])
        inputs.extend(
            metadata / f"GSE338525_{group}_metadata.csv.gz"
            for group in ["BA1", "BA2", "NL", "NonBA"]
        )
        plan.append(
            [
                "liver_raw.py",
                "--bulk-tar",
                str(bulk),
                "--spatial-tar",
                str(spatial),
                "--metadata-dir",
                str(metadata),
                "--output",
                str(out),
            ]
        )
        plan.append(
            [
                "liver_thresholds.py",
                "--spatial-tar",
                str(spatial),
                "--metadata-dir",
                str(metadata),
                "--output",
                str(out),
            ]
        )
        plan.append(
            [
                "liver_abundance.py",
                "--output",
                str(out),
                "--metadata-dir",
                str(metadata),
            ]
        )
    if a.workflow in ["weights", "all"]:
        raw = Path(config["tahoe_plate7_input"]).resolve()
        inputs.append(raw)
        plan.append(["tahoe_16line_raw.py", "--input", str(raw), "--output", str(out)])
        plan.append(["tahoe_weight_rules.py", "--output", str(out)])
    receipt = {
        "workflow": a.workflow,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "platform": platform.platform(),
        "environment": {},
        "inputs": [],
        "commands": [],
        "completed": False,
    }
    for name in ["numpy", "pandas", "scipy", "h5py", "anndata", "safedelta"]:
        try:
            receipt["environment"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            receipt["environment"][name] = "not installed"
    report = out / "run.json"
    try:
        for raw in sorted(set(inputs)):
            receipt["inputs"].append(
                {"path": str(raw), "bytes": raw.stat().st_size, "sha256": sha256(raw)}
            )
        env = dict(
            os.environ,
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        )
        for j, args in enumerate(plan):
            script = HERE / "workflows" / args[0]
            cmd = [sys.executable, str(script), *args[1:]]
            print("Running", args[0], flush=True)
            entry = {
                "script": args[0],
                "sha256": sha256(script),
                "arguments": args[1:],
                "log": f"{j:02d}_{script.stem}.log",
            }
            receipt["commands"].append(entry)
            with (out / entry["log"]).open("w") as log:
                result = subprocess.run(
                    cmd, stdout=log, stderr=subprocess.STDOUT, env=env, check=False
                )
            entry["exit_code"] = result.returncode
            if result.returncode:
                raise RuntimeError(f"{args[0]} failed: see {entry['log']}")
        comparison = verify(out, HERE / "reference_tables")
        (out / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
        receipt["completed"] = True
        receipt["comparisons_passed"] = comparison["passed"]
        if not comparison["passed"]:
            raise RuntimeError("result mismatch: see comparison.json")
    finally:
        receipt["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        report.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        "Audited workflow comparisons passed. Full-paper coverage: see the README coverage summary."
    )


if __name__ == "__main__":
    main()
