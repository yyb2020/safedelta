"""Portable NPZ interface; all input arrays must use identical target/gene axes."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import SplitControlAdapter, __version__


def main(argv=None):
    parser = argparse.ArgumentParser(prog="safedelta")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser(
        "fit", help="Fit a calibration-only release rule from NPZ arrays"
    )
    fit.add_argument("input", type=Path)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--adapter", choices=["offset", "mixture"], default="offset")
    fit.add_argument(
        "--convention", choices=["one_way", "symmetric"], default="symmetric"
    )
    args = parser.parse_args(argv)
    try:
        with np.load(args.input, allow_pickle=False) as data:
            model = SplitControlAdapter(
                adapter=args.adapter, convention=args.convention
            ).fit(
                **{
                    k: data[k]
                    for k in [
                        "source_responses",
                        "calibration_indices",
                        "treated_calibration",
                        "control_a",
                        "control_b",
                    ]
                }
            )
        args.output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output / "predictions.npz",
            fold_a=model.predict(fold=0),
            fold_b=model.predict(fold=1),
        )
        decision = asdict(model.decision_)
        if not np.isfinite(decision["evidence"]):
            decision["evidence"] = None
        (args.output / "decision.json").write_text(
            json.dumps(decision, indent=2, allow_nan=False) + "\n"
        )
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"safedelta: {error}\n")
    print(json.dumps(decision, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
