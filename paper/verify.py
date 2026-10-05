"""Strict table verification: row order, full schema, NA mask, text and numbers."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compare(
    actual, expected, *, atol=1e-9, rtol=1e-8, reference_filter=None, columns=None
):
    a = pd.read_csv(actual)
    b = pd.read_csv(expected)
    full = len(b)
    all_reference_columns = list(b)
    if columns is not None:
        if list(a) != columns:
            raise ValueError(
                "Unexpected generated columns for declared partial verification"
            )
        b = b[columns]
    if reference_filter:
        for column, value in reference_filter.items():
            b = b[b[column] == value]
        b = b.reset_index(drop=True)
    result = dict(
        table=Path(expected).name,
        actual_rows=len(a),
        reference_rows=full,
        compared_rows=len(b),
        scope="subset" if len(b) != full else "full",
        atol=atol,
        rtol=rtol,
        columns={},
        passed=False,
    )
    if columns is not None:
        result["scope"] = "column_subset"
        result["verified_columns"] = columns
        result["unverified_columns"] = [
            c for c in all_reference_columns if c not in columns
        ]
    if list(a) != list(b) or a.shape != b.shape:
        result["error"] = dict(
            actual_columns=list(a),
            reference_columns=list(b),
            actual_shape=list(a.shape),
            expected_shape=list(b.shape),
        )
        return result
    for c in a:
        na, nb = a[c].isna().to_numpy(), b[c].isna().to_numpy()
        if not np.array_equal(na, nb):
            result["columns"][c] = {"passed": False, "error": "NA masks differ"}
            continue
        if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
            av, bv = a[c].to_numpy(float), b[c].to_numpy(float)
            same = np.isclose(av, bv, atol=atol, rtol=rtol, equal_nan=True)
            finite = np.isfinite(av) & np.isfinite(bv)
            diff = np.abs(av[finite] - bv[finite])
            maxabs = float(diff.max()) if len(diff) else 0.0
            result["columns"][c] = {
                "passed": bool(same.all()),
                "max_abs": maxabs,
                "mismatched": int((~same).sum()),
            }
        else:
            same = (
                a[c].fillna("__NA__").astype(str) == b[c].fillna("__NA__").astype(str)
            ).to_numpy()
            result["columns"][c] = {
                "passed": bool(same.all()),
                "mismatched": int((~same).sum()),
            }
    result["passed"] = all(v["passed"] for v in result["columns"].values())
    return result


def verify(output, reference):
    result = []
    for actual in sorted(output.glob("*.csv")):
        expected = reference / actual.name
        if not expected.exists():
            continue
        filt = (
            {"base": "pooled_B"}
            if actual.name
            in ["criterion_external_panels.csv", "criterion_external_summary.csv"]
            else None
        )
        tol = (
            5e-6
            if actual.name
            in ["criterion_threshold_loco_v2.csv", "release_rule_validation.csv"]
            else 1e-9
        )
        columns = None
        if actual.name == "liver_core94_signature.csv":
            columns = ["ensg", "sym", "pHSC", "LX2"]
        if actual.name == "liver_spatial_spot_scores.csv":
            columns = [
                "section",
                "barcode",
                "x",
                "y",
                "umi",
                "region",
                "core94",
                "Bdom",
                "Ddom",
                "core94_h",
                "Bdom_h",
                "Ddom_h",
                "cell2loc_hep_prop",
                "cell2loc_stellate_prop",
                "cell2loc_chol_prop",
            ]
        result.append(
            compare(actual, expected, atol=tol, reference_filter=filt, columns=columns)
        )
    panels = json.loads(
        (Path(__file__).parent / "manifests/panel_tables.json").read_text()
    )
    required = {name for panel in panels for name in panel["tables"]}
    full = {row["table"] for row in result if row["passed"] and row["scope"] == "full"}
    outstanding = sorted(required - full)
    return dict(
        full_paper_passed=not outstanding,
        active_plot_tables=len(required),
        fully_verified_active_tables=len(required & full),
        outstanding_full_table_verification=outstanding,
        passed=bool(result) and all(x["passed"] for x in result),
        tables=result,
        note="Pass applies only to listed comparisons; subset coverage is explicit. Unlisted figure tables are not verified.",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generated", type=Path, required=True)
    p.add_argument(
        "--reference", type=Path, default=Path(__file__).parent / "reference_tables"
    )
    p.add_argument("--report", type=Path, required=True)
    p.add_argument(
        "--require-full-paper",
        action="store_true",
        help="Fail unless every active plot table is fully rebuilt and verified",
    )
    a = p.parse_args()
    result = verify(a.generated, a.reference)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    passed = result["passed"] and (
        not a.require_full_paper or result["full_paper_passed"]
    )
    print(json.dumps({
        "passed": passed,
        "comparisons_passed": result["passed"],
        "full_paper_passed": result["full_paper_passed"],
        "compared_tables": len(result["tables"]),
        "required_scope": "full_paper" if a.require_full_paper else "listed_comparisons",
    }))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
