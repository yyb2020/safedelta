# Changelog

## 1.0.0 — 2026-10-05

First public release. The internal 0.3.0 staging version
was not published; its contents are released as 1.0.0 with unchanged numerical code.

Renamed from `controlcoupling` to `safedelta`.

- Distribution name, import name and command-line entry point are now `safedelta`
  (`pip install safedelta`, `import safedelta`, `safedelta fit ...`).
- Environment variables used by the figure code were renamed from
  `CONTROLCOUPLING_TABLES` / `CONTROLCOUPLING_FIGURES` to
  `SAFEDELTA_TABLES` / `SAFEDELTA_FIGURES`.
- No numerical code was changed. The public API (`SplitControlAdapter`,
  `ReleaseDecision`, `measure_a`, `measure_a_directional`, `measure_lambda`,
  `dual_regime_scores`, `check_identity`, `split_control`, ...) is identical to
  `controlcoupling` 0.2.0.
- The audit receipts in `paper/audit/` were produced with `controlcoupling` 0.2.0
  and are kept unchanged as historical records; the package name and script hashes
  recorded there refer to the pre-rename files.
- Added `CITATION.cff` and this changelog; package metadata completed.
- Repository slimmed for release: figure-rendering code, per-panel plot tables, the
  manuscript-claim checker and internal review notes were removed. `paper/` keeps the
  public-matrix workflows, the 20 recomputed tables, the 36 stored tables they are
  compared with or that summarise the other main results, and the run scripts.
- `paper/legacy_sources/coupling_probe.py` added (helper imported by three run scripts).

## 0.2.0 (as `controlcoupling`) — 2026-10-01

Release candidate with the audited reproduction workflows under `paper/`.

## 0.1.0 (as `controlcoupling`) — 2026-09-22

First internal version of the split-control diagnostics.
