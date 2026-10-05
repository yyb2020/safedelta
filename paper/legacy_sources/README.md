# Recovered analysis sources

`weightrule2.py`, `weightrule.py` and `lib_ablate.py` were executed on the
newly raw-rebuilt 16-context, 86-condition, 7007-gene array by
`../workflows/tahoe_weight_rules.py`; their active output tables passed comparison.

The remaining files are recovered sources that have NOT been run from raw data
in this audit. They retain historical paths, input assumptions and dependencies;
placing them here is not a reproduction certificate. The release removes the
retired dataset registry entry from two training files; upstream/released hashes
are recorded in `../manifests/legacy_sources.json`.

The 16-line reconstruction and candidate library differ from the 15-context
main release experiment. Do not substitute their models, seeds or cohorts.

`coupling_probe.py` is the shared helper imported by `run_trvae.py`, `run_scgen.py`
and `run_w10_pairwise.py` (projection coefficient, noise-to-signal ratio, safe CSV
appends). It was added in the `safedelta` 0.3.0 release so that those scripts can be
imported; it is not listed in `../manifests/legacy_sources.json`.
