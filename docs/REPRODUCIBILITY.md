# Reproduction contract

The starting point is public expression/count matrices, as selected by the author.
FASTQ processing is outside this contract. `reference_tables/` are comparison
fixtures, never substitute raw inputs. A passing subset is not a complete-paper
reproduction certificate.

The audited scientific environment is Python 3.12; exact observed versions are
in `paper/requirements-tested.txt`. The lightweight package is tested separately
with NumPy 2.5.3 and declares Python 3.10+.

## Install and obtain inputs

```bash
python -m pip install '.[reproduce,test]'
python scripts/download_public_matrices.py --output /data/public_matrices
```

The four files are the publisher-preprocessed expression matrices in
[Zenodo record 14607156](https://zenodo.org/records/14607156). Their archive MD5s
are checked before decompression; the receipt includes decompressed SHA256s.
They already contain a selected gene axis and expression preprocessing. The
workflows preserve that published input scale. They do not claim to reconstruct
steps the publisher performed before depositing these matrices.

Tahoe starts from the publisher's count H5AD, or the original Plate 2 HDF5 chunk
cache containing raw CSR entries. See `paper/manifests/public_inputs.json` for
source locations and local audit hashes. Cohort metadata in `paper/specs/tahoe`
freezes context/condition/gene selection, and contains no expression outcomes.
The code reselects cells from raw metadata. Full-gene library sums are used for
per-cell log1p(10,000 × counts/library) **before** selecting genes and averaging.

## One command for audited workflows

Create a JSON config using your local paths:

```json
{
  "external_dir": "/data/public_matrices",
  "sciplex_input": "/data/public_matrices/sciplex3.h5ad",
  "tahoe_plate2_input": "/data/plate2_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad",
  "tahoe_plate7_input": "/data/plate7_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad",
  "liver_bulk_tar": "/data/geo/GSE253493_RAW.tar",
  "liver_spatial_tar": "/data/geo/GSE338525_RAW.tar",
  "liver_metadata_dir": "/data/geo"
}
```

Alternatively replace `tahoe_plate2_input` with `tahoe_plate2_cache`, pointing to
a directory containing `metadata-cache-v1/` and `x-chunk-cache-v1/`.

```bash
python paper/reproduce.py --config paths.json --workflow all --output paper/generated/run01
```

Use `--workflow external`, `sciplex`, `liver`, `weights`, or `tahoe` for one experiment family. The
runner requires a fresh output directory, hashes inputs, records dependency
versions, commands, script hashes, logs and exit codes, then compares numeric
values, NA patterns, text, full schemas and row order. Failed commands or
comparisons produce a nonzero exit. `all` means **all implemented workflows**,
not all experiments in the article. See the coverage summary in the README for exact coverage.

The external workflow recomputes the pooled-source additive branch on four
matrices. trVAE predictions require separately trained models; a pooled-source
pass does not certify trVAE. The external script retains its historical optional
NPZ prediction branch, but the runner's fresh output directory prevents inherited
prediction files from silently entering that run.

The sciPlex depth experiment uses fixed treated-cell samples, plate-disjoint
controls and nested depth levels. It writes the panel scores, centered numerator
gaps, and control-noise/signal table. `gxn=gap*cells_per_condition` and
`nxn=noise_centred*n0` are derived automatically for historical plot compatibility.
The historical lambda signal denominator is uncentered; this is an explicit
historical convention, distinct from the package's centered default.

Tahoe reconstructs the original 15-context Plate 2/Plate 7 comparison, uses a
source-only SVD basis, and reports calibration-only release evidence. Plate 7
selects 400 treated cells and 400 controls **per half**, with nested shallow
subsets of 20 treated and 10 controls per half. The historical Plate 7 path reads
the publisher's filtered file and does not apply a second `pass_filter` condition.
Gene symbols are remapped independently in Plate 7; Plate 2 column positions
are never reused as Plate 7 gene identities.

`derive_tables.py` regenerates release counts and LOCO threshold summaries from
panel rows, replacing historical hand-entered summary values. The deep-tuned
threshold is evaluated on shallow data separately from the zero-threshold rule.

## Release

```bash
python -m pip install '.[dev]'
python scripts/publish_github.py --remote git@github.com:OWNER/REPO.git --dry-run
python scripts/publish_github.py --remote git@github.com:OWNER/REPO.git
```

Use an empty GitHub repository and your configured git identity/authentication.
The script tests, builds, validates package metadata, commits only this repository,
and pushes without force. It refuses to replace a different origin. No tokens or
raw matrices are embedded. `--dry-run` runs validation but does not mutate git or
contact the destination repository. This does not publish to PyPI.

## Liver raw-count workflow

```bash
python scripts/download_geo_inputs.py --output /data/geo --include-counts
python paper/reproduce.py --config paths.json --workflow liver --output paper/generated/liver01
```

[GSE253493](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE253493) supplies
12 bulk count files and their public sample titles. Counts are grouped by
unversioned Ensembl identifier, normalized to log2(1+CPM), filtered for counts
>10 in at least six samples, and mapped using the spatial matrix's gene
annotations. Primary-HSC and LX-2 treatment effects determine the 438-gene
positive-response decomposition and 94-gene cross-system set directly. No
saved signature membership or effect sizes are used to make those decisions.

[GSE338525](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE338525) supplies
spatial count H5 files, coordinates and four official regional metadata files.
Those metadata contain nine fields despite an eight-field header; the parser
checks identities and uses the correct region and cell2location columns.
BA1_3 is absent from the author's official analysis metadata and excluded from
the active 15-section output. Gene scores use within-section standardization
of log1p(CP10k). Historical CSV rounding boundaries are preserved before
hepatocyte referencing.

The per-spot output verifies 15 of the 17 current columns. It deliberately omits
`hsc_identity` and `signed_hsc_state`, whose frozen marker/axis construction
remains unaudited. The 94-gene signature verifies four of five columns and
omits `in_frozen_axis`. These column subsets are explicitly listed in comparison
receipts. The decomposition and section-contrast tables are full comparisons.

`liver_thresholds.py` re-derives the 60 section/threshold rows from raw spatial
counts and the newly computed bulk effects. The separate expression-matched
null-draw summary is not silently certified by that per-section comparison.

## Preserved outputs

`paper/verified_tables/` stores recomputed CSVs, including explicitly partial
column tables. `paper/reference_tables/` stores the version under review.
`paper/audit/comparison.json` records numeric differences and exact coverage;
`paper/manifests/verified_tables.json` pins the regenerated files. Run
`python scripts/check_integrity.py` to detect file drift.

To require coverage of **every active plot table**, use the stricter audit:

```bash
python paper/verify.py --generated paper/generated/run01 \
  --report paper/generated/run01/full-paper.json --require-full-paper
```

This returns a nonzero exit while unverified/partial tables remain. A normal
workflow-level pass never changes `full_paper_passed` to true by implication.

## Separate 16-context weight comparison

`--workflow weights` reconstructs 16 contexts × 86 conditions × 7007 genes
from Plate 7 counts (seed 23, archived float32 boundaries), then runs three
recovered scripts: `weightrule2`, `weightrule`, and `lib_ablate`. These generate
the four active weight-rule/library-ablation tables plus supporting raw rows.
The nearest-control-context and calibration-rank-one candidates in this
comparison are distinct from the source-simplex/source-basis candidates in
the 15-context release experiment. The package exposes the latter convention.

The liver workflow also recomputes abundance-stratified contrasts from the raw
spot scores and official deconvolution covariates, with separate numerical
claim evidence in `liver_abundance_claims.json`. Numerical agreement does not
resolve within-patient/within-section clustering assumptions.
