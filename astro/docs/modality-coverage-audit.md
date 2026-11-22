# Modality coverage audit

Check catalogue coverage, usable scalar labels and selected payloads in finite
sky fields on CPU. The audit uses the configured North/South stream and its
published margins, with an independent reference for the emitted records.

## Run

From `astro/`, with the development environment installed:

```bash
uv run python scripts/audit_modalities.py \
  --recipe configs/nanotron/astropt3-70m.yaml \
  --fields configs/audit/modality-validation-fields.json \
  --max-payload-records 64 \
  --max-group-mib 256 --max-payload-mib-per-catalog 4096 \
  --metadata-cache ../cache/modality-audit \
  --output ../astroPTv3_telemetry/modality-audit/run-01

uv run python scripts/verify_modality_audit.py \
  --run ../astroPTv3_telemetry/modality-audit/run-01
```

The output directory must be new. Both paths above are covered by the existing
repository ignore rules. Reuse the metadata cache and choose a new run directory
for each audit. Catalogue revisions must be immutable 40-character commit pins.
Only metadata and coordinate/scalar projections are cached; image and spectrum
reads retain the selected rows instead of downloading a complete catalogue.

The run saves its recipe and executing source, input hashes, fields, per-field
coverage, selected payloads, decoder checks and generated plots. The offline
verifier reads these files and writes `verification.json`; it never executes
the archived source. Raw runs, caches and generated figures belong outside the
tracked contribution.

## Checks and limits

- Coverage counts use all emitted records, including unmatched partners, as
  their denominator. Scalar counts distinguish usable, missing, nonfinite,
  sentinel, quality-rejected and invalid-transform values using the production
  sequencer's rules. Catalogue presence and usable labels are separate measures.
- An independent Astropy reference checks anchor IDs, nearest partners and
  unmatched partner IDs against primary rows. For an aperture of radius `R`
  and matching radius `r`, it reads anchors to `R+r` and partners to `R+2r`.
  This includes competing matches outside the aperture. The reference respects
  the recipe footprint and checks paths with zero emitted rows too.
- Any nearest ID is accepted for a tied core-anchor match. Outer records whose
  status depends on a halo anchor's ambiguous tie are counted as unverified;
  the run returns nonzero instead of claiming a complete pass.
- Payload selection uses stable hashes within source combinations, then covers
  unseen sources and field/path pairs before filling the global cap. It is a
  deliberate coverage sample. Failed decoding, nonfinite tokens and unavailable
  payloads fail validation. `--payloads-per-combination 0` produces an explicit
  `coverage_only` result.
- Offline replay checks hashes, record accounting, duplicate partner IDs across
  anchor paths, and joined coordinates/scalars against the saved primary rows.
  It repeats decoding and verifies token order, position resets, finite values
  and modality-mask alignment in packed tensors.

The fixed manifest has eight seeded fields in each of North-only, South-only
and overlap coverage, plus separately targeted COSMOS. All 25 apertures are
disjoint and have radius 300 arcseconds. Seeded centres use seed 42 and at least
2 degrees of separation within each stratum. A field's label describes its
centre; the run also records coverage classifications for individual rows.
Omit `--fields` to use the selector; see `--help` for its options.

The default cap is 12 output partitions per field/path; upstream inputs can
span additional partitions. Default payload limits are 192 MiB per row group
and 512 MiB per catalogue. The command above raises those limits to accommodate
South image groups and the larger field sample. Reads are sequential. These
limits describe projected uncompressed sizes, not peak RAM or exact wire bytes.
An oversized group or exhausted budget is reported as a failed payload check.

Input fill fractions follow published South image and spectrum masks;
North/HSC use zero-fill proxies and RGB is checked after dequantization.
Entirely filled inputs are reported even if their tokens are finite. These
checks do not establish scientific calibration or label accuracy. Sequential
packing checks tensor integrity, not streaming-packer efficiency or throughput.

## Recorded validation (2026-11-22)

The [aggregate results](assets/modality-audit/results.json) record the exact
catalogue revisions, field-manifest hash, package versions and source hashes.
All 1,503 anchors, 6,012 anchor/partner comparisons and 519 unmatched records
agreed with the independent reference, with no ambiguous ties or cross-path
partner duplicates.

| Field-centre stratum | Fields | Records | Anchors | Unmatched | DESI | HSC | Galaxies |
|---|---:|---:|---:|---:|---:|---:|---:|
| North only | 8 | 248 | 168 | 80 | 105 | 0 | 41 |
| South only | 8 | 702 | 658 | 44 | 0 | 177 | 75 |
| Overlap | 8 | 778 | 450 | 328 | 0 | 462 | 84 |
| COSMOS (targeted) | 1 | 294 | 227 | 67 | 101 | 109 | 15 |

Catalogue columns overlap. In the 24 seeded fields, 1,251/1,728 records (72.4%)
contain one catalogue source, which can still supply multiple image bands.
Galaxies occurs in 200 records; 63 have usable SFR labels. COSMOS is kept
separate. These finite fields do not estimate the weighted training mixture
or establish the cause of model learning curves.

All 64 selected records passed decoding and offline replay, preserving 43,704
nonpad tokens in 12 packed rows. No budget skips, nonfinite tokens or entirely
filled inputs occurred. Input fill fractions reached 43.8% for South images and
52.7% for SDSS spectra. The live run took 86.7 minutes with partly warm caches
and returned 4.87 GiB in logged ranges; setup is excluded and peak RAM was not
measured. Source hashes and the byte-preserving recipe-snapshot correction are
recorded with the results; measurement logic is unchanged.

Validation passed with 485 Linux CPU tests, 4 skipped and 19 deselected, plus
all 19 audit tests on Windows. All named parameter counts were within tolerance:

```bash
uv run pytest -m 'not gpu and not network' -q
uv run python scripts/count_params.py
```

Linux validation used GNU `echo` for the existing launcher fixtures. Regression
checks cover real local HATS joins, boundary and halo cases, scalar mismatches,
artifact tampering, payload budgets and LF/CRLF recipe snapshots. A live check
with an intentionally undersized budget returned exit code 1 with four
unavailable payloads.
