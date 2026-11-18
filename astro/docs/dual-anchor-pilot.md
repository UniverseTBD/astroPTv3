# Dual-anchor training pilot (#58)

**The mixed tiny model learned both image modalities.** After eight CPU updates its loss on two fixed, non-updated objects per anchor fell from **292.50 to 246.88 (North)** and **1041.67 to 480.85 (South)**. The North-only control reached 233.63 on North; its South loss barely moved. This tests the decoder → sequencer → packed HF model → optimizer path with real pixels, not the live join, spatial generalization, or the optimal sampling weight.

## Learning comparison

One HF `test-tiny` architecture with both image modalities (420,768 parameters), fp32, seed 58, AdamW at 0.001, eight steps of two objects in one 320-position row per arm. All arms started with identical weights (SHA-256 of state tensor bytes: `57715c11db06d7f69879e71afca923f02678f63515407ef6bbf50b55c63a73ed`) and a fresh optimizer. For each anchor, samples 0–7 provide updates; 8–9 are fixed diagnostics from the same row group. Checked the 20 coordinates against `catalog_stream._split_sky('train').contains_lonlat`: 10/10 North and 10/10 South are in the buffered training sky. Diagnostics are ID-disjoint but **not** a spatial validation set: nearby cutouts can overlap. No hyperparameter selection or production checkpoint.

| Arm | Training loss, first → last batch | Fixed North loss, before → after | Fixed South loss, before → after | Equal-anchor mean after |
|---|---:|---:|---:|---:|
| North only | 290.26 → 226.06 | 292.50 → 233.63 | 1041.67 → 1039.29 | 636.46 |
| South only | 850.40 → 446.40 | 292.50 → 292.47 | 1041.67 → 479.43 | 385.95 |
| Mixed (one of each per row) | 457.53 → 365.16 | 292.50 → 246.88 | 1041.67 → 480.85 | 363.86 |

Loss is the model's patch NLL minus flow log determinant, averaged over patches. Compare a **given anchor across arms**, not the North and South magnitudes: a North patch has 192 target values and a South patch has 256. The mixed objective averages the two image modality losses. First and last training batches contain different objects; the fixed-probe columns are the like-for-like comparison. Mixed got half as many exposures per anchor as each single-anchor arm. These results are one tiny run, not a claim that mixing improves either anchor over its own control.

## Realized load and resources

Counts below are optimizer inputs, excluding diagnostics. Each single-anchor arm replayed the same eight objects twice; mixed saw eight distinct North and eight distinct South objects once. This deliberately balanced experiment is **not** a measurement of the recipe's North:South **1:2 draw weights** or its echo policy. A draw weight is not an object or token fraction.

| Arm | Object exposures N / S | Nonpad tokens N / S | Loss-bearing patches N / S | Target values N / S |
|---|---:|---:|---:|---:|
| North only | 16 / 0 | 2352 / 0 | 2304 / 0 | 442368 / 0 |
| South only | 0 / 16 | 0 / 2352 | 0 / 2304 | 0 / 589824 |
| Mixed | 8 / 8 | 1176 / 1176 | 1152 / 1152 | 221184 / 294912 |

North, South and pooled **p95 / p99 / max = 147 / 147 / 147** positions per image-only object (144 patches, BOS and two delimiters). Every arm packed 2352/2560 positions, **91.875%** utilization. Mixed was 50% South by objects, nonpad tokens and patches, but 57.14% South by target values. `observe_micro_batch`/`drain_step` counted actual packed inputs and masks; step logs include source-qualified IDs and individual lengths. Partner payloads and scalars were not projected: no match, missed-match, scalar-coverage, or full-corpus length rate can be inferred here.

| Arm | Local read+decode+sequence, both pools (s) | Eight timed updates (s) | Objects/s | Nonpad tokens/s | Peak process RSS (MiB) |
|---|---:|---:|---:|---:|---:|
| North only | 0.205 | 0.123 | 130 | 19118 | 1051 |
| South only | 0.211 | 0.137 | 117 | 17217 | 1052 |
| Mixed | 0.197 | 0.159 | 100 | 14765 | 1060 |

Local AMD EPYC 9455, two PyTorch threads, Python 3.13.15, torch 2.12.1, transformers 5.13.0, pyarrow 26.0.1. No GPU, DataLoader subprocess or Dask worker (worker RSS: not applicable). RSS is the Linux process high-water mark, including both score pools. Update timings include packing cached sequences, instrumentation, model forward/backward and optimizer, not staging, initialization or diagnostic passes. One timed window per arm, too short for a throughput benchmark.

## Pinned inputs, transfer and replay

One main-catalog partition per anchor, selected for small file/row-group size before looking at losses. Under `<catalog>/dataset/Norder=4/Dir=0/`, read only the listed row group, projected to `object_id`, `ra`, `dec`, image `band`, `scale`, `flux`, plus South `mask`, `ivar`. Both production decoders and the normal 96×96 center crop were used; North keeps g/r/z in `images`, South keeps g/r/i/z in `legacy_south_images`.

| Anchor (repository @ recipe revision) | File / zero-based row group | Rows fetched / used | Measured HF payload bytes (ranges) | Fetch + stage (s) |
|---|---|---:|---:|---:|
| `UniverseTBD/mmu_ssl_legacysurvey_north@f634744d3c44dd4fde0dee3172d4887c5e3c31c0` | `Npix=945.parquet` / 1 | 40 / 10 | 10,948,543 (2) | 1.026 |
| `hugging-science/mmu_legacysurvey_dr10_south_21@982f321784d8daf2a5f983eb3ec560f31b90d667` | `Npix=2069.parquet` / 0 | 10 / 10 | 6,720,731 (3) | 0.701 |

Total **17,669,274 bytes**, fetched once and shared by all arms. Preparation peak RSS 954 MiB. Byte counts are `install_byte_probe` returned payloads grouped by pinned path; they exclude headers, Hub tree discovery and exploratory metadata-only reads. No remote reads during optimization. This bounded projection bypasses partner joins, footprint selection, split routing and stochastic anchor selection; it does not measure end-to-end LSDB read amplification or reader RSS.

From `astro/` with the project dependencies installed, choose a **fresh** output directory. The four commands below are the full pilot, not a Nanotron launch:

```bash
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONPATH="$PWD/src"
export HF_HOME="$PWD/tmp/dual-pilot-hf"
out=tmp/pilot58-replay
mkdir -p "$out"
set -o pipefail
timeout 90s python scripts/dual_anchor_pilot.py prepare "$out" | tee "$out/prepare.log"
for arm in north south mixed; do
  HF_HUB_OFFLINE=1 timeout 75s python scripts/dual_anchor_pilot.py "$arm" "$out" | tee "$out/$arm.log" || break
done
```

The measured run's ignored artifacts are in `astro/tmp/pilot58/`: two projected Parquets, `prepare/bytes.rank0.main.jsonl`, `{north,south,mixed}/steps.rank0.jsonl`, and per-arm stdout. No trained weights, images or intermediate tensors are published.

## End-to-end LSDB follow-up: no completed draw

A separate CPU probe exercised the active 70M loader with the same pins, joins, split and seed, but only one partition per draw and one in-process reader. It planned both paths (308 North and 2,751 South aligned pixels), then hit its **1 GiB encoded HTTP response-body limit** during the first North draw. Catalog startup had already consumed about 453 MiB of that budget. The probe ended after 22.1 s (peak process RSS 2.9 GiB), with no decoded rows or packed batches. These are startup and failure measurements, not throughput or realized mix. No optimizer ran.

### Two-cell staging attempt

Attempted to stage full decoder inputs from the active `configs/nanotron/astropt3-70m.yaml` pins for two deliberately chosen NESTED order-10 cells: **4807411** in the audited overlap field and **6978115** around the audited South four-partner example. This is not random sky. Staging selected rows by `_healpix_29`, without matching or decoding. The CPU process used two-core affinity, one BLAS thread, a 12 GiB virtual-address ceiling and a 145 s timeout per staging invocation. No Dask worker was started.

| Cached main-table rows, not stream outputs | Overlap cell | South cell |
|---|---:|---:|
| North | 6 | no North leaf |
| South | 22 | 38 |
| SDSS | 5 | 2 |

The reader first encountered missing Parquet index statistics (`AttributeError: 'NoneType' object has no attribute 'max'`); reading just `_healpix_29` for those groups fixed that staging error. The next guard stopped before transferring the DESI payload: the complete `spectrum` struct in group 0 of `Norder=6/Dir=20000/Npix=27258.parquet` was part of a **301,141,118-byte** compressed projection, exceeding the **256 MiB** group cap. This is a metadata estimate of a **rejected** read, not bytes transferred. Traceback from `tmp/stream58/stage2.log`:

```text
stage desi 6978115 0 5137 301141118
Traceback (most recent call last):
  File ".../astro/tmp/stream58/stage.py", line 38, in <module>
    print('stage',name,cell,j,rg.num_rows,nbytes,flush=True); assert nbytes < 256*2**20 and rg.total_byte_size < 1024*2**20
AssertionError
```

The narrower `spectrum.{flux,lambda,mask}` retry did not execute (execution classifier timed out). Staging logs, returned-range telemetry (`tmp/stream58/fetch/`) and the completed pinned row projections remain local for resumption. **No mixed cycle or per-path control ran:** decoded matches, realized proportions, lengths, stream bytes, throughput and worker RSS are unmeasured. Neither the earlier image-only training metrics nor the positional audit fills that gap.
