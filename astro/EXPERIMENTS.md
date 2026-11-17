# AstroPTv3 lab book

The charter, fixed decisions, and experiment log. Current code is described
in `docs/`, decision records are in `docs/adr/`, verification gates are in
`AGENTS.md`, and git history is the change record.

## 0. Charter — MMU Streaming × AstroPT

**Context.** MMUv1.5 (July 2026) makes crossmatched Multimodal Universe
data reachable without staging the full 80TB+ locally — crossmatching
happens on the hub via lsdb:

```python
import lsdb

gz10 = lsdb.open_catalog("hf://datasets/UniverseTBD/mmu_gz10")
sdss = lsdb.open_catalog("hf://datasets/UniverseTBD/mmu_sdss_sdss")

xmatch = gz10.crossmatch(sdss, n_neighbors=1).compute()
```

The remaining gap: streaming crossmatched examples still cannot keep GPUs
busy, so training today requires locally staged or pre-crossmatched data.
Astronomy will soon wrangle far more than the MMU's 80TB — the goal of
this project is **pre-training on remotely hosted data that is
crossmatched on the fly**.

**End goal.** An importable, model-agnostic `mmu_streaming` package so
that anyone can train on crossmatched MMU data at a reasonable pace,
HF-streaming style:

```python
from datasets import IterableDataset
from mmu_streaming import crossmatch
from astropt import AstroPT

data: IterableDataset = crossmatch(["LegacySurvey", "DESI", "HSC-Wide"])
AstroPT().fit(data.shuffle(seed=42))
```

**AstroPTv3's role.** First test-case for the fast crossmatch streaming
interface. V3 builds on HF's SmolLM (SmolLM3 transformer backbone +
nanotron training infrastructure), takes patched astronomical data via
linear projection, and trains through a JetFormer-style autoregressive
reconstruction objective (tokeniser lineage: Sogol's `sogol_branch`). We
train a sweep of models at a fixed scaling ladder to stress the streaming
interface and study how astro-foundation models learn (and grok)
astronomical concepts under a generative objective. **The goal is not the
"best astronomical foundation model"** — it is understanding learning
dynamics. Following EleutherAI's Pythia, we release checkpoints throughout
training, with training configurations and the information needed to
reproduce the training streams.

### Roadmap and status

| # | Roadmap item | Status |
|---|---|---|
| 1 | LSDB outer-join crossmatch ([#1590](https://github.com/astronomy-commons/lsdb/pull/1590)) | In use from LSDB main (commit pinned in `uv.lock`) |
| 2 | LSDB crossmatch stream, `CrossMatchStream` ([#1584](https://github.com/astronomy-commons/lsdb/pull/1584)) | In use from LSDB main (commit pinned in `uv.lock`) |
| 3 | Many-modal (>2 catalog) crossmatching in LSDB | In use: every recipe streams five catalogs through one `CrossMatchStream` with `min_catalogs` |
| 4 | Streaming speedups upstream | Submit-overlap patch validated (+13.6% tokens/s, §9) on `Smith42/lsdb:xmatch-stream-overlap`; held local until upstream chooses the integration path |
| 5 | `mmu_streaming` package (model-agnostic, HF `IterableDataset`) | Not started. The logic it will extract lives in `astropt3.data.nanotron_loader` (catalog registry, column projection, join order, decode) — deliberately kept out of the nanotron fork for exactly this move |
| 6 | AstroPT sweep on Perlmutter (20k shared hours) or other compute; ~7B aspirational | Not started. Ladder configs exist for all eight sizes (Part I); 70M/160M shakeouts (Part I) and the 70M trimodal 10k trial (§11) done |
| 7 | Release: AstroPTv3 + MMU-Streaming — weights, GitHub, paper, blog post | Not started |

Upstream asks filed on #1584: precompute/reuse per-pixel crossmatch graphs,
and move the skip decision off the loader's critical path (§9). Re-pin LSDB to
a release once one carries the multi-catalog API.

# Part I — Fixed decisions and pilot results

## User decisions (fixed)

1. **Training framework: nanotron** (fork of `huggingface/nanotron@smollm3` —
   branch verified to exist, `run_train.py` at root). Parallelism strategy per
   the **Ultra-Scale Playbook**. The **transformers implementation of the model
   is kept as the release/probing artifact** (nanotron→HF conversion, exactly
   how SmolLM3 itself ships).
2. **From scratch** at all sizes; Pythia-style checkpoint schedule.
3. **Minimal special vocab** (64 ids); no natural-language text; no 128k
   lm_head — outputs are per-modality regression decoders.
4. **Data**: Multimodal Universe catalogs streamed live via LSDB, starting
   from `UniverseTBD/mmu_ssl_legacysurvey_north` × `UniverseTBD/mmu_desi_edr_sv3`
   (schemas verified below); new surveys are config-only extensions.
5. **The JetFormer tokeniser is the sole regression head** (flow + GMM head).
6. **uv** manages environments; new deps in `astro/pyproject.toml`; upgrading
   existing venv packages is fine.
7. **GPU work and training runs are allowed on the dev machine**. CPU
   unit/smoke tests stay the fast gate and must be green first; the node is
   shared, so pin a device and check `nvidia-smi` before claiming one.
   Multi-day production runs still belong on the training cluster by
   preference, not by prohibition.

## Verified pilot-data schemas (checked 2026-07-07 via HF datasets-server + MMU builders)

**`UniverseTBD/mmu_ssl_legacysurvey_north`** — 14,174,203 rows, ~4 TB:

- `image` struct: `band=["des-g","des-r","des-z"]`, `flux` float32 **(3, 152, 152)**
  (builder constant `_image_size=152`, `_pixel_scale=0.262`), `psf_fwhm`, `scale`.
- Scalars: `flux_{g,r,z}`, `fiberflux_{g,r,z}`, `psfdepth_{g,r,z}`, `ebv`, `z_spec`.
- `ra` f64, `dec` f64, `object_id` string, `_healpix_29` int64.
- ⚠ Raw flux (not JPG), 152 px — **not divisible by 16**; patch size must change vs AstroPTv1.

**`UniverseTBD/mmu_desi_edr_sv3`** — 1,126,441 rows, ~86 GB:

- `spectrum` struct of length-**7781** sequences: `flux`, `lambda`, `ivar`,
  `lsf_sigma` (float32), `mask` (bool).
- `Z`, `ZERR`, `ZWARN`, `FLUX_*`/`FIBERFLUX_*` photometry, `EBV`, `ra`, `dec`,
  `object_id`, `_healpix_29`.
- Crossmatched pilot corpus is bounded by DESI: expect ~0.5–1M matched pairs;
  the remaining ~13M image-only objects are still usable as single-modality
  sequences (design requirement below).

## Model-size table (named size ≈ TOTAL params; no vocab head)

GQA, NoPE interval 4, seq 4096. Small sizes gain a layer or two to hit nominal
totals. `scripts/count_params.py` prints current totals (flows in their own
column) and asserts ±10% of nominal. The measured column below was taken
without the JetFormer flows.

| Name | layers | hidden | heads | kv | head_dim | inter | total (measured) |
| ------ | -------- | -------- | ------- | ---- | ---------- | ------- | ------------------ |
| 70M | 23 | 512 | 8 | 2 | 64 | 1536 | 70.04M (+0.1%) |
| 160M | 25 | 768 | 12 | 4 | 64 | 2048 | 158.3M (−1.0%) |
| 410M | 27 | 1024 | 16 | 4 | 64 | 4096 | 411.9M (+0.5%) |
| 1B | 22 | 2048 | 16 | 4 | 128 | 5632 | 994.8M (−1.5%) |
| 1.4B | 31 | 2048 | 16 | 4 | 128 | 5632 | 1.401B (−0.7%) |
| 2.8B | 36 | 2048 | 16 | 4 | 128 | 11008 | 2.815B (+0.5%, exact SmolLM3-3B body) |
| 6.9B | 38 | 4096 | 32 | 8 | 128 | 11008 | 6.740B (−1.8%) |
| 12B | 42 | 5120 | 40 | 8 | 128 | 14336 | 11.90B (+0.4%) |

## Parallelism recipe (Ultra-Scale Playbook; H100 80GB, seq 4096, GBS 2M tokens = 512×4096)

| Size | GPUs (grant) | TP | DP | ZeRO | Activation recompute | Notes |
| ------ | -------------- | ---- | ---- | ------ | ---------------------- | ------- |
| 70M–410M | 32 | 1 | 32 | 1 | none | max micro-batch that fits |
| 1B–1.4B | 64 | 1 | 64 | 1 | selective | |
| 2.8B | 64 | 2 | 32 | 1 | selective | |
| 6.9B | 128 | 4 | 32 | 1 | selective/full | |
| 12B | 256 | 8 | 32 | 1 | full | TP inside NVLink node; **no PP** (24GB bf16 weights /8 + ZeRO-1 optim shards fit comfortably) |

Playbook practices baked into the plan: TP never crosses the node boundary;
prefer ZeRO-1 + activation recomputation before adding PP; benchmark
micro-batch size and grad-accum to hit GBS; profile the first ~100 steps
(torch profiler) before committing node-hours; track tokens/s/GPU + MFU
(target ~30–45% depending on size); FA2 varlen doc masking via
position_ids; bf16 + fused AdamW (β 0.9/0.95, wd 0.1 on ≥2D params), grad clip
1.0, cosine to 0.1× peak, warmup min(2000 steps, 1%).
Peak LR by size (Pythia): 1e-3 / 6e-4 / 3e-4 / 3e-4 / 2e-4 / 1.6e-4 / 1.2e-4 / 1.2e-4.
Multi-epoch training is accepted (per AstroPTv1 findings); the token budget per
size is set in the YAML. Playbook:
<https://huggingface.co/spaces/nanotron/ultrascale-playbook>.

## Pilot results (2026-07)

Measured on an earlier data path (local Legacy × DESI shards); the numbers
characterise the model and trainer, not the current live stream.

- HF↔nanotron parity is bf16-tolerance, not bitwise: nanotron applies RoPE at
  absolute row positions in the packed row while HF restarts per object. RoPE
  is relative, so attention agrees; the weight-conversion round trip is bitwise.
- Kill/resume at step 137: the resumed step-138 loss matched the killed run's
  own value at log precision and tracked an independent uninterrupted run to
  0.7% mean over steps 138–200. Compare a resumed run against its own
  trajectory: two independent runs drift a few % by step 137 (nondeterministic
  backward).
- Upstream nanotron's DDP + fp32-accum + ZeRO-1 comm hook routed to a dead
  reduce-scatter branch, so every ZeRO-1 DDP run crashed at step 1; the fork
  uses the all-reduce path.
- 70M, 100-step DP=2 dry run: loss 0.459→0.259, ~240k tok/s total,
  123 model TFLOPs/GPU (~39% MFU), peak 31.7 GiB/80.
- 70M and 160M 20k-step shakeouts: 70M val loss plateaued at 0.203 from
  ~step 18k with redshift probe R² 0.28–0.32; 160M final lm_loss 0.197 at
  132 TFLOPs/GPU. Both mixes carried almost no spectra, so these are
  effectively image-only numbers.

# Part II — Data-loader throughput experiments (2026-09-02 → present)

Section numbers are stable; code comments cite them.

## Context

`astropt3-70m-jetformer-crossmatch.yaml` (DP=2, 8 loading workers/rank,
streaming `mmu_desi_edr_sv3 x mmu_ssl_legacysurvey_north` live via
`nanotron_loader.py`'s direct-lsdb path) was observed running with the GPU
mostly idle. This log covers the investigation into why, what was tried,
what was measured on real runs, and what's still open. All "real run"
numbers below come from `scripts/bench_report.py` reading the
`ASTROPT3_TELEMETRY_DIR` output of short verification runs on spare GPUs
(2,3), using a scratch checkpoint path and (mostly) `WANDB_MODE=disabled` so
they don't collide with production training. Two runs were logged to wandb
(`astropt3-loader-verify` project) for visual inspection.

## 1. Where the stall was coming from

`cProfile` against the real `InfiniteStream`/`PackedMicroBatches` path
(`profile_stream.py`, ad hoc) showed two distinct causes:

- **Legacy-only stream**: cold start (catalog open + first partition fetch)
  ≈ 40s, of which 34.6s was pure `_ssl._SSLSocket.read` — genuine network
  wait. Steady-state decode/pack of an already-fetched partition was
  ~30ms/micro-batch — negligible.
- **Crossmatch stream**: steady-state was 0.47s/micro-batch, ~89% of it
  inside `_as_mapping`'s `DataFrame.to_dict()` fallback. Each DESI
  spectrum arrives as a 7781-row × 5-column per-record `DataFrame`;
  `to_dict(orient="list")` boxes every one of those ~39k cells
  individually via pandas' `maybe_box_native`. This was a real, fixable
  CPU bug, not network cost.

## 2. Fix: fast-path the spectrum DataFrame, then migrate to `map_rows`

- First fix: `_decode_spectrum` reads columns directly via `.to_numpy()`
  when handed a `DataFrame`, skipping `to_dict()` entirely.
  **0.47s → 0.02s/micro-batch.**
- Investigated `nested_pandas.NestedFrame.map_rows` (the library's own
  purpose-built row-wise accessor for nested/struct columns — delivers
  nested sub-columns as numpy arrays straight from Arrow storage, never
  materializing a per-row `DataFrame` at all). Benchmarked in isolation:
  **19x faster** than the already-fixed `.iterrows()` path for spectrum
  extraction alone (0.241s → 0.013s over 1017 rows), byte-identical output.
- Migrated `_open_records`'s full decode loop to `map_rows`
  (`_row_from_map_rows` reassembles the dotted-key output back into the
  nested-mapping shape `decode_legacy_row`/`decode_crossmatch_row` already
  expect, so their public contracts didn't change). Offline test fixtures
  were upgraded from plain `pandas.DataFrame` to real
  `nested_pandas.NestedFrame`s (`legacy_fixture.nested_frame`, via
  `join_nested`) so the CPU suite exercises the same code path as
  production — the old fake fixtures didn't support `map_rows` at all,
  which the test suite alone wouldn't have caught.
- **Net: crossmatch decode 0.47s → 0.01s/micro-batch (~47x)**, legacy-only
  2.6x faster. Decode dropped out of the profile entirely; packing
  (`torch.arcsinh`, `spectral_normalize`) became the dominant CPU cost at
  ~10-20ms/micro-batch.

## 3. Fetch overlap: does prefetching help?

`lsdb.streams.catalog_streams.InfiniteStream` only prefetches in the
background when given a real `dask.distributed.Client` — with the
`client=None` default it was using, `submit_next_partitions` calls
`.compute()` synchronously, so partition N+1's fetch fully blocks the call
that returns partition N. Zero overlap with anything.

Added a lightweight per-worker `dask.distributed.Client` backed by
`LocalCluster(processes=False, ...)` — an in-process scheduler + thread
pool, not a distributed cluster (~30ms startup, ~1.5MiB RSS, confirmed via
direct measurement). Verified the mechanism works correctly in isolation
(non-blocking submit, genuine background progress on a `Future` checked
mid-flight).

**Result on a real run: no measurable improvement.** `stall_share` stayed
flat at ~85% before and after. Root cause: a worker only stays
`DataLoader`'s `prefetch_factor` micro-batches ahead of training-step
consumption before it blocks — a depth-1 partition lookahead barely dents
a 10-50s fetch, because the wall-clock window it gets to run in isn't "however
long draining a partition takes," it's bounded by how fast the main loop
is actually consuming.

## 4. `partitions_per_chunk` tuning (pre-outer-crossmatch)

Widening the window via `InfiniteStream(partitions_per_chunk=N)` — fetch N
partitions per draw instead of 1, giving the background prefetch more time
to finish before the worker needs it. Real-run sweep (same config, DP=2,
8 workers/rank):

| chunk | stall_share | p95 step | p99 step | rows/s | MFU | RAM (box-wide) |
|---|---|---|---|---|---|---|
| 1 | ~85% | ~13s | ~51s | ~1,000 | 2.6% | not measured |
| 4 | 65.4% | 0.46s | 24.0s | 2,044 | 5.5% | ~161GB |
| **8** | **45.5%** | **0.45s** | **6.4s** | **3,324** | **8.6%** | ~237GB |
| 16 | 61.7% | 0.45s | 6.8s | 2,312 | 6.0% | ~595GB |

`chunk=8` was a real optimum, not a plateau — 16 regressed on every metric
(worse stall, worse throughput) while RAM more than doubled, consistent
with 16 workers × 2 chunks-in-flight × 16 partitions pushing into genuine
memory/scheduling pressure.

## 5. Wire-efficiency investigation

With the network confirmed as the dominant remaining cost (measured ~815
MiB/s aggregate at chunk=8, ≈65% of the box's 10Gbps NIC — not clearly
link-limited, but not far off either), the next question was whether the
bytes being pulled were actually useful.

- **Byte breakdown at chunk=8**: 66.8% of wire bytes were Legacy image
  data.
- **Measured match rate** (live sample, 18,355 DESI rows across 4
  partitions): only **10.0%** have a Legacy image match within 1
  arcsec; **90.0%** are spectrum-only.
- **Read `lsdb`'s crossmatch source** (`crossmatch_catalog_data.py`,
  `kdtree_match.py`): the KDTree match algorithm reads the *full* right
  (Legacy) partition — including the large `image` struct column — for
  every candidate row in an overlapping pixel pair, *before* running the
  distance filter. The ~90% of rows that don't match within radius get
  their image bytes fetched and then silently dropped by
  `AbstractCrossmatchAlgorithm._create_crossmatch_df`.
- **Net finding**: roughly **60% of total wire bytes** (66.8% × ~90%)
  were being spent downloading Legacy images that get discarded.
- **Tried and rejected**: dotted sub-column projection
  (`columns=["spectrum.flux", "spectrum.lambda", "spectrum.mask"]`
  instead of the whole `spectrum` struct) to drop the unused `ivar`/
  `lsf_sigma` fields. Confirmed by direct measurement this does **not**
  reduce wire bytes — HATS/parquet reads the whole nested column's
  row-group regardless of which sub-fields are kept afterward. In-memory
  schema pruning only, no wire savings. Negative result, not pursued
  further.
- **Not pursued**: `MFU_busy` (compute efficiency while *not* stalled) sat
  at ~16-18% across every run measured, against a 989 TFLOP/s peak — a
  separate, wire-independent lever (kernel/model-shape efficiency, not
  data pipeline). Flash attention, fused RMSNorm, fused rotary embeddings
  and QKV packing are already on in this config; no GPU-level profiling
  (torch profiler / nsys) was done this session to look further.

## 6. `OuterKdTreeCrossmatch`: recovering the wasted 60% for free

Since the unmatched Legacy image bytes are already being read into memory
before being discarded, the fix is to stop discarding them rather than to
fetch less — zero extra network cost either way.

`outer_crossmatch.py`'s `OuterKdTreeCrossmatch` subclasses `KdTreeCrossmatch`
and extends `how="left"` row assembly to also emit unmatched right-side
(Legacy) rows as image-only records (NA spectrum/DESI columns), instead of
silently dropping them. Two real bugs were caught by testing against live
data rather than shipping on inspection alone:

- **Margin-cache double-counting**: the right catalog's margin cache
  (candidate rows borrowed from a neighboring partition, included only to
  catch boundary-crossing matches) had to be excluded via spatial-index
  filtering (`healpix_to_spatial_index` bounds check against the row's own
  native pixel) — otherwise a margin row gets emitted here *and* again
  when its home partition is processed.
- **`pandas.NA` vs `None`**: a null `object_id` (pyarrow string) scalar
  comes through as `pandas.NA`, not Python `None`, both via raw pandas
  access and via `map_rows`. The original `decode_crossmatch_row` used
  `is None`, which would have silently decoded these rows' `object_id` as
  the literal string `"<NA>"` rather than falling back to
  `object_id_legacy` or raising. Fixed by switching to `pd.isna()` (a
  strict superset of `is None`, so plain-`None` test fixtures are
  unaffected).

**Verified on real data**: 6,703 rows split exactly into 1,403 matched +
4,333 spectrum-only + 967 image-only — zero loss, zero duplication, zero
rows with neither modality. `ObjectSequencer.build()` succeeds on all
sampled image-only objects.

Scope is deliberately narrow: this only recovers Legacy rows *within pixel
pairs the crossmatch already visits* (near some DESI pointing). A Legacy
partition with no DESI coverage nearby is never fetched by this join at
all — that's the majority of unmatched Legacy imagery, and recovering it
would need a genuinely separate, full-wire-cost stream (not built).

## 7. Re-tuning `partitions_per_chunk` after the outer join

`OuterKdTreeCrossmatch` recovers real extra data, but each recovered image
row is real extra weight (~277KB/row) buffered per chunk. This pushed
`chunk=8` into the same RAM-pressure regression `chunk=16` hit in section
4. Re-tuned with two equally-mature real-run readings (~552-575s each):

| | chunk=8 + outer-join | chunk=4 + outer-join |
|---|---|---|
| stall_share | 49.8% | **32.8%** |
| rows/s | 2,324 | **2,997** |
| MFU | 8.90% | **12.13%** |
| E_values/MiB | 66,218 | **89,458** |
| p99 step | 7.75s | **3.50s** |
| slow steps (>60s) | 1 | **0** |
| RAM (box-wide) | 529GB | **349GB** |

`chunk=4` won on *every* metric, not just RAM — confirms this wasn't a
tradeoff, `chunk=8` was genuinely past the regression point once the outer
join's extra data is counted. `chunk=2` was not tested (stopped once the
win was decisive); worth checking if further tuning is ever revisited.

## 8. Settings at the end of the investigation

Settings at the end of this investigation: `map_rows`-based decode, per-worker
lightweight `dask.distributed` prefetch client, `partitions_per_chunk=4`,
`OuterKdTreeCrossmatch` for the crossmatch stream. Confirmed with a
wandb-logged run of this exact configuration
([`astropt3-loader-verify/runs/jsoi53bq`](https://wandb.ai/smith42/astropt3-loader-verify/runs/jsoi53bq),
separate project, scratch checkpoint path — not production), read at a
comparable maturity (540s) to the section-7 numbers:

| | section 7 reading | wandb-confirmed reading |
|---|---|---|
| stall_share | 32.8% | 35.4% |
| rows/s | 2,997 | 2,881 |
| MFU | 12.13% | 11.64% |
| E_values/MiB | 89,458 | 87,992 |

Consistent within normal run-to-run variance — the section-7 numbers hold.

**End-to-end from the original ~85% stall_share baseline (section 1) to
here: stall_share 85% → 35%, rows/s ~1,000 → ~2,900 (~2.9x), MFU 2.6% →
11.6% (~4.5x).**

## 9. v3 adoption bench: CrossMatchStream vs manual crossmatch (2026-09-24)

Setup: paired fresh runs, 2,000 steps/arm, DP=2 per arm on 2×H200 each
(both arms live simultaneously — shared link by design). Same working tree
except the stream: the v2 arm ran from a detached worktree of HEAD
(`../AstroPTv3-v2bench`, same retry fix applied) shadowed via PYTHONPATH,
because the training venv's editable `astropt3` points at the main tree.
Steady state = steps 21–2000. The v3 arms are
`lsdb.streams.CrossMatchStream` (astronomy-commons/lsdb#1584) at two
`count_fraction_threshold` settings.

| arm | link conditions | non-pad tokens/s | rows/s | img values/s | spec values/s | img token share |
|---|---|---|---|---|---|---|
| v2 r1 | paired | 100,644 | 2,498 | 26.6M | 14.0M | 71.7% |
| v3 thr=0.5 r1 | paired | 77,006 | 2,342 | 16.4M | 15.6M | 58.4% |
| v2 r2 | ~24% contended | 117,769 | 2,923 | 31.1M | 16.4M | 71.7% |
| v3 thr=0.0 | solo | 110,584 | 2,715 | 29.5M | 15.1M | 72.2% |

Findings:

1. **Mechanism tax ≥ 6% (floor; contention-corrected ~10%).** Even with
   the skip off, CrossMatchStream is slower than streaming a pre-
   crossmatched catalog: every drawn pixel pays `PixelSearch` + a fresh
   crossmatch graph build + the density estimate synchronously on the
   loader's critical path, where v2 builds the crossmatch graph once. The
   record mix is identical (72.2% vs 71.7% image share; per-modality value
   rates track) — pure overhead, semantics verified including
   OuterKdTreeCrossmatch's unmatched-right recovery.
2. **The skip at 0.5 is strictly harmful here.** Legacy bytes −31%, but
   tokens/s −~20% more and image share 72% → 58%: the loader is
   latency-bound, not bandwidth-bound, skipped pixels still pay the
   per-pixel fixed costs, and the skip discards the image-heavy rows.
   (0.1 previously showed no effect either direction.) Threshold set to
   0.0 on adoption.
3. **Repeatability**: v2-r2 rows/s 2,923 ≈ §8's solo 2,881–2,997.
   Pairing costs ~15% (v2 paired 100.6k vs r2 117.8k tokens/s).

**Decision:** adopt v3 at thr=0.0 — for the multi-catalog capability and
ADR 0015 upstream alignment — accepting the measured tax at 70M scale,
where 0.36s steps leave the GPU maximally exposed to loader latency; at
larger model sizes step time grows and the tax should shrink. Two asks
filed upstream on #1584: precompute/reuse per-pixel crossmatch graphs, and
move the skip decision off the critical path (which would make the skip a
pure win whenever bytes bind).

Reliability note from the same bench: a transient `HfHubHTTPError` can be
swallowed by fsspec's parquet reader (cat_ranges results are gathered with
`return_exceptions=True` but consumed without exception checks) and
surfaces as `TypeError: can't concat X to bytes` (block merge) or
`'X' object is not subscriptable` (footer slicing). The first killed the
original v3 run at step 46,767; the second killed a bench arm at step 302.
`_retryable` now classifies the whole family; regression tests cover both
shapes plus a live-fsspec message canary. Upstream issue drafted against
fsspec (root fix: check `is_exception` in `_transfer_ranges`).


## 11. Trimodal HSC trials: 1k and 10k steps (2026-09-28)

Third-catalog validation for roadmap item 3: `hsc_images` (HSC PDR3 Wide,
matched-only left join on the DESI × Legacy `CrossMatchStream`) trained
end-to-end on 70M with wandb online. Fresh cosine schedules; DP=2 × 8
loading workers; checkpoints every 1000 steps.

- **1k trial** (https://wandb.ai/smith42/astropt3/runs/u53mgt1i): 1000/1000
  steps, zero Arrow errors after the `Array2DExtensionType` unregister fix.
- **10k trial** (https://wandb.ai/smith42/astropt3/runs/qmnq2gd1):
  10,000/10,000 steps, checkpoint 10000, zero errors. Median losses
  early→late: overall 381→188, HSC 1110→886, Legacy images 99.6→−88.7,
  spectra −24.5→−216. HSC token share 5.87% (present in 86% of
  rank-steps; matched-only join, ~8% as many HSC spans as Legacy spans).
  Throughput 66.6k non-pad tokens/s vs 110.6k two-catalog upstream v3 —
  the gap is p99 HSC partition stalls (257 steps >5s; §10 shows leaf
  projection cannot fix them; the fix is upstream leaf-level range reads).

## 12. JetFormer noise and convergence diagnosis (2026-07-14)

Measured on the step-20000 checkpoint of the low-LR 70M run
(`astropt3-70m-jetformer-lowlr`, wandb `y3oak0l0`, DeltaAI GH200), using
the HF-converted checkpoint + real val shards. Two independent problems:

1. **Optimisation drift.** `grad_norm` sat at 1000–1700 against
   `clip_grad=1.0` for the entire second half; `images_loss` bottomed at
   −0.76 (step 16k) then climbed back to +1.0 by 20k — the sampled
   checkpoint was past the best point. Negative loss is a tell: the flow
   can push `NLL_GMM(z) − logdet` below zero by inflating logdet without
   improving samples (a plausible gradient-blow-up source; the
   OLMo-style grad-norm growth persisted at 10× lower LR, so it is not an
   LR-knob issue).
2. **Uncalibrated noise generation at any temperature.** In model output
   space (noise isolated by a 3×3 high-pass): per-channel noise std ~5×
   too small at T=0.2 and ~15–20× too large at T=1.0 — no temperature
   matches the real amplitude; T=1 noise is also spatially anti-correlated
   where real noise is PSF-correlated. Teacher-forced `log_sigma` is NOT
   saturated (0% at the ceiling) and the mixture is not collapsed — the
   shallow 4-step flow reaches low z-space NLL while its inverse amplifies
   and whitens noise; exact z-likelihood does not constrain sample quality.

The transformer body was fine — teacher-forced GMM-mode reconstructions
looked reasonable; the sampling/variance path was the broken part.
Priorities recorded at the time: instrument the loss per term (GMM-NLL vs
logdet separately) and regularise `|logdet|`; stop memorising sensor noise
(whiten/down-weight by the known noise model, or denoise the target and
re-add calibrated noise at sample time); do not expect temperature to
produce realistic noise.

## 13. Standard throughput benchmark validation (issue #45)

`scripts/benchmark_throughput.py` bounds the real nanotron training loop
without changing the YAML horizon or LR/noise schedules; usage and clock
definitions are in [`docs/training.md` §8](docs/training.md#8-standard-throughput-benchmark).

Functional checks (2026-10-20) ran on idle H200 NVL GPUs of a shared host, 2
warmup + 5 measured steps, with the tiny YAML's 50-step horizon unchanged. DP1
used that YAML as is (no loader workers); DP2 and TP2 used copies with that
topology, one loader worker per rank and console logging every third step.
All streamed live LegacySurvey data; cache state unknown.

| topology | step records | measured non-pad tokens | returned payload bytes | result |
|---|---:|---:|---:|---|
| DP1 / TP1 | 7 | 15,300 | 8,708,962,755 | complete |
| DP2 / TP1 | 7 per rank | 30,600 | 13,745,150,931 | complete |
| DP1 / TP2 | 7 per rank | 15,300 | 17,417,925,510 | complete |

TP2 counts tokens once but both replicas' downloads (2 × 8,708,962,755 B).
Warmup was ~44–47 s, almost all of it the first batch, while the measured
windows were 0.1–0.7 s: **these are execution/accounting checks, not
throughput or scaling measurements** (the MFU estimates, 0.01–0.07% on a
538k-parameter model, mean nothing). No checkpoints were written, and
launching into an existing output directory failed before any write.
Payload counts only what returned before training ended on each rank: one
earlier DP1 check ended with a 178 MB fetch in flight and recorded 57 fetches
instead of 58, in both the in-run and the regenerated report.

Environment: Python 3.13.15, torch 2.8.0+cu128, flash-attn 2.8.3, driver
615.71.09, LSDB `5102a8a9`; `rankN.json` records the rest. Not validated live:
multi-node, mixed DP+TP, remote checkpoint staging.

**Standard 70M runs** (10 warmup + 200 measured steps, GPUs 2-3; run dirs
`astroPTv3_telemetry/benchmark-70m-*-20261020T*`):

| config | GPUs | useful tokens/s (/GPU) | stall share | images / spectra | MFU est. |
|---|---:|---:|---:|---|---:|
| `70m-jetformer-crossmatch-v3`, pre-cut runner | 2 | 221,338 (110,669) | 38.1% | 74.5 / 25.5 | n/a |
| `70m-jetformer-crossmatch-v3`, final runner | 2 | 253,793 (126,897) | 28.8% | 74.5 / 25.5 | 15.2% |
| `astropt3-70m-dp1` (crossmatch) | 1 | 145,067 | 20.6% | 75.0 / 25.0 | 17.4% |

The two v3 runs differ only in loader stalls: non-stalled window time was
71.7 s vs 71.9 s, so the simplified runner measures the same work. The DP1
run attributed 19.1 of 62.9 GiB returned payload to DESI, the first live
check of DESI attribution.

## 14. Gaussian factor-out bounded pilot (issue #51, 2026-11-17)

**Question:** does assigning the tail of each flowed patch to N(0, I), with
only `d_ar` coordinates embedded/GMM-predicted, help early learning or cost?
**Result: mixed, not evidence to change the full-width default.**

Used the saved `isqm4s6c` five-catalog `effective.yaml` from
`astroPTv3_telemetry/astropt3-70m-gwh-dequant-echo4-20261116T230151Z/`.
A bounded live sample supplied 64 unique training objects and 32 disjoint
validation IDs, shared across all arms. Each arm trained from scratch for
100 steps (4 objects/batch, seq 4096), seed 0. Matching-shaped parameters
started identically; narrower encoders/heads were freshly initialized.
Only patch `d_ar` differs between arms; scalar widths and full flow widths
are unchanged. The fractions round down (`max(1, int(D * fraction))`).

**Pilot deviations from production:** HF FP32/SDPA rather than Nanotron
bf16; constant peak AdamW LR 0.001 rather than the production LR schedule;
fixed small data and span order rather than live streaming/echo4. Betas
0.9/0.95, epsilon 1e-8, weight decay 0.1 and global clip 1 are shared. Noise
uses the original 10,000-step horizon. This is not a continuation or a
10,000-step replication. The held-out objects share the sampled sky window;
this is not a spatially held-out generalization test.

One initially idle H200 NVL, pinned as GPU 2; torch 2.8.0+cu128,
transformers 5.13.0, LSDB 0.11.3.dev3+g8aa1e63f9. Timings synchronize CUDA
and exclude 5 warmup steps; fixed-batch host-to-device transfers are included,
network collection is excluded. Single run per arm, no timing confidence
interval. Working branch `feat/51-gaussian-factor-out`, based on Astro
`54b7093` and Nanotron `48fb180`.

| AR fraction (HSC d_ar) | Parameters incl. flows | Mean step (s) | Peak allocated GiB | Total NLL | All-image source NLL |
|---|---:|---:|---:|---:|---:|
| 1.0 (320) | 113,407,488 | 0.278 | 6.873 | −152.914 | 39.021 |
| 0.4 (128) | 108,444,672 | 0.294 | 6.799 | −137.224 | 9.981 |
| 0.2 (64) | 106,795,008 | 0.293 | 6.771 | −129.283 | 3.028 |

All likelihoods are **nats/token, lower is better**, summed over the complete
patch (AR + prior − logdet), then token-weighted. Total covers 4,889 tokens;
it is not the family-balanced training objective. Source/sky masks are
fixed per object across arms: invert the image arcsinh normalization, take
channel-wise median and 1.4826×MAD, then mark a patch as source if any pixel
is more than 4 sigma from its channel median (absolute deviation). This is
an approximate segmentation, not the issue author's original mask.

| Image group / held-out tokens | Full width | 0.4 fraction | 0.2 fraction |
|---|---:|---:|---:|
| HSC all / 1,008 | 150.885 | 148.328 | 147.846 |
| HSC sky / 871 | 144.090 | 138.867 | 140.728 |
| HSC source / 137 | 194.084 | 208.480 | 193.096 |
| Legacy all / 2,304 | 62.175 | 82.284 | 84.739 |
| Legacy source / 303 | 96.635 | 117.008 | 111.865 |
| Galaxies RGB all / 1,024 | −1063.928 | −1058.535 | −1012.620 |
| Galaxies RGB source / 150 | −218.983 | −387.510 | −390.417 |

The source aggregate improves mainly through galaxies RGB; it does not imply
uniform source improvement. HSC has only 7 held-out cutouts, Legacy 16 and
RGB 4. DESI and SDSS total NLL also worsen in both split arms. The 0.4/0.2
arms save 4.4%/5.8% of parameters and 1.1%/1.5% of peak allocated memory,
but are 6.1%/5.4% slower in this run. Scalar NLLs are mixed and have at most
four validation targets per scalar. **No linear probe, scalar-head accuracy,
multiple-seed study, or full-training representation claim is supported.**
LSDB warned that a right catalog lacks a margin cache; all arms share the
same potentially incomplete boundary matches.

Local artifacts: `astro/tmp/issue51-real100/{recipe.yaml,hf_config.json,data.pt,result.json}`.
The result includes AR/prior/logdet components, training curves, exact object
IDs and dimensions. `data.pt` contains the exact sequences and train/eval
batches (SHA256 `704e979a20a4894bf0a9ad137f5ac9a09c5dfb379edc7aae027ea678d37c8b0f`).
The output directory is ignored scratch, not a committed corpus. Command:

```bash
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv-astropt3/bin/python astro/scripts/benchmark_factor_out.py \
  --config-file /h/msmith/projects/astropt/astroPTv3_telemetry/astropt3-70m-gwh-dequant-echo4-20261116T230151Z/effective.yaml \
  --output astro/tmp/issue51-real100 --steps 100
```

**Next scientific gate:** matched multi-seed, full-horizon Nanotron runs with
held-out source NLL, scalar-free linear probes and scalar-head accuracy on
converted checkpoints. Keep factor-out opt-in until then.

## Open items / not pursued

- **`chunk=2`**: not tested after the section-7 re-tune; the win at 4 was
  decisive enough to stop there.
- **Compute-side MFU (`MFU_busy` ~16-18%)**: a real, wire-independent
  lever, not investigated this session. Needs GPU-level profiling
  (torch profiler / nsys), not data-pipeline telemetry.
- **Legacy imagery outside DESI's footprint**: the majority of unmatched
  Legacy images (partitions with no nearby DESI coverage at all) are still
  never fetched by this stream. Recovering them needs a genuinely separate
  stream at full wire cost — a real design decision (more corpus coverage
  vs. more bandwidth spent), not attempted here.
- **lsdb docstring accuracy**: `InfiniteStream`/`CatalogStream`'s docstring
  claims background prefetching happens generically ("derived from
  `client` object"), which overstates the `client=None` case (fully
  synchronous). Worth a small upstream issue/PR; not filed.
