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

## Source-and-band packing probe (PR #65)

The first CPU probe used the 70M recipe's four partitions per draw and hit
its 4 GiB download cap before decoding a record. A follow-up at `f9d0d3c`
completed one live draw per anchor path, using one partition per draw, seed
42, the training split, and the recipe's 16 × 4096 packer. The North draw
was capped at 512 of 819 joined rows; all 261 South rows were used.

| Anchor path (order-7 pixel) | Records (partner-only) | Nonpad / slots | Fill | Object length p95 / p99 |
|---|---:|---:|---:|---:|
| Legacy North (39606) | 512 (248) | 396,187 / 458,752 | 86.36% | 1169 / 1460 |
| Legacy South (108789) | 261 (9) | 222,494 / 327,680 | 67.90% | 1315 / 1588 |

Pooled fill was 78.67%, including padding in each draw's final batch.
This is a live spot check, not a weighted production estimate or a
throughput benchmark. Before merging #65, run a short 70M training pilot
with the default four partitions per draw. Report packing fill, realized
anchor mix, useful tokens/sec, and whether loss stays finite and trends down.

# Part III — Model scaling (2026-11)

## 14. 1M vs 70M at 10k steps, and family-loss reweighting (2026-11-17/18)

**Setup.** AstroPTv3 `15478d2` / nanotron `c3dfb40` (clean), one H200 NVL
each, run dirs in `astroPTv3_telemetry/`. Both runs used the seven-catalog
`min_catalogs: 3` stream, seed 42, echo factor 4, seq 4096 × micro-batch
16, a cosine LR schedule (peak 1e-3, 2k warmup, floor 1e-4), `clip_grad` 1.0
and 10k steps (655M tokens, ~164M of them distinct). The only differences
were the body size and `jetformer_flow_hidden`.

| run | body | flow hidden | params: body / flows / heads / total | tok/s | wandb |
|---|---|---|---|---:|---|
| 70M (`astropt3-70m.yaml`) | 512 × 23 | 128 | 69.4M / 43.1M / 11.0M / 125.4M | 63K | `g99ek571` |
| 1M (`astropt3-1m.yaml`, `train_steps` 10000) | 128 × 5 | 32 | 0.94M / 3.7M / 2.8M / 7.9M | 127K | `1u7eaz9d` |

The "1M" is 88% flows and heads; its transformer body is under 1M.

**Result: no scaling at this budget.** Training loss only; no held-out
eval was run. Medians over steps 9500–10000:

| loss | 70M | 1M |
|---|---:|---:|
| `lm_loss` | −0.687 | −0.692 |
| image / spectrum / scalar family | −0.47 / −0.71 / −2.65 | −0.49 / −0.70 / −2.58 |
| Legacy `images` / `galaxies` / `sdss_spectra` | −64.7 / −2960 / −167 | −78.5 / −2740 / −163 |
| `gwh_redshift` / `sdss_Z` / `provabgs_LOG_MSTAR` | −2.42 / −2.36 / −0.81 | −2.51 / −2.60 / −0.89 |

The curves overlap from ~1k steps, and both still fall at the same rate
over 5k→10k (Δ`lm_loss` −0.07 for the 70M, −0.09 for the 1M). Neither run
diverged. In the second half the 70M's median `grad_norm` was ~20 vs 3–7
for the 1M, and both had ~250–400 steps with a norm above 200. The 1M is
only 2× faster, so the loader bounds it.

Token mix per step (70M telemetry, ~58.8K non-pad tokens; 75% of them
echo replays): HSC images 36%, Legacy images 33%, galaxies-with-hats images
10%, DESI spectra 10%, SDSS spectra 0.3%, all scalars together ~3%,
Galaxy10 ~0.1%.

**Family reweighting (70M, stopped early).** The same recipe was rerun
with configurable `family_loss_weights` (default image:spectrum:scalar
1:1:0.1). The code was reverted after these runs.

| run | weights | clip | stopped at | wandb |
|---|---|---:|---:|---|
| equalfam | 1 : 1 : 1 | 1 | 4,119 | `ntrwkvbk` |
| imghalf | 0.5 : 1 : 1 | 1 | 4,247 | `n2z6iui4` |
| equalfam-clip10 | 1 : 1 : 1 | 10 | 2,088 | `2m43ykqf` |
| imghalf-clip10 | 0.5 : 1 : 1 | 10 | 2,201 | `uuepgsj7` |

Medians over steps 3600–4000 (baseline = the 70M above; the baseline's
IQR-based standard error is ~0.01–0.05):

| loss | baseline | equalfam | imghalf |
|---|---:|---:|---:|
| image / spectrum / scalar family | **−0.40 / −0.61 / −1.47** | −0.28 / −0.50 / −1.22 | −0.19 / −0.52 / −1.14 |
| `sdss_Z` / `provabgs_LOG_MSTAR` / `hsc_cmodel_mag` | **−1.21 / −0.50 / −3.58** | −1.16 / −0.35 / −2.94 | −1.05 / −0.33 / −2.97 |
| `grad_norm` | 7.9 | 97 | 89 |

Both reweightings were worse than 1:1:0.1 on every family, including the
scalars they upweighted. At step ~2k (LR peak) imghalf was briefly level
or slightly ahead on spectra and scalars; this reversed by 4k. Raising
`clip_grad` to 10 did not help: at ~1.9k steps the clip-10 runs were
level with or worse than the clip-1 runs, at a 4× rather than 8× clip
ratio. Under AdamW a uniform clip rescale mostly cancels, so clipping was
never a good explanation. More likely, upweighting ~2K spiky scalar
tokens per step lets their gradients dominate the shared body and slows
every family. **1:1:0.1 stays the default.**

**Hypotheses for why 70M ≈ 1M** (untested; most informative first):

1. **The flows, not the body, carry the likelihood.** Both sizes put
   ~6.5M (1M) and ~54M (70M) parameters into per-modality flows and GMM
   heads. The loss is `NLL_GMM(z) − logdet`, and §12 showed the flow can
   lower it by inflating logdet without the body's prediction improving.
   If most of the gain is per-patch marginal density (logdet), body size
   barely matters. *Test:* log GMM-NLL and logdet separately (§12's open
   priority) and see which term separates the two sizes.
2. **The context is one object, and within-object prediction is local.**
   `position_ids` restart per object, so the body never sees more than one
   galaxy (~150–500 tokens). Next-patch prediction in a spiral-ordered
   image may be mostly local smoothness plus noise, which a 1M body
   already handles. *Test:* loss by patch index and by span order for both
   sizes. If the curves coincide everywhere, there is little structure
   left for capacity to exploit.
3. **Irreducible sensor noise sets a floor.** Images are ~80% of the
   tokens, and §12 found the flow fitting per-pixel noise. If both models
   sit near the noise-entropy floor, extra capacity cannot show up in
   training NLL. *Test:* compute the floor implied by `ivar`, or evaluate
   on noise-whitened targets.
4. **The noise curriculum moves both curves together.** The z noise fed to
   the encoders anneals linearly from 0.1 to 0 over the full run, so the
   conditioning gets cleaner every step regardless of model size. The
   late, equal-rate decline of both runs may be the schedule, not
   learning. *Test:* held-out eval of the step-1k…10k checkpoints at fixed
   noise 0.
5. **Most scalar targets are effectively unconditional.** Span order is
   uniform random, so a scalar comes before its image or spectrum about
   half the time and is predicted from the population prior. Only the
   conditional half can reward capacity, and the scalar family is 0.1 of
   the objective. *Test:* split the scalar loss by whether a
   high-information span (image or spectrum) came first.
6. **Budget and optimisation.** 5 tokens per parameter for the 70M, 75%
   echo replays, training loss only, and one LR (1e-3) for both sizes. The
   70M's rising `grad_norm` suggests its LR is on the high side. *Test:* a
   2–3k-step LR sweep for the 70M (3e-4, 1e-4) and a held-out eval, before
   any longer run.

Hypotheses 1, 2 and 4 can be tested on the existing step-1k…10k
checkpoints of both runs without new training.

**Held-out tests of H1, H2 and H4 (2026-11-18).** The step-1k/2k/5k/10k
checkpoints of both runs were converted to HF and scored on one fixed
eval set: 24 micro-batches drawn from the same stream with seed 12345 and
no echo. It holds ~1.29M patch tokens (Legacy 439K, HSC 541K, galaxies
134K, DESI spectra 171K) plus scalars. Hooks on the flows and on
`gmm_nll` split each token's loss into GMM NLL and −logdet; the rebuilt
totals match the model's own per-modality loss within bf16 rounding.
Scripts, eval set and JSON are in
`astroPTv3_checkpoints/hf-eval/scaling-20261118/`. Some eval objects may
have been seen in training, and there are no error bars (one eval set).

Held-out loss at step 10k (nats/dim; it agrees with the training loss):

| | 1M | 70M |
|---|---:|---:|
| Legacy images | **−0.454** | −0.379 |
| HSC images | 0.075 | **0.067** |
| galaxies images | −3.57 | **−3.85** |
| DESI spectra | −0.735 | **−0.750** |

- **H4 rejected.** At every step and for both sizes, scoring at the
  training noise σ instead of σ = 0 changes the loss by ≤0.002 nats/dim.
  The late decline is learning, not the schedule.
- **H1 not decidable from the split.** The two sizes split the same total
  very differently. On Legacy at 10k the 1M has NLL −0.04 and −logdet
  −0.42; the 70M has NLL 1.32 and −logdet −1.70. Its flow stretches z ~4×
  more, and the GMM pays for it in NLL. The split is a free rescaling of
  z, so it cannot say which part does the work. Both terms drift during
  training (70M Legacy NLL 0.61 → 1.64 → 1.32 over 1k/5k/10k), the §12
  logdet-inflation pattern. Testing H1 needs an ablation instead (the
  same flow with the body's input removed).
- **H2 partly supported.** The 70M's HSC gain sits in the first patches
  of each span (−0.053 nats/dim at patch 0, −0.006 over patches 36+,
  which are 108 of the 144 per image). Seeing another image or spectrum
  first improves image loss by only 0.02–0.04 nats/dim in either model.
  But the 70M is better on every galaxies patch and worse on every Legacy
  patch after the first, so "nothing to learn" is not the whole story.
- **Conditional scalars do not reward the 70M.** An image or spectrum
  span earlier in the object helps scalars a lot (DESI `Z` 0.21 → −2.64,
  HSC cmodel mag −1.66 → −8.73 for the 1M), which confirms H5's premise.
  In exactly that conditioned case, though, the 70M is worse: `Z` −2.17
  vs −2.64, HSC cmodel mag −8.19 vs −8.73, photometry −3.29 vs −3.53. It
  is better only on stellar mass (−1.13 vs −1.03) and on unconditioned
  `Z` (−0.85 vs 0.21), which amounts to learning the prior. The order
  split is confounded for spectra, many of which come "first" only
  because their object has no image. **This points at H6 (70M
  optimisation) over task saturation**, so the 70M LR sweep is next.

**70M peak-LR sweep (2026-11-18, stopped early).** The 70M recipe
was rerun at peak LR 3e-4 (floor 3e-5) and otherwise unchanged: the
same 10k-step schedule, warmup and noise curriculum. It ran on one
H200 shared with another job. Loader waits of ~1.3 s/step (vs 0.36 s in
the baseline) halved the throughput. The run was stopped by hand at
step 2,088 (checkpoints through 2,000 kept in
`astroPTv3_telemetry/astropt3-70m-lr3e-4-echoconfig-*`). The planned
1e-4 arm never started, and no held-out eval was run.

Training loss, medians over steps 1701–2001 (all three runs at peak LR):

| loss | 1M (1e-3) | 70M (1e-3) | 70M (3e-4) |
|---|---:|---:|---:|
| image / spectrum / scalar family | −0.30 / **−0.52** / **−1.15** | −0.31 / −0.38 / −0.87 | **−0.33** / −0.49 / −1.13 |
| `sdss_Z` / `gwh_redshift` | **−1.27** / **−1.77** | −0.96 / −1.45 | −1.11 / −1.71 |
| `hsc_cmodel_mag` / `provabgs_LOG_MSTAR` | −3.11 / −0.35 | −2.55 / −0.25 | **−3.34** / **−0.41** |
| `grad_norm` | 3.9 | 7.8 | 5.7 |

- **1e-3 was costing the 70M.** At 3e-4 it beat the 1e-3 run on every
  family by step 2k (spectra +0.12, scalars +0.26), even though its LR
  was 0.3× the baseline's throughout warmup. It already led on scalars
  and spectra by step ~1.2k.
- **The fix made the 70M level with the 1M, not better.** It was ahead on
  images, HSC magnitudes and stellar mass, and behind on spectra and both
  redshifts. Over steps 1254–1554 it had led the 1M on spectra and
  scalars, so 300-step training-loss windows move by ~±0.1 between
  runs. That is too noisy to call a scaling gap at this size.
- **H6 is supported only in part.** Training LR is one cause of the 70M's
  weak result, but on this evidence it does not account for the missing
  scaling. H1 (flows and heads do the work; testing it needs a
  body-ablation) and H2 (most tokens are noise) remain the main
  candidates. Still open: the held-out eval of the step-2000 checkpoint
  and the 1e-4 arm.

## 15. PR #63 recipe: sizes, LR, single-survey controls and cross-modal tests (2026-11-19/23)

**Setup.** AstroPTv3 `b627214` / nanotron `4b18a5b` (PR #63, which contains
#60, plus nanotron #15) unless noted. One H200 per run except 976 (DP=2).
Run dirs are in `astroPTv3_telemetry/`; each has a README with provenance.
Recipes are the 70M/1M/1B YAMLs of that commit: LR 1e-3 (1B 3e-4), 2k
warmup, cosine to 0.1×, seq 4096 × 16 rows, echo 4, sky-scatter pre-flow
noise.

| Slurm | run | W&B |
|---:|---|---|
| 956 / 959 / 962 | 70M / 1M / 1B, full multimodal stream, 10k steps | `85tt2442` / `qdspn9uz` / `io9mjrpm` |
| 964 | 70M pilot: first span unscored, scalar weight 1, 2k→4k | `hy7o4di5` |
| 971 | 1M with pre-flow noise σ0 = measured pixel spread | `wpjtswfq` |
| 976 | 70M, DP=2, 80k steps (`run_train.py`) | `lcq37ho8` |
| 985 | 70M, galaxies-with-hats RGB only | `ccotv2t4` |
| 986 / 990 | 70M / 1B, Legacy South g/r/i/z only | `6vqenbxq` / `zhn2t4nl` |
| 989 | 70M at LR 1e-3 vs 3e-4 on byte-identical replayed batches | — |
| 991 | 70M, Legacy South g/r/i/z + DESI EDR spectra, `min_catalogs: 2` | `7he89t59` |

The single-catalog runs (985/986/990) needed `catalogs: []` to pass
nanotron's config check and `partitions_per_chunk: 1`: four galaxies or
South partitions exceed Arrow's 2 GiB offset limit when an anchor-only
stream shuffles whole chunks.

**Sizes at 10k (956/959/962).** The 1M had lower scalar-family training
loss than the 70M (gap 0.70 at 10k). On small held-out panels the 70M was
ahead or level: redshift point error 0.056 vs 0.078, Legacy South flux NLL
2.70 vs 3.15. The 1B's logged per-task losses average per-micro-batch means
with absent tasks as zero (#68), so its curves cannot be ranked.

**Peak LR (989).** Two DP=1 trainers on the 956 recipe, differing only in
peak/floor LR, consumed the same batches (the control streamed them and
replayed each to the other arm; batch hashes matched for all 10k steps).
Held-out metrics are on 4 Legacy South objects.

| | 1e-3 | 3e-4 |
|---|---:|---:|
| training loss, paired steps where this arm is lower (median gap) | 0–2% | 98–100% (0.05–0.07) |
| held-out r-band NLL, step 4k / 10k (nats/dim) | 0.61 / −0.04 | 0.10 / −0.20 |
| own-vs-wrong g context gain on r, step 4k / 10k | +0.0002 / +0.007 | +0.017 / +0.011 |
| mean cross-band correlation of free samples, step 10k | 0.17 | 0.66 |

**3e-4 is now the 70M default.** This completes the §14 sweep. Related
symptoms of the hot LR: the 70M's Galaxy10 image NLL went from −0.19 to
47 nats/dim between steps 256 and 3k, and swapping only the Galaxy10 flow
moved the failure with it (Galaxy10 was 0.024% of image tokens; it has
since been removed from the recipes). In 976 the galaxies RGB loss jumped
between two levels in bursts on 13–20% of steps until ~20k, with the other
image losses worse during bursts.

**Sample quality.** No implementation fault was found: the flow inverts
exactly, generation from the true tokens reproduces them, and a fresh
1M-config model trained on one image through the HF forward memorises it
in 3k steps, with free-running error equal to teacher-forced error. Poor
samples are under-fitting:
- *RGB is starved in the multimodal mix.* It is ~2% of tokens, sits at ~0
  nats/dim (1.85 above the noise floor), and with 768-dim tokens gets 1/12
  of a 64-dim band's loss weight per dimension. The same 70M recipe on
  galaxies-with-hats only (985) gives seam-free, plausible free samples by
  10k after 2.4M RGB presentations (vs ~47k in 956). Step 128 (31k images)
  looks as bad as 956; step 512 already shows galaxies. Exposure and
  cross-modal interference are not separated.
- *Single-survey South works.* 986 gives clean, centred galaxies. The 1B
  (990, LR 3e-4, 2.3 s/step vs 0.45) is behind it at 10k: median
  `lm_loss` over the last 100 steps 0.50 vs 0.40, behind on every band.
  The 1B has seen 0.06 tokens per parameter and runs at a different LR.
- *Strong pre-flow noise (971)* removed the ~1e5 spectrum spikes (worst
  one-token deviation 29 → 8 arcsinh units) but not the image patch seams
  (edge/within ratio 1.1–1.8 vs ~0.93 in data), and 66–69% of its spectra
  collapse to zero flux at the red end. Not adopted.

**Cross-modal conditioning.** Gain = NLL after a wrong object's context
minus NLL after the object's own, in nats/dim. In 956, 959 and 971 it is
≈0 in both directions (SDSS spectra were 0.06% of tokens). 991 restricts
the sky to the DESI SV3 overlap (82 deg² train), where 17% of streamed
objects carry both an image and a spectrum. On 32 held-out paired objects:

| step | image → spectrum | spectrum → r, first token | spectrum → r, whole span | objects with gain > 0 |
|---:|---:|---:|---:|---|
| 1,000 | −0.0001 ± 0.0003 | +0.007 ± 0.006 | +0.0006 ± 0.0004 | 17/32, 17/32 |
| 10,000 | **+0.0057 ± 0.0012** | **+0.205 ± 0.047** | **+0.0016 ± 0.0004** | 29/32, 27/32 |

That is ~45 nats per spectrum. Samples do not show it yet: spectrum →
image draws match the unconditional ones, and image → spectrum draws have
no emission lines at the template's z = 0.288. `min_catalogs` selects the
footprint only; rows stay outer-joined (56% South-only, 27% DESI-only, 17%
paired in train).

**70M at 80k (976).** Medians per 10k-step bin, first vs last:

| | 0–10k | 70–80k |
|---|---:|---:|
| `lm_loss` | 0.16 | −0.47 |
| image / spectrum / scalar family | 0.67 / −0.29 / −0.85 | 0.11 / −0.75 / −3.10 |

All families still fell at the end. Legacy South free samples first look
galaxy-like at 74k (compact, centred, smaller than the data); RGB samples
are still noise at 80k, and swapping the conditioning spectrum changes
the rendered image draws by a mean |Δpixel| of only 3–6 (20–27 at 40k). Step 80k, with 10k–70k
as `stepN` branches, is public as `Smith42/astroPT_v3.0-pre-alpha-70m`
(no held-out evaluation). Its checkpoints did not resume as saved: the
per-dataset token ledger is empty under `run_train.py`.

**Suffix-only pilot (964), rejected.** Leaving each object's first span
unscored and raising the scalar weight to 1 (resumed from 956 at 2k) was
worse at 4k on 432 held-out objects in every family: image +0.53 nats/dim,
spectrum +0.34, scalar +2.35 per token. Masking and reweighting are not
separated.

**Streaming bytes (991 vs 986, from `telemetry/bytes.*.jsonl`).** Loader
wait was 3% of step time at ~20 MB/s from HF. 991 read 654 kB per fresh
object (986: 523 kB). Partitions were downloaded more than once: DESI
2.22× (1.72× from different workers reading the same partitions, 1.29×
within a worker), South 1.21×, against 1.00× in 986. An inner join would
have cost ~4 MB per paired object. Dotted sub-field projection
(`spectrum.flux`, …) still downloads the whole struct (298.7 MB per DESI
partition either way; cf. §5): fsspec's `open_parquet_file` prefetch, used
by nested-pandas for remote files, truncates column names to the
top-level field. Plain pyarrow skips the unused `ivar` (152 → 84 MB). Of
the bytes read, unused `ivar` is ~49% of DESI and ~47% of HSC; South uses
`ivar` only as a validity test (~34% of its bytes).

## Open items / not pursued

- **Paired-object echo weighting**: echo paired objects more often than
  single-modality ones (e.g. 12× vs 4×) to raise the cross-modal share
  without more downloads; needs object-level echo in packing (§15).
- **Per-worker sky sharding**: give each loader worker and DP rank its own
  aligned pixels to remove the 1.2–2.2× duplicate partition reads (§15).
- **Sub-field reads**: fix the fsspec/nested-pandas prefetch so dotted
  struct columns skip unused `ivar` (§15).
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
