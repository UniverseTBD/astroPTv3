# ADR 0006: Stream MMU upstream instead of pre-resharding local parquet

- **Status:** **Closed** (accepted 2026-07-17, implemented and closed
  2026-08-04). The decision — stream MMU's own catalogs at train time instead
  of pre-resharding them locally — stood and was in production through two
  revisions that reshaped *what* is streamed without reopening that choice:
  ADR 0011 (row-group streaming, a
  precomputed ids-only cell-keyed match index, the crossmatch-scan demux)
  and its 2026-08-04 amendment (**crossmatch-only**: the match index defines
  the corpus, one scan, no weights). **Retired 2026-09-01:**
  [ADR 0015](0015-lsdb-infinite-stream-training.md) deletes `streaming.py`
  and the whole match-index mechanism in favor of `lsdb.streams.InfiniteStream`
  over a single uncrossmatched catalog; this ADR is now the historical record
  of the streaming decision it made, not a description of current code.
- **Date:** 2026-07-17
- **References:**
  - `astro/EXPERIMENTS.md` Part I "Data pipeline" — the local-shard + `HF_DATASETS_OFFLINE=1`
    design this ADR **supersedes**
  - `astro/src/astropt3/data/streaming.py` — the implementation
  - `astro/scripts/prepare_pilot_data.py` — the lsdb crossmatch → parquet
    reshard, deleted (its `row_to_record` moved into `streaming.py`)
  - `astro/src/astropt3/data/mmu.py` — `PILOT_FEATURES`, `MMUIterableDataset`,
    `assign_split` being deleted
  - `astro/src/astropt3/data/nanotron_loader.py` — `PackedMicroBatches`, rewired
    to stream
  - MMU's own streaming-crossmatch workflow:
    <https://huggingface.co/blog/hugging-science/multimodal-universe-hats>

## Question

The training corpus is built by lsdb-crossmatching two Multimodal Universe
(MMU) HATS catalogs on a login node and **rewriting** the result into a
bespoke `PILOT_FEATURES` parquet schema, which the compute nodes then stream
back offline. That custom schema is a second, hand-maintained source of truth
that drifts from what MMU actually publishes, and onboarding any new dataset
means editing schema code in three places (`mmu.py`, `synthetic.py`, the
crossmatch). Can the loader instead consume MMU **in its native shape**, so we
stop maintaining a parallel schema and adding a dataset becomes configuration,
not schema code?

## Context

- MMU publishes each survey as a HATS collection on the HF hub and provides
  crossmatch utilities; there is **no single pre-joined images×spectra
  release** — you crossmatch yourself. The endorsed way is lazy and streamed:
  `lsdb.open_catalog("hf://datasets/UniverseTBD/mmu_*")` +
  `.crossmatch(other, n_neighbors=1)` builds a task graph, and `CatalogStream`
  yields **one HEALPix partition at a time with background prefetch**, loading
  only overlapping tiles (~4 GB RAM for the whole join). **Internet is required
  at read time.**
- Today's pipeline (`prepare_pilot_data.py`) materializes that join into
  `PILOT_FEATURES` shards under `{root}/{train,val}/`; `MMUIterableDataset`
  streams them back with `HF_DATASETS_OFFLINE=1`; `synthetic.py` mirrors the
  same record shape so everything runs offline in CPU tests.
- The pilot trains exactly **one** corpus (LegacySurvey × DESI). There is no
  second corpus to mix yet.
- Checkpoint-resume today is exact ("never replays a trained record"), built
  entirely on HF `datasets`' stateful streaming iterator — the machinery this
  ADR deletes.
- Evaluation demands a **fixed, deterministic** val set: `eval/val_loss` reuses
  fixed val batches and the probe sweep re-evaluates it on every checkpoint.
- `synthetic.py` and the frozen contracts (special-token ids in
  `tokenization.py`, the `ObjectSequencer`/`PackedCollator` packing contract)
  are load-bearing and must not move.

## Decision drivers

- One schema, MMU's — no hand-maintained parallel `PILOT_FEATURES`.
- Adding a dataset should be config, not schema code.
- Keep CPU tests **offline and lsdb-free** (a hard repo constraint) even though
  training now needs the network.
- Don't build a multi-source mixing engine for a repo that trains one corpus.
- The frozen tokenization/packing contracts and `synthetic.py` stay untouched.

## Decision

**Stream the MMU catalogs directly at train time via lsdb/HATS + `CatalogStream`;
delete the local reshard and its custom schema entirely.** The compute nodes now
have internet — this knowingly supersedes PLAN's offline-node/local-shard data
design.

1. **Corpus = three interleaved streams**, all streamed live:
   - **images-only** — ~14M LegacySurvey images as single-modality image
     sequences,
   - **spectra-only** — ~1.1M DESI spectra as single-modality spectrum
     sequences,
   - **crossmatch (inner)** — ~0.7M matched pairs as multimodal image+spectrum
     sequences. The pairs are **precomputed offline into a published
     match-index** (see Performance); this stream reads native image + spectrum
     partitions and joins them on the index in memory — no live spatial join.

   The join flips `how="left"` → `how="inner"`: image-only coverage now comes
   from the standalone images stream, so the join carries pairs only. A matched
   object's image appears both standalone (stream 1) and paired (stream 3); this
   redundancy is inherent to combining MMU datasets and is accepted.

2. **Interleave by fixed provisional per-record weights** — **0.60 / 0.15 /
   0.25** (images-only / spectra-only / pairs). Images stay dominant (bulk,
   cleanest signal); pairs are up-weighted ~5× above their ~5% natural share
   because cross-modal learning is the point; spectra sit between so the
   spectrum head sees standalone signal without ~13× over-memorizing the small
   corpus. **These weights are provisional** — tuning them is the mixing issue
   below.

3. **Three sources hardcoded**, no config registry. General, weighted,
   config-driven multi-source mixing is deferred to a **GitHub issue**, to be
   built the day there is a second corpus to mix.

4. **Resume is partition-granularity.** Per-source cursor
   `(epoch, partition_index)` over the deterministic, per-epoch-shuffled
   partition order; on resume, reopen the native catalogs and reload the
   match-index (cheap; no spatial join to rebuild), skip whole partitions up to
   `partition_index` (no download), and restart the in-flight partition from its
   beginning. This **replays ≤ 1 partition of already-trained
   records per kill** (a partition is thousands of objects). The Phase 4
   kill/resume gate and the `object_id_log` audit are rewritten from "no replay"
   to **"no replay beyond the in-flight partition."** The weighted-sampler RNG
   state is part of the saved cursor so the source draw sequence resumes
   identically.

5. **Val = reserved partitions.** A fixed set of the first **K** HEALPix
   partitions (chosen to overlap the spectra catalog's coverage, so val contains
   matched pairs and the redshift linear-probe has `Z` labels) is reserved for
   validation and excluded from train. Val is streamed each eval — only K
   partitions, not the whole catalog — keeping it fixed and spatially disjoint
   (whole partitions ⇒ no leakage). Replaces `assign_split`'s hash-by-tile
   split.

6. **Float to latest upstream revision** (no pinning). This knowingly softens
   the reproducibility goal: if MMU pushes a new revision mid-project the corpus
   changes silently. Mitigation left open (see Open issues): logging each
   catalog's resolved revision at run start costs nothing and restores
   after-the-fact traceability without config upkeep.

7. **Schema demolition.** Delete `PILOT_FEATURES`, `normalize_record`,
   `write_shard`, `decode_record`, `assign_split`, `MMUIterableDataset`,
   `prepare_pilot_data.py`, `check_pilot_data.py`. The one thing kept is the
   lsdb-row → record-dict logic (`row_to_record`), which **moves into the
   streaming loader as the sole decode adapter** over MMU-native fields.
   `ObjectSequencer`, `PackedCollator`, `tokenization.py`, `band_registry.py`,
   `transforms.py` are untouched. `synthetic.py` is unchanged and remains the
   offline test source via the `data_root == "synthetic"` sentinel. "Kill the
   custom schema" is satisfied: there is no longer a second hand-curated Arrow
   schema distinct from what MMU publishes — the loader consumes native rows and
   adapts minimally.

8. **lsdb is offline-only; train-time streaming is `pyarrow` + `hats`.** lsdb
   runs only in the prep env, to compute the match-index once; no train-time
   path needs its dask/KD-tree machinery. `pyarrow`/`hats` are lazy-imported on
   the streaming path only (behind the `synthetic` sentinel), so the CPU suite
   stays offline; they — not lsdb — move into the nanotron training env's
   dependencies. lsdb stays in the `[data]`/prep extra.

9. **Adapter test = checked-in cassette.** A handful of real crossmatch rows are
   captured once (online, login node) into a small repo fixture and replayed
   offline through `row_to_record` in a CPU test — covering the non-trivial
   decode (struct coercion, `_desi` suffix handling, null handling) without
   lsdb/network, and doubling as documentation of the real upstream row shape.

10. **Sharding/shuffle mechanism.** Partition-level DP/worker split: assign the
    per-epoch-shuffled partition index across `(world_size × num_workers)` by
    modulo — deterministic, disjoint, no cross-rank coordination. Per-epoch
    seeded partition-order shuffle (identical across ranks, so the modulo split
    stays disjoint) plus a per-worker buffer shuffle break the spatial
    clustering of HEALPix-ordered partitions. Because the stream is endless,
    uneven partition sizes never starve a rank at the optimizer all-reduce — each
    rank always has another partition to draw.

## Performance

Streaming must sustain **≥2× training-consumption per rank** so the GPU never
starves; beyond that floor, throughput work stops ("as fast as possible" is not
the target). The 75 obj/s of the old serial `prepare_pilot_data.py` prep is
**not** that number — training runs `DP × num_workers` (~256 at pilot scale)
loader processes, so the naive live path may already clear the floor by worker
parallelism, or 256 concurrent engines may exhaust RAM. **A profiling spike
settles it before anything is built** (see Open issues): one live worker over
~5 real partitions, coarse timers on `partition read` vs `row decode` vs
`tokenize`, so the optimization is aimed at the measured bottleneck rather than
a guessed one.

Three refinements shape the streaming path — chosen so they also survive scale
(a 62 TB Legacy South imaging corpus grows the *images-only* stream; the
crossmatch stays bounded by the smaller spectroscopic side, so imaging volume
never inflates it):

- **Precompute the crossmatch into a published match-index, not joined
  records.** `lsdb.crossmatch` runs offline, once; the artifact published to HF
  is only `(image_healpix, image_id, spectrum_healpix, spectrum_id)` —
  ~0.7M rows, tens of MB, **no pixels duplicated** (joined
  records would be ~280 GB of images already hosted in the LegacySurvey
  catalog, and a second bespoke schema). The index is bounded by matched pairs,
  so it survives arbitrarily large imaging corpora unchanged. This keeps the
  "one native schema" win: the published artifact is pointers into MMU's data,
  not a copy of it.
- **lsdb offline-only; train-time streaming is `pyarrow` + `hats`.** Once the
  join is an id-dict lookup, no train-time path needs lsdb's dask/KD-tree
  engine: `hats` enumerates partitions + HEALPix order + `hf://` resolution,
  `pyarrow`/`fsspec` stream partition files, and matches resolve through the
  in-memory index. Removes the 256-dask-scheduler feasibility risk and drops a
  distributed-join engine from a path that only reads files sequentially.
- ~~**Optimize the shared decode, images-first.**~~ **Dropped — the spike
  measured decode at 5% of the time.** The reasoning below was sound but aimed
  at the wrong stage: read dominates at 85%, so a perfect decode buys ~5% on
  the images path. The precondition held (image flux *is* fixed-shape
  `(3, 152, 152)`, so a columnar decode is viable), and this can be revived if
  read ever stops dominating — but it is not built now. Original rationale: the
  60% images-only stream does no join and shares its decode/tokenize path with
  the other two, so a decode win lifts all three.

### Spike results (2026-07-20)

`scripts/spike_profile_streaming.py`, one worker, 5 real partitions per source:

| stage | images | spectra |
|---|---|---|
| read | 10.67 ms/obj (**84.8%**) | 3.25 ms/obj (**76.6%**) |
| decode | 0.63 ms/obj (5.0%) | 0.32 ms/obj (7.5%) |
| tokenize | 1.28 ms/obj (10.2%) | 0.67 ms/obj (15.9%) |
| **rate** | **79.5 obj/s/worker** | **235.6 obj/s/worker** |

Four conclusions, each changing a decision above:

1. **Read dominates; the decode optimization is dropped.** Read is
   network-bound — ~170 MB per images partition at ~10 MB/s per worker. This
   is a hub-bandwidth problem, not a CPU one.
2. **The ≥2× floor is cleared by worker parallelism alone.** The 70M reference
   (~240k tokens/s total, `docs/training.md`) is ~1,570 obj/s at 153 tokens
   per image object, so the floor is ~3,100 obj/s ≈ **40 workers** across the
   job at the measured rate. No throughput work is needed to start.
3. **256 workers is NOT RAM-feasible, and whole-partition buffering is the
   cause.** An images partition is up to 774 MB in Arrow; buffering one per
   source is ~2.3 GB/worker, i.e. ~74 GB at only 32 workers. **Partitions are
   therefore streamed by row group** (~200 rows / ~56 MB each), which caps a
   source at ~56 MB and keeps resume exact — the cursor gains a row-group
   index. This supersedes the "restart the in-flight partition" language in
   decision 4.
4. **The scaling table is not linear in practice.** These are single-worker
   extrapolations of a network-bound stage; aggregate hub bandwidth will cap
   throughput well below 256× (256 × 10 MB/s = 2.5 GB/s from HF is not
   realistic). Treat worker counts above ~40 as unvalidated.

### Measured in production (2026-07-23/24, closing numbers)

`astropt3-70m-jetformer-crossmatch-only` on gpu5 (dp=2, 8 workers/rank,
seq 4096, mbs 16) reached **step 13,354 / 20,000**: mean **3,221 ms/step**,
**17.5 %** of iterations ≥ 5 s, fast-step floor 0.59 s (~105 K tok/s/GPU),
flat start to finish (first 2,000 mean 3,356 ms vs last 2,000 mean 3,412 ms),
all five losses converging. It ended on a DataLoader-worker OOM traced to the
per-cell unmatched-spectrum buffer, now bounded — see ADR 0011's 2026-08-04
revision.

The ≥2× floor of this section is met by worker parallelism as the spike
predicted, but the run is **NIC-bound, not compute-bound**: the 0.59 s
fast steps are what the GPU can do, and the 3.2 s mean is what the 1 Gbit
link allows. That gap, not the code, is the throughput story on this box.

Prefetch is the existing DataLoader worker queue (`prefetch_factor` /
`num_workers`); no custom prefetcher is built. Network-stall resilience and
cross-epoch local partition caching are **out of scope** for the sub-epoch
pilot — they live under the "network is a hard dependency" negative below, not
the speed work.

## Options considered

### Option A — Native-schema local shards (keep the reshard, mirror MMU)

Keep `prepare_pilot_data.py` but make its output schema the faithful union of
MMU's two native schemas instead of a curated subset. **Rejected:** still a
second on-disk schema to maintain and still a login-node prep step; it trims the
drift but not the maintenance surface, and it keeps the whole offline-shard
machinery this ADR is trying to delete.

### Option B — Stream upstream at train time (chosen)

Drop local shards; stream the crossmatch live with lsdb/`CatalogStream`.
**Chosen:** removes the parallel schema and the prep step entirely, consumes MMU
exactly as published, and is MMU's own recommended workflow. Costs the offline
guarantee and exact resume (see Consequences), both accepted.

### Option C — Multi-source mixing engine now

Build the general N-dataset, weighted, config-driven mixing loader immediately.
**Rejected as premature:** the pilot trains one corpus; sampling weights,
interleave, and per-source modality routing are speculative machinery. Deferred
to a GitHub issue; the interim is three hardcoded sources with fixed weights.

### Option D — Status quo (local reshard, offline nodes)

Keep PLAN's design. **Rejected:** it is exactly the hand-maintained parallel
schema and three-place dataset-onboarding this ADR removes.

## Consequences

### Positive

- One schema — MMU's. No `PILOT_FEATURES`, no `normalize_record`/`write_shard`,
  no login-node reshard step; a large, drift-prone slice of the data layer is
  deleted outright.
- Adding a dataset (once mixing lands) is pointing at an `hf://` catalog, not
  editing schema code in three files.
- Single-modality *and* paired records both fall out of the same live stream
  set; the corpus mirrors how MMU is actually published.
- CPU tests stay offline and lsdb-free (lazy import behind the synthetic
  sentinel); the frozen tokenization/packing contracts and `synthetic.py` are
  untouched.

### Negative / tradeoffs

- **Network is a hard training dependency.** HF hub downtime or rate-limiting
  stalls training with no local fallback — the deliberate price of
  streaming-only.
- **Resume replays ≤ 1 partition per kill** (thousands of objects); the exact
  no-replay guarantee and its Phase 4 gate are relaxed accordingly.
- **Reproducibility softened** by floating revisions — the corpus can change
  under a long run if MMU pushes upstream.
- **`pyarrow`/`hats` become training-env dependencies** (lsdb stays offline).
  That `hats` alone — without lsdb — exposes deterministic partition
  enumeration, HEALPix order, and `hf://` resolution is an **implementation
  risk** assumed here, not yet verified in code (see Open issues).
- Interim fixed weights (0.60/0.15/0.25) are a guess; the small corpora
  (spectra, pairs) still carry over-memorization risk until the mixing issue
  tunes them.

## Open issues

- **Multi-source weighted mixing** (GitHub issue): general, config-driven source
  list with tunable sampling weights, interleave, and per-source modality
  routing — supersedes the hardcoded three sources and the provisional weights.
  Moot under crossmatch-only (one source, emergent mix); it returns with the
  second survey.
- ~~**Profiling spike**~~ — **done**, see Spike results above.
- ~~**`hats`-without-lsdb partition API**~~ — **verified**. `hats.read_hats`
  on an `hf://` URL returns a `CatalogCollection`; `.main_catalog`
  `.get_healpix_pixels()` gives deterministic HEALPix-ordered partitions and
  `hats.io.paths.pixel_catalog_file` resolves each to its parquet path, read
  through `HfFileSystem` + `pyarrow`. No lsdb at train time.
- **Aggregate throughput at realistic worker counts** is unmeasured; the spike
  extrapolated from one worker on a network-bound stage. Measure before
  trusting any figure above ~40 workers.
- ~~**Match-index build**~~ — **built** (`mmu_stream.build_match_index`), schema
  `(image_partition, image_id, spectrum_partition, spectrum_id)` where the
  partition columns are HEALPix `(order, pixel)` cells, so the artifact
  survives MMU adding or dropping partitions. Partition-locality **confirmed**: every image partition in
  the smoke index maps to exactly ONE spectrum partition, so the in-memory
  join is partition-local. The full build is ~200 partitions (the *inner*
  crossmatch, not the 5,596 of the LEFT join) at ~17 s/partition ≈ **1 hour**.
  **Published** at
  `hf://datasets/Smith42/mmu_desi_edr_sv3_x_mmu_ssl_legacysurvey_north/match_index.parquet`
  — 137,906 pairs over 173 image cells — and wired into every `data_root: mmu`
  config; `$ASTROPT3_MATCH_INDEX` overrides it for a locally built index.
- **Revision logging:** cheap run-start capture of each catalog's resolved HF
  revision into a `provenance.json`, restoring after-the-fact traceability under
  the float-to-latest decision.
- **Val coverage:** confirm the reserved K partitions actually contain enough
  matched pairs for a stable redshift probe. Under crossmatch-only every
  reserved cell is a matched cell by construction, so the question is only
  whether K=8 of 173 is enough pairs, not whether any are present.

## References

- `astro/src/astropt3/data/nanotron_loader.py` — `PackedMicroBatches` (rewired
  to the single `pyarrow`/`hats` crossmatch scan), the partition-ceiling
  check, resume.
- `astro/src/astropt3/data/streaming.py` — `decode_record` (absorbed the old
  `row_to_record`), `_crossmatch_examples`, `owned_by_rank`, `open_stream`.
  `astro/src/astropt3/data/mmu.py` is deleted, as listed above.
- `astro/src/astropt3/data/synthetic.py` — unchanged offline record source.
- MMU streaming-crossmatch blog (above) — the endorsed lsdb/`CatalogStream`
  workflow.
- [ADR 0003](0003-checkpoint-samples-in-eval-sidecar.md) — house format
  precedent.

## Implementation notes (2026-07-20)

Implemented in `astro/src/astropt3/data/streaming.py`. Several decisions
changed once the code met the real API; the rest landed as written.

**Superseded by the Performance revision:** the first implementation used
lsdb's `CatalogStream` at train time and buffered whole partitions. Both are
gone — train-time streaming is `pyarrow` + `hats` over row groups, and lsdb
runs only in `mmu_stream.build_match_index`. Notes 1-3 below still hold.

1. **Weights apply per record, not per partition draw** (§2). Partitions hold
   wildly unequal row counts — measured on the real catalogs: ~5100 rows in a
   DESI partition, ~2447 in a LegacySurvey one, ~290 in a crossmatch one — so
   drawing whole partitions by weight would realize a corpus mix nothing like
   0.60/0.15/0.25. Each source buffers its current partition and only fetches
   the next when that buffer drains.

2. **Resume is exact; the ≤1-partition replay budget was not needed** (§4).
   Because a source buffers its partition whole, the row offset into it is
   checkpointable, so resume re-fetches the in-flight partition but re-trains
   none of it. The Phase 4 no-replay gate and the `object_id_log` audit stand
   unchanged. The saved state is `(draw, per-source (epoch, cursor,
   row_off))` — all ints. There is **no sampler RNG state in the checkpoint**:
   the source draw order is a fixed repeating length-20 pattern derived from
   the weights, not a random draw.

3. **No cassette fixture** (§9). The CPU suite may now use the network, so a
   `network`-marked test decodes real hub rows end-to-end instead. A cassette
   would have been a second artifact to capture, refresh, and let drift. The
   cursor logic is covered offline by injecting fake fetchers behind the
   `Source` API (`tests/fake_mmu.py`), which is faster and deterministic.

Resolved open issues:

- **`CatalogStream` resume API** — it exposes no cursor and no
  skip-to-partition, but `_delayed_partitions` is a plain list indexed by
  partition index and `submit_next_partitions` takes explicit indices, so the
  partition order is ours to own. `CatalogStream` is constructed only for its
  culled per-partition dask graphs; its own iterator is never used.
- **Val coverage** — the pairs source is an inner join, so every reserved val
  partition of it carries matched pairs with `Z`. K is
  `streaming.VAL_PARTITIONS`.

Closed at 2026-08-04:

- **Prefetch — recorded non-decision, not a TODO.** The old framing ("requires
  a dask client, `open_sources(client=...)`") described an API the train path
  no longer has: lsdb and `CatalogStream` left with the Performance revision.
  Prefetch is the DataLoader worker queue (`num_workers` / `prefetch_factor`)
  and nothing custom is built. **Synchronous per-worker reads are accepted
  while training is link-bound**: at 1 Gbit the link saturates around
  1.8 s/step against a 0.55 s/step compute demand, so overlapping I/O cannot
  buy what the NIC will not deliver. Revisit at ≥ 10 Gbit, or when a measured
  run is compute-bound with the link below saturation — not before.
  ADR 0012's Gate 8 holds the design if that day comes.
- **Sharding ceiling — now enforced.** The claim above was aspirational: no
  such check existed, and `datasets` merely warned and stopped the surplus
  workers. The numbers also changed under crossmatch-only — the corpus is the
  match index's 173 cells (165 train), not ~200 pairs partitions — and
  partitions are dealt to DP ranks (`owned_by_rank`) rather than node-split,
  so the bound is per rank: `num_loading_workers <= floor(165 / dp)`.
  `nanotron_loader._mmu_records` raises with cause and remedy, and
  `test_run_configs_fit_the_crossmatch_partition_ceiling` sweeps
  `configs/nanotron/*.yaml` for violations. The lasting fix is sharding by
  parquet row group rather than by cell — deferred, not needed at dp ≤ 32.

Still open (do not block this ADR):

- **Revision logging** — unchanged from above, still worth doing. Note that a
  suspected mid-project drift was investigated and was a measurement error
  (`images.npartitions` = 5488 vs the LEFT-crossmatch's 5596 partitions);
  upstream had not moved.
- **Aggregate throughput above ~40 workers** is still unmeasured, and the
  partition ceiling now caps a dp=64 rank at 2 workers anyway. Whether that
  starves a 1B run is unknown: per-GPU bandwidth demand falls ~10× from 70M
  to 1B, so the 8-workers/rank figure from the 70M shakeouts does not
  transfer. Measure at the first pilot topology.
- **Multi-source weighted mixing** (GitHub issue) — moot for the current
  single-source corpus, but the mechanism returns with the second survey.

Out of scope, deferred to the corpus-expansion work (`mmu-corpus-expansion`
branch; see ADR 0013, and
`docs/mmu_crossmatch_research.md` (deleted 2026-08-06; in git history)): Legacy South DR10 × DESI,
HSC PDR3 × SDSS, Gaia DR3 × APOGEE, and JWST. Each needs its own gate — a
band registry that is source-aware, an SDSS spectrum normalizer, epoch
propagation for the stellar branch — and none of them is implemented here.
