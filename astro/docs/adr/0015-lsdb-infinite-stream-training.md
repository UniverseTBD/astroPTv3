# ADR 0015: Use LSDB InfiniteStream as the sole training source

- **Status:** Accepted for an experimental branch; incomplete under the current
  phase gates
- **Date:** 2026-09-01
- **Scope:** Initial uncrossmatched LegacySurvey North ingestion

## Context

AstroPTv3 currently assembles its training corpus through `mmu-stream`, a
mandatory precomputed match index, custom DP/worker ownership, TorchData
checkpoint state, and a separate synthetic source. That implementation owns
behavior that should eventually live upstream in LSDB: catalog operations,
crossmatching, distributed stream ownership, and cursor resume.

The immediate goal is a hard cutover to LSDB's official
`lsdb.streams.InfiniteStream`, starting with one uncrossmatched catalog while
those richer upstream capabilities are developed. This is intentionally an
experimental image-catalog phase, not a preservation of the current
crossmatch-defined corpus.

## Decision

Use this as the only training-record path:

```text
lsdb.open_catalog("hf://datasets/UniverseTBD/mmu_ssl_legacysurvey_north")
→ lsdb.streams.InfiniteStream(
      client=None,
      partitions_per_chunk=1,
      seed=consumer_seed,
  )
→ minimal LegacySurvey row decode
→ existing ObjectSequencer and packing
→ torch.utils.data.DataLoader(batch_size=None)
→ nanotron
```

### Source and model

- Start with the uncrossmatched LegacySurvey North catalog.
- No spectra or crossmatched partners are present. Decode image-side scalar
  fields already provided by the catalog when the active registries use them.
- Keep the full multimodal model shape so later crossmatches do not require a
  checkpoint architecture migration.
- Float both LSDB and the catalog revision. `uv.lock` records the package graph
  resolved for a checkout; runs log the resolved LSDB version and catalog
  provenance when available.

### Concurrency and sampling

- Torch DataLoader workers own concurrency; LSDB runs synchronously with
  `client=None`. Do not create nested Dask worker pools.
- Retain the current configured 6–8 workers per DP rank.
- Derive each stream seed from the run seed, DP rank, worker id, and retry
  generation.
- Each consumer (DP rank × worker) owns every N-th anchor HATS partition of the
  footprint, N = DP size × workers, so consumers do not fetch the same anchor
  files; consumers share partitions only when N exceeds the partition count.
- Accept repeated records within a consumer, no epochs, and no
  complete-coverage guarantee.

### Resume and failures

- Remove TorchData and all dataset-position checkpoint state.
- Trainer checkpoints restore model, optimizer, scheduler, and trainer state,
  then start a fresh LSDB stream.
- Retry recognized transport/storage failures with the existing bounded
  exponential-backoff policy, discarding the failed iterator and opening a
  fresh stream.
- Decode, validation, and unknown runtime errors fail immediately.

### Hard cutover

Delete old source paths without compatibility shims:

- `data_root`, `match_index`, synthetic fraction controls, and
  `object_id_log`;
- the synthetic training source;
- `mmu-stream`, TorchData, match-index assembly, and exact stream state;
- match-index builders and launch checks;
- old stream fakes/tests and exact-resume/no-replay assertions.

Keep `ar_replicas` and `replica_placement`; they are independent of the record
source.

### Evaluation and generation

Defer source-backed validation loss, probes, sample sweeps, and their CLIs and
co-launch wiring. Keep pure model-side embedding, metric, sampling, and
rendering functions that consume already-provided records, objects, or
batches. Future LSDB-backed evaluation reconnects at that seam.

## Consequences

### Positive

- LSDB becomes the sole astronomy catalog and partition-streaming dependency.
- Most bespoke corpus, sharding, and resume machinery is deleted rather than
  wrapped.
- The model-facing decode/sequence/pack seam stays small and reusable.
- Future crossmatching and resume improvements can arrive through LSDB instead
  of another local framework.

### Negative and accepted risks

- `InfiniteStream` may retain roughly two whole pandas partitions per worker.
  With LegacySurvey partitions previously observed near 774 MiB, 6–8 workers
  per rank can consume many GiB and may OOM.
- There is no formal RSS or unused-gradient preflight gate; worker counts are
  adjusted only after an observed failure.
- Spectrum modules remain allocated but untrained and may expose a live
  synchronization issue.
- Retries and checkpoint resumes can revisit data immediately.
- Floating package and catalog revisions can change behavior, schema, or data.
- No fixed validation, quality signal, or automated learnability assertion
  exists during this interval.

## Deferred

- Crossmatching and multimodal source assembly.
- Strict distributed partition ownership.
- Dataset cursor and RNG resume.
- Fixed validation, probes, and sample sweeps.
- Automated LSDB-backed learnability smoke.

## Interim evidence

- Offline LegacySurvey-shaped row decode → sequence → pack tests.
- One bounded `pytest.mark.network` integration test that opens the live
  catalog and consumes `InfiniteStream` output.
- `count_params.py` remains applicable because the model shape is unchanged.

The current repository contract requires `pytest`, `count_params`, and
`train_smoke` before a phase is declared complete. This decision temporarily
removes the synthetic-dependent `train_smoke`, so the branch remains explicitly
**experimental/incomplete** until an LSDB-backed smoke returns or the governing
contract changes.

## Implementation plan

### 1. Cut dependencies

Update `astro/pyproject.toml` and `astro/uv.lock`:

- add unpinned `lsdb` to main dependencies;
- remove `torchdata` and `mmu-stream`;
- remove direct HATS, Arrow, or nested-pandas declarations only when no
  surviving code imports them directly.

### 2. Collapse the dataset configuration

Update `nanotron/src/nanotron/config/astropt3_config.py` and
`astro/configs/nanotron/*.yaml`:

- remove `data_root`, `match_index`, synthetic fractions, and
  `object_id_log`;
- keep `ar_replicas` and `replica_placement`;
- remove match-index and old source checks from launch scripts.

Keep the existing `astropt3_streaming` nanotron dataset type; renaming it adds
churn without changing its role.

### 3. Rewrite the record source

Update `astro/src/astropt3/data/nanotron_loader.py`:

- open the fixed LSDB catalog inside each worker iterator;
- project only columns required by the surviving image/scalar registries;
- create `InfiniteStream(client=None, partitions_per_chunk=1, ...)`;
- convert each yielded DataFrame row to the existing record contract;
- feed records through the unchanged sequencer, replica placement, and packer;
- use an ephemeral partition-draw counter as the modality-order nonce because
  `InfiniteStream` has no epoch; it is intentionally not checkpointed;
- add bounded fresh-stream retry and provenance logging;
- return a plain `torch.utils.data.DataLoader(batch_size=None)`.

Do not introduce a generic catalog registry, source interface hierarchy, or a
second DataLoader wrapper.

### 4. Remove dataset checkpoint state

- Delete dataset `state_dict`/`load_state_dict`, snapshots,
  `loader_state_dict`, and `resume_state_dir` handling.
- Remove dataset-state lookup from `nanotron/run_train.py`.
- Remove `dataset_state/dp_*.pt` writes from
  `nanotron/src/nanotron/trainer.py`.
- Continue loading model/optimizer checkpoints; ignore any old dataset-state
  files.

### 5. Delete retired source machinery

- Remove the synthetic source and source-specific tests.
- Remove match-index builders and related scripts.
- Remove old MMU stream fakes, streaming tests, resume tests, and GPU assertions
  tied to dataset-state files.
- Adapt or remove telemetry that imports `mmu-stream` catalog constants.

### 6. Trim evaluation wiring

- Remove synthetic/MMU collectors, source-driven eval CLIs,
  `run_probe_sweep.py`, and eval co-launch hooks.
- Retain pure embedding, metric, sampling, and rendering functions.
- Keep pure-function tests; delete tests that only exercise retired collectors.

### 7. Add interim checks

- Add a small offline LegacySurvey-shaped row fixture covering decode,
  sequencing, and packing. It is a unit fixture, not an alternate training
  source.
- Add a bounded network-marked test that opens the real catalog, consumes a
  stream chunk, decodes a few rows, and validates tensor shapes and finite
  values.
- Confirm ordinary CPU model tests accept batches with no spectra.

### 8. Update active documentation

Update `astro/README.md`, `astro/docs/training.md`, architecture documentation,
and launch guidance. Preserve earlier ADRs as historical records and point the
active documentation to this superseding experimental decision.

### 9. Verify honestly

Run from `astro/`:

```bash
uv run pytest
uv run python scripts/count_params.py
```

Do not declare the phase complete while the mandated `train_smoke` gate is
absent. No GPU or training run is performed on this machine.

## References

- [LSDB `InfiniteStream`](https://docs.lsdb.io/en/stable/reference/api/lsdb.streams.InfiniteStream.html)
- [LegacySurvey North HATS catalog](https://huggingface.co/datasets/UniverseTBD/mmu_ssl_legacysurvey_north)
