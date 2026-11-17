# ADR 0012: Gate MMU streaming throughput by measured byte economics

- **Status:** Proposed
- **Date:** 2026-07-22
- **References:**
  - [ADR 0006](0006-stream-mmu-upstream.md)
  - ADR 0011
  - [ADR 0014](0014-byte-efficiency-and-mfu-programme.md) — the
    byte-economics measurements this gate asked for
  - `src/astropt3/data/streaming.py`
  - `src/astropt3/data/nanotron_loader.py`

## Question

How should AstroPTv3 improve useful training throughput over a fixed Internet
connection without republishing MMU data, while preserving scientifically
valuable crossmatches and leaving room for future modalities?

## Context

The measured 70M run is network-bound on a 1 Gbit NIC. Images cost about
250 KB each on the wire; `image.flux` occupies about 50 MB of each 56 MB
row group. Compute-bound training would require roughly 3 Gbit/s, while the
link-saturated floor is about 1.7–1.9 seconds/step.

ADR 0011 already removed a redundant standalone image scan by skimming useful
unmatched rows from the crossmatch scan. It cannot remove the dominant cost:
upweighted pairs require matched partitions to be served repeatedly.

The current implementation also has deliberate constraints learned during
shakeout:

- nested HF Dataset readers inside workers break worker sharding;
- whole-row-group Python conversion exceeds the 96 GiB cgroup;
- direct PyArrow row reads avoid nested worker splitting;
- excessive workers can saturate shared egress and impair other users;
- exact train-stream replay adds complexity, while fixed validation remains
  non-negotiable.

## Decision drivers

- Optimize end-to-end non-padding training tokens/sec, not raw bytes/sec.
- Preserve a configurable scientific floor: at least `P` of distinct emitted
  row bundles are multimodal.
- Report source choices on a frontier of tokens/sec, `P`, and distinct
  standalone-row coverage.
- Use current public HATS files unchanged.
- Prefer existing HF Datasets, Arrow, fsspec, and OS facilities over custom
  transport machinery.
- Keep memory, cache, concurrency, and resume drift explicitly bounded.
- Do not retain complexity that fails to improve real-host end-to-end
  throughput by at least 10%.

## Decision

Adopt a sequential, measurement-gated client-side optimization policy.

### 1. Scientific accounting

A source row is the atomic observation. A multimodal training object is a
stable bundle identified by its member `(source, row-id)` pairs. Re-emitting a
bundle does not increase distinct coverage.

For future multiway matches, choose a cost-aware anchor and attach at most the
nearest valid row from each selected source. Do not form Cartesian products.

Keep `P` symbolic. Benchmark source sets as a Pareto frontier rather than
embedding one universal source mix or utility multiplier.

### 2. Optimize in byte-economic order

Apply and benchmark these gates in order:

1. instrument the current path;
2. tune existing native read concurrency, buffering, ranges, and batches;
3. avoid repeated remote reads using source/anchor policy and lightweight
   row-group location metadata where it can skip physical column chunks;
4. test an optional bounded per-host SSD cache for demonstrated rereads;
5. optimize Arrow-to-Python conversion only if CPU becomes material;
6. add a custom asynchronous queue only if native controls cannot keep the
   fixed link busy.

Column projection remains a verification step, not a presumed major win:
the audit found the required image flux column dominates bytes and the
remaining fields are small. Keep projection only if actual nested-leaf
measurements show useful savings or the change is essentially free.

### 3. Cache contract

This ADR reopens ADR 0011's blanket rejection of local materialization only
for an optional bounded ephemeral cache.

The cache:

- is never required for correctness;
- has an explicit per-host byte budget;
- is keyed by immutable catalog revision and file/range identity;
- may retain hot matched partitions or ranges, but is not a corpus mirror;
- is safe to delete;
- reports cold and warm performance separately;
- adds cross-process locking only if duplicate downloads are measured.

### 4. Resume and validation

Validation remains bit-exact and spatially disjoint.

Training resume need only preserve the configured statistical source
distribution. At most one configured, byte-bounded prefetch window per worker
may replay or be skipped. Report this bound in bytes, rows, and bundles.

### 5. Deployment and ownership

Support:

- one host with shared egress,
- multiple hosts sharing institutional egress,
- hosts with independent links.

Repository maintainers own safe loader behavior, instrumentation, and
conservative cache-disabled defaults. Training operators declare topology,
RAM/SSD/concurrency budgets, choose `P` and a source-set frontier point, and
promote settings after end-to-end validation.

Do not add runtime auto-tuning until static measured settings prove
insufficient.

### 6. Acceptance and stop rule

Use loader-only controlled-link tests for attribution, then short real-host
training trials.

Added loader complexity is accepted only when it provides at least 10% higher
real-host end-to-end non-padding training tokens/sec at the same model,
source policy, `P`, and cache condition, with bounded memory and no stability,
distribution, or spatial-split regression.

Stop optimizing when the loader no longer causes accelerator data wait or
when the next gate fails the 10% threshold.

## Options considered

### Republish a cropped or prejoined training dataset

Rejected by scope. It could reduce bytes substantially, but creates another
hosted artifact and schema lifecycle.

### Reduce the multimodal share

Rejected as a transport optimization. `P` is a science/operator decision and
must remain visible on the frontier.

### Replace HF Datasets with a custom reader

Rejected unless native controls fail. The current stack already supplies
sharding, retry, interleave, and state handling; previous custom HTTP handling
failed in production.

### Always preserve exact no-replay training order

Rejected as a hard requirement. Distributional resume with one bounded
prefetch window permits simpler overlap while preserving validation and
source statistics.

### Build the generalized multiway loader now

Rejected for milestone 1. Document the row/bundle seam, but optimize the
current image/spectrum/scalar path first.

## Consequences

### Positive

- Targets dominant transferred bytes and repeated reads before CPU polish.
- Makes scientific crossmatch value explicit and auditable.
- Supports varying deployments without speculative auto-tuning.
- Allows bounded cache gains without reintroducing a bulk local corpus.
- Provides a hard deletion gate for ineffective complexity.

### Negative

- Results depend on controlled and real-host benchmarks.
- Relaxed resume can replay or skip one prefetch window per worker.
- Operators must choose `P`, source policy, and topology budgets.
- Bounded caching introduces revision keys and cold/warm reporting.
- Future multiway modality support still requires later implementation.

## Follow-up

**Closed by [ADR 0014](0014-byte-efficiency-and-mfu-programme.md)
(2026-08-06):** the throughput plan was executed, the corpus is transfer-bound,
and the measured ceiling plus the surviving levers are recorded there.
