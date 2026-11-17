# ADR 0014 — Byte-efficiency and MFU programme for five-spoke training (crop retained)

- **Status:** Accepted (2026-08-06). Rewritten the same date to fold the
  measurement study into the gated byte-efficiency + MFU programme below;
  the programme is the committed plan and its gates govern execution, not
  adoption. **Amended 2026-08-06 (r3)** to add the HSC image projection,
  partner-row scalar attachments, byte-balanced dealing, the bounded wire
  experiment, cache-era decode work, compute-side follow-ons, and the
  upstream publication ask. **Amended 2026-08-07 (r4)** to ship the HSC
  image-struct projection, record the Legacy projection as a measured null
  result, and adopt seven free scalar modalities — telescope measurements
  included by owner decision — from columns already on the wire (A7, A8).
  **Amended 2026-11-17 (r5)** to adopt the issue #53 value-added scalar
  triage and the PROVABGS scalar spoke (A9).
  The implementation amendments at the foot of
  this document record eight corrections the code found. Supersedes and absorbs
  the labbook working document *"Useful bytes per megabyte downloaded"*
  (2026-08-06, amended same day for the five-spoke corpus), which was
  deleted when the first version of this ADR was extracted — recover it from
  git history if a derivation needs
  checking.
- **References:** [ADR 0006](0006-stream-mmu-upstream.md), [ADR
  0008](0008-scalar-modalities.md), ADR 0011,
  ADR 0013, PR #31.
- **Owner decision recorded here (dated 2026-08-06):** retain the 96×96
  central image crop; full-frame and foveated tokenisation are out of scope
  for this programme (§1).

## Context

Training on the ADR 0006/0011/0013 live stream is **transfer-bound, not
compute-bound**. On the five-spoke North corpus the GPU idles at every cell
boundary while ~1.18 GB of partner data arrives: median 0.68 s/step against a
mean of 7.80 s/step, with 93% of wall clock inside stalls. Three independent
attempts to fix that with *concurrency* all failed (§11). The only lever that
matters is therefore **loss-bearing signal per megabyte transferred**.

### What a megabyte buys (measured, single row group per catalogue)

- LegacySurvey North anchor row group: 283 KB/object, of which **`image.flux`
  is 92.3%**. Every other anchor column is <0.1% — projecting them is
  pointless.
- `image.flux` is 3×152×152; `packing.IMAGE_CROP = 96` tokenises the central
  96×96. **60% of every image byte is discarded before it reaches a token.**
- DESI spectra: 114 KB/spectrum, of which `spectrum.ivar` is **41%** and is
  never read outside `synthetic.py`.
- KB per loss-bearing token: anchor/PROVABGS scalars ~0 (same row we already
  pay for), full-frame image 0.73, ivar-projected spectrum 1.15, today's 96×96
  crop 1.82, today's spectrum 1.93, **HSC 3.56** — the most expensive token in
  the corpus by 5×.
- **HSC (verifiable from published metadata):** the catalog reports 335.6 GiB
  across 156 partitions (about 2.2 GiB/partition). Its `image` struct carries
  `flux`, `ivar`, `mask`, `psf_fwhm`, and `scale`; the adapter currently
  consumes only `flux` and band metadata. `image.ivar` is a 5×160×160
  float32 plane, the same raw size as `flux` (~512 KB/row), and `mask` is
  another transferred plane (compressible, but not free). Unless normalization
  uses `psf_fwhm`/`scale`,
  roughly half of each HSC payload is fetched and discarded.
- The discarded periphery is **not** empty sky: on 40 real North objects the
  median peripheral patch carries the same arcsinh-normalised variance as the
  median central patch (ratio 1.01×), and only 12.7% of full-frame patches
  fall below the crop's own 10th percentile. Full-frame tokenisation would be
  2.5× more tokens of roughly equal quality at zero extra bytes.

## Question

The five-spoke corpus is transfer-bound: ~93% of wall clock on the
`ar_replicas: 1` baseline is stalls waiting for bytes, and three separate
concurrency attempts (workers beyond 8, cell prefetch, fsspec/pyarrow
readahead) produced no improvement — the prefetch is actively harmful at 8
workers. Given that the binding constraint is loss-bearing signal per
megabyte transferred, and given the owner constraint that anchor images
remain cropped to 96×96, what is the ordered set of changes we commit to,
what metrics and gates govern each one, and what do we explicitly refuse to
build?

## Decision summary

Adopt a gated programme, in this order:

1. **Immediately, in parallel:** fetch-boundary instrumentation, MFU
   accounting, and a frozen benchmark; the sequence-assembly fingerprint
   fix; nested spectrum-leaf projection for DESI and SDSS plus HSC image-struct
   projection; validation-set repair; the replay (`ar_replicas`) quality
   validation; open the transient-cache ADR conversation against ADR 0006/0011;
   send the upstream publication request.
2. **Next:** per-band tokenisation A/B **on the cropped image** — now the
   only image-representation experiment in scope.
3. **Last:** transient cross-epoch cache prototype under stable ownership;
   re-measure partner-fetch locality on the 5,488-cell index before any
   ordering/caching-adjacent work beyond that.

The r3 additions fit around that order rather than replacing the cache:
HSC projection ships with §6, scalar attachments are a Phase 1 shortlist in
§6a, byte-balanced dealing follows the §3 cell-cost measurement, and the
single-fetch wire test in §10a runs only when telemetry shows headroom below
NIC line rate. Decode vectorisation waits for the cache prototype, while
scalar-head fusion and the other MFU hygiene wait for the instrumented
baseline. The publisher request in §10b goes out immediately because its lead
time is external.

Progress is judged on two headline axes — **useful bits per byte
transferred** and **model FLOPs utilisation (MFU)** — with scientific
quality per resource as the deciding gate. Full-frame tokenisation,
foveated tokenisation, and HSC full-frame are **rejected for this
programme** by owner decision (§1). Everything on the stop list (§11) is
refused with reasons.

---

## 1. Owner decision: retain the 96×96 image crop

`packing.IMAGE_CROP = 96` stands, for Legacy North anchors and HSC alike.
Full-frame tokenisation of the 152×152 frame (0.73 KB/token vs 1.82 cropped,
per §Context), foveated tokenisation (patch-8 centre + patch-16 annulus ≈
198 tokens for all pixels), and any patch-size change that alters image
tokenisation are out of scope.

**Owner rationale as stated:** the discarded periphery is mostly empty sky.

**Evidence note (recorded for honesty, per governance):** the periphery
measurement in §Context measured the opposite — the median peripheral patch
carries the same arcsinh-normalised variance as the median central patch.
Patch variance is a proxy for structure, not learnable signal — a
high-variance pure-noise patch is still irreducible — so the rationale may
yet be vindicated by a sharper measurement (spatial autocorrelation within
a patch, or variance against the per-pixel noise estimate). As measured
today, the "black sky" claim is not supported.

**Grounds on which the decision stands regardless:**

- **Sequence budget.** Full frame is 361 tokens/image vs 144 — fewer
  objects per packed row, fewer unique objects per step, and pressure on
  `sequence_length: 4096` p95/p99 object lengths once per-band tokenisation
  (§8) is layered on.
- **Checkpoint compatibility.** Any crop change invalidates every
  checkpoint and downstream number; it is a modelling change demanding its
  own ADR and A/B, a large evidence burden the programme does not need to
  pay.
- **Distribution shift.** Full frame changes what the model sees (more
  neighbours and background per object), entangling a byte-efficiency
  change with a modelling change.

**Consequence accepted with this decision:** the image byte-efficiency
ceiling is fixed. 92.3% of anchor row-group bytes are `image.flux`, of
which 60.1% are fetched and discarded before tokenisation; images stay at
~1.82 KB per fused token. No lever in this programme recovers those bytes.
All remaining gains come from factorisation (replay §7, per-band §8),
projection and partner signal (§6–§6a), scheduling (§10–§10a), and
cross-epoch reuse (§9).

**Reopening clause:** this decision may be revisited through an evidenced
amendment if (a) the sharper periphery measurement shows the discarded
region is predominantly irreducible noise (strengthening the decision) or
predominantly learnable structure (weakening it), or (b) the cache (§9)
lands and the corpus stops being transfer-bound, at which point the
byte-efficiency argument for full frame lapses anyway.

## 2. Metrics: useful bits per byte, and MFU

Every change in this programme is judged on four metrics, not on token
counts:

- **Wire efficiency** `E_values` = valid loss-target scalar values per MiB
  actually fetched. Counts observed target dimensions; replay and per-band
  do **not** increase it.
- **Factorisation efficiency** `E_AR` = loss-bearing AR tokens per MiB
  fetched, decomposed into primary tokens, replay tokens, distinct span
  permutations, and identical duplicates.
- **Model FLOPs utilisation (MFU)** = achieved model FLOPs per second
  divided by the accelerator's peak (bf16, dense — no sparsity credit).
  Defined and decomposed in §2a below.
- **Scientific efficiency** `E_science` = Δ validation quality (family and
  modality losses, fixed redshift/morphology probes) per TiB downloaded
  **and per GPU-hour**, at matched bytes and matched wall clock.

Partner-row scalar attachments are the one r3 lever intended to raise
`E_values`; projection, replay, per-band, scheduling, and caching primarily
change bytes, `E_AR`, or stall time. No change is accepted for raising `E_AR`
or MFU alone — both are gameable by repetition (identical replicas raise MFU
and `E_AR` while adding zero information; so would padding). They are
**necessary-but-not-sufficient** gates; `E_science` is the deciding metric
wherever they conflict.

### 2a. MFU: definition, decomposition, and role

**Definition.** Per step: model FLOPs = FLOPs/token × non-padding tokens in
the step (padding earns no MFU credit — a padded-out packed row is wasted
compute exactly as a stall is wasted time). FLOPs/token uses the standard
decoder estimate (≈ 6·N_params + attention term at `seq_len` 4096) plus the
modality encoder/decoder and jetformer flow/GMM terms, which are **not**
negligible for a 70M backbone with 47 modality heads — the accounting must
include them or small-model MFU will be flattered. Divide by wall time and
peak FLOPs for the reserved GPUs.

**Decomposition (mandatory in every report):**

```text
MFU = MFU_busy × (1 − stall_share) × utilisation_packing
```

- `MFU_busy` — MFU over compute-busy time only (excludes loader waits).
  Moved by representation and model changes: per-band's narrower 64-dim
  heads, kernel efficiency, micro-batch shape.
- `stall_share` — fraction of wall clock waiting on the loader. Moved by
  byte/loader levers: replay, projection, cache, workers.
- `utilisation_packing` — non-padding fraction of packed rows. Moved by
  packing policy and object-length distributions.

The decomposition is the point: an undecomposed MFU comparison between a
fused and a per-band arm confounds a data-pipeline effect with a
model-shape effect and supports the wrong conclusion.

Alongside the three MFU factors, report `loss_bearing_token_fraction`: the
fraction of non-padding tokens whose positions participate in a loss. A
representative galaxies-only anchor is about 255 tokens, of which roughly 72
begin/end specials carry no loss — about 28% delimiter overhead. This is a
diagnostic, not a fourth multiplier in the MFU identity; it makes structural
waste visible before proposing a token-layout change.

**Where the baseline stands — MEASURED 2026-08-06** (superseding the
wall-clock-ratio estimate this paragraph used to carry; full table in
`docs/evidence/adr0014-benchmark-2026-08-06/benchmark.md`). A frozen 120-step
window, dp=2 × 8 workers:

| Arm | mean s/step | MFU | MFU_busy | stall_share | packing |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 (replicas 1) | 8.53 | **2.42%** | 27.29% | 90.9% | 0.979 |
| B1 (replicas 2, adjacent) | 4.06 | 4.98% | 21.63% | 75.9% | 0.954 |
| B1-D (replicas 2, decorrelated) | 3.81 | **5.42%** | 27.67% | 80.0% | 0.980 |

The estimate held: B0 runs at **9.06% of its own stall-free MFU** against the
"roughly 9%" predicted here, with a time-weighted stall share of 90.9%
against the implied ~91%. B0 is frozen as the reference MFU.

**Why MFU belongs alongside the byte metrics.** The byte metrics price the
resource we are short of (wire bytes); MFU prices the resource we have in
surplus but are wasting (reserved FLOPs). They are duals under the
fixed-step arithmetic below: at fixed tokens/step, cutting stall time
raises MFU exactly as it cuts bytes per unit time. A change that raises
`E_AR` without raising MFU (or vice versa) is a flag that something else
moved — packing, model shape, or duplication — and the decomposition says
which.

**Fixed-step arithmetic (applies throughout):** at fixed `train_steps`,
tokens per step are pinned by `seq_len × batch`, so inflating tokens per
object (replay, per-band) proportionally deflates unique objects per step,
epochs per run, and total bytes transferred — and, by shrinking
`stall_share`, raises MFU. Total exposures per object are unchanged
(measured for replay: 8.7 exposures at replicas 1, 2, and 4). The single
governing hypothesis of this programme is therefore: **does quality at
fixed step count survive the reduced unique-object diversity?** Every A/B
in §7–§8 is an instance of that question; MFU and the byte metrics measure
the benefit side, `E_science` measures the risk side.

## 2b. Compute-side follow-ons (after the instrumented baseline)

These are independent of the transfer levers and are deliberately deferred
until §3 provides a trustworthy `MFU_busy` baseline:

- **Fuse scalar heads — proposed, then DROPPED (owner decision,
  2026-08-07).** The idea was to group the ~40 (now 50) tiny `input_size: 1`
  encoder/decoder pairs into a batched GEMM or block-diagonal path, removing
  dozens of small kernel launches. It does **not** share weights between
  scalars — each field keeps its own parameters, exactly as an LLM's embedding
  table holds a distinct row per token id in one tensor, or an MoE layer runs
  separate experts through one grouped GEMM. It was dropped anyway: it renames
  parameters (new checkpoint family), it touches the fork's tied-parameter
  replication of modality modules, and the gain is unprofiled — we do not
  actually know how much of the missing `MFU_busy` is launch overhead. The
  scalars stay unmerged. Record `MFU_busy` before and after A8's seven new
  modalities land; reopen only if that number moves.
- **Standard hygiene:** run `torch.compile` on modality paths and a
  micro-batch shape sweep only after the instrumented baseline exists. Report
  `MFU_busy`, launch/compile effects, and scientific quality separately from
  wire changes.
- **Scalar-span consolidation (parked):** consolidating the 34 `gwh_*`
  fractions into one 34-dimensional span would replace 102 tokens with one,
  but conflicts with ADR 0013's one-field-one-modality rule. Under fixed-step
  arithmetic it also creates more unique objects per step and therefore more
  bytes while the run is transfer-bound. Reopen only after the cache changes
  the regime and a dated governance amendment accepts the semantic trade.

## 3. Instrumentation and benchmark (Phase 0)

Byte accounting through `HfFileSystemFile._fetch_range` is the only
accepted instrument — HTTP-request counting and contiguous-log-run counting
are both recorded failures (range requests are not downloads; 16 workers
interleave one log). Extend the instrument with structured fields: DP rank,
loader worker, source, partition path, anchor cell, byte range and payload
bytes, fetch wait duration, and projected columns where known. Also record
HTTP response bytes, requested/payload byte ratios, effective throughput for
each logical fetch, and the fsspec block/coalescing settings used.

Per step (or short window), record: transferred bytes by source;
non-padding and loss-bearing tokens by family/modality; valid target
dimensions; distinct base object ids; replica count and distinct span-order
count; cells consumed and boundaries crossed; packing utilisation; **step
wall time split into compute-busy and loader-wait; achieved model FLOPs and
the three MFU factors of §2a**; `loss_bearing_token_fraction`; rolling
100-step source composition (the `object_id_log` already carries this); RSS
per process. Persist measured byte cost per cell and its source/partition
breakdown; §10's dealing and §10a's line-rate decision use this record rather
than row-count estimates.

MFU accounting notes: FLOPs/token is computed once per (model config,
tokenisation policy) pair and pinned in the benchmark record — per-band
changes it (more tokens, narrower heads), so arms must not silently share a
constant. GPU-idle attribution uses the trainer's own step timing plus the
loader-wait instrumentation; no profiler in the measurement path (profiler
runs are separate, unfrozen).

Freeze a benchmark: pinned five-spoke index and source revisions, fixed
dp=2 × 8-worker layout, fixed cell order and starting state, one ordinary
window and one HSC-enriched window, several repetitions. Report mean, p50,
p95, p99, stall count, stall share, **and decomposed MFU** — the 0.68 s
median / 7.80 s mean baseline shows the median alone is blind to the
problem, and undecomposed MFU would be equally blind to its cause.

## 4. Validation repair (Phase 0b — gates everything downstream)

> **Amended 2026-08-06 (A1 below): the premise of this section was wrong.**
> `VAL_PARTITIONS` does not govern the five-spoke corpus at all. The
> source-graph path splits on `split_of_cell`, which already reserves
> **252 of 5,488 cells (4.6%), 4.0% of anchors** — above the 1–2% asked for
> here. The split change and its assembly bump are **cancelled**; the
> composition work below stands and is delivered by §3's instrument.

Every acceptance gate below routes through validation quality, and the
validation set has to carry that weight against a corpus whose realised
composition swings 0.1% → 16.4% for HSC between runs (§Mechanism notes).
Therefore, before the benchmark freezes:

- ~~raise `VAL_PARTITIONS`~~ — cancelled; the split is already defensible
  (A1). Confirm per-family val-loss stability across reruns instead, and
  only revisit the split if a family proves unstable;
- land the composition-per-100-step-window measurement. **Delivered by §3**:
  per-modality loss-bearing token counts per step are exact, where inferring
  composition from `object_id_log` cannot tell whether a matched anchor
  carried HSC. One instrument, not two;
- require composition-matched or composition-stratified comparison windows
  for every A/B; decide HSC-touching questions on HSC-enriched cells, since
  0.7% coverage means random short windows cannot evidence that spoke.

## 5. Sequence-assembly fingerprint (immediate, independent of experiments)

`source_assembly` protects record *order* only. A checkpoint saved at
`ar_replicas: 1` and resumed at `ar_replicas: 2` today passes the assembly
check while silently changing the emitted sequence stream. Extend the
resume-state tag to a fingerprint over: source-graph assembly and
revisions; modality-config and scalar-field-registry hash (§6a);
image-crop policy; band-tokenisation policy (§8); `ar_replicas` and
replica-separation policy (§7); span-order
algorithm version; sequence length. Mismatch rejects the stream state
(weights still load), exactly like an assembly bump. This protects every
A/B in the programme from contaminated resumes.

## 6. Nested leaf and HSC-image projection (ship now)

Change `_SOURCE_COLUMNS["desi"]` and `["sdss"]` from the whole `spectrum`
struct to the leaf paths `spectrum.flux`, `spectrum.lambda`,
`spectrum.mask`, and change `_spectrum_part()` to a whitelist of those
three children rather than copying every struct child present.

Measured saving: 40% of spectrum bytes per read (60.7 → 36.4 MB, counted at
`_fetch_range`). **Gate met and exceeded, 2026-08-06** (A2): on a 436.8 MB
single-row-group DESI partition (`Npix=2356`) the projected read pulls
229.6 MB against 436.9 MB whole-struct — **47.5%** — and SDSS, measured
independently on `Npix=268`, saves **57.9%** (72.9 → 30.7 MB). Both beat the
40% headline because the unread siblings (`lsf_sigma` and friends) drop with
`ivar`; footer overhead is 0.01 MB and does not scale with partition size.
Demoted, correctly, from headline to hygiene: DESI is 6.3%
of five-spoke anchors (was 92.2% at three spokes), so the corpus-level
effect is small — but it is free, reversible, and scientifically neutral
(`ivar` is read nowhere outside synthetic fixtures; ADR 0007 normalisation
does not use it).

Gates: decoded records byte-identical before/after; the saving re-verified
against a **520 MB single-row-group partition**, not only the 60.7 MB one
(footer and range-coalescing overheads need not scale linearly); SDSS
saving measured independently; record order and resume unchanged.

### HSC image-struct projection (verifiable from published metadata; ship now)

`_SOURCE_COLUMNS["hsc"]` currently requests the whole `image` struct, while
`attach_source` consumes only `image.flux` and `image.band`. Project the
consumed leaves instead; include `image.psf_fwhm` or `image.scale` only if a
future normalization actually uses them. This drops the unread `image.ivar`
and `image.mask` planes from the wire. The published 335.6 GiB / 156-partition
catalog size and the equal raw `flux`/`ivar` plane sizes make the source-local
saving verifiable; the exact compressed/HTTP saving remains a §3 measurement.

Gate it exactly like the spectrum projection: compare `_fetch_range` bytes on
an HSC-enriched window and a representative large partition, keep decoded
flux and band values byte-identical, verify optional normalization fields are
not silently required, and require unchanged record order and resume state.
Do not generalise the result from the 2.2 GiB partition average to the corpus
without measuring HSC composition.

> **Measured and shipped 2026-08-07 (A7).** 47.0% of a 1,849 MB HSC partition
> is `image.ivar`, never read. This section's field list was correct; `mask`
> exists but compresses to 0.001%, so the saving is `ivar` alone. See A7 for
> the column table and the Legacy null result.

## 6a. Partner-row scalar attachments (Phase 1)

Rows already fetched for a matched partner can carry useful scalar targets at
near-zero marginal wire cost. HSC rows expose 50+ candidate scalar columns;
candidate HSC fields include
`{g,r,i,z,y}_cmodel_mag`, their `magerr`s, extendedness, and extinction
`a_*`; DESI candidates include `FLUX_*` and `EBV` photometry. Add fields one at
a time under ADR 0013's field-predicate governance: documented units,
provenance, fixed transform and inverse, missingness, valid range, quality
predicate, and source revision. A field that fails its predicate is omitted;
there are no standalone partner scans.

This lever raises `E_values`, not merely `E_AR` or MFU. Measure its actual
incremental `_fetch_range` bytes, valid values per MiB, object-length and
packing effects, family loss share, and validation/scalar metrics. Accept a
field only when the marginal bytes are consistent with the fetched-row
assumption, ADR 0013's family weighting remains bounded, and
`E_science` improves or holds at matched bytes and GPU-hours.

> **First batch adopted 2026-08-07 (A8):** seven modalities over 12 columns,
> from the Legacy anchor as well as the HSC partner, chosen against measured
> distributions and with 45 candidate columns refused as redundant or as pure
> measurement error. The `E_science` acceptance above still governs — A8 is
> the shortlist and its evidence, not a quality verdict.

## 7. Replay validation (`ar_replicas`) — the headline lever

With the crop retained, replay is the largest measured lever in scope. It
is already built: `ar_replicas` in `nanotron_loader.py` re-emits each
downloaded record under a different ADR 0008 span order — the shuffle is
already a pure function of `(object_id, epoch)`, so the suffixed id *is*
the reseed, with no new RNG and nothing extra to checkpoint. Replica 0
keeps the original `object_id`, replicas get `#n`, so the no-replay audit
still catches accidental duplication. Measured over identical first 117
steps, five-spoke, 8 workers, dp=2: `ar_replicas: 1` → 7.80 s/step mean,
21/116 slow; `ar_replicas: 2` → 2.51 s/step mean, 11/116 slow — **~3.1×
faster per step with slow steps halved**, an implied stall-share drop from
~91% to ~72% and a ~3× MFU improvement, to be confirmed by the §3
instrument. It is promoted to production default only through the
following gates.

**7a. Distinct-order enforcement.** Suffixing the object id reseeds the ADR
0008 shuffle but does not guarantee a different permutation. Before
emitting a replica: skip replay for one-span records (no alternative order
exists); cap replicas at the number of distinct useful permutations;
enforce that emitted replicas carry distinct span orders. Note the corpus
asymmetry: 94% of anchors are galaxies-only with ~37 spans (image + 34
`gwh_*` + ebv + photometry) — a huge permutation space where replay is most
defensible — while unmatched two-span records support at most one extra
ordering. This gate is precisely what keeps replay's MFU gain honest:
identical duplicates would raise MFU and `E_AR` while training on nothing
new.

**7b. Decorrelation.** Replicas currently land adjacent in the same packed
row; document masking prevents cross-attention but not gradient repetition
within the batch. Add a deterministic bounded buffer that separates
replicas from their base record — at most one replica of a base object per
packed row, preferably per micro-batch — reconstructible from the saved
stream position (or carried in checkpoint state) so resume stays exact.

**7c. Quality A/B.** Arms: replicas 1 (B0); replicas 2 adjacent (B1 —
continuity with the measured result); replicas 2 decorrelated (B1-D).
Replicas 4 is refused until B1-D passes. Compare at matched bytes and
matched wall clock on the repaired validation set. Accept if quality per
TiB **and per GPU-hour** is preserved or improved, the audit stays
exactly-once (replica 0 keeps the original id; every logged line unique),
resume stays exact, and most of the measured MFU/stall gain survives
decorrelation.

> **Throughput half measured 2026-08-06 (A6); the quality half is still
> open.** Decorrelation does not merely preserve the gain — B1-D beats B1 on
> every axis (MFU 5.42% vs 4.98%, mean step 3.81 s vs 4.06 s, p99 34 s vs
> 46 s). The decomposition attributes it: adjacent placement stacks a
> record's replicas into one row and fragments the row tail, dropping packing
> to 0.954 and `MFU_busy` to 21.63%, while B1-D's `MFU_busy` of 27.67%
> matches B0's 27.29% as identical model shapes must. `E_values` stays flat
> across all three arms and `E_AR` doubles with primary held constant, so the
> gain is not the repetition artefact §2 warns about. Exactly-once holds on
> every arm. The replay factor measures 1.99×, not 2.00× — §7a correctly
> withholds a replica from one-span records. **Still required before B1-D is
> adopted:** the `E_science` comparison, composition-matched windows (the B0
> window drew HSC, the B1 windows did not), and repetitions.

## 8. Per-band tokenisation — on the cropped image

With full frame out of scope, per-band is the only image-representation
experiment: 144 × 192-float fused tokens → 432 × 64-float per-band tokens
from identical bytes. State it correctly: a finer autoregressive
factorisation with narrower heads — `E_AR` rises 3×, `E_values` does not
move (27,648 target values either way), and cross-band structure is learned
autoregressively instead of pre-fused.

MFU note: per-band moves **two factors in opposite directions**. It
shrinks `stall_share` (3× tokens per object → fewer unique objects per
step → fewer boundary fetches) but may lower `MFU_busy` (64-dim heads and a
narrower jetformer flow do less arithmetic per token against the same
kernel overheads; more tokens per object also lengthens attention). The
decomposed report decides whether the net is a win; an undecomposed MFU
number would hide the trade.

> **Both factors measured, 2026-08-06 (arm P0, throughput only).**
> `stall_share` 90.9% → **76.2%**; `MFU_busy` 27.29% → **25.14%** (−7.9%
> relative); `utilisation_packing` 0.979 → 0.970 at 355 non-padding tokens
> per emitted sequence, so the `seq_len` length gate is comfortable. Net MFU
> 2.42% → **5.81%** at identical bytes, `E_AR` 388 → 1,076 per MiB (2.77×,
> below 3× because scalar spans do not triple), `E_values` flat. Amendment A5
> sharpens the `MFU_busy` explanation: the heads are exactly cost-neutral, so
> the loss is kernel efficiency at width 64 and attention length, not reduced
> arithmetic. **This is not acceptance** — §8 accepts only on `E_science`,
> and the window was not composition-matched to B0.

Design for the first A/B: one modality per survey image; **fixed** band
order (g,r,z; HSC g,r,i,z,y if it ever reaches this stage), centre-out
spatial order within band; no band shuffling (one variable at a time — band
order as an ADR 0008 ordering decision is a follow-up). Config carries
`channel_tokenization: fused | per_band` and `band_order`; 432 positions
for the Legacy crop; the fingerprint (§5) covers it; checkpoints
incompatible — new configs, never edits to historical ones.

Gates: p95/p99 object length and packing utilisation against `seq_len`
4096 (a cropped per-band image object is ~437 tokens before scalars — fine
alone, check the packed distribution); jetformer flow/GMM behaviour at
`input_size` 64, including the noise curriculum's useful range; validation
loss **per pixel** at matched bytes, probes at matched wall clock;
decomposed MFU with per-arm FLOPs/token accounting (§3). Accept only on
`E_science`, never on token count or headline MFU. If accepted, combine
with the §7 winner (arm P1) and gate the combination the same way.

## 9. Transient cross-epoch cache (conversation now, prototype last)

Each epoch re-downloads the corpus; at ~262 KB per anchor row group,
2,182,875 anchors ≈ 0.57 TB/epoch (indicative, from single-partition
sampling), and 8.7 exposures per object over 20k steps ≈ 5 TB of anchor
traffic without reuse. This dominates every single-epoch optimisation for
any multi-epoch future — and it is the only lever in the programme that can
push MFU toward compute-bound territory rather than trimming the stall
share, since epoch-2+ reads come from local disk. It is in tension with
ADR 0006 (live streaming, reshard deleted) and ADR 0011 (no local cache).
Open the amendment conversation **now** (longest lead time); build
**last**.

Framing: a *transient* cache — evictable, reconstructible, no schema of its
own — never a reshard or second corpus. Keyed by pinned revision, path,
byte-range/projection, and decode version; checksum-validated;
byte-bounded; safe under concurrent readers. Partial-by-construction: at
dp=2 a full per-node share is ~285 GB (likely infeasible); at dp=32 it is
~18 GB (trivial); hottest-cells-first under a byte budget covers both.

Prerequisite: **stable cell ownership.** Today's shuffle-then-deal changes
which rank sees a cell every epoch, defeating node-local reuse. Evaluate
stable cell→rank ownership with shuffling only within each rank's set
(assembly bump; composition and balance measured — the composition-variance
concern in §Mechanism notes interacts here). Go/no-go: ≥80% epoch-2 byte
reduction, bounded disk/RSS, no cross-split or duplicate emission,
acceptable rank balance, and a measured epoch-2 MFU report (expected to be
the largest single MFU movement in the programme).

**Cache-era decode path (inferred; build with the cache prototype).**
`_rows()` currently materializes every row with
`table.slice(j, 1).to_pylist()[0]`, including large nested image lists. That
Python conversion is hidden by today's network stalls but becomes the next
loader bottleneck when epoch-2 reads are local. Replace it then with
batch-level Arrow→NumPy conversion, using zero-copy `to_numpy` on primitive
leaves such as `image.flux` where Arrow permits; keep the row adapter only at
the schema boundary. Gate the change on decoded-value identity, RSS, resume,
and a measured reduction in post-cache decode/loader time. Do not build this
before the cache prototype makes the CPU cost visible.

## 10. Worker and scheduling policy

Baseline: 8 loader workers, exactly one large read in flight per worker, no
cell-prefetch thread, no fsspec background cache. Worker count is a large
lever (2 workers: 19.95 s/step vs 9.02 at 8, because a ~100 s fetch
amortises over `num_workers` steps of the loader rotation) — never reduce
it as a congestion workaround; the prefetch is fatal at 8 workers (NCCL
watchdog kill) and merely useless at 2. After the representation
experiments, a bounded sweep (4/8/12) and, separately, a controlled
ready-first delivery experiment are permitted — noting ready-first changes
record order (assembly bump) and biases short-window composition toward
fast cells. Adaptive replay *during* a measured boundary wait (replaying
only while the single next-cell fetch is incomplete) is the designated
successor to fixed replay, after §7 passes — it preserves unique-record
rate whenever data is ready and converts residual stall time to MFU
directly.

**Byte-balanced cell dealing (inferred; after Phase 0).** The current
count-based deal can put several expensive HSC-referencing cells on the same
worker even when another worker receives cheap, galaxy-only cells. Once §3
has measured per-cell byte costs, use the match-index partner partition
membership as a prior, greedily bin-pack cells by measured bytes rather than
count, then interleave heavy and cheap cells within each worker's queue.
This changes no read concurrency; it only prevents one worker's rotation slot
from repeatedly owning the heavy tail. It is a scheduling
change, not a second prefetch path.

Because cell order is part of the stream contract, bump `SOURCE_ASSEMBLY` and
re-measure composition, rank/worker balance, resume, RSS, p95/p99 step time,
and stall share. Build it only if the five-spoke evidence shows lower tail
stalling without skewing family exposure; otherwise keep the simpler
count-based deal.

Before any locality/block-ordering work: re-measure partner-fetch
redundancy on the 5,488-cell index with the §3 instrument (the 39% /
434-vs-266 figures are three-spoke and superseded; the 8–16-cell block
suggestion is smaller than the 16 consumers of a dp=2 × 8 run and buys
nothing — mechanically, `shuffled()` runs *before* `owned_by_rank` deals
cells round-robin to `dp × num_loading_workers` consumers, so any block
smaller than the consumer count has its locality stripped on arrival). Only
build if the five-spoke trace shows a material byte opportunity at
consumer-aware block sizes — the null result of the reverted block-shuffle
run and DESI's exit from the byte budget both argue it will not.

## 10a. Segmented single-fetch experiment (bounded, inferred)

This is not the failed prefetch. Test whether one large HSC column-chunk
fetch is below link rate by segmenting that one logical fetch into bounded
parallel range requests (an `hf_transfer`-style implementation), without
adding another logical column read or another large read per worker. First
compare aggregate throughput at the fixed eight-worker layout with the
available NIC capacity. A single stream at roughly 100–200 Mbit/s on a
1 Gbit/s link is evidence for the experiment; a stream already at line rate
is a stop condition.

In the same bounded experiment, audit fsspec block size and readahead against
the actual column-chunk offsets, using payload-versus-HTTP bytes to expose
over-fetch. Compare the ordinary and segmented arms on an HSC-enriched
window with fixed order, workers, and starting state. Proceed only if there is
measured line-rate headroom and segmentation lowers stall share or p95/p99
tail time without increasing total bytes, RSS, errors, or concurrent logical
reads; decoded values and resume must remain exact.

## 10b. Upstream publication request (send now)

Extend the MMU publisher request to include:

- `BYTE_STREAM_SPLIT` for float columns, with compression settings and
  representative before/after sizes recorded;
- approximately 64 MB row groups and page indexes; and
- splitting the outlier HSC partitions, whose published average is about
  2.2 GiB, into smaller independently fetchable chunks.

The request should quantify the current PLAIN+ZSTD result (about 5% on
`image.flux`) against the expected 15–30% float32-imaging improvement from
`BYTE_STREAM_SPLIT`. Treat that replacement ratio as an upstream measurement,
not a client-side acceptance assumption: if achieved, it is a corpus-wide
wire reduction unavailable to local projection changes.

## 11. Stop list (refused, with reasons)

- **Full-frame and foveated tokenisation** — owner decision, §1.
- **Cell-boundary prefetch thread** — not merely neutral: at 8 workers on
  the five-spoke corpus it hung a rank for 20 minutes and NCCL's watchdog
  killed the job. It adds a *second* concurrent large read per loader
  process (~32 in flight instead of 16) and creates no bandwidth.
- **fsspec `cache_type=background` + pyarrow `pre_buffer`** — within noise
  of baseline (3.73 vs 2.88 s/step mean, identical slow-step count).
- **Row-group / page skipping on published files** — every DESI partition
  is a *single* row group (up to 520 MB) and no page index is written;
  there is no sub-file granularity to skip to. Remains an external publication
  request in §10b, which also asks publishers to split the outlier HSC
  partitions; it is not a client-side build.
- **Request-count or log-run "redundancy" metrics** — recorded measurement
  failures (HTTP requests count *ranges*, and 16 loader processes
  interleave in one log with no worker id).
- **Headline (undecomposed) MFU as an acceptance criterion** — gameable by
  duplication and padding, and confounds pipeline effects with model-shape
  effects; only the §2a decomposition is admissible.
- **Dropping unused anchor scalars** — <0.1% of the row group; keep them.
- **Filtering "empty-sky" patches** — refuted by the §Context periphery
  measurement (and §1's evidence note applies equally here).
- **HATS index tables** — solve partition lookup, which the match index
  already provides.
- **Standalone unmatched-source scans; strict N-way joins** — ADR 0013
  fetched-only policy stands; unmatched rows are kept (36.3% of tokens at
  zero marginal byte cost; dropping them saves no wall clock — and they
  raise MFU for free, since the stall is the fetch, not the step).
- **8–16-cell block shuffles from the 181-cell simulation** — superseded
  numbers, the consumer-count error above, and an inconclusive live run
  (block 256 over 36 steps: identical 0.69 vs 0.68 s median, identical
  slow-step count; reverted unmerged in favour of replay — no signal
  either way, and no five-spoke re-measurement yet).
- **`ar_replicas: 4`** — until B1-D passes distinctness and correlation
  gates.

## Experiment matrix (gated, in order)

| Arm | Image | Bands | Replay | Gate to run |
| --- | --- | --- | ---: | --- |
| B0 | 96 crop | fused | 1 | frozen baseline (reference MFU) — **throughput measured 2026-08-06** |
| B1 | 96 crop | fused | 2 adjacent | reproduce measured result — **throughput measured** |
| B1-D | 96 crop | fused | 2 decorrelated | after §7a/7b land — **throughput measured; `E_science` open** |
| P0 | 96 crop | per-band | 1 | after B-arms decided — **throughput measured; `E_science` open** |
| P1 | 96 crop | per-band | §7 winner | only if B1-D and P0 both pass — **not run** (passing means `E_science`) |

Configs: `configs/nanotron/bench-north-5spoke-{b0,b1,b1d,p0-perband}.yaml`;
run with `scripts/run_benchmark.sh`, report with `scripts/bench_report.py`.
Throughput results: `docs/evidence/adr0014-benchmark-2026-08-06/benchmark.md`.

Each arm reports the §3 per-step record plus `E_values`, `E_AR`
(decomposed), **decomposed MFU (MFU_busy, stall_share,
utilisation_packing, with per-arm FLOPs/token)**, and `E_science` at
matched bytes and matched wall clock.

### R3 lever matrix

| Rank | Lever | Confidence | Primary movement | Placement |
| ---: | --- | --- | --- | --- |
| 1 | HSC image-struct projection | **Measured: 47.0% of an HSC partition** | bytes / stall share | §6, **shipped 2026-08-07 (A7)** |
| 1b | Legacy image-struct projection | **Measured: 0.0% — null result** | none | §6, **refused (A7)** |
| 2 | Partner-row and anchor scalar attachments | **Measured, 7 modalities adopted** | `E_values` | §6a, **shipped 2026-08-07 (A8)**; `E_science` open |
| 3 | Byte-balanced cell dealing | Inferred | stall share / tail steps | §10, after Phase 0 cell costs |
| 4 | Segmented single-fetch download | Inferred | stall share / tail steps | §10a, only below line rate |
| 5 | Vectorised decode | Inferred | post-cache stall share | §9, with cache prototype |
| 6 | Fused scalar heads and MFU hygiene | Inferred | `MFU_busy` | §2b, after baseline |
| 7 | Upstream publication changes | Measured current / inferred gain | all byte metrics | §10b, send now |

The cache remains the dominant regime-changing lever; this table only shortens
the transfer-bound interval before it lands.

## Mechanism notes worth keeping

- **Stalls arrive on the loader rotation, not at random.** At
  `num_loading_workers: 8`, every >60 s stall from step 80 landed at exactly
  `step % 8 == 6`. Each worker's cell-boundary fetch blocks its own turn and
  nothing amortises it. This is why adding workers cannot help: each new
  worker adds its own serialised fetch to the same rotation.
- **The partition ceiling is gone.** `num_loading_workers <= floor(cells/dp)`
  bound the pilot recipes at 165 train cells; at 5,488 cells it no longer
  binds, and the deferred "shard by row group instead of by cell" loader
  change is not worth building.
- **Unmatched rows stay** (ADR 0005/0011), and the economics back it: 36.3%
  of training tokens come from rows never fetched for their own sake, at
  zero marginal bytes, spending compute we have in surplus. The real threat
  is **composition variance**, not the matched/unmatched ratio — HSC was
  0.1% of objects in one run and 16.4% in another because its matches live
  in a small part of the footprint. Per-family loss weights normalise
  *within* a batch, so a stretch of spectrum-only batches trains no image
  head at all. Block ordering would make this worse. `object_id_log`
  already records what is needed to measure it (§4 lands that measurement).

## Caveats

Column shares come from one row group of one partition per catalogue; bytes
are compressed on-disk sizes, and pyarrow also fetches footers and coalesces
ranges, so real HTTP bytes are somewhat higher. Runs compared here hit a
shared hub at different times — the *absence* of benefit is the finding, not
the ordering between arms.

## Consequences

- The image byte-efficiency ceiling is accepted: ~1.82 KB per fused image
  token and a 60% pixel discard stand for the life of this decision (§1
  reopening clause notwithstanding).
- The programme now carries a compute-side ledger as well as a wire-side
  one. Expected movements, to be confirmed by instrumented MFU: baseline
  ~9% of stall-free MFU; replay-2 ~3× that; the cache the only path toward
  compute-bound MFU; per-band an open trade between stall share and
  MFU_busy.
- The programme's expected byte wins are: replay ~2× fewer unique-record
  bytes per fixed-step run (measured ~3.1× wall-clock mean); per-band up to
  a further token-inflation factor **if and only if** quality per pixel
  holds; DESI/SDSS projection as a free hygiene win; HSC projection at the
  source-local ~half-payload ceiling, with corpus impact set by HSC
  composition; and the cache, if adopted, ~an epoch's bytes for every epoch
  after the first — the dominant term for multi-epoch runs. Scalar
  attachments instead add `E_values` at marginal row bytes and add bounded
  sequence/head overhead.
- Validation cost: `VAL_PARTITIONS` does not rise for the five-spoke path;
  every arm still pays for composition-controlled windows, and any
  byte-balanced order pays an additional composition/resume re-measurement.
  This is the price of decisions that mean anything on a corpus whose
  modality mix swings 160× between short runs.
- One assembly bump remains scheduled for stable ownership in §9; any
  byte-balanced dealing also bumps `SOURCE_ASSEMBLY`. Both reject stale
  stream states while keeping weights loadable, per the documented mechanism.
- Old checkpoints are unaffected by everything except per-band adoption,
  which creates a new model family by design.

---

## Implementation amendments (2026-08-06)

Recorded when Phase 0/0b, §7a/7b and §8 were built. Each is a place where
the code found the plan wrong or under-specified; none changes the
programme's direction.

**A1 — §4's premise was wrong; one of the two scheduled assembly bumps is
cancelled.** The five-spoke corpus runs the *source-graph* path
(`_source_graph_dataset`), which splits on `split_of_cell` — crc32 buckets
over order-4 parents, 1-in-20 — not on `VAL_PARTITIONS`. Measured on the
live index: 252 of 5,488 cells (4.6%) and 4.0% of anchors are val, already
above the 1–2% §4 prescribed. `VAL_PARTITIONS = 8` binds only the retired
DESI-only path. Consequence: no split change, no assembly bump, and
§Consequences' "two assembly bumps are scheduled" becomes one (stable cell
ownership, §9). The composition measurement §4 wanted is delivered by the §3
instrument as per-modality token counts per step.

**A2 — §6's saving is larger than estimated, and the large-partition gate is
met.** 47.5% on a 436.8 MB single-row-group DESI partition, 57.9% on SDSS,
both measured through `_fetch_range` (see §6). Still hygiene, not headline —
DESI is 6.3% of five-spoke anchors — but a bigger free win than booked.

**A3 — the checkpoint unit moved from the partial row to the partial
micro-batch, and B1 needed a placement switch.** §7b requires a base
object's replicas to land in different packed rows, so the loader now opens
all `micro_batch_size` rows at once and assigns emptiest-first. That makes
the natural resume unit the open micro-batch rather than the open row —
equally exact, since nothing in an open batch has been yielded, and simpler
than the old partial-row bookkeeping. Two consequences the ADR did not
anticipate:

- packing at `ar_replicas: 1` is no longer byte-identical to the historical
  runs (worst-fit across open rows instead of next-fit down one). B0 is
  being re-frozen as the reference anyway, and `utilisation_packing` is
  reported per arm, so the change is visible rather than hidden;
- B1 ("2 adjacent") and B1-D ("2 decorrelated") would otherwise be the same
  config. A `replica_placement: adjacent | decorrelated` knob keeps B1 as a
  real continuity arm; the fingerprint covers it, so the arms cannot resume
  onto each other.

**A4 — §7a's distinctness cap binds harder than §7a implies, and MFU has a
stated approximation.** A record's replicas are capped at `n_spans!`, so the
94%-of-anchors galaxies-only case (~37 spans) is unconstrained while a
two-span unmatched record supports exactly one extra ordering and a one-span
record supports none — `ar_replicas: 4` therefore does *not* mean 4× tokens
on the unmatched tail. Separately, the MFU backbone term is scaled linearly
by the non-padding fraction although attention is quadratic in `seq_len`;
this errs toward over-counting (document masking already makes real
attention block-diagonal), so reported MFU is an upper bound. Model the
block structure only if an arm is ever decided on the attention term alone.

**A5 — per-band's compute cost is entirely in the backbone; the heads are
cost-neutral.** Measured from the implemented accounting: image FLOPs per
token fall to **6.92e5 from 2.07e6** — exactly 1/3, because encoder, GMM head
and flow all scale linearly in `input_size` — while tokens triple. The
modality modules therefore cost the *same* per object; every extra FLOP is
144 → 432 tokens through the transformer body. §8's warning that per-band
"may lower `MFU_busy`" is right but for a sharper reason than stated: not
less arithmetic per token against fixed kernel overheads in aggregate, but
**kernel efficiency at width 64** and a longer attention span per object.
Object length measures 463 mean / 477 p95 against `seq_len: 4096` (~8 objects
per row), close to §8's ~437 estimate.

**A6 — two arithmetic traps in the reporting, both found by measurement.**
Neither changes the instrument, both changed the answer:

- **`stall_share` must be time-weighted.** A mean of per-step ratios reported
  12.4% where the run actually paid 90.9%. On a corpus this bimodal — most
  steps stall for nothing, a few for a minute and a half — averaging ratios
  buries exactly the behaviour under measurement. Same for MFU: aggregate
  `total_flops / total_wall`, never a mean of per-step MFU.
- **`utilisation_packing` is already inside the flops.** Since padding earns
  no credit, `MFU_busy` computed from those flops double-counts the packing
  factor and the §2a identity does not close. `MFU_busy` must be deflated by
  packing to be the full-occupancy number the decomposition means. Both traps
  are now pinned by `tests/test_bench_report.py`, and the report prints an
  identity check.

A third, smaller: `E_values` counted *presented* target dimensions, so it
doubled under replay — the exact immunity §2 requires it to have. It is now
deflated by the measured replay factor, and both figures are reported.

**Also recorded:** the surviving five-spoke run config uses
`ar_replicas: 4`, which §11 refuses until B1-D passes. That is deliberate —
it is a *running job*, not a default. The benchmark arms are new configs
(`bench-north-5spoke-{b0,b1,b1d,p0-perband}.yaml`) and the running config
was left untouched.

---

## Implementation amendments (2026-08-07, r4)

**A7 — image-struct projection measured on both catalogues: HSC is a large
win, Legacy is a null result.** Column-chunk sizes from the parquet footers of
one representative partition per catalogue, so these are exact on-disk
compressed shares, not estimates:

| Catalogue | Partition | Size | `image.flux` | `image.ivar` | `image.mask` | Other leaves |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| HSC | `Npix=70744` (2,394 rows) | 1,849 MB | 53.001% | **46.961%** | 0.001% | 0.003% |
| Legacy North | `Npix=5072` (2,465 rows) | 645 MB | **99.973%** | — | — | 0.002% |

Two findings, one of which corrects §6 and one of which corrects an earlier
draft of this amendment:

- **§6's field list was right; an earlier draft of A7 wrongly "corrected" it.**
  The HSC struct is `band`, `flux`, `ivar`, `mask`, `psf_fwhm`, `scale`,
  exactly as §6 said. The draft claim that there is no `mask` plane came from
  a truncated schema print and is withdrawn. §6's hedge on `mask`
  ("compressible, but not free") is also settled by measurement: it is 0.011 MB
  in a 1,849 MB partition, i.e. free. **The entire saving is `ivar`**, and
  projecting to `image.flux` + `image.band` halves every HSC fetch.
- **Legacy image projection saves nothing and is not built.** The Legacy
  `image` struct is `band`, `flux`, `psf_fwhm`, `scale` — no `ivar`, no
  `mask`, no second plane of any kind. At
  99.973% `flux`, the projectable remainder is 0.014 MB per 645 MB partition.
  Writing the whitelist anyway would add a code path and a test to defend
  0.0%, which is the kind of hygiene that is only hygiene when it is free.
  Recorded here so the question is not reopened; if upstream ever adds an
  `ivar` plane, the HSC whitelist is the pattern to copy.

The **anchor's** `psf_fwhm` is now consumed as a scalar target (A8); it rides
in free inside the unprojected Legacy `image` struct. HSC's own `psf_fwhm`
stays off the wire, since projecting to leaves nothing reads is the whole
point of the change.

**A8 — free scalar attachments: 7 new modalities from 12 already-fetched
columns, chosen against measured distributions.** §6a's premise holds and is
now exercised on both the anchor and the HSC partner: the rows are already on
the wire, so these fields raise `E_values` at a marginal byte cost of 0.0%
(each is ~0.01 MB per partition against 645–1,849 MB of pixels). §6a asked for
one field at a time under ADR 0013's field-predicate governance; what follows
is the shortlist, the evidence for each cut, and the predicates.

*Grouping.* Bands of the same physical quantity are one modality with
`input_size = n_bands`, following the existing `photometry` precedent
(`input_size: 3` over `flux_g/r/z`) rather than ADR 0013's
one-field-one-modality rule, which exists to stop *unlike* quantities being
lumped together (the `gwh_*` case in §2b). This turns 19 candidate columns
into 7 heads and 21 tokens instead of 57.

| Modality | Size | Source columns | Coverage | Transform |
| --- | ---: | --- | ---: | --- |
| `fiberflux` | 3 | `fiberflux_{g,r,z}` | every anchor | `arcsinh(f/0.01)` |
| `psfdepth` | 3 | `psfdepth_{g,r,z}` | every anchor | `log10(1+x) − 2.5` |
| `psf_fwhm` | 3 | `image.psf_fwhm` | every anchor | `x − 1.5` |
| `hsc_cmodel_mag` | 5 | `{g,r,i,z,y}_cmodel_mag` | HSC | `(m − 21.5)/2` |
| `hsc_extendedness` | 1 | `i_extendedness_value` | HSC | `2x − 1` |
| `hsc_shape` | 3 | `i_sdssshape_shape{11,22,12}` | HSC | `arcsinh(x/0.3)` |
| `hsc_psf_shape` | 3 | `i_sdssshape_psf_shape{11,22,12}` | HSC | `arcsinh(x/0.1)` |

Every transform is fixed, invertible and carries no corpus-fitted constant,
per the ADR 0007 discipline `scalar_registry.py` already enforces. The knees
are pinned to published units, and the percentiles below are reported only to
show each transform lands O(1) — they are not fitted to.

*Evidence for what was kept* (n ≈ 2,400 objects per catalogue, one partition
each):

- `fiberflux` correlates only **0.64–0.67** with the `flux_{g,r,z}` we already
  train on. Fibre-aperture flux against total flux is a light-concentration
  measure, so this is genuinely new information rather than a rescaling. Same
  units as `photometry` (nMgy), so it takes the identical band-registry knee.
- `hsc_cmodel_mag` has signal-to-noise **89–254** (population spread over
  median reported error) across the five bands. The 22.5-mag DUD cut means
  nothing here is faint. Kept in all five bands.
- `hsc_shape` second moments: about **80% of objects are resolved** in every
  band (`shape11 > 1.5 × psf_shape11`). The distribution is heavy-tailed
  (p50 0.28, p99 24.7 px²) and `shape12` is signed, so `arcsinh` on a 0.3 px²
  knee is the transform that keeps both tails and the sign.

*Evidence for what was refused*, all of it free on the wire and refused
anyway, because a target that cannot be learned is an irreducible loss term
and three wasted tokens:

- **`{g,r,i,z,y}_cmodel_magerr` (5 columns).** Median 0.004–0.013 mag — this
  *is* the noise, not a property of the object. Refused as a target but
  **fetched and used as a quality predicate** (see below), which is the honest
  use of an error column.
- **`a_{g,r,i,z,y}` (5 columns).** Their pairwise correlation is exactly
  **1.0000**: one dust map times a fixed per-band coefficient. `ebv` on the
  anchor is that same map a third time. Five columns carrying one number.
- **`{g,r,z,y}_extendedness_value` (4 columns).** The value is binary (only
  0.0 and 1.0 occur; 85.5% are 1.0) and near-identical across bands. One band
  is the whole field.
- **Object shape moments in four of five bands (12 columns).** ~80% resolved
  in *every* band, so these are twelve correlated views of one shape. `i` is
  HSC's reference band.
- **PSF shape moments in four of five bands (12 columns).** Same argument.

*Telescope measurements are included by owner decision (2026-08-07).* An
earlier draft refused `psfdepth`, `psf_fwhm` and the PSF shape moments on the
grounds that they describe the observing conditions rather than the object,
and are largely predictable from sky position — a model that learns them has
learned the survey footprint. The counter-argument, which the owner took, is
that they are exactly what a model needs in order to *separate* instrument
from sky, they cost nothing on the wire, and refusing them presumes an answer
we have not measured. Two honesty notes attach to the decision:

- these are the fields most likely to inflate a headline scalar-family loss
  improvement while teaching nothing about galaxies, so scalar-family loss
  must be read per modality, which §3's instrument already does; and
- `hsc_psf_shape` is a **very narrow** target (p1–p99 spans 0.094–0.147 px²
  on `shape11`). A GMM head can score well on it by predicting a constant.
  If it is ever cited as evidence of anything, it needs a
  predict-the-marginal baseline first.

*Predicates* (ADR 0013 governance; a field failing its predicate is omitted,
not defaulted, and a group is all-or-nothing because `_scalar_value` returns
`None` if any member key is missing):

| Modality | Predicate |
| --- | --- |
| `fiberflux` | all three finite |
| `psfdepth` | all three finite and `> 0` |
| `psf_fwhm` | all three finite and `0 < x < 5` arcsec |
| `hsc_cmodel_mag` | all five finite, `10 < m < 30`, **and** the band's `cmodel_magerr` finite and `< 0.1` mag |
| `hsc_extendedness` | finite and in `{0, 1}` |
| `hsc_shape` | all three finite |
| `hsc_psf_shape` | all three finite |

The magerr screen passes **99.8%** of rows in the measured partition, so it is
a guard against pathological rows rather than a cut.

*Costs, stated up front.* Sequence length grows by 9 tokens on every anchor
(3 modalities × 3 tokens) and a further 12 on HSC-matched rows; against §2a's
~255-token galaxies-only anchor that is ~3.5%, and it worsens the 28%
delimiter overhead §2a already flags. Scalar heads go from 43 to 50. §2b's
head-fusion idea was **considered and dropped** on 2026-08-07: it would rename
parameters (new checkpoint family) and touch the fork's tied-parameter
replication, for an unprofiled gain. The scalars stay unmerged; `MFU_busy` is
recorded before and after these fields land, and fusion is reopened only if
that number moves.

*Model-family consequence.* Seven new modalities means 21 new token ids
(`vocab_size` 145 → 166) and seven new head/encoder pairs. Checkpoints from
before this change are a different model family and cannot be resumed onto —
the §5 fingerprint covers the modality-config hash and rejects the stream
state, and the parameter-shape change stops the weights loading, which is the
correct outcome rather than a silent one. Record *order* is unchanged, so
`SOURCE_ASSEMBLY` does **not** bump.

*Caveat carried from §Caveats.* Every number above is one partition per
catalogue, one patch of sky. The extinction degeneracy and the magnitude
signal-to-noise are structural and will hold corpus-wide; the shape and
resolved-fraction statistics are the least safe to generalise, and the HSC
figures describe 0.7% of the corpus.

## Implementation amendments (2026-11-17, r5)

**A9 — value-added scalar triage (issue #53): 33 new scalar modalities
across six catalogues, PROVABGS as a left-joined scalar spoke, and Galaxy10
as a left-joined image + categorical-label spoke.** The
method is A8's: measured distributions on 2–3 partitions per catalogue (HATS
revisions pinned in the recipes; PROVABGS at `a50ea5c`, Galaxy10 at `e999187`), keep a field only if
it carries information the stream does not already hold, and record its
transform, range and gate in `scalar_registry.py`. Every recipe now wires 42
scalars (48 modalities, ids to 147, `vocab_size: 256` for headroom).

*One predicate for every scalar.* `scalar_registry.scalar_valid` is the single
gate `has_modality` and the sequencer use: every value finite and inside the
scalar's open `VALID_RANGES` interval, and every `QUALITY_GATES` flag present
and at most its limit. A missing flag fails. `CatalogStream` rejects a recipe
whose gated scalar lacks its flag column from the same catalogue. The range
table absorbs the earlier `-99` sentinel list and A8's predicates.

*Kept:*

| Modality | Columns | Coverage | Transform | Range / gate |
| --- | --- | ---: | --- | --- |
| `ebv`, `photometry`, `fiberflux`, `psfdepth` | Legacy anchor | 100% | ADR 0008 / A8 | `psfdepth > 0` |
| `Z` | DESI `Z` | all DESI | `log(1+z)` | `ZWARN ≤ 0` |
| `sdss_Z` | SDSS `Z` | all SDSS | `log(1+z)` | `ZWARNING ≤ 0` |
| `sdss_vdisp` | SDSS `VDISP` | 82% | `log10(v) − 2` | `0 < v < 850`, `ZWARNING ≤ 0` |
| `hsc_cmodel_mag`, `hsc_extendedness`, `hsc_shape`, `hsc_psf_shape` | HSC (A8) | all HSC | A8 | `10 < m < 30`, each `magerr ≤ 0.1` |
| `gwh_gz_<question>` ×10 | Zoobot fractions, n−1 per question | 100% / 5.7% follow-ups | `2x − 1` | — |
| `gwh_total_sfr_median`, `gwh_total_ssfr_median` | MPA-JHU | 11% | unchanged | `> −99`, MPA flag `≤ 0` (drops 12%) |
| `gwh_fibre_sfr_median`, `gwh_fibre_ssfr_median` | MPA-JHU | 11% | as total | `> −99` |
| `gwh_ssfr_med_photoz` | photo-z SED | 93% | `x + 11` | `> −99` |
| `gwh_est_petro_th50` | half-light radius, arcsec | 100% | `log10` | `> 0` |
| `gwh_sersic_n`, `gwh_sersic_ba` | NSA | 7.4% | `log10`, `2x − 1` | `n > 0` |
| `gwh_petro_radius` | NSA `petro_th50`, `petro_th90` (joint: concentration) | 7.4% | `log10` | `> 0` |
| `gwh_petro_ba` | NSA `petro_ba50`, `petro_ba90` (joint: inner vs. outer axis ratio) | 7.4% | `2x − 1` | `> 0` |
| `gwh_nsa_mag` | NSA GALEX F, N, SDSS u, i (bands Legacy lacks) | ~6.5% | `(m − 17)/2` | `10 < m < 30` |
| `gwh_logMH`, `gwh_W50` | ALFALFA HI mass, line width | 1.1% | `(x − 10)/2`, `log10(W) − 2` | `W > 0` |
| `provabgs_LOG_MSTAR`, `_Z_MW`, `_TAGE_MW`, `_AVG_SFR` | PROVABGS SED fit | 223k objects | registry maps | — |
| `provabgs_MAG_W1` | WISE W1 | 223k objects | `(m − 18)/2` | `10 < m < 30` |
| `gz10_label` | Galaxy10 DECaLS class 0–9 (categorical) | ≤ ~7k objects | one-hot | integer in `0..9` |

*Galaxy Zoo grouping (owner decision, 2026-11-17).* Each question's Zoobot
fractions sum to 1 to float precision, so a question is one joint span over
its non-null answers. The dropped answer (no, none, artifact or can't-tell,
else the last) is `1 − Σ kept`; keeping it would hand the diagonal GMM a
degenerate direction, and joint samples stay on the simplex the way joint
`photometry` stays colour-consistent. This is A8's grouping rule (one
quantity, several components), and it costs 30 tokens per object, not the
102 of 34 separate spans. Unreached follow-up questions are NaN upstream,
which is the ordinary absent span.

*Refused:*

- **Copies of a kept field (ADR 0008's `z_spec` rule).** galaxies-with-hats
  `spec_z`, `photo_z` and `redshift_nsa`: `redshift` is `spec_z` where
  present (97%), else `photo_z` (98%). `mag_{g,r,z}_desi`: galaxies-with-hats
  joins only onto Legacy rows, so these are the anchor's fluxes. DESI
  `FLUX_*`, `FIBERFLUX_*`, `FIBERTOTFLUX_*` and `EBV` are Legacy targeting
  photometry and the same dust map. PROVABGS `Z_HP` equals DESI `Z` on 97% of
  matches (99.9% share an `object_id`), and `MAG_G/R/Z` and `FIBMAG_R` are
  Legacy photometry. Legacy `z_spec` (4.8% coverage) is the redshift again.
- **Derivable from kept fields.** `elpetro_mass_log` (NSA mass beside the
  photo-z mass), `elpetro_absmag_*` and `mag_abs_*_photoz` (apparent
  magnitude plus redshift), `u_minus_r`, `est_petro_th50_kpc` (arcsec size
  plus redshift), `HIflux` (`logMH` plus distance), `log_bolometric_l`.
  `elpetro_ba` equals `petro_ba90`, and `elpetro_theta_r` duplicates the
  Petrosian radii.
- **Measurement error, posterior summaries, bookkeeping.** Every `*ERR`,
  `*_IVAR`, `sig*`, `SNR`, `RMS`, percentile, mode, mean and entropy column;
  `photo_zerr`; PROVABGS weights, `TSNR2_BGS`, `HPIX_64`, `SCHLEGEL_COLOR`,
  `PROVABGS_Z_MAX`; `galaxy_size` (undocumented); position angles
  (orientation is not a galaxy property). HSC `a_*` and the other bands'
  extendedness and shape moments stay refused as in A8.
- **Functions of a spectrum already modelled.** SDSS `SPECTROFLUX_*` and
  `SPECTROSYNFLUX_*`.
- **Coverage.** OSSY AGN fields (`log_l_oiii`, `log_l_ha`, `log_m_bh`, line
  widths): 0.04% of galaxies-with-hats, about 3k objects in the stream.

*PROVABGS joins left.* It has a HATS margin, but an outer join would emit
right-only rows that are label-only objects, and its objects are DESI
targets anyway. Left-joined, it decorates Legacy-anchored rows and stays out
of the `min_catalogs` footprint selection, like galaxies-with-hats. Its
values come from SED fits to DESI spectra and photometry, so a model
predicting them partly distils PROVABGS; read them as that, not as truth.

*Same quantity, different method, kept on purpose.* `gwh_mass_med_photoz`
and `provabgs_LOG_MSTAR` (stellar mass), and `gwh_ssfr_med_photoz` beside
MPA-JHU `gwh_total_ssfr_median`. These disagree for physical reasons (SED
from photometry vs. from spectra), so the conditional between them is
information, not copying. If either head looks too good, compare it against
the other source as the baseline first.

*Measured cost (23,043 live records, 70M recipe, seed 0).* "Before" is the
same records with the previous 9 scalars. Each scalar span is 3 tokens.

| | mean | p50 | p95 | p99 | max | next-fit packing at 4096 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| before | 145.9 | 147 | 444 | 590 | 629 | 0.969 |
| after | 158.8 | 159 | 474 | 643 | 710 | 0.966 |

Tokens per record rise 8.8%; every object still fits in one row. Record
shares for the new spans: HSC scalars 40–41%, Legacy scalars 35%, PROVABGS
10%, photo-z sSFR 6.1%, MPA-JHU fibre 2.0%, `sdss_vdisp` 1.9%, NSA 1.1–1.2%,
Galaxy Zoo follow-ups 0.1–0.6%, ALFALFA under 0.05%. The rare heads will see
few targets per step; read their losses with that in mind.

*Galaxy10 (`UniverseTBD/mmu_gz10`).* 17,736 objects, ten classes
(1,081 / 1,853 / 2,645 / 2,027 / 334 / 2,043 / 1,829 / 2,628 / 1,423 /
1,873), 256×256 RGB PNGs at 0.262″/px in `rgb_image`. Measured:

- 36% lie above dec 32.375 and 40% inside Legacy North's HEALPix coverage,
  so at most about 7k can match the anchor. All of it lies inside
  galaxies-with-hats coverage, and 91% of a 97-object sample matches a
  galaxies-with-hats object within 1″.
- The two cutouts of a shared galaxy are **not** the same image: pixel
  correlation 0.35–0.99, mean |Δ| ≈ 20/255. `gz10_images` is a second view,
  not a copy, so it is kept as its own image modality with the `galaxies`
  encoder configuration (patch 16, `input_size` 768, `image_crop` 0, 256
  positions) and its own flow. Decoder `gz10_png` is the `rgb_png` contract
  on the `rgb_image` column.
- Galaxy10 `redshift` equals `gwh_redshift` on 18 of 19 matched pairs:
  **refused** as a copy. `rgb_pixel_scale` is a constant.
- The label is derived from Galaxy Zoo DECaLS volunteer votes, which the
  `gwh_gz_*` Zoobot fractions are trained on, so it is predictable from them.
  That conditional is real information (thresholded votes vs. a network's
  fractions), but a high label accuracy from objects carrying `gwh_gz_*`
  spans is not evidence of image understanding; score it on objects without
  them.

Galaxy10 joins **left**: an outer join would make it count toward
`min_catalogs` and pull South sky, with its SDSS/DESI/HSC rows, into the
stream. The 60% outside Legacy North needs a South anchor.

*Categorical scalars.* `ModalityConfig.categories > 0` marks a scalar whose
one record key is a class index. The token value is its one-hot vector
(`input_size == categories`) through the ordinary scalar encoder, and a
`CategoricalHead` (one linear layer to class logits) replaces the GMM: the
loss is cross-entropy, generation samples the softmax (temperature scales the
logits; 0 or `argmax` takes the mode). It stays in the scalar family, so it
shares the 0.1 weight. `scalar_registry.CATEGORICAL` owns the class count,
and `CatalogStream` rejects a modality whose `categories` disagrees. A GMM over
class indices was rejected because it would order the classes. Both
implementations carry it, with a GPU parity and conversion test.

*Deferred to follow-ups on issue #53:* nested struct paths (Legacy and HSC
`image.psf_fwhm`, Gaia, MaNGA maps), filling a record key from a second
catalogue when the first lacks it (DESI/SDSS photometry and HSC `a_*` for
objects without a Legacy match), a South anchor for the rest of Galaxy10,
Gaia and APOGEE.
