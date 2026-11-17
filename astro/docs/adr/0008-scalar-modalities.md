# ADR 0008: Scalar modalities for autoregressive label prediction

- **Status:** Accepted (2026-07-18) — implemented same day (`scalar_registry.py`,
  scalar spans + uniform span shuffle in `packing.py`, GMM heads both
  implementations, `eval/scalar_head.py`, probe scalar-free)
- **Date:** 2026-07-18
- **Superseded for expanded-corpus loss aggregation:** ADR 0013
  replaces independent scalar `loss_weight` terms with one collectively bounded
  scalar-family mean. Historical configs retain this ADR's objective.
- **References:**
  - `astro/src/astropt3/tokenization.py` — `_MODALITY_ID_BLOCKS`, the frozen
    64-id reservation this ADR spends ids 8–16 of
  - `astro/src/astropt3/modalities.py` — `ModalityConfig`, `ModalityRegistry`,
    `Encoder`/`PositionEmbedder`, `GMMHead`, `gmm_nll`
  - `astro/src/astropt3/data/packing.py` — `ObjectSequencer.build`, the
    modality-optional path and the span-order rule this ADR supersedes
  - `astro/src/astropt3/data/band_registry.py` — the normalization pattern
    `scalar_registry.py` mirrors
  - `astro/src/astropt3/data/mmu.py` — `PILOT_FEATURES`, `_SCALAR_KEYS`,
    the scalar fields already carried per record
  - `astro/src/astropt3/modeling_astropt3.py` — the per-modality loss
    aggregation (`Σ loss_weight_m · loss_m / n_present`)
  - `astro/src/astropt3/eval/linear_probe.py` — the `Z` probe whose
    comparability this ADR must preserve
  - [ADR 0004](0004-spiral-token-order-for-imagery.md) — intra-span token
    order; unaffected (scalar spans are 1 token)
  - [ADR 0005](0005-include-spectra-from-non-crossmatched-desi.md) — the
    bimodal parity span-order rule **superseded** here, and the `1:1`
    `loss_weight` principle **overridden** here
  - [ADR 0007](0007-physical-spectra-normalization.md) — the fixed,
    invertible, no-fitted-constants normalization precedent

## Question

The model can only be probed for physical labels, never asked. `Z` is
recovered post-hoc by a ridge probe on mean-pooled hidden states
(`eval/linear_probe.py`, R²≈0.66–0.71 on the pilot_v2 70M runs); there is
no way to condition on an image and have the model *emit* a redshift, and no
way to sample a joint (image, spectrum, redshift). How do we give AstroPT3
autoregressive scalar prediction — redshift now, sSFR and morphology later —
without inventing machinery outside the existing modality contract?

## Context

- **The 64-id reservation exists for this.** `tokenization.py` allocates 3
  consecutive ids per modality and reserves ids 8–63 explicitly "for future
  modalities (time series, tabular, ...)". Nothing about a scalar is special
  at the token level: it is a modality whose span happens to be one token.
- **The sequencer is already modality-optional** (ADR 0005): a modality is
  included only if its key is non-None. Per-scalar missingness — most rows
  have no spectroscopic redshift — therefore needs **no new masking logic**,
  it is an absent span, exactly as a missing spectrum already is.
- **The scalars already exist in the record.** `PILOT_FEATURES` carries
  `ebv, flux_g, flux_r, flux_z, z_spec, Z, ZERR`, and `normalize_record`
  already emits `None` for missing or NaN values via `_SCALAR_KEYS`. No
  corpus change, no new crossmatch, no `pilot_v3`.
- **`GMMHead` and `gmm_nll` are already written and tested** in
  `modalities.py`, currently reachable only from the `jetformer` tokeniser
  path. They are independent of `TinyFlow1D`.
- **Z coverage equals spectrum coverage** in `pilot_v2` (both come from the
  DESI side), so ADR 0005's shard-level oversampler already lifts
  `Z`-bearing objects to ~25–30% of per-epoch draws. The draw-starvation
  problem ADR 0005 diagnosed does not need re-solving here.
- **Loss aggregation is `n_present`-normalized.** `total = Σ loss_weight_m ·
  loss_m`, then `loss = total / n_present`. Each modality's own loss is a
  per-token mean, so a 1-token span and a 144-token span carry *equal*
  weight by construction.

## Decision drivers

- Ship a capability that does not exist today (emitting labels), not a
  better version of one that does (probing for them).
- Spend zero new plumbing: reuse the registry, the sequencer, the collator,
  the loss path, the nanotron loader, and `GMMHead`.
- Keep the headline `Z` probe metric comparable to every run already logged.
- Do not silently degrade image/spectra quality — the model's primary job —
  in exchange for label heads.
- Make adding the *next* label (sSFR, morphology) a data change with no
  model change. That is the test of whether this design is right.
- Honour ADR 0007's normalization discipline: fixed, invertible, physical,
  no per-corpus calibration, unknown names raise.

## Options considered

### Option A — Improve the linear probe, add nothing

Tune the ridge probe, probe from a better layer, use attention pooling.
**Rejected:** it cannot produce the capability. A probe is a post-hoc
read-out; it can never be sampled from, never conditions the rest of the
sequence, and never participates in joint generation. This ADR's goal is
explicitly *inference capability*, not representation quality.

### Option B — A head off `<|bos|>` or the final hidden state

Hang a regression head outside the token sequence. **Rejected:** it forfeits
everything the modality contract already provides — packing invariants,
resume exactness, TP replication in the nanotron fork, the generation path —
in exchange for nothing. It is also not autoregressive: labels could not
condition on each other or be conditioned on by later spans.

### Option C — One `scalars` modality, one token per quantity, shared head

A single span with a learned index position embedding identifying which
scalar occupies each slot, and one shared `Linear(hidden → 1)`. Cheapest in
token ids. **Rejected:** a shared 1-dim head cannot express heterogeneous
output distributions (a categorical morphology label alongside a continuous
redshift), and a single span cannot be permuted to teach the conditionals
*between* scalars without extra intra-span shuffling machinery.

### Option D — One modality per scalar quantity (chosen)

Each scalar is an ordinary registry modality with its own token-id block,
`Encoder`, `PositionEmbedder`, and head. Under the uniform span shuffle
(below), N independent spans in random order teach every conditional among
them — `p(Z | photometry)`, `p(photometry | Z)`, `p(Z | image, spectrum)` —
which is ADR 0005's bidirectional-conditioning argument applied one level
down. Per-scalar heads may have per-scalar output distributions.

## Decision

**Add scalars as ordinary modalities in the existing registry — one modality
per physical quantity, each with a `GMMHead`, normalized through a new fixed
`scalar_registry.py`, serialized under a uniform random span order that
supersedes ADR 0005's parity rule.**

### The three new modalities

| Modality | ids | `input_size` | Normalization | Source field |
| --- | --- | --- | --- | --- |
| `Z` | 8, 9, 10 | 1 | `log(1 + z)` | `Z`, gated `ZWARN == 0` |
| `ebv` | 11, 12, 13 | 1 | `ebv / 0.1` | `ebv` |
| `photometry` | 14, 15, 16 | 3 | `arcsinh(f / 0.01 nMgy)` | `flux_g/r/z` |

- **`photometry` is one joint 3-dim span, not three modalities.** The three
  fluxes are never independently missing (one catalogue row), and **colour**
  — not flux — is what photometric redshift actually depends on. A joint
  `GMMHead(..., out_size=3)` models the correlated quantity directly; three
  independent marginals would generate colour-inconsistent samples.
- **`z_spec` and `ZERR` are deliberately excluded.** `z_spec` is the same
  physical quantity as `Z` from a second source: with both present, uniform
  shuffle would sometimes show one and ask for the other, producing
  spectacular head accuracy that measures copying. `ZERR` is a property of
  the DESI pipeline's fit, not of the galaxy. Both remain plain record
  fields for probes and metadata.
- **`Z` spans are gated on `ZWARN == 0`**, reusing ADR 0005's reliability
  flag rather than inventing a new quality cut. Rows failing the gate emit
  no `Z` span — the ordinary modality-optional path.

### Heads: `GMMHead`, K = 5

Every scalar modality predicts a K=5 diagonal Gaussian mixture over its
normalized target; loss is `gmm_nll` at the `starts − 1` position, the same
`left_shift_mask` alignment every other modality uses.

- **Why a mixture, not a point estimate.** Photometric-redshift posteriors
  are genuinely multimodal — colour degeneracies place real probability mass
  at several redshifts. A Huber point estimate lands *between* the modes,
  producing a confidently wrong answer for exactly the objects that matter.
  A mixture gives the full conditional, samples for generation, and
  calibrated uncertainty.
- **Why this costs nothing.** `GMMHead` and `gmm_nll` already exist and are
  already exercised by the jetformer path.
- **Scalars work identically under `affine` and `jetformer`.** `GMMHead`
  does not require `TinyFlow1D`; the flow exists to make high-dimensional
  patch tokens invertible, which a scalar does not need. There is therefore
  **no odd-dimension problem** and no tokeniser-specific exclusion.
- **The `log_sigma` clamp is now load-bearing.** GMM NLL is unbounded below
  and a 1-dim head can drive σ→0 on an easy target for arbitrarily negative
  loss — far easier in 1 dimension than across 192 image dimensions.
  `GMMHead`'s existing `clamp(-7.0, 2.0)` is the floor that prevents this;
  it must not be loosened without revisiting this ADR.
- **Per-patch standardization is skipped for scalar modalities** (as it is
  for jetformer configs): the mean and std of a single value are degenerate.
  The `scalar_registry` transform is the whole normalization.

### Normalization: `data/scalar_registry.py`

Mirrors `band_registry.py` and honours ADR 0007: **fixed, invertible,
physical, no fitted constants, unknown scalar names raise.**

- `Z → log(1 + z)` — the standard photometric-redshift working variable. It
  compresses the high-z tail and makes errors naturally fractional, so the
  reported σ maps onto the literature's `Δz/(1+z)` with no conversion.
- `ebv → ebv / 0.1` — a fixed knee putting typical Galactic extinction at
  O(1), matching the band registry's philosophy.
- `flux_g/r/z → arcsinh(f / 0.01 nMgy)` — literally the band registry's
  existing transform, so aperture photometry and image pixels live in the
  same units.

**Fitted standardization (subtract corpus mean / divide by corpus std) is
rejected** for ADR 0007's reason: it couples a checkpoint to the corpus it
was fitted on, so a later corpus with different sky coverage shifts the
target distribution under a model that cannot know.

### Span order: uniform shuffle (supersedes ADR 0005)

**All present spans of an object are serialized in uniform random order,
seeded on `crc32(object_id) ^ epoch`.** ADR 0005's "bimodal objects reverse
on parity" rule is undefined at more than two spans and is replaced.

- **This is a strict generalization, not a reversal.** At two spans, a
  uniform shuffle *is* a 50/50 flip — ADR 0005's amendment becomes the N=2
  special case of this rule rather than an exception to it. The
  distributional behaviour of existing bimodal objects is unchanged.
- **Resume exactness carries over unchanged.** The seed is derived from
  `object_id` and `epoch`, so there is no RNG state to checkpoint and the
  Phase-4 contract holds verbatim.
- **Always on, no config knob**, following ADR 0005's precedent: one
  behaviour, no per-run option to drift.
- **A scalar span sometimes lands first.** The model is then asked to
  predict `Z` from `<|bos|>` alone and will learn the marginal `p(Z)`. This
  is correct behaviour, not a defect — it is the prior the conditional
  posteriors are updates of.
- The rest of the pipeline is already order-agnostic: `<|begin_m|>` tokens
  make the order self-describing, value alignment is per-modality boolean
  masks in row-major order, and the `starts − 1` loss alignment keys off the
  masks rather than the span order.

### Loss weight: 0.1 for scalar modalities (overrides ADR 0005's 1:1)

`loss = Σ loss_weight_m · loss_m / n_present` means a fully-labelled object
has five modalities present, so **images would fall from 1/2 of the
objective to 1/5**, with three of those five being spans of one to three
tokens. Sixty percent of the gradient would go to five numbers and twenty
percent to 144 image patches.

**Scalar modalities take `loss_weight = 0.1`**, putting all three together
at roughly 1/8 of the objective and leaving images and spectra dominant.

**This explicitly overrides ADR 0005's "no objective tilt" principle**, and
the override is scoped, not a repudiation: 0005 reasoned about two
modalities of comparable token count (364 vs 31), where the per-token mean
genuinely did normalize the imbalance. At 144:1 the per-token mean is what
*creates* the imbalance. `0.1` is a **starting point for a sweep**, recorded
here in the same spirit as 0005's 25–30% oversample ratio: a self-stated
first-principles number flagged as an empirically-tuned hyperparameter.

### Evaluation: the probe sees scalar-free sequences

`eval/linear_probe.py` builds its batches with **all scalar spans omitted**
(a flag on the eval record builder; the sequencer is already
modality-optional, so this is near-zero code).

- Without this the probe mean-pools over a sequence that *contains* the
  redshift token, so R² stops measuring representation quality and starts
  measuring copying — destroying comparability with every run already
  logged, which is the metric's entire value.
- It also matches downstream use: at inference on unlabelled data there is
  no `Z` span to include.
- Because the model was *trained* with `Z` spans present but is *probed*
  without them, an improvement in probe R² is uncontaminated evidence for
  the representation-quality byproduct — a cleaner result than a
  contaminated probe could ever give.

**A new autoregressive-`Z` metric is the actual success gate**: condition on
the observation spans, force `<|begin_Z|>`, read the head, score against
truth on the fixed val split. Report both — probe R² for representation
quality (comparable backward), head accuracy for the ADR 0008 capability.

## Consequences

### Positive

- The capability exists: the model can be *asked* for a redshift, with a
  full multimodal posterior rather than a point estimate.
- Near-zero new code. Three registry entries, one `scalar_registry.py`
  mirroring an existing file, one `GMMHead` wiring outside the jetformer
  path, one span-order function, one eval flag.
- **Adding the next label is a data change with no model change** — sSFR and
  morphology land in ADR 0009 as new registry entries and a new crossmatch.
  That is the design's own falsification test.
- Uniform shuffle teaches every conditional direction among observations and
  labels, generalizing ADR 0005's argument rather than special-casing it.
- ADR 0005's oversampler already lifts `Z`-bearing objects to ~25–30% of
  draws, so the head is not draw-starved and no new knob is needed.
- Scalars behave identically under `affine` and `jetformer`.

### Negative / tradeoffs

- **Checkpoint break for every `pilot_v2` run.** New spans and a new
  serialization mean existing checkpoints have never seen these sequences;
  retrain from step 1, Pythia schedule restarts. (Mirrors the ADR 0005 and
  0007 precedents.) Pre-0008 runs stay frozen baselines evaluated under
  pre-0008 code.
- **Affine configs now mix a negative NLL term with positive Huber terms**,
  so the scalar total loss is no longer interpretable as a single quantity
  and a confident scalar head can drive it down while image loss stagnates.
  Mitigated by logging per-modality losses (`val_loss` already does) and by
  gate 3 below. In `jetformer` configs the mix is homogeneous in kind — all
  terms are exact NLL — though not in scale.
- Three more spans per object is ~9 more tokens of delimiter overhead on
  fully-labelled objects, and a packing-composition shift the collator
  handles but which changes batch statistics slightly.
- `ebv` is a function of sky position rather than of the object, so the
  model can only predict it by inferring position from colour. It is
  included as cheap auxiliary supervision, not because it is astrophysically
  interesting; if it proves to be a distraction it is one registry entry to
  drop.
- Ids 8–16 of the frozen block are spent, leaving 47 (≈15 modalities).
- `eval/samples.py` and `generation.py` need observation→scalar modes and
  scalar-bearing templates.

## Validation / success criteria

1. **Scalar losses non-zero and decreasing** across checkpoints for all
   three modalities.
2. **Autoregressive `Z` beats the linear probe.** Head accuracy on the fixed
   val split, reported as `Δz/(1+z)` scatter (`nmad`) and outlier fraction,
   must improve on the probe's R² ≈ 0.66–0.71 pilot_v2 baseline. This is the
   ADR's reason to exist.
3. **Image and spectra val loss not degraded** versus the `pilot_v2`
   baseline. This is the guard on `loss_weight = 0.1` — if it fails, the
   sweep moves the weight down, not the gate.
4. **GMM calibration**: fraction of val objects whose true `Z` falls within
   the predicted 1σ interval ≈ 0.68. An uncalibrated head that scores well
   on point accuracy has not delivered the capability.

## Non-goals / scope (deferred)

- **No sSFR or morphology in this ADR** — they require new catalogue
  crossmatches (`prepare_pilot_data.py`, a `pilot_v3`) and belong in a
  data-only ADR 0009 with no model change.
- **No categorical heads.** Morphology labels will need one; the per-scalar
  head design accommodates it, but nothing is built here for a target that
  does not yet exist in the corpus.
- **No change to the loss aggregation rule** (`n_present` normalization)
  — token-count weighting would alter the objective for every existing
  config and re-break the bimodal balance ADR 0005 reasoned about.
  Revisit only if `loss_weight` tuning proves insufficient.
- **No back-port to pre-0008 checkpoints** — incompatible; frozen.

## Open issues

- The `K = 5` mixture count is unswept; K is cheap to change before the
  first real run and expensive after.
- Whether `ebv` earns its span at all (see tradeoffs) — decide from the
  first sweep rather than in advance.
- Confirm the uniform shuffle's N=2 behaviour matches the ADR 0005 parity
  rule in distribution under test, so bimodal runs remain comparable.
- Re-verify the Phase-4 checkpoint-resume exact-continuation test under the
  new span-order function (extend `test_loader_resume.py`).

## References

- `astro/src/astropt3/tokenization.py` — frozen id blocks; ids 8–16 spent here.
- `astro/src/astropt3/modalities.py` — `GMMHead`, `gmm_nll`, `ModalityConfig`.
- `astro/src/astropt3/data/packing.py` — `ObjectSequencer.build`, span order.
- `astro/src/astropt3/data/band_registry.py` — normalization pattern mirrored.
- `astro/src/astropt3/data/mmu.py` — `PILOT_FEATURES`, `_SCALAR_KEYS`.
- `astro/src/astropt3/modeling_astropt3.py` — loss aggregation.
- `astro/src/astropt3/eval/linear_probe.py` — the probe whose comparability
  the scalar-free sequence rule preserves.
- [ADR 0005](0005-include-spectra-from-non-crossmatched-desi.md) — span-order
  rule superseded; `1:1` `loss_weight` principle overridden.
- [ADR 0007](0007-physical-spectra-normalization.md) — normalization discipline.
