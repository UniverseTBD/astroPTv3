# ADR 0005: Include non-crossmatched DESI spectra as spectrum-only rows

- **Status:** Implemented
- **Date:** 2026-07-16
- **References:**
  - `astro/EXPERIMENTS.md` Part I Phase 5 — the spectra-starved 70M/160M shakeout diagnosis motivating this decision
  - `astro/configs/data/pilot_images_spectra.yaml` — the `pilot_v1` crossmatch config this supersedes
  - `astro/scripts/prepare_pilot_data.py` — the LEFT-crossmatch the second pass extends
  - `astro/src/astropt3/data/mmu.py` — `PILOT_FEATURES`, `normalize_record`, `decode_record`, `assign_split`
  - `astro/src/astropt3/data/packing.py` — `ObjectSequencer.build` (already modality-optional)
  - `astro/src/astropt3/data/nanotron_loader.py` — `PackedMicroBatches` / `build_astropt3_dataloader`
  - `astro/src/astropt3/modeling_astropt3.py` — the per-token-mean Huber loss + `loss_weight`
  - [ADR 0001](0001-jetformer-inverse-variance-loss.md), ADR 0002, [ADR 0003](0003-checkpoint-samples-in-eval-sidecar.md) — house format precedents

## Question

The training corpus is severely spectrum-starved: lsdb `LEFT`-crossmatches the ~14.2M LS-North images against the ~1.1M DESI SV3 spectra, so only the ~0.5–1M matched objects carry a spectrum (~7%), and **every unmatched DESI row is dropped**. The 70M/160M shakeouts reflected this — `spectra_loss` logged 0 in most iterations; the runs were "effectively image-only." How do we include more spectra in the next dataset run so the spectrum head trains on a defensible mix?

## Context

- **Today's pipeline.** `prepare_pilot_data.py` iterates *image* partitions of the LEFT join; the right (DESI) side contributes a spectrum only where the 1-arcsec match succeeds. `PILOT_FEATURES` makes `image` **required** and `spectrum` optional (None when unmatched). `ObjectSequencer.build` is already **modality-optional** — it includes a modality only if the relevant key is non-None and raises only if *no* modality is present — so a **spectrum-only** object already flows through the sequencer, collator, and model unchanged.
- **Per-object token budget.** Images → 364 tokens (144 patch-8 + delimiters); spectra → 31 tokens (~12× imbalance per paired object).
- **Loss mechanic.** The loss is `F.huber_loss` with default `reduction="mean"` per modality (mean over all tokens of that modality in the batch, plus patch elements), then `total += loss_weight_m * mod_loss_m` summed over present modalities, finally `loss = total / n_present`. **The ~12× per-object token imbalance is already normalized out within each modality** — a 364-token image and a 31-token spectrum each contribute a per-object *mean* loss of comparable scale. What is *not* balanced is **cumulative gradient across an epoch**: image-only objects fill most batches, so the spectra head updates only when spectra are present (~7% of draws). The real starver is **object-draw frequency**, not token counts.
- **The DESI ceiling.** DESI SV3 is only ~1.1M rows total; ~0.5–1M already matched, so non-crossmatched DESI is at most ~0.1–0.6M additional spectra. Even if *every* DESI row were a spectrum, the raw-corpus *token* fraction is ~0.7% (31·1.1M ≈ 34M spectra-tokens vs 364·13M ≈ 4.7B image-tokens). A balanced raw-corpus fraction is **infeasible from DESI alone**; any "recommended mix" must be met by **oversampling** at draw time, not corpus composition.
- **The non-crossmatched rows.** DESI rows the LEFT join dropped lack an LS-North image (either outside the LS-North footprint, or inside it but failed the 1-arcsec match). They carry `ra/dec/_healpix_29` (so they slot into the existing spatial split deterministically) and `FLUX_*/FIBERFLUX_*` aperture photometry, but **no** 152×152 cutout image.
- **Split.** `assign_split` is purely position-based on the `order=7` HEALPix ancestor of `_healpix_29`, so DESI-only rows slot into train/val by their own position with zero code change. One object → one position → one split; no reverse leakage.

## Decision drivers

- Directly target the diagnosed spectrum starvation (the only stated Phase-5 quality gap).
- Defend the resulting mix — there is no canonical "literature-recommended mix" to cite (see Research, below), so the ADR must own its own number and justify it first-principles.
- Stay inside one ADR's scope: image+spectra only; no new modalities, no new imaging survey crossmatches.
- Reuse the existing modality-optional sequencer path; minimize blast radius and new dependencies.
- Keep checkpoint-resume exact (Phase 4 contract) under any new draw behavior.
- Don't double-correct: if the per-token mean already normalizes tokens and oversampling already balances draws, an extra objective tilt is redundant.

## Research: there is no canonical mix ratio to cite

We searched the AstroPT / Multimodal Universe (MMU) / general multimodal-pretraining literature for a citable "modality mix" number for continuous-token astronomical image+spectra regression. **None exists:**

- **MMU dataset paper** (Multimodal Universe Collaboration 2024, NeurIPS Datasets & Benchmarks, arXiv:2412.02527) — compiles 100 TB of images/spectra/light curves but prescribes **no training-mix ratio**; it is a *dataset*, not a *recipe*.
- **AstroPT on Euclid Q1** (Siudek et al. 2026, A&A 711 A13) — the AstroPT-applied paper. It uses photometric **SEDs** via simple token-chaining (**not** full spectra), and explicitly *"leave[s] the addition of the spectra for future work."* Only ~5% of its sample has DESI spec-z, used for *validation*, not a mix target. It does give one strong, citable *conceptual* quote we will lean on: *"autoregressive models can operate on the union of available observations without being restricted solely to cross-matched datasets, which typically represent only a small fraction of the available data."* — direct support for adding **spectrum-only** rows.
- **Chameleon / early-fusion unified-token multimodal LLMs** (arXiv:2405.09818) — the closest ML analog to our continuous-token autoregression. They use **empirically-tuned** image/text token mixes, no single canonical ratio, and are discrete-token LLMs — a weak analogy.

**Conclusion:** the ADR frames its target as a self-stated, first-principles number that is *flagged as an empirically-tuned hyperparameter* and scheduled for a small sweep — citing only the Euclid-AstroPT quote and the MMU corpus as *conceptual* justification, not a numeric source.

## Options considered

### Option A — Rebalance only via `loss_weight` (no new data)

Keep the raw ~7% spectra corpus; set `spectra loss_weight > 1` so spectra contribute a chosen share of total loss. **Rejected:** because the loss is already per-token-mean, the per-object signal is already comparable; tilting `loss_weight` beyond 1:1 would double-correct (bias gradient toward an under-trained head) and, more importantly, it does *not* add the diversity/coverage of non-crossmatched DESI rows, which is half of the stated goal ("include more spectra").

### Option B — Pair DESI-only rows with an image from another survey

Crossmatch the non-crossmatched DESI rows against LS-South / DECaLS so every object stays bimodal, preserving per-object image↔spectra co-attention everywhere. **Rejected for this ADR:** heavy lift — a new HATS image catalog, a band-registry extension (LS-South bands ≠ `des-g/r/z`), and prep-script coupling — exceeds one ADR's scope. Recorded as a **future cross-modal ADR**.

### Option C — Oversample the existing matched pairs only

Don't add new spectra; just oversample the ~1M existing matched image+spectra objects to ~25–30% of draws. Avoids the schema change entirely. **Rejected:** contradicts the stated goal of *including more spectra* (diversity and out-of-footprint coverage), and caps rebalance to the matched set.

### Option D — Token-balanced extreme

Draw objects proportional to tokens ⇒ ~92% spectra object draws (since spectra are 31 vs 364 tokens). The literal first-principles extreme of "equal per-modality token budget per epoch." **Rejected as a strawman:** it would mostly-duplicate the small spectra set through extreme repetition, inviting severe overfit, and ignores that the *per-token mean already normalizes tokens* — object-draw balance, not token balance, is the gap.

### Option E — Spectrum-only rows + train-time oversampling (chosen)

Emit the non-crossmatched DESI rows as **spectrum-only** objects (no image), and rebalance draws at train time by a **weighted sampler** so spectrum-bearing objects are ~25–30% of per-epoch object draws. Keep `loss_weight=1:1` (the per-token mean plus balanced draws is sufficient; no objective tilt). Cap oversampling and reshuffle per epoch to bound overfit.

## Decision

**Add non-crossmatched DESI spectra as spectrum-only rows in a new `pilot_v2` corpus, and oversample spectrum-bearing objects to ~25–30% of per-epoch draws at train time, with `loss_weight` left at 1:1.**

### The three-knob mix

1. **Loss aggregation — rely on the existing per-token mean.** The Huber loss already means over each modality's tokens, so the 364-vs-31 per-object token imbalance is absorbed; each object's per-token signal is already comparable. No new token-weighting logic.
2. **`loss_weight = 1:1`.** Leave the per-modality `loss_weight` at its existing equal value. The per-token mean normalizes tokens and oversampling balances draws; an additional objective tilt would double-correct. The two remaining knobs stay non-redundant: the corpus fixes *what objects exist*, the weighted sampler fixes *how often they're drawn*, the objective stays neutral.
3. **Oversampling — ~25–30% of per-epoch draws are spectrum-bearing.** Boost spectrum-bearing objects (matched pairs + new spectrum-only rows) from today's ~7–8% to ~25–30% — a ~3–4× lift off the starved baseline, kept modest to bound the per-spectrum repetition to ~3–4× per epoch. The exact ratio is an **empirically-tuned hyperparameter**: start at ~25–30%, tune **up** only if `spectra_loss` still stalls. Record the chosen value and its effect.

### Structural choice — spectrum-only rows

- Non-crossmatched DESI rows are emitted **spectrum-only** (no image), directly endorsed by the Euclid-AstroPT "union of available observations" quote. The `ObjectSequencer` already supports modality-optional sequences, so no model change is required.
- Per-object image↔spectra co-attention on these ~0.1–0.6M rows is absent; cross-modal co-attention continues to happen on the ~1M existing matched pairs.
- **The non-crossmatched set** = *all* DESI SV3 rows the LEFT join dropped (both out-of-footprint and in-footprint failed matches), gated by **`ZWARN==0`** (DESI's reliable-spectrum flag). Per-bin `mask==True` flux is already zeroed in `_spectra_tokens`, so no extra SNR cut.

### Schema and corpus mechanics

- **Schema:** break backward compatibility. Make `image` *optional* in `PILOT_FEATURES`, mirroring how `spectrum` is already optional. Spectrum-only records drop `image` exactly as image-only records already drop `spectrum`; the existing modality-optional sequencer path is the only path. `normalize_record` emits `image=None` for spectrum-only rows; `decode_record` drops `image` when absent; synthetic fixtures gain a spectrum-only generator. `pilot_v1` continues to use the old image-required schema.
- **Prepare script:** add a **second pass** to `prepare_pilot_data.py` — an lsdb **right-anti-join** (spectra ⋈ images, keep unmatched right) or a full DESI scan de-duped against the matched set, filtered to `ZWARN==0`, normalized with `image=None`, and written as per-partition parquet shards into `pilot_v2/` alongside the re-emitted matched + image-only shards. One script, one corpus.
- **Versioning:** new corpus **`pilot_v2`**; `pilot_v1` stays frozen. Retrain **from scratch** (Pythia schedule restarts at step 1) since both corpus content and schema change — `pilot_v1` checkpoints are not forward-compatible (mirrors the pre-physical-norm incompatibility precedent). `pilot_v2`'s val split now contains spectrum-only objects.

### Amendment (2026-07-18) — randomized bimodal ordering for bidirectional conditioning

> **Superseded by [ADR 0008](0008-scalar-modalities.md) (2026-07-18).** The
> parity rule below is undefined once an object can carry more than two
> spans, so ADR 0008 replaces it with a **uniform random shuffle of all
> present spans**, seeded on the same `crc32(object_id) ^ epoch`. This is a
> strict generalization, not a reversal: at two spans a uniform shuffle *is*
> the 50/50 flip described here, so the behaviour documented below remains
> accurate for bimodal objects and resume-exactness carries over unchanged.
> ADR 0008 also **overrides** this ADR's `loss_weight = 1:1` principle for
> scalar modalities only — the "no objective tilt" argument assumed
> modalities of comparable token count (364 vs 31), which fails at 144:1.

**Decision: bimodal (matched) objects serialize their two modality spans in
randomized order, 50/50 per draw.** The original fixed alphabetical order
(`images` then `spectra`) means the causal model only ever learns
`p(spectra | images)`; image tokens never attend to spectra, so
spectra→image prediction does not exist in the model at any temperature.
Randomizing the span order per object teaches both conditionals with the
same corpus and loss.

- 50/50 split, uniform per object draw — the simplest defensible choice;
  no evidence yet that either direction deserves a protective bias (revisit
  with a biased split only if the image→spectra direction measurably
  degrades).
- Single-modality objects (image-only, spectrum-only) are unaffected —
  there is nothing to permute.
- The rest of the pipeline is already order-agnostic: `<|begin_m|>` tokens
  make the order self-describing, value alignment is per-modality boolean
  masks in row-major order, and the `starts−1` loss alignment keys off the
  masks, not the span order.
- **Always on, no config knob** (user decision 2026-07-18, revising the
  first draft's config-carried flag): every bimodal sequence everywhere is
  built under the parity rule — one behavior, no per-run option to drift.
- **Consequence: checkpoint break for spectra-bearing models** — fixed-order
  checkpoints have never seen a spectra-first sequence; retrain (mirrors
  the ADR 0007 precedent). Their post-rule evals (val batches, probe
  embeddings, panels) are order-OOD; treat pre-rule runs
  (pilotv2 baseline / specnorm / clip100) as frozen baselines evaluated
  under pre-rule code. Generation gains a `spectra-to-images` mode.

### Oversampling realization (known subtlety)

`MMUIterableDataset` streams parquet **shards** in order, sharded by DP rank via `split_dataset_by_node`; there is no per-record weighting intercept today. A weighted sampler to ~25–30% spectra draws must therefore operate at **shard-granularity** — favoring spectrum-bearing shards ~3–4× per epoch in the shard-order shuffle inside `_load`, or as a mix-aware wrapper around the record generator in `PackedMicroBatches._mmu_records` (the synthetic stream already has a `synthetic_image_only_fraction` knob to mirror). This argues for writing spectrum-only rows as a **distinct shard set** (they are spatially clustered in the DESI footprint, so this is natural) rather than interleaving them, so shard-level weighting stays clean. **Checkpoint-resume (Phase 4, state captured at the partial-row start) must be re-verified to remain exact under the weighted re-emit** — the one subtlety to test explicitly. Oversampling is strictly a **train-time** concern (the corpus is written once, oversampled at draw time; no physical duplication, so the ratio is retunable without re-prep).

### Overfitting mitigation

Per-epoch reshuffle (already in `MMUIterableDataset.set_epoch`) plus a **cap** on the oversample factor (~don't exceed ~4×). Monitor train vs val `spectra_loss` divergence as the overfit-detection signal. **No spectra augmentation** in this ADR — adding flux noise / wavelength jitter would complicate the jetformer exact-likelihood path (the invertible record→token map), and is explicitly deferred.

## Consequences

### Positive

- The spectra head trains on a defensible draw distribution (~25–30% vs ~8% today), directly closing the Phase-5 spectra-starvation gap.
- New spectra diversity — the full (~0.1–0.6M) non-crossmatched DESI population enters the corpus, including out-of-LS-North coverage.
- Minimal new code: the sequencer already handles modality-optional objects; the schema change mirrors the existing `spectrum`-optional path; the prepare pass reuses the existing script machinery.
- Two non-redundant knobs (`loss_weight` neutral, oversampling tuned) on a clear first-principles footing, with the one literature claim cited honestly as *conceptual*, not numeric, support.

### Negative / tradeoffs

- **Overfit from ~3–4× spectra repetition** (headline risk). Mitigated by reshuffle + cap, monitored via train/val `spectra_loss`; if 25–30% proves too aggressive, the cap and ratio retune without re-prep.
- Spectrum-only rows never co-attend with an image of the same object — cross-modal alignment on those rows is absent (noted as a known limitation; pairing is the future ADR).
- Spectrum-only objects are ~31 tokens, so packed rows become spectra-dense — a packing-efficiency and per-batch token-distribution shift the collator handles but which changes batch composition (minor).
- Backward compat broken for the shard schema: `pilot_v1` and `pilot_v2` use different `PILOT_FEATURES` (image required vs optional); consumers select by corpus version.
- The weighted-re-emit must be verified against the Phase-4 checkpoint-resume contract (the one explicit test burden).
- `eval/val_loss`, the redshift probe, and generation panels must handle spectrum-only batches and spectrum-only templates (the existing image-centric sample templates need extending).

## Validation / success criteria

1. `eval/val_loss` reports a **non-zero, decreasing per-modality spectra** component across checkpoints (no more `spectra_loss=0`); image and spectra losses within ~5× after warmup (the PLAN Phase-5 gate).
2. **Redshift (Z) linear-probe R² improves** vs the 70M/160M shakeouts (0.28–0.32), since spectra-bearing objects are now denser and more often drawn.
3. **Spectrum generation quality** via `eval/samples` panels — requires new *spectrum-only* templates (the existing image-centric templates need extending).
4. Train/val `spectra_loss` divergence tracked across checkpoints as the overfit signal tied to the oversampling cap.

## Non-goals / scope (deferred to future ADRs)

- **No new MMU modalities** (time series, tabular) in this ADR — deferred to Phase-6 modality-extension ADRs; recorded as a future-work note.
- **No LS-South / DECaLS image pairing** of the spectrum-only rows — a separate future cross-modal ADR.
- **No spectra augmentation** (flux noise / wavelength jitter) — out of scope here (avoids the jetformer invertibility complication); mitigation is reshuffle + oversample cap.
- **No back-port to `pilot_v1`** — frozen; its checkpoints are not forward-compatible.
- **No objective (`loss_weight`) tilt** — `1:1` is the principled default given the per-token mean + balanced draws.

## Open issues

- Confirm the exact right-anti-join realization in `prepare_pilot_data.py`'s second pass (lsdb `how` support for anti-join vs a full-DESI-scan de-dup).
- Decide whether spectrum-only shards are written as a separate `/spectra/` subdir of `pilot_v2` (cleaner for shard-level weighting) or interleaved by partition.
- Re-verify the Phase-4 checkpoint-resume exact-continuation test under the weighted-re-emit draw path (extend `test_loader_resume.py`).

## References

- `astro/configs/data/pilot_images_spectra.yaml` — `pilot_v1` config superseded.
- `astro/scripts/prepare_pilot_data.py` — the LEFT-crossmatch the second pass extends.
- `astro/src/astropt3/data/mmu.py` — `PILOT_FEATURES`, `normalize_record`, `decode_record`, `assign_split`.
- `astro/src/astropt3/data/packing.py` — `ObjectSequencer.build` (modality-optional path).
- `astro/src/astropt3/data/nanotron_loader.py` — `PackedMicroBatches`, `build_astropt3_dataloader`.
- `astro/src/astropt3/modeling_astropt3.py` — per-token-mean Huber loss + `loss_weight`.
- Multimodal Universe Collaboration 2024, arXiv:2412.02527 (NeurIPS Datasets & Benchmarks) — corpus source; no mix ratio prescribed.
- Siudek et al. 2026, A&A 711 A13 (AstroPT on Euclid Q1) — "union of available observations" conceptual quote.
- Chameleon, arXiv:2405.09818 — early-fusion unified-token multimodal LLM (empirically-tuned mix; weak analog).