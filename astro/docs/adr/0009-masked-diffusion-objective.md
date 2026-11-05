# ADR 0009: Masked-diffusion training objective (jetformer-only)

- **Status:** Proposed
- **Date:** 2026-07-18
- **References:**
  - `astro/src/astropt3/modeling_astropt3.py` — the `tokeniser: jetformer`
    path (z computed once up front, embedding and target), `left_shift_mask`,
    the `attention_mask=None` causal-packing trick this ADR branches away from
  - `astro/src/astropt3/modalities.py` — `TinyFlow1D`, `GMMHead`, `gmm_nll`;
    the sampler this objective's generation loop is built on
  - `astro/src/astropt3/data/packing.py` — `position_ids` restart at 0 per
    object; pads at position 0 — the segment information the bidirectional
    mask is derived from
  - `astro/src/astropt3/tokenization.py` — frozen ids, deliberately untouched
  - `astro/src/astropt3/generation.py`, `astro/scripts/generate.py` — where
    the unmasking sampler lands
  - `astro/src/astropt3/eval/` — `val_loss` (determinism rule), `linear_probe`
    (unmasked forward), `samples` (needs the sampler)
  - `nanotron/src/nanotron/models/astropt3.py` — `cu_seqlens` derived from
    position-id restarts, consumed by varlen flash attention with a causal
    flag; the mirror's whole diff is that flag plus the masking/loss changes
  - [ADR 0002](0002-ivar-weighted-huber-loss.md) — ivar weighting, unchanged
  - [ADR 0003](0003-checkpoint-samples-in-eval-sidecar.md) — the sample
    panels the minimal sampler must keep alive
  - [ADR 0005](0005-include-spectra-from-non-crossmatched-desi.md) — the
    span-order rule whose *purpose* this objective makes moot
  - LLaDA (Nie et al. 2025), MDLM (Sahoo et al. 2024), UniDisc (Swerdlow et
    al. 2025) — the estimator, the ELBO, and the per-modality mask ratios

## Question

The AR factorization conditions one way per draw. ADR 0005's span-order
shuffle teaches both cross-modal directions *in aggregate*, but every
individual sequence is still a single left-to-right chain: generation is
strictly prefix-conditioned, infilling is impossible, and representations
at token *i* never see tokens *> i*. Diffusion language models (LLaDA,
Mercury, Dream) get any-order conditioning — arbitrary infill, either
conditional direction at inference, bidirectional features — from the same
transformer body by swapping the objective. Can AstroPT3 have a
masked-diffusion training regime as an opt-in A/B against the AR baseline,
with minimal changes to the existing implementation?

## Context

- **Absorbing-state masked diffusion is just weighted masked prediction.**
  LLaDA's training loop is: sample a mask ratio `t ~ U(0,1)`, mask each
  token independently with probability `t`, predict the masked tokens with
  bidirectional attention, weight the loss by `1/t`. That estimator is an
  unbiased bound (ELBO) on the data log-likelihood. No timesteps, no noise
  schedules on values, no new heads — the machinery is which positions are
  targets and what fills the holes.
- **The jetformer path already has everything a diffusion sampler needs.**
  `z = flow(values)` is computed once up front and serves as both embedding
  input and prediction target; `GMMHead` gives a real per-token sampler and
  a likelihood. The affine/regression path has neither: iterative unmasking
  with a Huber head can only commit point estimates (regression-to-mean
  blur) and has no confidence signal. The tokeniser axis and the objective
  axis are otherwise orthogonal — the branch point ("which positions are
  targets, what embeds at masked slots") is identical for both.
- **Object boundaries are already in `position_ids`.** The HF
  implementation passes `attention_mask=None` and lets `create_causal_mask`
  detect packing from position restarts; the nanotron fork derives
  `cu_seqlens` from the same restarts and calls varlen flash attention with
  a causal flag. Both sides can build the bidirectional-within-object,
  isolated-pads attention pattern from information already in the batch —
  nothing new flows through the data pipeline.
- **There is no discrete head.** The model has a 64-id `embed_tokens` and
  no lm_head, so special tokens (`<|bos|>`, `<|begin_m|>`, `<|end_m|>`)
  cannot be prediction targets even in principle. Only modality value
  tokens are maskable; the scaffolding stays visible.
- **The span-order rule exists solely because attention is causal.** ADR
  0005's 50/50 shuffle puts both conditioning directions in the training
  distribution because a causal model can only condition left-to-right.
  Under bidirectional attention, span order no longer determines
  conditioning direction — the mask pattern does.

## Decision drivers

- Opt-in A/B against the AR baseline, not a replacement: AR is the
  validated regime and the Pythia-mirror suite is mid-flight.
- Smallest possible diff: no data-pipeline changes, no new tensors through
  the collator or nanotron's flat-dict device mover, frozen token ids
  untouched, packing/resume/TP contracts intact.
- Both cross-modal conditionals (image→spectrum, spectrum→image) must be
  *in the training distribution*, not just reachable at inference.
- The eval sidecar must keep working: deterministic val loss, comparable
  linear probe, sample panels per checkpoint (ADR 0003).
- Follow LLaDA where a published choice exists rather than inventing local
  variants — the estimator, the `U(0,1)` ratio, the normalization.

## Options considered

### Option A — Masked (absorbing-state) diffusion, LLaDA-style (chosen)

Mask value tokens at a sampled ratio, predict them bidirectionally at the
masked positions, weight by the inverse ratio. Reuses the flow, the GMM
heads, and the loss aggregation; the structural changes are the attention
pattern and a mask embedding.

### Option B — Continuous Gaussian diffusion on token values

Noise the patch values themselves, condition on a timestep, predict ε or v.
**Rejected:** needs timestep embeddings, a new prediction target, and a
noise schedule — the furthest of the three from the current code, and it
abandons the flow+GMM likelihood machinery rather than reusing it.

### Option C — Diffusion forcing (per-token noise, causal attention)

Keeps the causal packing trick untouched. **Rejected:** it is the least
standard reading of "diffusion LM", does not give bidirectional
representations (half the motivation), and its per-token noise levels are
a new tensor through the whole pipeline.

### Option D — Diffusion under the affine/regression tokeniser too

**Rejected as degenerate:** no sampler (point-estimate fills), no
confidence, and the masked-Huber loss is no longer interpretable as a
likelihood bound. `objective: diffusion` asserts `tokeniser: jetformer`.

### Option E — Separate `AstroPT3DiffusionModel` class

**Rejected:** duplicates the model to avoid one branch; two classes to keep
in parity across two implementations.

## Decision

**Add `objective: ar | diffusion` (default `ar`) to `AstroPT3Config`.
`diffusion` is valid only with `tokeniser: jetformer` and trains the
existing architecture as a LLaDA-style masked diffusion model:
bidirectional attention within each object, per-modality mask ratios,
GMM-NLL on masked latents, minus the full-span flow logdet.**

### Attention: bidirectional within each object, derived in-model

When `objective: diffusion`, `forward` derives segment ids from
`position_ids` (cumulative count of restarts — pads each restart at 0, so
every pad is its own segment, exactly the isolation the causal trick
provides today) and builds the block-diagonal bidirectional mask
`segment_i == segment_j`.

- HF side: a 4D additive mask passed to `SmolLM3Model` under SDPA. The
  O(T²) materialization is acceptable — this implementation is the CPU
  test/probing artifact, not the training path.
- nanotron side: the existing `cu_seqlens` (already derived from the same
  restarts) with `causal=False` on the varlen flash-attention call. No
  materialized mask, no new inputs.
- Nothing changes in the collator, the loader, or the flat-dict contract.

### Masking: a learned embedding, no new token ids

`input_ids` are untouched — the per-modality placeholder id already carries
modality identity, and `tokenization.py` stays frozen. Masking withholds
the *value*: at masked slots the additive delta `encoder_m(z) +
pos_embed_m(position)` becomes `mask_embed + pos_embed_m(position)`, where
`mask_embed` is one shared learned vector (modality identity comes from
`embed_tokens`; position embedding stays so the model knows *where* the
hole is and, for spectra, at what wavelength). Only modality value tokens
are maskable; special tokens are always-visible scaffolding.

Masking happens in the model in training mode, not in the data pipeline —
the same record produces the same packed batch under either objective.

### Mask ratios: independent per modality (UniDisc-style)

Each (object, modality) draws its own `t_m ~ U(0,1)`; each value token of
that span is masked independently with probability `t_m`. This is
load-bearing, not a refinement: with a single shared `t` per object, the
regime "spectrum fully masked, image fully clean" has probability ≈ 0 in
training, yet it is exactly the image→spectrum conditional generation the
model exists to do. Independent ratios put both cross-modal conditionals —
and everything between — in the training distribution.

**This supersedes the *purpose* of ADR 0005's span-order rule** (and of any
successor uniform-shuffle rule): under symmetric attention with per-object
positions, span order is unobservable up to position embedding and no
longer determines conditioning direction; the mask pattern does that job.
The shuffle stays on — it is harmless, keeps the packed batches identical
across objectives, and AR configs still need it.

### Loss: the LLaDA estimator in z-space

Per modality span:

```
loss_m = (1/t_m) · Σ_masked NLL_GMM(z_i) / L_m
```

read at the masked positions themselves — no `left_shift_mask` — where
`L_m` is the full span length (LLaDA's normalization; dividing by the
masked count instead biases the estimator). The full-span flow logdet is
subtracted once per token, masked or not, keeping the bound anchored to
data-space likelihood. Then the existing aggregation, unchanged:
`Σ loss_weight_m · loss_m / n_present`.

- **Exact likelihood becomes an ELBO.** The AR jetformer loss is the exact
  NLL via the chain rule; the diffusion loss is a bound on it. Stated
  plainly here so nobody compares the two numbers (see Evaluation).
- `t_m ~ U(0,1)` unclamped, per LLaDA. A draw that masks zero tokens
  contributes zero loss but the modality still counts in `n_present` —
  that is what keeps the estimator unbiased.
- ADR 0002's ivar weighting applies inside the per-token NLL term exactly
  as it does today.
- The jetformer noise curriculum composes untouched: it perturbs only the
  embedded z copy, which under diffusion means only the *unmasked* context
  embeddings. Runs that want it off set the existing knobs to 0.

### Generation: minimal random-order unmasking

`astropt3.generation` gains one sampler: start from the target scaffold
with all value tokens masked (or any subset — conditioning is "leave it
unmasked"), and for a fixed number of steps, predict all still-masked
positions, sample z for a random subset from the GMM heads, commit, repeat;
invert the flow at the end. Enough to keep ADR 0003's per-checkpoint sample
panels alive for diffusion runs and to do image↔spectrum conditional
generation in both directions. Confidence-ordered remasking, semi-AR block
decoding, and step-schedule tuning are inference-time refinements deferred
to a follow-up — they need no training change.

### Evaluation

- **`val_loss`:** the diffusion loss is a stochastic estimator, so the
  fixed val batches use seeded, fixed `t_m` and mask draws — the same masks
  at every checkpoint. Diffusion curves are ELBO-scale: comparable across
  diffusion runs, **never against AR runs' curves**.
- **`linear_probe`:** the forward runs with nothing masked (all values
  visible, bidirectional) and mean-pools as today. Probe R² remains the
  cross-objective comparator — and is where bidirectional attention is
  expected to pay off.
- **`samples`:** the minimal sampler above, same fixed templates.

### Scope: both implementations, from scratch

The nanotron mirror is in scope — a training regime that exists only in
the artifact never trained is a design document cosplaying as code. The
mirror's diff: `causal=False` on the varlen call, the mask
embedding/substitution on the embedding block, the masked-position loss on
the head block, TP-synced `t_m`/mask draws (same mechanism as the TP-synced
jet noise). A gpu-marked HF↔nanotron parity test under fixed seeded masks
extends the existing `test_nanotron_gpu.py` pattern.

First runs are **from scratch** at 70M/160M, matching the shakeout pattern
— scratch keeps the A/B against AR clean (same data budget, one variable).
Warm-starting from an AR jetformer checkpoint (Dream-style) is free to try
later: the architectures differ by one mask vector, so an AR checkpoint
loads with `strict=False` and only `mask_embed` initializes fresh. Noted,
not committed to.

## Consequences

### Positive

- Any-order conditioning: image→spectrum and spectrum→image generation,
  arbitrary infill (mask a sky region, mask a wavelength range), joint
  sampling — from the same body, data pipeline, and checkpoint format.
- Bidirectional representations for the probe and downstream encoders.
- Near-minimal diff: one config knob, one learned vector, one mask-building
  function per implementation, one loss branch, one sampler. No new
  tensors through the pipeline; token ids, packing, resume, and TP
  contracts untouched.
- The A/B is honest: identical packed batches under either objective (the
  shuffle stays on), identical probe protocol, one changed variable.

### Negative / tradeoffs

- **Objective incompatibility is total.** An AR checkpoint fine-tuned or
  evaluated under `diffusion` (or vice versa) is meaningless without
  retraining; the config knob rides the checkpoint to prevent silent
  mixing. Loss curves are not comparable across objectives.
- The exact-likelihood claim — jetformer's selling point — becomes a bound
  under this objective.
- The `attention_mask=None` auto-detection elegance is abandoned on the
  diffusion path; the HF side materializes an O(T²) 4D mask (acceptable:
  test artifact).
- `1/t_m` weighting is high-variance at small `t_m`; LLaDA lives with it
  and so do we (no clamp), but expect noisier loss curves than AR.
- Sampler quality is unproven: random-order unmasking with K-component
  GMM commits may need the deferred confidence ordering to produce
  competitive panels. The training objective is unaffected either way.
- One more axis in the config matrix (`objective` × size), though only
  jetformer configs can take it.

## Validation / success criteria

1. `uv run python -m astropt3.train_smoke --config
   configs/model/test-tiny-diffusion.yaml --steps 50 --assert-decrease`
   passes (new config: `test-tiny-jetformer` + `objective: diffusion`).
2. CPU tests: mask derivation isolates pads and objects exactly as the
   causal path does (same segment partition); zero-masked draws leave the
   loss finite and `n_present` correct; the AR path is bit-identical to
   pre-0009 behaviour when `objective: ar`.
3. gpu-marked HF↔nanotron parity under fixed seeded masks, extending
   `test_nanotron_gpu.py`.
4. 70M/160M scratch shakeout: diffusion ELBO decreasing; sample panels
   render in both conditional directions; **probe R² within noise of or
   better than the AR jetformer baseline** — this is the A/B readout and
   the ADR's reason to exist.

## Non-goals / scope (deferred)

- **No regression×diffusion** (Option D) — asserted out.
- **No confidence-ordered remasking or semi-AR decoding** — inference-time
  follow-up, no training change required.
- **No mask-ratio or step-schedule sweeps** — LLaDA defaults first.
- **No interaction with the in-flight scalar-modalities ADR** (0008, on
  its own branch): per-modality `t_m` extends to 1-token scalar spans with
  no new mechanism (the span is masked with probability `t_m`), but
  wiring the two together is that ADR's merge problem, not this one's.
- **No AR↔diffusion checkpoint conversion tooling** — `strict=False`
  loading is the whole mechanism, and warm-start is an experiment, not a
  deliverable.

## Open issues

- Default sampling step count for the panels (LLaDA uses ≈ sequence
  length; our spans are short — 175 value tokens — so start there).
- Whether the shared `mask_embed` should be per-modality instead; identity
  already flows from `embed_tokens`, so shared is the starting point.
- Whether ADR 0005's oversample ratio (tuned for AR draw starvation) is
  still right when spectra are also learned through cross-modal infill.
- RoPE under bidirectional attention: per-object positions are unchanged
  and symmetric attention handles relative offsets in both signs; confirm
  no long-range artifact at the 96×96→144-token scale in the shakeout.

## References

- `astro/src/astropt3/modeling_astropt3.py` — jetformer path, loss
  aggregation, the causal-packing trick.
- `astro/src/astropt3/modalities.py` — `TinyFlow1D`, `GMMHead`, `gmm_nll`.
- `astro/src/astropt3/data/packing.py` — position-id restarts, span order.
- `nanotron/src/nanotron/models/astropt3.py` — `cu_seqlens` derivation.
- [ADR 0002](0002-ivar-weighted-huber-loss.md), [ADR
  0003](0003-checkpoint-samples-in-eval-sidecar.md), [ADR
  0005](0005-include-spectra-from-non-crossmatched-desi.md).
- Nie et al. 2025, *Large Language Diffusion Models* (LLaDA); Sahoo et al.
  2024, *Simple and Effective Masked Diffusion Language Models* (MDLM);
  Swerdlow et al. 2025, *Unified Multimodal Discrete Diffusion* (UniDisc);
  Ye et al. 2025, *Dream 7B* (AR warm-start precedent).
