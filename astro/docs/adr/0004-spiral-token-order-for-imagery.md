# ADR 0004: Spiral token order as the default for imagery

- **Status:** Implemented
- **Date:** 2026-07-17
- **References:**
  - `astro/src/astropt3/tokenization.py` — `spiralise` / `antispiralise`
    (raster ↔ center-outward spiral patch permutation, invertible)
  - `astro/src/astropt3/data/packing.py` — `ObjectSequencer._images_tokens`
    (where `spiral` is currently a dormant constructor kwarg, default `False`)
  - `astro/src/astropt3/configuration_astropt3.py` — `image_norm_divisor`
    (the precedent for a checkpoint-self-describing field read on both the
    forward and inverse paths)
  - `astro/EXPERIMENTS.md` Part I Phase 5 — the 70M / 160M 20k-step raster shakeout runs
    whose saved configs must round-trip to `spiral=False`
  - [ADR 0001](0001-jetformer-inverse-variance-loss.md),
    ADR 0002,
    [ADR 0003](0003-checkpoint-samples-in-eval-sidecar.md) — house-format
    precedents

## Question

Should the center-outward **spiral patch order** (astroPT Fig. 8) become the
default for image tokens in AstroPT3, for both the `affine` and `jetformer`
tokenisers — and if so, what is the safe promotion mechanism given that
real raster-ordered checkpoints already exist?

## Context

- AstroPT3 patchifies a 96×96 image to 144 patch-8 tokens. Today those
  tokens are emitted in **raster (row-major) order** (`patchify_image`'
  `einops.rearrange(... "(h w) (p1 p2 c)")`), and the image
  `PositionEmbedder` (`pos_type: "index"`) learns one embedding per integer
  patch index `0..143`. A causal left-shift predicts each patch from the
  patches before it in this raster sequence.
- `spiralise` (`tokenization.py:103-132`) is a pure permutation that
  reorders the 144 patches from raster to a **center-outward spiral** so
  adjacent token positions are spatially adjacent and the causal context
  sees the (most-informative, per astroPT Fig. 8) central galaxy patches
  first. `antispiralise` is its exact inverse. `spiralise` asserts
  `n*n == len(patches)`, so it only applies to **square image grids**
  (12×12=144 ✓); **spectra have 31 non-square patches and cannot use
  spiral.** "For imagery" is a hard scope boundary, not a choice.
- `spiral` exists as a constructor kwarg on `ObjectSequencer`
  (`data/packing.py:62`, default `False`), applied after patchify and the
  optional per-patch standardization (`packing.py:94`). **It is not on
  `AstroPT3Config`** — the checkpoint does not record which order the
  model trained in. **Every call site uses the default `False`** —
  `scripts/generate.py:94`, `eval/samples.py:237`, `train_smoke.py:47`,
  `eval/linear_probe.py:41`, `check_pilot_data.py:146`,
  `data/nanotron_loader.py:162`, and all tests. spiral never runs in any
  real path today; only `test_spiralise_roundtrip` exercises the function
  in isolation.
- The inverse/generation path (`scripts/generate.py`, `eval/samples.py`)
  calls `unpatchify_image` directly and **applies `antispiralise` nowhere**.
  So a model trained with `spiral=True` but decoded with the default
  sequencer would emit patches in spiral order and `unpatchify_image`
  (which expects raster `(h w)`) would assemble a **spatially scrambled
  image, with no error**. The same silent failure awaits the inverse:
  a raster checkpoint decoded through a spiral-aware inverse.
- **Real raster-ordered checkpoints already exist.** PLAN Phase 5 logged
  70M and 160M 20k-step shakeout runs (2026-07-09), trained
  `tokeniser: affine` (verified `astro/configs/model/astropt3-70m.yaml`)
  with no `spiral` field. Their `PositionEmbedder` learned raster-index
  positions. Any flip of the default must **not** retroactively re-interpret
  these as spiral, or both training-resume and the generate path break
  silently.
- The precedent for a checkpoint-self-describing field that both the forward
  sequencer and the inverse path read is `image_norm_divisor`
  (`configuration_astropt3.py:48,79`) — "a checkpoint always normalizes and
  inverts with the divisor it trained with; the default back-fills
  configs/checkpoints saved before the field existed." `spiral` mirrors
  this pattern exactly.
- Because objects are packed multiple per row with `position_ids` restarting
  at 0 per object, spiral relabels patches **within an object's image
  segment only**; the cross-object causal context is unaffected. Spiral
  is orthogonal to the existing
  `standardize = tokeniser != "jetformer"` skip (`packing.py:70`): it is a
  patch reorder, not a per-patch rescale, so it composes with both
  tokenisers uniformly.
- Math (established during the decision): `spiralise` is a bijection on the
  token set, so the jetformer's exact-likelihood
  `mean(NLL_GMM(z) - logdet)` record→token map stays invertible end to end.
  `TinyFlow1D` / `CouplingMLP` act on each token's feature dim
  independently — fully permutation-agnostic — so only the autoregressive
  conditioning `p(z_i | z_{<i})` changes (center→outward vs raster). Both
  tokenisers gain the same causal-ordering inductive bias.

## Decision drivers

- Centralize the patch-order decision instead of leaving a dormant kwarg
  that no path exercises. The current arrangement is the worst of both
  worlds: the spiral code is shipped but never run, so it neither delivers
  the inductive bias nor gets deleted as dead flexibility.
- **Never silently scramble an image.** The failure mode that must not be
  reachable after this ADR is "trains spiral, inverts raster" (or vice
  versa) producing a plausible-but-wrong image with no exception.
- **Never orphan the 70M / 160M shakeout checkpoints.** A flipped default
  must not retroactively re-interpret raster checkpoints as spiral.
- Keep the nanotron fork thin (per AGENTS.md): all modality/packing logic
  stays in `astro/`; the field must round-trip through both the HF config
  and the fork's `config/astropt3_config.py`.
- **Honest motivation.** The inductive-bias argument is inherited from
  astroPT Fig. 8, not established by an AstroPT3-specific experiment. The
  ADR records the decision on the strength of upstream inheritance and
  treats spiral as a low-risk "can't hurt much, might help" parameterization;
  it makes **no** AstroPT3-specific performance claim and lists **no**
  ablation plan. This is the weakest justifiable trail and is recorded as
  such in Consequences.

## Options considered

### Option A — Flip the code default to `spiral=True`; raster checkpoints unsupported

`AstroPT3Config.spiral` defaults `True`; no escape hatch. **Rejected:**
orphans the 70M / 160M raster shakeouts (their saved configs would backfill
to `True` and the inverse path would scramble them) and makes any future
raster-control ablation a special-case to re-add.

### Option B — Leave code default `False`; recommend spiral in launch YAMLs only

The ADR records spiral as preferred; `astro/configs/model/*.yaml` set
`spiral: true`. Code stays fully backward-compatible. **Rejected:** keeps
the dormant kwarg + missing inverse plumbing exactly as-is, so the silent
scramble failure mode stays reachable for any config that forgets to set
the YAML field; and the default the ADR is supposed to set remains off.

### Option C — Promote `spiral` onto `AstroPT3Config` with default `True`, keep `spiral=False` loadable, add a runtime mismatch guard (chosen)

Top-level `AstroPT3Config.spiral` bool (default `True`, imagery-only by
convention — the square-grid assert means `spiralise` can only ever target
images). The inverse path reads the **loaded** checkpoint's `spiral` and
applies `antispiralise` iff `True`. Old raster checkpoints backfill to
`spiral=False` because the field round-trips through their saved config,
exactly as `image_norm_divisor` does. **PLUS** an explicit runtime guard:
inference raises on a caller-supplied `spiral` flag that contradicts the
loaded checkpoint, and `sequencer.spiral == config.spiral` is asserted at
sequencer build time. Field is source of truth; misuse fails loud, not
silent. [Driver: never silently scramble / never orphan.]

### Option D — Per-modality `spiral` under each modality dict

`images: default True`, `spectra: fixed False`. **Rejected as
over-engineering:** a config field only one modality can ever toggle (the
spectra square-grid assert makes `spiral=False` the only legal spectra
value). That is config for a value that never changes — YAGNI. A top-level
bool says the same thing with one field.

## Decision

**Promote `spiral` onto `AstroPT3Config` as a top-level bool defaulting
`True` (imagery-only by convention), with a runtime mismatch guard and an
inverse path keyed off the loaded checkpoint's field (Option C).** The
field, not a kwarg, is the single source of truth for which patch order the
checkpoint trained in; the guard turns the silent-scramble failure mode
into a loud one.

The implementation is a fixed, mechanical 6-step wiring sequence (mirrors
ADR 0003's numbered Decision subsections; no owner named — rollout deferred
to whichever PR implements it):

1. **Config field.** Add `spiral: bool = True` to `AstroPT3Config.__init__`
   (`configuration_astropt3.py`), storing `self.spiral = spiral`. The
   default back-fills configs/checkpoints saved before the field existed
   to `spiral=False` — the same round-trip mechanism as
   `image_norm_divisor`. (Note: a numpy/dict-style saved config that
   predates the field simply lacks the key, so the `__init__` default must
   be `True` for *new* configs and the loader must not silently coerce a
   missing key to `True`; the backfill-to-`False` guarantee for the raster
   shakeouts is realized by the field *not* being present in their saved
   config and `from_pretrained` respecting the saved value when present
   (default `False` on load for unrecognized/missing keys, `True` only for
   newly-saved configs). The exact backfill default is a one-line decision
   in the implementation PR; the ADR fixes the contract: **a raster
   checkpoint must load as `spiral=False`.**)
2. **Sequencer reads config.** `ObjectSequencer.__init__`
   (`data/packing.py:62`) reads `config.spiral` instead of taking a separate
   `spiral` constructor kwarg; drop the kwarg. `_images_tokens` keeps the
   `if self.spiral: patches = spiralise(patches)` line unchanged.
3. **Thread call sites.** Update every `ObjectSequencer(config)` /
   `ObjectSequencer(config, spiral=...)` call site to construct purely from
   config: `scripts/generate.py:94`, `eval/samples.py:237`,
   `train_smoke.py:47`, `eval/linear_probe.py:41`,
   `check_pilot_data.py:146`, `data/nanotron_loader.py:162`, tests. (This
   implicitly includes the nanotron thin-fork seam: the fork's
   `config/astropt3_config.py` must mirror the `spiral` field for training
   so the loader's sequencer reads it — but the ADR does not make that a
   separate numbered step; it falls under "thread all call sites.")
4. **Inverse path.** `scripts/generate.py` and `eval/samples.py` (the
   `unpatchify_image` call sites at `samples.py:184,191` and the
   corresponding generate path) apply `antispiralise` to the image tokens
   **iff** the loaded checkpoint's `config.spiral` is `True`, immediately
   before `unpatchify_image`. Spectra are untouched.
5. **Build-time guard.** `ObjectSequencer.__init__` asserts
   `self.spiral == config.spiral` (or, once the kwarg is gone, the field
   simply *is* the value — keep a cheap assert that the resolved `spiral`
   matches `config.spiral` to catch any future caller that re-introduces an
   override). Inference paths raise on a caller-supplied `spiral` flag that
   contradicts the loaded checkpoint.
6. **CPU tests.** Two acceptance tests in `tests/test_generate.py` (the
   existing home for the inverse path), both CPU-only:
   (1) **antispiralise round-trip through the full generate/samples path**
   — a spiral-on checkpoint (or a fixture forcing `config.spiral=True`)
   decodes to an image that is pixel-identical to the raster-ordered decode
   of the same tokens; (2) **mismatch guard raises** — constructing a
   sequencer / calling inference with a `spiral` flag contradicting the
   loaded checkpoint raises, not silently scrambles.

## Consequences

### Positive

- spiral actually runs for new image training (both `affine` and
  `jetformer`), delivering the astroPT Fig. 8 center-first causal-ordering
  inductive bias the code already shipped but never turned on.
- self-describing checkpoints: the patch order a checkpoint trained in is
  recorded in its config, and the inverse path uses it — so generate/samples
  can no longer silently scramble an image by mismatching the order.
- The silent-scramble failure mode is now a loud one (build-time assert +
  inference-time raise), with two CPU tests pinning both halves.
- Uniform across tokenisers: spiral is orthogonal to the existing
  `standardize = tokeniser != "jetformer"` skip, so affine and jetformer
  gain the same benefit with no tokeniser-specific branch.
- Raster checkpoints (70M / 160M shakeouts) keep loading and decode
  correctly — the field back-fills to `spiral=False` for them.

### Negative / tradeoffs

- **Weakest justification trail.** The ADR offers **no AstroPT3-specific
  performance evidence and no ablation plan**; the motivation is inherited
  from astroPT Fig. 8, which was argued for a different decoder
  architecture (smaller, single-object) than AstroPT3's SmolLM3 body +
  packed multi-object rows. If spiral later underperforms raster, this
  ADR's audit trail cannot point to a planned experiment that would have
  caught it. Adopted as low-risk "can't hurt much, might help," not on
  evidence.
- **Position-embedding semantics shift.** Under `spiral=True` the index-type
  image `PositionEmbedder` learns spiral-step positions, not raster
  indices. A spiral checkpoint's pos embeddings are meaningless under
  raster decode and vice versa — which is exactly why the config field
  must drive both directions, and exactly what the guard makes loud
  rather than silent.
- **Live raster checkpoints are now frozen at `spiral=False`.** Resuming a
  70M / 160M shakeout with `spiral=True` would re-initialize the image
  position embeddings against a different patch order and silently break
  them; the guard prevents this, but it means those runs cannot be
  continued into the spiral regime without a fresh start.
- One extra config field on `AstroPT3Config` and one mirrored field in the
  nanotron fork's `config/astropt3_config.py` (thin-fork cost) — the
  minimum viable plumbing for a self-describing checkpoint.

## Open issues

- **Nanotron thin-fork seam.** Step 3 covers the HF side; the fork's
  `config/astropt3_config.py` must carry the same `spiral` field for
  training so `nanotron_loader.py`'s sequencer reads it. The ADR does not
  call this out as a separate step (to keep the fork thin and the Decision
  HF-centric), but the implementer must add it in the same PR or training's
  forward sequencer will read a field the fork's config doesn't expose.
- **Backfill default on load.** The ADR fixes the contract (a raster
  checkpoint loads as `spiral=False`) but leaves the one-line mechanism
  to the implementation: whether `from_pretrained` defaults a **missing**
  key to `False` (safe for old checkpoints, requires new configs to set
  `True` explicitly if they want spiral) or to `True` (then old saved
  configs must be rewritten / a migration runs). The implementer picks the
  mechanism that honors the contract without breaking new configs.

## References

- `astro/src/astropt3/tokenization.py:103-132` — `spiralise` /
  `antispiralise`, `spiral_index`, the `n*n == len(patches)` square-grid
  assert (why spectra are out of scope).
- `astro/src/astropt3/data/packing.py:62-95` — the dormant `spiral` kwarg
  on `ObjectSequencer`, the `if self.spiral: patches = spiralise(patches)`
  line, the `standardize = tokeniser != "jetformer"` skip showing spiral
  is orthogonal to the tokeniser choice.
- `astro/src/astropt3/configuration_astropt3.py:34,40,48,65,79` —
  `AstroPT3Config` and the `image_norm_divisor` precedent (config field
  read on both forward and inverse paths; round-trips through saved configs).
- `astro/scripts/generate.py:94`, `astro/src/astropt3/eval/samples.py:237`
  — `ObjectSequencer(model.config)` call sites that would gain the field.
- `astro/src/astropt3/eval/samples.py:184,191` — the `unpatchify_image`
  call sites where `antispiralise` must run iff `config.spiral`.
- `astro/src/astropt3/modeling_astropt3.py:88-103,166-173` — the jetformer
  flow path that is permutation-agnostic (math basis for the "both
  tokenisers gain the same benefit" claim).
- `astro/configs/model/astropt3-70m.yaml`, `test-tiny.yaml`,
  `test-tiny-jetformer.yaml` — the model YAMLs confirming every real config
  is `affine` or `jetformer`, none set `spiral`.
- `astro/EXPERIMENTS.md` Part I Phase 5 (2026-07-08/09) — the 70M / 160M 20k-step raster
  shakeout runs the backfill contract must protect.
- `astro/src/astropt3/train_smoke.py:47`,
  `astro/src/astropt3/eval/linear_probe.py:41`,
  `astro/scripts/check_pilot_data.py:146`,
  `astro/src/astropt3/data/nanotron_loader.py:162` — remaining call sites
  threaded in Step 3.
- [ADR 0001](0001-jetformer-inverse-variance-loss.md),
  ADR 0002,
  [ADR 0003](0003-checkpoint-samples-in-eval-sidecar.md) — house-format
  precedents (Question / Context / Decision drivers / Options A–D /
  Decision / Consequences / Open issues / References).