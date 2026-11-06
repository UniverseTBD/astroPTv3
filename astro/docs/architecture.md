# AstroPTv3 architecture

*Background for anyone picking up the project. The operational counterpart is
[`training.md`](training.md); the historical phase plan is
[`../EXPERIMENTS.md`](../EXPERIMENTS.md) Part I.*

Training uses live LSDB access to YAML-configured catalogs and modality lists.
The default recipe combines LegacySurvey North and HSC images with DESI and SDSS
spectra; the size ladder retains its two-survey recipes on the same N-modal path.
[ADR 0015](adr/0015-lsdb-infinite-stream-training.md) records the historical
image-only cutover. This document focuses on the survey-to-model contract.

## What this is

AstroPTv3 is a from-scratch suite of **multimodal astronomical
foundation models** spanning 70M–12B parameters, mirroring the Pythia suite's
sizes and checkpoint schedule so that scaling behaviour can be studied across
the whole family. The recipe combines two lineages:

- **AstroPT** (Smith et al.): autoregressive *next-token regression* over
  continuous embeddings of astronomical data — no quantization, no text
  vocabulary. The model predicts the next image/spectrum patch directly,
  via the jetformer (JetFormer/GIVT-style) flow + GMM regression head.
- **SmolLM3**: the transformer body — a modern decoder stack with grouped-
  query attention (GQA), RoPE with NoPE every 4th layer, RMSNorm, SwiGLU,
  and document-masked packed sequences.

The **Multimodal Universe** catalogs supply partially overlapping sources.
Each record contributes whichever configured modalities it carries; unmatched
right-side records remain separate objects. A join does not guarantee a complete
survey pass. The 70M–12B family studies how modeling changes with scale, not a
claim that every size has already been trained.

## From a survey source to a token sequence

Every training example ("object") is one astronomical source. The path from
survey data to model input (using LegacySurvey and DESI as examples):

1. **Record**: an MMU-schema dict. `image.flux` is float32 `(3, 152, 152)`
   (g/r/z bands); `spectrum` (when present) has 7781-bin `flux`, `lambda`,
   and `mask`. `data/catalog_stream.py` decodes live LSDB/HATS catalog rows
   into configured record keys; `data/nanotron_loader.py` packs them for training.
   Absent modalities stay absent.

2. **Physical normalization** (`data/band_registry.py`, ported from
   galactiktok `feat/norm`): image flux is normalized physically, keyed on
   the record's band names — rescale to LegacySurvey nanomaggies
   (zeropoint + pixel-area factors from the surveys' documentation) →
   clamp survey-flagged bright pixels (per-band `m_bright` ceiling) →
   `arcsinh(flux/0.01)` (flux in knee units: 0.01 nMgy = 10 picomaggies,
   so tokens are O(1)). No data-driven calibration; unknown bands
   raise, `rgb-*` composites pass through raw. Invertible up to the clamp
   (`physical_inverse`). Spectra get the symmetric treatment
   (`data/spectral.py`, ADR 0007): masked bins zeroed → DESI f_λ converted
   to AB nanomaggies per-pixel (`f_ν = f_λ·λ²/c`; the DESI grid is a format
   constant, 3600–9824 Å × 0.8 Å, so the map inverts with no side info) →
   `arcsinh(f_ν/10)` (knee = 10 nMgy, the fiber sky-noise scale). Exactly
   invertible (`spectral_inverse`); unknown grids raise.

   > **Migration note:** checkpoints trained before this change (on the
   > percentile-calibrated Platonic-Universe asinh stretch) are
   > **incompatible** — the normalization target is different. The config
   > field `image_norm_divisor` is back-filled on load, but the weights were
   > trained on a different target: declare incompatible, retrain. The same
   > applies to physnorm checkpoints trained before the switch to knee-unit
   > tokens (dropping the trailing `·divisor`) and the 96×96 central crop
   > (361 → 144 image tokens): incompatible, retrain. Likewise all
   > spectra-bearing checkpoints trained before the physical spectra
   > normalization (ADR 0007, raw-DESI-flux tokens): the config field
   > `spectra_norm_divisor` back-fills, but the record→token map changed —
   > incompatible, retrain.

3. **Patchify** (`tokenization.py`):
   - images → central 96×96 crop (`packing.IMAGE_CROP`; JWST cubes are
     already 96×96) → **144 patches of 192 floats** (8×8×3), integer
     patch-index positions;
   - spectra → pad 7781→7936 → **31 patches of 256 floats**, each with a
     *continuous* position: the patch's mean wavelength, normalized
     `(λ−3000)/7000`.

4. **Sequence assembly** (`data/packing.py::ObjectSequencer`): the
   physically normalized patches are **not** standardized per patch; losing
   each patch's mean and variance would break the flow's likelihood in patch
   space. Present modality spans are wrapped in frozen special tokens and
   shuffled in a deterministic order based on the source id and stream draw.
   An image contributes 144 patches and a spectrum 31; optional scalar
   modalities each contribute a one-token span when configured and present.
   For the image+spectrum-only config, that is 180 tokens per paired source
   or 147 per image-only source, including span markers and `<|bos|>`.
   `tokenization.py` defines the frozen 64-id marker vocabulary; there is no
   text vocabulary or lm_head.

5. **Packing** (`PackedCollator`): whole objects (never split) are packed
   greedily into fixed rows of `sequence_length` (4096);
   the tail is padded. Two invariants the model depends on:
   - `position_ids` restart at 0 for each object and **double as the
     document mask**: the HF model passes `attention_mask=None` and
     transformers builds the block-diagonal causal mask from the restarts;
     nanotron's flash-attention varlen path uses the same signal
     (`_use_doc_masking: true`). Pads get position 0, so each pad is its own
     one-token document, unable to attend to (or be attended by) anything.
   - flattened per-modality `values`/`positions` are concatenated in
     row-major (batch, time) order — exactly the order boolean-mask indexing
     produces — so the model aligns values to slots without index tensors.

## The model

Both implementations share this spec (`modeling_astropt3.py` is the readable
reference):

- **Input assembly**: `embed_tokens(input_ids)` over the 64-id vocab, plus an
  *additive* delta at each modality-placeholder slot:
  `embed(<|m|>) + encoder_m(value) + pos_embed_m(position)`. The placeholder
  embedding acts as a learned modality-type embedding; nothing is overwritten
  in place.
- **Encoders** are a single per-modality `nn.Linear` (data space -> embedding
  space). Image positions go through an `nn.Embedding` (index type), spectra
  positions through a small affine layer (continuous type). Each patch
  modality also has a per-modality normalizing flow (`TinyFlow1D`) that maps
  its raw patch value to a latent z — z is both what gets embedded and what
  the `GMMHead` decoder predicts; scalar modalities (`Z`, `ebv`, `photometry`)
  skip the flow and feed a `GMMHead` directly on the raw normalized value.
- **Body**: the stock SmolLM3 decoder stack, consuming `inputs_embeds`.
  RoPE θ=100k, **NoPE every 4th layer** (`no_rope_layer: 4` /
  `no_rope_layer_interval: 4`), GQA, RMSNorm ε=1e-6, SwiGLU, bf16 training.
- **Loss**: the exact patch-space likelihood `mean(NLL_GMM(z) - logdet)`
  (may be negative) computed at positions **one to the left** of each
  modality token — `<|begin_m|>` predicts patch 0, patch *i* predicts patch
  *i+1* (AstroPT's `starts−1` alignment, implemented via `left_shift_mask`).
  ADR 0013 configs average present modalities within image/spectrum/scalar,
  then combine present family means at 1:1:0.1. Historical configs retain the
  former per-modality `loss_weight` mean. Special tokens and pads carry no
  loss.
- **Init**: stock SmolLM3 `_init_weights`, normal(0, 0.02).

### Size family (Pythia-mirrored nominal sizes, no vocab head)

The 70M, 160M, 410M, 1B, 1.4B, 2.8B, 6.9B, and 12B model configs live in
`configs/model/`. `scripts/count_params.py` calculates their **current**
parameter totals, including JetFormer modality modules, and checks the
nominal-size tolerance. The configuration ladder is not evidence of a
completed training or scientific scaling study at every size.

## Two implementations, one weight source of truth

- **nanotron fork** (`Smith42/nanotron`, branch `astropt3`, git submodule at
  repo root `nanotron/`): the *training* implementation.
  `src/nanotron/models/astropt3.py` is the branch's Qwen2/SmolLM3 stack with
  the vocab-embedding block replaced by the 64-id + modality assembly, and
  the lm_head + sharded-CE loss replaced by modality flows/GMM heads +
  masked exact-likelihood loss.
  `run_train.py` gains an `astropt3_streaming` dataset type that calls back
  into this package's `data/nanotron_loader.py`.
- **transformers implementation** (`src/astropt3/`): the release/probing
  artifact and CPU test target. `import astropt3` registers the Auto
  classes, so `AutoModel.from_pretrained(<converted checkpoint>)` works.
- **Converters** (`nanotron/tools/astropt3/convert_{nanotron_to_hf,
  hf_to_nanotron}.py`): every released checkpoint is converted
  nanotron→HF, exactly as SmolLM3 ships. The conversion roundtrip is
  bitwise; *forward* parity between the two stacks is bf16-tolerance only,
  because nanotron applies RoPE at absolute packed-row positions while HF
  restarts per object (RoPE is relative, so attention agrees, but float
  trajectories differ), and flash-attn vs sdpa kernels differ.

The design rule that keeps this manageable: **all modality/packing logic
lives in `astro/`**; the fork only consumes flat micro-batch dicts
(`{m}_values`, `{m}_positions`, `{m}_mask`, `input_ids`, `position_ids` —
flat because nanotron's device mover only transfers top-level tensors).
`nanotron_loader.py` must stay importable without nanotron installed.

## Parallelism semantics

- **PP = 1 everywhere** (asserted). Even the 12B fits without pipeline
  stages (24GB bf16 weights /TP8 + ZeRO-1 optimizer shards); this keeps
  modality tensors on every rank and avoids pipeline complexity.
- **TP**: the transformer body shards as upstream. The tiny modality
  encoders/decoders/position-embedders stay **replicated** across TP ranks,
  riding nanotron's stock tied-parameter mechanism
  (`mark_unsharded_params_as_tied_across_tp`, `reduce_op=None` — identical
  inputs give identical grads by construction; `scripts/tp2_grad_check.py`
  asserts this). This is why **`tp_mode: ALL_REDUCE` is asserted by the
  model**: REDUCE_SCATTER would shard the hidden stream over the sequence
  and break replication.
- **DP + ZeRO-1**: standard data parallelism with optimizer-state sharding.
  Note that the flattened modality tensors have *different shapes on each DP
  rank* (different objects → different patch counts), which is why
  `general.ignore_sanity_checks: true` is required — nanotron's DP
  input-difference check all-gathers tensors assuming equal shapes.
- **Live survey source**: `data/nanotron_loader.py` uses LSDB inside torch
  DataLoader workers, decodes catalog rows to the shared record contract, and
  packs them before nanotron sees a batch. Crossmatch-enabled configs include
  DESI spectra; other configs can consume LegacySurvey images alone. Seeds
  separate consumers, but there is no cross-rank ownership or exact record
  coverage guarantee. Stream details belong to the active run config and
  [training guide](training.md), not the model contract.

## Training routine

Pythia-style, adapted to a smaller corpus:

- bf16, fused AdamW (β=0.9/0.95, ε=1e-8), weight decay 0.1 on ≥2D params
  (norms excluded), grad clip 1.0, fp32 gradient accumulation.
- LR: linear warmup min(2000 steps, 1%), cosine decay to 0.1× peak.
  Peak LR by size (Pythia values): 1e-3, 6e-4, 3e-4, 3e-4, 2e-4, 1.6e-4,
  1.2e-4, 1.2e-4 for 70M → 12B.
- GBS 2M tokens (512×4096) at cluster scale; live LSDB partition sampling
  is cursorless and does not define a complete corpus epoch.
- **Checkpointing**: `checkpoint_schedule: pythia` saves at steps
  1, 2, 4, …, 512 and then every `checkpoint_interval` (1000) — the log2-
  spaced early checkpoints are what make learning-dynamics studies possible.
  Checkpoints hold model/optimizer/scheduler/RNG weight state only (ADR
  0015): the cursorless LSDB stream restarts fresh on resume.
- **Evaluation (deferred, ADR 0015)**: `astropt3.eval` retains pure
  model-side functions over provided records/objects/batches (`evaluate`,
  `embed_objects`, `ridge_r2`, `scalar_head_metrics`, sampling/rendering).
  Source-backed evaluation and checkpoint sweeps are not currently wired.

## Roadmap context

Earlier phases established the model and training fork; the live LSDB
cutover is explicitly experimental and does not yet satisfy the former
synthetic smoke gate. Current work is on the 70M multimodal pilot, ahead of
full size-ladder scale-up. See [`../EXPERIMENTS.md`](../EXPERIMENTS.md) Part I for the historical
phase log and [`../EXPERIMENTS.md`](../EXPERIMENTS.md) for loader measurements.
