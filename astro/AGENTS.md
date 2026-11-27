# AGENTS.md (`astro/`)

Guidance for agents working in `astro/`, the AstroPTv3 project package.
Repo-wide layout (fork of `huggingface/smollm`; `text/`, `vision/`, and
`tools/` are read-only upstream reference) is in the root `AGENTS.md`.

## Hard constraints

- **Do not assume machine topology** — this repo runs on dev boxes and
  training clusters with different GPU counts, memory, and schedulers.
  The CPU suite is the fast gate and must pass with no GPUs present;
  GPU-marked tests and training runs go wherever a suitable GPU
  environment exists, and multi-day runs belong on the training cluster
  by preference.
- Tests may use the network, but only `network`-marked ones may require it
  (ADR 0015 streams the corpus live via LSDB; deselect with
  `-m 'not network'` when the HF hub is down).
- Special-token ids in `src/astropt3/tokenization.py` are **frozen**.
- Dependencies are managed only through uv in `pyproject.toml`.

## Commands

```bash
uv sync --extra dev                       # create/update the venv
uv run pytest                             # CPU suite (gpu-marked tests excluded via addopts)
uv run pytest tests/test_model.py::test_pad_invariance   # single test
uv run python scripts/count_params.py     # size table; asserts ±10% of nominal
uv run pytest -m network tests/test_lsdb_stream.py   # live hub stream check
```

These two (pytest, count_params) are the phase verification gates and must
pass before any phase is declared done. `@pytest.mark.gpu`
tests run wherever a GPU environment exists (`uv run pytest -m gpu` in
the GPU venv); on any shared node, pin a device
(`CUDA_VISIBLE_DEVICES`) and check `nvidia-smi` before claiming one.

## Architecture

Data flows record → sequence → packed batch → model; the contract between
stages is implicit and easy to break, so understand it before editing:

1. **Records** are MMU-schema dicts (`image.flux` float32 (3,152,152);
   `spectrum` with 7781-bin `flux/lambda/ivar/mask`); both modalities are
   optional per record (image-only is the common case). Offline fixtures
   live in `tests/legacy_fixture.py`. Real records are **streamed
   live from the HF hub inside each DataLoader worker** by
   `data/catalog_stream.py` via merged LSDB main (commit pinned in
   `uv.lock`): `InfiniteStream` for one catalog, `CrossMatchStream` for several.
   Explicit YAML `catalog_stream.anchors` and `catalog_stream.catalogs`
   define anchors and shared partners. Generation uses the same stream.
   AstroPT composes the sky masks for anchor ownership and evaluation splits;
   LSDB handles catalog coverage, footprint filtering and boundary matching.
   AstroPT retains decoding, calibration, quality gates and bounded retries.
   `min_catalogs` selects sky coverage, not a per-record modality count;
   unknown settings are rejected.
   Each worker owns a dask `LocalCluster`
   (threads only, default `partitions_per_chunk=4` for joins, 1 for anchor-only)
   so the next draw's fetch overlaps the current one's decode; rows decode
   through `NestedFrame.map_rows`, never `iterrows`. `num_loading_workers`
   counts concurrent LSDB readers per DP rank — watch box-wide RSS. Native
   Nanotron loads on one rank per TP group and broadcasts the packed batch. The stream is
   **cursorless**: checkpoints restore weights/optimizer/scheduler/RNG,
   never the record position, so resume opens a fresh stream and may
   revisit records (ADR 0015; `docs/training.md` §5). Recognized
   transport/storage errors discard the iterator and reopen under bounded
   backoff; decode/validation and unknown errors fail immediately.
   Startup logs report reader identity and footprint counts; the local
   benchmark report records catalog configuration and package versions.
2. **`ObjectSequencer`** (`data/packing.py`) turns a record into an
   `ObjectSeq`: central 96×96 crop (`packing.IMAGE_CROP`; JWST cubes are
   already 96×96) + physical band-registry normalization
   (`data/band_registry.py`: rescale to nanomaggies → bright-pixel clamp →
   `arcsinh(x/0.01)` — tokens are flux in knee units of 0.01 nMgy = 10 pMgy,
   O(1) values, keyed on the record's band names — no per-corpus
   calibration; unknown bands raise; spectra get the symmetric ADR 0007
   treatment in `data/spectral.py`: DESI f_λ → AB nMgy via `f_ν = f_λ·λ²/c`
   on the fixed DESI grid, clamped to ±10,000 nMgy, then `arcsinh(f_ν/10 nMgy)`,
   invertible up to the clamp with no side info; unknown grids raise) + patchify (`tokenization.py`) per
   modality — no per-patch standardization: the jetformer tokeniser's
   exact-likelihood loss needs an invertible record -> token map, and
   standardization would discard each patch's mean/std — wrapped in frozen
   special tokens: `<|bos|> <|begin_m|> …placeholders… <|end_m|>` per
   modality, spans serialized in a **uniform random order seeded on
   `crc32(object_id) ^ epoch`** (resume-exact, always on — every
   conditional among the present spans lands in training; ADR 0008,
   generalizing 0005's bimodal 50/50 flip). Calibrated images and display RGB
   channels have separate source-and-band spans, each with 144 patch-8 tokens
   (64 floats; galaxies-with-hats RGB downsamples 256×256 to 96×96 per
   `rgb-{r,g,b}` channel). DESI spectra → 122 patch-64 tokens with normalized per-patch
   mean wavelength as a continuous position (masked, exact-zero, and tail-padded
   bins are noise-filled along the linearly interpolated local continuum); ADR 0008 adds one-token
   **scalar modalities** (`Z` gated on ZWARN==0, `ebv`, joint 3-band
   `photometry`; `data/scalar_registry.py` fixed transforms; the scalar family
   is weighted 0.1) predicted by `GMMHead`s (`scalar_gmm_k`) directly (categorical scalars, `categories > 0`, take a one-hot input and a `CategoricalHead` with cross-entropy instead) — no flow, no
   logdet, plain `gmm_nll` on the raw normalized value. The
   linear probe builds scalar-free sequences (`include_scalars=False`)
   so R² stays a representation metric; `eval/scalar_head.py` is the
   ask-the-model metric (`nmad`/`outlier_frac`/`coverage_1sig`).
3. **`PackedCollator`** greedily packs whole objects (never split) into
   fixed-length rows. Two invariants matter:
   - `position_ids` restart at 0 per object and **are the document mask**:
     the model passes `attention_mask=None` and transformers'
     `create_causal_mask` detects the packed format from the restarts
     (torch ≥ 2.6). Pads get position 0, isolating each as its own segment.
   - Flattened `modality_values`/`modality_positions` are concatenated in
     row-major (batch, time) order — exactly the order boolean-mask
     indexing produces. The model relies on this to align values without
     indices.
4. **`AstroPT3Model`** (`modeling_astropt3.py`): marker-only `embed_tokens` (no
   text vocab, no lm_head) + additive deltas
   `encoder_m(value) + pos_embed_m(position)` at placeholder slots →
   `SmolLM3Model(inputs_embeds=…)` → per-modality regression heads. The
   jetformer tokeniser (JetFormer/GIVT, ported from astroPT v2's
   sogol_branch) is the sole regression head: each patch modality's tokens
   flow through a per-modality `JetFormerFlow` (`flows.py`: affine couplings
   with ViT conditioners attending across each object's modality span;
   `docs/jetformer-flow.md`) to a latent z that is both
   embedded and predicted by a per-modality `GMMHead`, with loss the exact
   patch-space likelihood `mean(NLL_GMM(z) - logdet)` at positions one left
   of each modality token (`<|begin_m|>` predicts patch 0 — astroPT's
   `starts-1` alignment), via `left_shift_mask`; the loss can go negative.
   Each family's loss is one mean over all its tokens (patch-token NLL
   divided by `input_size`, i.e. nats per dimension; scalar tokens as-is),
   and present families combine at image:spectrum:scalar 1:1:0.1. Training can
   add pre-flow noise (`pre_flow_noise_max` per modality), cosine-decayed to
   zero, so the embedded z, target z and logdet use the same flow input. Separate
   linear latent noise (`jetformer_noise_max` -> `_min`) perturbs only the
   embedded z copy. Both schedules use `set_jet_noise_frac`. Scalar modalities
   (`Z`, `ebv`, `photometry`) bypass the flow and go straight to a
   `GMMHead` on the raw normalized value. The nanotron fork mirrors all of
   it (flows/GMM heads on the embedding/head blocks, z+logdet routed to
   the loss, TP-synced noise); sampling lives in `astropt3.generation` +
   `scripts/generate.py`.
5. **Config**: `AstroPT3Config(SmolLM3Config)` carries a `modalities` list
   of dicts; `import astropt3` registers the Auto classes, so it must be
   imported before `AutoModel.from_pretrained` on a checkpoint. Size YAMLs
   in `configs/model/` are loaded by `config_io.load_model_config`.

**Two implementations, one weight source of truth**: this transformers
implementation is the release/probing artifact and CPU test target; actual
pretraining happens in the nanotron fork (`../nanotron` git submodule)
that consumes flat micro-batch dicts built by `data/nanotron_loader.py`
(`{m}_values`/`{m}_positions`/`{m}_mask` + `input_ids`/`position_ids` —
flat because nanotron's device mover only transfers top-level tensors).
The fork adds `src/nanotron/{models/astropt3.py,config/astropt3_config.py}`,
the `astropt3_streaming` dataset type in `run_train.py`, and
`tools/astropt3/convert_{nanotron_to_hf,hf_to_nanotron}.py`; PP=1 and
`tp_mode: ALL_REDUCE` are asserted (modality modules are TP-replicated via
nanotron's tied-parameter mechanism). `nanotron_loader.py` must stay
importable without nanotron; keep all modality/packing logic in `astro/`
so the fork stays thin. gpu-marked tests (`tests/test_nanotron_gpu.py`)
cover HF↔nanotron forward/loss parity,
exact weight-conversion round-trips, TP=2 replicated gradients (including
the noise curriculum), and a 50-step smoke run with checkpoint conversion —
see `docs/training.md` §1 for the GPU venv recipe.

**Checkpointing & eval**: `checkpoints.checkpoint_schedule: pythia` saves
at steps 1,2,4,…,512 plus every `checkpoint_interval` (canonical schedule
in `checkpoint_schedule.py`, lazy-imported by the fork's trainer);
`latest.txt` is written last, so any step dir it covers is complete.
Checkpoints carry weights/optimizer/scheduler/RNG and **nothing else** —
the LSDB stream is cursorless, so resume opens a fresh stream and records
may be revisited (ADR 0015; `docs/training.md` §5). The trainer only scores
the fixed held-out panel (`val/*`, `docs/training.md` §6); for other
evaluation, convert a checkpoint with the fork's
`tools/astropt3/convert_nanotron_to_hf.py`, then score the HF artifact
with `astropt3.eval` (`val_loss`, `linear_probe`, `scalar_head`,
`samples`).

A behavior to remember when touching data or fixtures: per-patch
standardization turns flat/noise-only patches into irreducible N(0,1)
targets — synthetic fixtures must contain patch-scale structure or smoke
training cannot learn. The fixtures' image flux is nMgy-scale (cores ~0.1,
noise ~0.001) so the physical normalization's fixed 0.01 nMgy arcsinh knee
lands in the same regime as on real LegacySurvey data.
