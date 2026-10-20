# AstroPTv3 lab book

One document for the project: the charter and roadmap, the plan/phase
history, and the experiment log. Verification gates live in
`AGENTS.md` (repo root and `astro/`); architecture detail in
`docs/architecture.md`; decision records in `docs/adr/`.

## 0. Charter — MMU Streaming × AstroPT

**Context.** MMUv1.5 (July 2026) makes crossmatched Multimodal Universe
data reachable without staging the full 80TB+ locally — crossmatching
happens on the hub via lsdb:

```python
import lsdb

gz10 = lsdb.open_catalog("hf://datasets/UniverseTBD/mmu_gz10")
sdss = lsdb.open_catalog("hf://datasets/UniverseTBD/mmu_sdss_sdss")

xmatch = gz10.crossmatch(sdss, n_neighbors=1).compute()
```

The remaining gap: streaming crossmatched examples still cannot keep GPUs
busy, so training today requires locally staged or pre-crossmatched data.
Astronomy will soon wrangle far more than the MMU's 80TB — the goal of
this project is **pre-training on remotely hosted data that is
crossmatched on the fly**.

**End goal.** An importable, model-agnostic `mmu_streaming` package so
that anyone can train on crossmatched MMU data at a reasonable pace,
HF-streaming style:

```python
from datasets import IterableDataset
from mmu_streaming import crossmatch
from astropt import AstroPT

data: IterableDataset = crossmatch(["LegacySurvey", "DESI", "HSC-Wide"])
AstroPT().fit(data.shuffle(seed=42))
```

**AstroPTv3's role.** First test-case for the fast crossmatch streaming
interface. V3 builds on HF's SmolLM (SmolLM3 transformer backbone +
nanotron training infrastructure), takes patched astronomical data via
linear projection, and trains through a JetFormer-style autoregressive
reconstruction objective (tokeniser lineage: Sogol's `sogol_branch`). We
train a sweep of models at a fixed scaling ladder to stress the streaming
interface and study how astro-foundation models learn (and grok)
astronomical concepts under a generative objective. **The goal is not the
"best astronomical foundation model"** — it is understanding learning
dynamics. Following EleutherAI's Pythia, we release checkpoints throughout
training, with training configurations and the information needed to
reproduce the training streams.

### Roadmap and status (2026-09-28)

| # | Roadmap item | Status |
|---|---|---|
| 1 | LSDB [#1590](https://github.com/astronomy-commons/lsdb/pull/1590) — outer-join crossmatch | In flight upstream. Local stopgap in production here: `data/outer_crossmatch.py` (`OuterKdTreeCrossmatch`) |
| 2 | LSDB [#1584](https://github.com/astronomy-commons/lsdb/pull/1584) — filtering crossmatch stream (`CrossMatchStream`) | In flight upstream; tracked as a git dependency (`selective-x-match`); adopted at `count_fraction_threshold: 0.0` (§9) |
| 3 | Many-modal (>2 catalog) crossmatching in LSDB | Not started upstream. Existence proof + API shape: this repo streams DESI × HSC × Legacy through one `CrossMatchStream` (sequential right-catalog joins) — validated by the 10k-step trimodal trial (§11, PR #33) |
| 4 | Streaming speedups upstream | Submit-overlap patch validated (+13.6% tokens/s, §9) on `Smith42/lsdb:xmatch-stream-overlap`; held local until upstream chooses the integration path |
| 5 | `mmu_streaming` package (model-agnostic, HF `IterableDataset`) | Not started. The logic it will extract lives in `astropt3.data.nanotron_loader` (catalog registry, column projection, join order, decode) — deliberately kept out of the nanotron fork for exactly this move |
| 6 | AstroPT sweep on Perlmutter (20k shared hours) or other compute; ~7B aspirational | Not started. Ladder configs exist for all eight sizes (Part I); historical 70M/160M 20k-step shakeouts (Part I, Phase 5); 70M v3 trimodal 10k trial done (§11) |
| 7 | Release: AstroPTv3 + MMU-Streaming — weights, GitHub, paper, blog post | Not started |

Upstream asks already filed/queued on #1584: precompute/reuse per-pixel
crossmatch graphs, and move the skip decision off the loader's critical
path (§9). Re-pin lsdb to a release once #1584 ships.

# Part I — Plan & phase history

The fixed project plan and phase record: decisions are final, and later
ADRs (0006, 0011, 0013, 0014, 0015) supersede sections in place with
banners — read top-down as a record, not as current-state documentation.


## Context

AstroPTv3 trains an open, from-scratch suite of multimodal
astronomical foundation models (70M–12B, Pythia-mirrored sizes and
checkpointing) by porting the AstroPT approach — autoregressive next-token
**regression** over continuous embeddings of images/spectra — onto the SmolLM3
architecture. This repo (the `astroPTv3` checkout)
is a fork of `huggingface/smollm`. Scope: **architecture + pre-training only**.

Reference implementation: `../astroPT` (branch `multi-gpu-llm`) —
`src/astropt/model.py` has `ModalityConfig`/`ModalityRegistry`,
`Encoder`/`Decoder`, Huber next-token regression, and an LLM-backbone path
using `<|begin_mod|>`/placeholder/`<|end_mod|>` special tokens;
`src/astropt/local_datasets.py` has patchify + collate semantics
(targets shifted to `starts-1`).

## User decisions (fixed)

1. **Training framework: nanotron** (fork of `huggingface/nanotron@smollm3` —
   branch verified to exist, `run_train.py` at root). Parallelism strategy per
   the **Ultra-Scale Playbook**. The **transformers implementation of the model
   is kept as the release/probing artifact** (nanotron→HF conversion, exactly
   how SmolLM3 itself ships).
2. **From scratch** at all sizes; Pythia-style checkpoint schedule.
3. **Minimal special vocab** (64 ids); no natural-language text; no 128k
   lm_head — outputs are per-modality regression decoders.
4. **Pilot data**: `UniverseTBD/mmu_ssl_legacysurvey_north` ×
   `UniverseTBD/mmu_desi_edr_sv3` via lsdb crossmatch (schemas verified below);
   time series + tabular later as config-only extensions.
5. **Jetformer tokeniser default and sole regression head** (flow + GMM head;
   decision rewritten 2026-09-01 — it previously read "affine tokeniser
   default, single `nn.Linear` per direction; aim MLP selectable via
   config." Both the affine `Decoder` and the never-wired-into-HF `aim` MLP
   variant are removed from the codebase).
6. **uv** manages environments; new deps in `astro/pyproject.toml`; upgrading
   existing venv packages is fine.
7. **GPU work and training runs are allowed on this machine** (2×A100 80GB
   plus slurm; rule rewritten 2026-08-06, it previously forbade both). CPU
   unit/smoke tests stay the fast gate and must be green first; the node is
   shared, so pin a device and check `nvidia-smi` before claiming one.
   Multi-day production runs still belong on the training cluster by
   preference, not by prohibition.

## Verified pilot-data schemas (checked 2026-07-07 via HF datasets-server + MMU builders)

**`UniverseTBD/mmu_ssl_legacysurvey_north`** — 14,174,203 rows, ~4 TB:

- `image` struct: `band=["des-g","des-r","des-z"]`, `flux` float32 **(3, 152, 152)**
  (builder constant `_image_size=152`, `_pixel_scale=0.262`), `psf_fwhm`, `scale`.
- Scalars: `flux_{g,r,z}`, `fiberflux_{g,r,z}`, `psfdepth_{g,r,z}`, `ebv`, `z_spec`.
- `ra` f64, `dec` f64, `object_id` string, `_healpix_29` int64.
- ⚠ Raw flux (not JPG), 152 px — **not divisible by 16**; patch size must change vs AstroPTv1.

**`UniverseTBD/mmu_desi_edr_sv3`** — 1,126,441 rows, ~86 GB:

- `spectrum` struct of length-**7781** sequences: `flux`, `lambda`, `ivar`,
  `lsf_sigma` (float32), `mask` (bool).
- `Z`, `ZERR`, `ZWARN`, `FLUX_*`/`FIBERFLUX_*` photometry, `EBV`, `ra`, `dec`,
  `object_id`, `_healpix_29`.
- Crossmatched pilot corpus is bounded by DESI: expect ~0.5–1M matched pairs;
  the remaining ~13M image-only objects are still usable as single-modality
  sequences (design requirement below).

## Design

### Two model implementations, one weight source of truth

- **nanotron fork** (`Smith42/nanotron`, branch `astropt3`, forked from
  `huggingface/nanotron@smollm3`): training-time model with TP + ZeRO-1 DP.
  Added as a **git submodule at repo root `nanotron/`**, installed editable via
  a `[train]` extra (needs flash-attn; any GPU box, this one included).
- **transformers implementation** (`astro/src/astropt3/`): release artifact,
  probing/eval, and CPU tests. `AstroPT3Config(SmolLM3Config)` +
  `AstroPT3Model(SmolLM3PreTrainedModel)`, registered with Auto classes
  (SmolLM3 classes verified present in transformers 4.57.1).
- **Conversion scripts both ways** + tiny-config forward-equivalence test.
  Every released checkpoint is converted nanotron→HF (as SmolLM3 does).

### Repo layout

> **Superseded in part by [ADR 0015](docs/adr/0015-lsdb-infinite-stream-training.md)
> (2026-09-01).** The listing below was written against the match-index/
> `mmu-stream` corpus (ADR 0006/0011/0013). ADR 0015's hard cutover deleted
> `data/streaming.py`, `data/match_index.py`, `data/synthetic.py`,
> `train_smoke.py`, `scripts/build_match_index.py`, and
> `scripts/run_probe_sweep.py` outright, and replaced the training-record path
> with `lsdb.streams.InfiniteStream` over a single uncrossmatched catalog
> feeding `data/nanotron_loader.py` directly. The listing has been updated to
> the current tree; see the ADR for what changed and why.

```
astroPTv3/
├── nanotron/                        # git submodule → Smith42/nanotron@astropt3
│   └── (fork adds:)
│       src/nanotron/models/astropt3.py        # SmolLM3-style body; modality embed/loss blocks
│       src/nanotron/config/astropt3_config.py # modalities section in nanotron YAML config
│       tools/astropt3/convert_{nanotron_to_hf,hf_to_nanotron}.py
├── astro/
│   ├── pyproject.toml               # uv project "astropt3": torch, transformers>=4.57,
│   │                                # datasets>=4.3, einops, pyyaml, pytest, wandb, lsdb
│   │                                # [train]: nanotron (editable ../nanotron), flash-attn (training machine)
│   ├── src/astropt3/
│   │   ├── __init__.py              # AutoConfig/AutoModel registration
│   │   ├── configuration_astropt3.py
│   │   ├── modeling_astropt3.py     # HF release/probing model
│   │   ├── modalities.py            # ModalityConfig/Registry, Encoder, Decoder, PositionEmbedder
│   │   ├── tokenization.py          # patchify/unpatchify, normalization, SPECIAL_TOKENS (frozen)
│   │   ├── generation.py            # jetformer/GIVT sampling
│   │   ├── checkpoint_schedule.py   # Pythia checkpoint step schedule
│   │   ├── data/
│   │   │   ├── nanotron_loader.py   # opens the LSDB InfiniteStream per worker, feeds the sequencer/packer
│   │   │   ├── packing.py           # ObjectSequencer + PackedCollator (shared by HF & nanotron paths)
│   │   │   ├── band_registry.py     # physical per-band normalization (rescale → clamp → arcsinh)
│   │   │   ├── spectral.py          # DESI spectral normalization (ADR 0007)
│   │   │   ├── scalar_registry.py   # scalar-modality transforms (ADR 0008)
│   │   │   ├── transforms.py        # per-patch standardization
│   │   │   └── telemetry.py
│   │   └── eval/{val_loss.py,linear_probe.py,scalar_head.py,samples.py}   # pure model-side functions (ADR 0015)
│   ├── configs/
│   │   ├── nanotron/astropt3-{70m,160m,410m,1b,1p4b,2p8b,6p9b,12b}.yaml   # full nanotron configs
│   │   └── model/…yaml (HF-side mirrors + test-tiny)
│   ├── scripts/
│   │   ├── count_params.py          # asserts each size within 10% of nominal
│   │   └── launch_slurm.sbatch      # torchrun → nanotron run_train.py, multi-node
│   └── tests/                       # CPU-only by default; @pytest.mark.gpu for nanotron parity
└── text/, vision/, tools/           # upstream smollm, untouched (reference)
```

### Model (both implementations share this spec)

> **Updated 2026-09-01.** The affine `Decoder`/Huber-loss regression head
> this section originally described is removed: the jetformer flow + GMM
> head (originally an alternative `tokeniser: jetformer` option) is now the
> sole regression head, made the Pythia ladder's default. See
> [`adr/0015-lsdb-infinite-stream-training.md`](docs/adr/0015-lsdb-infinite-stream-training.md)
> for the corpus cutover in the same commit; the tokeniser change is
> unrelated to that ADR but landed alongside it.

Tiny 64-id special-token embedding + SmolLM3 decoder stack (GQA, NoPE every
4th layer, RMSNorm, SwiGLU, doc masking) + per-modality `Encoder` (a single
`nn.Linear`) + per-modality `TinyFlow1D` + `GMMHead` + `PositionEmbedder`
dicts. No lm_head.

- Slot embedding is **additive**: `embed(<|m|>) + encoder_m(value) + pos_m(position)`
  (placeholder = learned modality-type embedding; no in-place overwrite).
  Patch modalities embed the flow's latent z; scalar modalities (`Z`, `ebv`,
  `photometry`) bypass the flow and embed the raw normalized value.
- Loss: the exact patch-space likelihood `mean(NLL_GMM(z) - logdet)` (may be
  negative) on each modality span, predictions taken one position left of
  each modality token (`<|begin_m|>` predicts patch 0 — astroPT `starts-1`
  semantics). ADR 0013 configs average modalities within each
  image/spectrum/scalar family and combine present family means at 1:1:0.1;
  historical configs retain the prior `loss_weight` mean. No loss on
  special/pad tokens (`special_token_ce_weight` hook kept for later
  variable-length modalities).
- Positions: SmolLM3 RoPE/NoPE unchanged over the flat packed sequence with
  per-object-reset `position_ids` (doubles as nanotron's doc-masking signal for
  FA varlen); per-modality embeddings added at input.
- Init: stock SmolLM3 `_init_weights` (normal 0.02).

Special vocab (frozen in `tokenization.py`): `0 <|pad|>`, `1 <|bos|>`, then
three ids per modality in addition order. Existing ids 2–16 never move; ADR
0013 configs consume complete blocks in 17–63 before explicitly enlarging the
vocabulary above 63.

### Modality tokenization (pinned to verified schemas)

- **Images**: `image.flux` (3,152,152) float32 → physical band-registry
  normalization (rescale → clamp → arcsinh; superseded the asinh stretch —
  `data/band_registry.py`, `docs/architecture.md`) → einops
  `"c (h p1) (w p2) -> (h w) (p1 p2 c)"` with **patch 8** → **361 tokens** of
  `input_size=192`; per-patch standardization; integer patch-index positions
  (spiral option ported). Patch 8 chosen because 152 = 8×19 (16 doesn't divide
  152) and higher tokens/object stretches the limited corpus.
- **Spectra**: `spectrum.flux` (7781) → pad to 7936 → **31 patches** of
  `input_size=256`; per-patch standardization; position = per-patch mean
  `lambda` normalized `(λ-3000)/7000` through a small affine PositionEmbedder
  (`pos_type="continuous"`). `mask==True` bins zeroed before patching.
  Option (not pilot-default): ivar-weighted Huber.
- **Missing modalities are allowed**: `ObjectSequencer` emits only the
  modalities present, so image-only objects (~13M) train alongside ~1M
  image+spectra pairs.
- Per object: `<|bos|> <|begin_images|> ×361 <|end_images|> <|begin_spectra|> ×31 <|end_spectra|>`
  = 397 tokens (image-only: 364). Greedy packing into seq 4096 (~10
  objects/seq), tail padded, pad excluded from loss/attention.

### Model-size table (named size ≈ TOTAL params; no vocab head)

GQA, NoPE interval 4, seq 4096. With the affine default, modality extras are
~2560×hidden (≈1M at 70M → ≈13M at 12B); small sizes gain a layer or two to hit
nominal totals — finalized in Phase 1 by `count_params.py` (±10% assert).

FINAL (Phase 1, verified by count_params.py on meta device; all within 1.8% of nominal):

| Name | layers | hidden | heads | kv | head_dim | inter | total (measured) |
| ------ | -------- | -------- | ------- | ---- | ---------- | ------- | ------------------ |
| 70M | 23 | 512 | 8 | 2 | 64 | 1536 | 70.04M (+0.1%) |
| 160M | 25 | 768 | 12 | 4 | 64 | 2048 | 158.3M (−1.0%) |
| 410M | 27 | 1024 | 16 | 4 | 64 | 4096 | 411.9M (+0.5%) |
| 1B | 22 | 2048 | 16 | 4 | 128 | 5632 | 994.8M (−1.5%) |
| 1.4B | 31 | 2048 | 16 | 4 | 128 | 5632 | 1.401B (−0.7%) |
| 2.8B | 36 | 2048 | 16 | 4 | 128 | 11008 | 2.815B (+0.5%, exact SmolLM3-3B body) |
| 6.9B | 38 | 4096 | 32 | 8 | 128 | 11008 | 6.740B (−1.8%) |
| 12B | 42 | 5120 | 40 | 8 | 128 | 14336 | 11.90B (+0.4%) |

### Nanotron fork surgery (branch `astropt3`)

1. `src/nanotron/models/astropt3.py`: copy the branch's SmolLM3 (qwen2-style)
   model; replace the vocab `TensorParallelEmbedding` block with the 64-id
   embedding + modality-encoder assembly; replace the lm_head
   `TensorParallelColumnLinear` + sharded-CE `Loss` block with per-modality
   affine decoders + masked Huber loss block.
2. **PP=1 everywhere** (playbook: don't take on pipeline complexity unless
   memory forces it — it doesn't, see recipe table). This means modality
   tensors never cross pipeline stages; the whole batch dict goes to every rank.
3. **TP**: transformer body sharded as upstream; modality encoders/decoders are
   tiny affine layers kept **replicated** across TP ranks (identical inputs →
   identical grads; verify nanotron's `NanotronParameter` reduce semantics for
   replicated modules in the tiny parity test — this is the one subtle bit).
4. Config: `modalities` section added to the nanotron model config dataclass;
   our YAMLs otherwise mirror `text/pretraining/smollm3/stage1_8T.yaml`
   conventions (`_use_doc_masking: true`, etc.).
5. Dataloader: new dataset type `astropt3_streaming` wired into
   `run_train.py`'s `get_dataloader`, implemented by
   `astro/src/astropt3/data/nanotron_loader.py`: MMUIterableDataset →
   ObjectSequencer → PackedCollator → micro-batch dicts
   (`input_ids`, modality values/masks/positions, `position_ids`,
   `label_*` targets), sharded by **DP rank** (`split_dataset_by_node`),
   identical stream within a TP group.
6. Checkpointing: patch the trainer's interval check with
   `should_checkpoint(step)` — Pythia schedule {1,2,4,…,512} then every 1000
   steps (~2B tokens at GBS 2M) + final. Save `datasets` stateful-iterable
   `state_dict()` alongside nanotron's checkpoint for resume.
7. Conversion: `tools/astropt3/convert_nanotron_to_hf.py` (backbone weight map
   - modality modules), modeled on the fork's existing llama converters.

### Parallelism recipe (Ultra-Scale Playbook; H100 80GB, seq 4096, GBS 2M tokens = 512×4096)

| Size | GPUs (grant) | TP | DP | ZeRO | Activation recompute | Notes |
| ------ | -------------- | ---- | ---- | ------ | ---------------------- | ------- |
| 70M–410M | 32 | 1 | 32 | 1 | none | max micro-batch that fits |
| 1B–1.4B | 64 | 1 | 64 | 1 | selective | |
| 2.8B | 64 | 2 | 32 | 1 | selective | |
| 6.9B | 128 | 4 | 32 | 1 | selective/full | |
| 12B | 256 | 8 | 32 | 1 | full | TP inside NVLink node; **no PP** (24GB bf16 weights /8 + ZeRO-1 optim shards fit comfortably) |

Playbook practices baked into the plan: TP never crosses the node boundary;
prefer ZeRO-1 + activation recomputation before adding PP; benchmark
micro-batch size and grad-accum to hit GBS; profile the first ~100 steps
(torch profiler) before committing node-hours; track tokens/s/GPU + MFU
(target ~30–45% depending on size); FA2 varlen doc masking via
position_ids; bf16 + fused AdamW (β 0.9/0.95, wd 0.1 on ≥2D params), grad clip
1.0, cosine to 0.1× peak, warmup min(2000 steps, 1%).
Peak LR by size (Pythia): 1e-3 / 6e-4 / 3e-4 / 3e-4 / 2e-4 / 1.6e-4 / 1.2e-4 / 1.2e-4.
Corpus is small vs Pythia's 300B tokens (~5.7B tokens/epoch pilot); multi-epoch
training is accepted (per AstroPTv1 findings); token budget per size set at
launch time in the YAML.

### Data pipeline

> **Superseded by [ADR 0006](docs/adr/0006-stream-mmu-upstream.md) (2026-07-17).**
> The local-reshard + offline-compute-node design below (lsdb crossmatch →
> `PILOT_FEATURES` parquet shards → offline `MMUIterableDataset`) is replaced by
> streaming the MMU catalogs directly at train time (lsdb/HATS + `CatalogStream`,
> internet required on compute nodes). Local shards, `prepare_pilot_data.py`,
> `PILOT_FEATURES`, and `assign_split` are removed; see the ADR for the three
> interleaved streams, partition-granularity resume, and reserved-partition val.
> The description below is retained for historical context.

- **Prep (login node, `[data]` env)**: `prepare_pilot_data.py` —
  `lsdb.open_catalog("hf://datasets/UniverseTBD/mmu_ssl_legacysurvey_north")`
  LEFT-crossmatch `mmu_desi_edr_sv3` (≤1″): all images kept, spectra attached
  where matched → ~256MB parquet shards at
  `$ASTROPT3_DATA_ROOT/{train,val}/` (default `../astroPTv3_data/pilot_v1`
  beside the repo) + provenance json.
  Fallback if lsdb won't install: HF streaming + homemade HEALPix join (same
  output schema).
- **Train/val split by coarse HEALPix pixel** (from `_healpix_29`) — spatially
  disjoint, no near-duplicate leakage.
- Training streams the local shards with `load_dataset("parquet", ...,
  streaming=True)` + `HF_DATASETS_OFFLINE=1` — no network/lsdb on compute nodes.
- Image normalization is physical (band-registry constants, no per-corpus
  calibration; superseded the original `compute_norm_stats.py` percentile
  calibration — `data/band_registry.py`).
- `synthetic.py` generates records matching the **verified schemas** above —
  all tests and the CPU smoke loop run networkless.

## Phases (each a reviewable PR)

### Phase 1 — `astro/` package: modalities, tokenization, packing, HF model

Create `astro/` scaffold (uv), port from `../astroPT/src/astropt/model.py`
(ModalityConfig/Registry l.41-71; Encoder/Decoder l.272-315 — affine default)
and `local_datasets.py` (patchify/spiralise); implement `tokenization.py`
against the verified MMU shapes, `packing.py`, `synthetic.py`,
`configuration_astropt3.py` + `modeling_astropt3.py`, size YAMLs,
`count_params.py`, `train_smoke.py`.
**Verify (this machine, CPU)**: pytest green, no network — patchify↔unpatchify
roundtrip exact on (3,152,152) and (7781,) fixtures; packing never splits an
object and handles image-only objects; masked positions contribute zero loss;
doc mask blocks cross-object attention (perturb object A → object B hidden
states identical); target alignment on a hand-built example; tiny-model
forward/backward finite with grads on every param; `save_pretrained` →
`AutoModel.from_pretrained` → identical outputs; `count_params.py` ±10% per
size; 50-step CPU smoke on synthetic: loss < 0.7× initial.

### Phase 2 — Pilot data prep + streaming dataset

`prepare_pilot_data.py`, `data/mmu.py`, `configs/data/pilot_images_spectra.yaml`
(all three since deleted — ADR 0006 replaced the local reshard with the live
stream; the original `compute_norm_stats.py` calibration step was retired for
physical normalization).
**Verify**: crossmatch logs matched/unmatched counts (expect ~0.5–1M matched,
~13M image-only); decoded-object sanity print (patch stats ~N(0,1) after
stretch, λ range 3600–9824Å); dataloader-only throughput ≥2× training
consumption; 2 ranks × 2 workers yield disjoint object_ids. (lsdb runs on the
login node here; nothing GPU.)

### Phase 3 — Nanotron fork: model, config, dataloader, conversion

Fork `huggingface/nanotron@smollm3` → `Smith42/nanotron@astropt3`; submodule at
`nanotron/`; implement fork items 1–7 above; `nanotron_loader.py`;
`convert_nanotron_to_hf.py` (+ reverse); nanotron YAMLs for test-tiny + 70M.
**Verify (training machine, 1 GPU, gpu-marked tests)**: tiny-config parity —
convert HF↔nanotron and match forward losses on a fixed synthetic batch to
bf16 tolerance; replicated-module gradient check across TP=2; 50-step nanotron
run on synthetic data: loss decreases; conversion of that checkpoint loads via
`AutoModel.from_pretrained` and reproduces val loss.

DONE (verified on a shared A100 node, GPU-pinned, tiny configs only). Fork
items 1–5 + 7 delivered; item 6 (Pythia checkpoint schedule) is Phase 4 per
the phase split. Notes:

- Modality encoders/decoders/pos-embedders ride nanotron's stock
  `mark_unsharded_params_as_tied_across_tp` (replicated across TP,
  `reduce_op=None` under ALL_REDUCE — "synced by design", asserted identical
  grads in `astro/scripts/tp2_grad_check.py`). `tp_mode: ALL_REDUCE` is
  asserted by the model: REDUCE_SCATTER shards the hidden stream over the
  sequence and would break replication (revisit only if throughput demands).
- nanotron applies RoPE at absolute row positions in the packed row; HF
  restarts per object. RoPE is relative so attention agrees — parity is
  therefore bf16-tolerance, not bitwise (conversion roundtrip IS bitwise).
- GPU env recipe (no prebuilt flash-attn for torch 2.12/cu13 yet):
  `uv venv --python 3.13; uv pip install torch==2.8.0
  <flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp313 wheel from GitHub>
  -e nanotron -e astro psutil` then `pytest -m gpu astro/tests/test_nanotron_gpu.py`.
- Fork carries 4 small compat/bug fixes vs upstream (functorch tree_map,
  flash-attn 2.8 rotary signature, unbound `tied_name` in weight-decay
  exclusion, consumption-stats hasattr guards) + relaxed numpy pin, psutil dep.
- `Smith42/nanotron` fork + `astropt3` branch exist locally in `nanotron/`;
  pushing to GitHub needs credentials (create the fork, then
  `git -C nanotron push origin astropt3` and push the recorded gitlink).

### Phase 4 — Checkpoint schedule, resume, eval hooks

Pythia `should_checkpoint` patch + dataset-state save; `eval/val_loss.py`
(fixed 512 val batches per eval interval) + `eval/linear_probe.py` +
`run_probe_sweep.py` (async over converted HF checkpoints — never blocks
training; ridge probe → redshift `Z`, which the pilot data carries natively).
**Verify (training machine)**: kill-at-step-137/resume overlays the
uninterrupted loss curve without sample replay (object_id hash log);
checkpoint dirs at exactly steps 1,2,4,…,512,1000,…; each converts and loads.

DONE (verified on the reserved A100s, tiny synthetic configs,
2026-07-08; gpu gates in `tests/test_phase4_gpu.py`). Notes:

- Schedule: `checkpoints.checkpoint_schedule: pythia` (new optional
  `CheckpointsArgs` field) = powers of two ≤ 512 ∪ multiples of
  `checkpoint_interval`; canonical implementation in
  `astropt3/checkpoint_schedule.py`, lazy-imported by the fork's trainer.
  1000-step run produced dirs at exactly 1,2,4,…,512,1000; all 11 convert
  and load via `AutoModel.from_pretrained`.
- Stream state: `PackedMicroBatches.state_dict()` = position at the START
  of the current partial packing row (partial rows are untrained, so resume
  re-draws them — no tensor serialization, exact continuation). Saved per
  DP rank to `{ckpt}/dataset_state/dp_{rank}.pt` before `latest.txt`;
  loaded in `run_train.py` from `trainer.init_checkpoint_path`. Requires
  `num_loading_workers: 0` (else `state_dict()` is None and the trainer
  skips it — revisit with torchdata StatefulDataLoader if Phase 5
  throughput demands workers). MMU resume is exact for
  `shuffle_buffer_size: 0`; with a buffer it skips at most the in-flight
  buffer and never replays a trained record (HF shuffle semantics).
- Kill/resume gate: checkpoint at 137, SIGKILL a few steps later, resume →
  the step-138 loss (pure forward on restored state) matches the killed
  run's own value at log precision (0.0791 = 0.0791) and tracks the
  independent uninterrupted run to 0.7% mean over steps 138–200;
  `object_id_log` (new dataset arg, yield-time append) proves the resumed
  stream is exactly the uninterrupted tail, no replay. Note: compare a
  resumed run against its OWN trajectory for tight tolerances — two
  independent runs drift a few % by step 137 (nondeterministic backward).
- 2026-07-08 (later): rebased the fork commit onto `origin/astropt3` after
  the user merged upstream `main` there (fork `main` now exists,
  PR #1 = astropt3). The merge required two more compat fixes: moe.py must
  not require `grouped_gemm` at import time (it is imported for dense
  models via scaling.parametrization), and the flash-attn ≥ 2.8 rotary
  signature fix from 179c18f0 had to be restored (merge took upstream's
  rotary.py wholesale). Full GPU suite re-verified post-rebase.
- Fork also needed: per-stage `consumed_tokens_per_dataset_folder` kept in
  sync by hand for astropt3 streams, else `TrainingMetadata`'s
  consumed-tokens invariant fails on checkpoint LOAD (upstream only updates
  it for BlendableDataset).
- Eval is fully outside the trainer: `run_probe_sweep.py` polls a run dir
  (gated on `latest.txt`), converts, then `eval/val_loss.py` (fixed
  deterministic batches; synthetic val = record indices ≥ 10M) and
  `eval/linear_probe.py` (closed-form ridge, inner-split λ, test R²).
  Tiny-run sweep: val loss 0.456→0.051 monotone, images/spectra within 2×,
  redshift probe R² 0.42→0.79 across checkpoints.

### Phase 5 — Slurm launch + 70M/160M pilots

`launch_slurm.sbatch` (torchrun → `nanotron/run_train.py --config-file
astro/configs/nanotron/astropt3-70m.yaml`), per-size YAMLs from the recipe
table, profiling run instructions.
**Verify (training machine)**: 100-step dry run at target node count first
(playbook rule) with tokens/s/GPU + MFU logged; then 70M and 160M pilots to
completion: monotone-ish val loss per modality; image/spectra losses within
~5× after warmup (else tune loss_weight); probe R² for redshift improves
across checkpoints.

IN PROGRESS (2026-07-08, dev node, user's GPU reservation): real-data
70M shakeout ahead of the cluster pilots.

- Pilot data prep RUNS ON THE DEV NODE too (network confirmed): the full
  `prepare_pilot_data.py` run is filling `../astroPTv3_data/pilot_v1`
  (5,596 partitions, ~75 obj/s ≈ multi-day; journalled, resume any time,
  any machine). A finished 2°-cone prep around the DESI SV3 rosette
  (217.97, 32.62) sits in `../astroPTv3_data/pilot_sv3cone`: 26,452
  objects, 7,354 with spectra (27.8%) — spectra-rich subset for probing.
  NEVER point a cone run at the canonical dir: cone partitions are
  row-filtered and would poison the resume journal.
- `compute_norm_stats.py` ran on 10k real images → asinh p1/p99 into
  the data yaml (historical: both retired when physical normalization
  landed — `data/band_registry.py`). `check_pilot_data.py`: real images decode to exact
  N(0,1) patches, spectra to 31 patches λ 3702–9784 Å; dataloader ~1,000
  obj/s ≈ 400k tok/s per process at 8 workers (≥2× gate passes at DP=2).
- Blocker found+fixed by the first 70M execution: upstream nanotron's
  DDP + fp32-accum + ZeRO-1 comm hook routes to a dead reduce-scatter
  branch (NotImplementedError) — every ZeRO-1 DDP run would have crashed
  at step 1. Fork fix: all-reduce path (nanotron commit 0668f369).
- 100-step DP=2 dry run (astropt3-70m-shakeout.yaml): loss 0.459→0.259,
  ~240k tok/s total, 123 model TFLOPs/GPU (~39% MFU), peak 31.7 GiB/80.
  20k-step run (2.6B tokens, Pythia checkpoints) + async probe sweep
  launched on the reserved pair.
- 2026-07-09: 70M 20k-step shakeout DONE; sweep: val loss plateaus at
  0.203 from ~step 18k, redshift probe R² 0.28–0.32. 160M 20k-step
  shakeout DONE the same day (astropt3-160m-shakeout.yaml; wandb + new
  per-modality loss logging; final lm_loss 0.197 at 132 TFLOPs/GPU;
  included an unplanned mid-run kill+resume). CAVEAT on both: the
  shakeout mixes and their val splits carried almost no spectra (the prep
  had not yet reached the DESI footprint), so these are effectively
  image-only numbers — spectra_loss logged 0 in most 160M iterations and
  the 70M val loss has no spectra component at all.
- 2026-07-10 resume gap found by audit and FIXED: the shakeouts ran with
  num_loading_workers: 8, and with workers the dataset state_dict() path
  never engaged — NO checkpoint of either 20k run carries dataset_state/
  (the 160M mid-run resume silently restarted the stream). Fix: with
  workers > 0 `build_astropt3_dataloader` now returns torchdata's
  StatefulDataLoader (new hard dep `torchdata>=0.10`; installed in the
  gpuenv) whose state_dict embeds per-worker row-start snapshots; the
  trainer saves it via the new `loader_state_dict()` helper. Workers > 0
  without torchdata now refuses to start instead of training unresumably;
  resume asserts the same worker count; legacy dataset-format states
  still load at workers 0. CPU-verified by 4 new tests in
  test_loader_resume.py (exact continuation at workers 0/2 × synthetic/
  MMU); the GPU kill/resume gate in test_phase4_gpu.py is now
  parametrized over workers {0,2} and also asserts dataset_state exists.
- 2026-07-10 data prep: restarted (died ~02:34 without a traceback at
  939/5596 partitions, all spectra-free — the DESI-footprint pixels
  simply sort late in the HEALPix order). prepare_pilot_data.py now
  processes partitions overlapping the spectra catalog's coverage FIRST
  (`--spectra-first`, default on; journal-keyed resume makes reordering
  safe): matched spectra went 0 → ~138k within hours and pilot_v1/val
  gained its first ~2k spectra objects. Remaining ~4.3k partitions are
  image-only tail, ~63 obj/s.
- 2026-07-10 Phase 5 deliverables landed: `scripts/launch_slurm.sbatch`
  (multi-node srun+torchrun, per-size node counts, DRY_RUN_STEPS=100
  dry-run mode) and full nanotron pilot YAMLs for all eight sizes
  (recipe-table parallelism, Pythia LRs, GBS 512×4096,
  checkpoint_schedule: pythia; 70m updated to match). 160M probe sweep
  re-launched against a frozen hardlink snapshot of pilot_v1/val
  (../astroPTv3_data/pilot_v1_val_frozen_20260710 — the live val dir
  grows while prep runs, and the sweep needs one fixed val set).

### Phase 6 — Scale-up + modality extension

410M → 1.4B (TP=1), then 2.8B/6.9B/12B per the recipe table (dry run before
each). Add time series (`mmu_tess_spoc`) and tabular scalars (`mmu_gaia_gaia`)
**config-only** via reserved token ids + registry entries; add a config-only
test proving no model-code change was needed.
**Scientific sanity check**: loss-vs-tokens curves order correctly across
sizes (bigger = lower at matched tokens).

## Risks

- **Nanotron surgery** is the deep end: embedding/loss pipeline blocks, TP
  semantics for replicated modality modules, custom dataset type. Mitigated by
  PP=1, the tiny HF↔nanotron parity test, and keeping all modality/packing
  logic in the shared `astro` package (nanotron only consumes batch dicts).
- **lsdb not installed anywhere yet**; only the login-node prep env needs it;
  HEALPix-join fallback specified.
- **Pilot corpus is small** (~5.7B tokens/epoch): fine for 70M–410M; larger
  sizes need the Phase 6 modality/survey extensions or many epochs — flag at
  the Month-3 scope checkpoint.
- **flash-attn / nanotron env** is a separate GPU venv (see
  `docs/training.md`); gpu-marked tests run wherever it exists, this box
  included. CPU tests must stay green first.
- **BeeGFS checkpoint pressure**: prune non-schedule intermediates; convert +
  upload scheduled checkpoints to HF hub as they land.
- **Streaming-resume fidelity**: pin `datasets`, record the pin in checkpoint
  metadata.

## Reference files

- `../astroPT/src/astropt/model.py` — ModalityConfig/Encoder/Decoder/optimizer groups to port
- `../astroPT/src/astropt/local_datasets.py` — patchify, sequence structure, collate semantics
- `text/pretraining/smollm3/stage1_8T.yaml` — SmolLM3-3B nanotron config conventions
- `vision/m4/models/vllama3/modeling_vllama3.py` — in-fork placeholder-scatter reference
- transformers 4.57.1 `models/smollm3/` — SmolLM3Config/Model (verified present)
- Ultra-Scale Playbook: <https://huggingface.co/spaces/nanotron/ultrascale-playbook>

# Part II — Data-loader throughput experiments (2026-09-02 → present)

Moved verbatim from the former EXPERIMENTS.md top matter; section numbering preserved (§ references in code comments and AGENTS.md point here).


## Context

`astropt3-70m-jetformer-crossmatch.yaml` (DP=2, 8 loading workers/rank,
streaming `mmu_desi_edr_sv3 x mmu_ssl_legacysurvey_north` live via
`nanotron_loader.py`'s direct-lsdb path) was observed running with the GPU
mostly idle. This log covers the investigation into why, what was tried,
what was measured on real runs, and what's still open. All "real run"
numbers below come from `scripts/bench_report.py` reading the
`ASTROPT3_TELEMETRY_DIR` output of short verification runs on spare GPUs
(2,3), using a scratch checkpoint path and (mostly) `WANDB_MODE=disabled` so
they don't collide with production training. Two runs were logged to wandb
(`astropt3-loader-verify` project) for visual inspection.

## 1. Where the stall was coming from

`cProfile` against the real `InfiniteStream`/`PackedMicroBatches` path
(`profile_stream.py`, ad hoc) showed two distinct causes:

- **Legacy-only stream**: cold start (catalog open + first partition fetch)
  ≈ 40s, of which 34.6s was pure `_ssl._SSLSocket.read` — genuine network
  wait. Steady-state decode/pack of an already-fetched partition was
  ~30ms/micro-batch — negligible.
- **Crossmatch stream**: steady-state was 0.47s/micro-batch, ~89% of it
  inside `_as_mapping`'s `DataFrame.to_dict()` fallback. Each DESI
  spectrum arrives as a 7781-row × 5-column per-record `DataFrame`;
  `to_dict(orient="list")` boxes every one of those ~39k cells
  individually via pandas' `maybe_box_native`. This was a real, fixable
  CPU bug, not network cost.

## 2. Fix: fast-path the spectrum DataFrame, then migrate to `map_rows`

- First fix: `_decode_spectrum` reads columns directly via `.to_numpy()`
  when handed a `DataFrame`, skipping `to_dict()` entirely.
  **0.47s → 0.02s/micro-batch.**
- Investigated `nested_pandas.NestedFrame.map_rows` (the library's own
  purpose-built row-wise accessor for nested/struct columns — delivers
  nested sub-columns as numpy arrays straight from Arrow storage, never
  materializing a per-row `DataFrame` at all). Benchmarked in isolation:
  **19x faster** than the already-fixed `.iterrows()` path for spectrum
  extraction alone (0.241s → 0.013s over 1017 rows), byte-identical output.
- Migrated `_open_records`'s full decode loop to `map_rows`
  (`_row_from_map_rows` reassembles the dotted-key output back into the
  nested-mapping shape `decode_legacy_row`/`decode_crossmatch_row` already
  expect, so their public contracts didn't change). Offline test fixtures
  were upgraded from plain `pandas.DataFrame` to real
  `nested_pandas.NestedFrame`s (`legacy_fixture.nested_frame`, via
  `join_nested`) so the CPU suite exercises the same code path as
  production — the old fake fixtures didn't support `map_rows` at all,
  which the test suite alone wouldn't have caught.
- **Net: crossmatch decode 0.47s → 0.01s/micro-batch (~47x)**, legacy-only
  2.6x faster. Decode dropped out of the profile entirely; packing
  (`torch.arcsinh`, `spectral_normalize`) became the dominant CPU cost at
  ~10-20ms/micro-batch.

Commits: `637ba68` (rows/rows_per_s telemetry, used throughout this
investigation), `e881e9b` (crossmatch decoding baseline), `c330446`
(`map_rows` migration).

## 3. Fetch overlap: does prefetching help?

`lsdb.streams.catalog_streams.InfiniteStream` only prefetches in the
background when given a real `dask.distributed.Client` — with the
`client=None` default it was using, `submit_next_partitions` calls
`.compute()` synchronously, so partition N+1's fetch fully blocks the call
that returns partition N. Zero overlap with anything.

Added a lightweight per-worker `dask.distributed.Client` backed by
`LocalCluster(processes=False, ...)` — an in-process scheduler + thread
pool, not a distributed cluster (~30ms startup, ~1.5MiB RSS, confirmed via
direct measurement). Verified the mechanism works correctly in isolation
(non-blocking submit, genuine background progress on a `Future` checked
mid-flight).

**Result on a real run: no measurable improvement.** `stall_share` stayed
flat at ~85% before and after. Root cause: a worker only stays
`DataLoader`'s `prefetch_factor` micro-batches ahead of training-step
consumption before it blocks — a depth-1 partition lookahead barely dents
a 10-50s fetch, because the wall-clock window it gets to run in isn't "however
long draining a partition takes," it's bounded by how fast the main loop
is actually consuming.

Commit: `9800176` (kept — correct mechanism, real value once paired with
a wider window; see next section).

## 4. `partitions_per_chunk` tuning (pre-outer-crossmatch)

Widening the window via `InfiniteStream(partitions_per_chunk=N)` — fetch N
partitions per draw instead of 1, giving the background prefetch more time
to finish before the worker needs it. Real-run sweep (same config, DP=2,
8 workers/rank):

| chunk | stall_share | p95 step | p99 step | rows/s | MFU | RAM (box-wide) |
|---|---|---|---|---|---|---|
| 1 | ~85% | ~13s | ~51s | ~1,000 | 2.6% | not measured |
| 4 | 65.4% | 0.46s | 24.0s | 2,044 | 5.5% | ~161GB |
| **8** | **45.5%** | **0.45s** | **6.4s** | **3,324** | **8.6%** | ~237GB |
| 16 | 61.7% | 0.45s | 6.8s | 2,312 | 6.0% | ~595GB |

`chunk=8` was a real optimum, not a plateau — 16 regressed on every metric
(worse stall, worse throughput) while RAM more than doubled, consistent
with 16 workers × 2 chunks-in-flight × 16 partitions pushing into genuine
memory/scheduling pressure.

Commit: `74851c3` (`_PARTITIONS_PER_CHUNK = 8`).

## 5. Wire-efficiency investigation

With the network confirmed as the dominant remaining cost (measured ~815
MiB/s aggregate at chunk=8, ≈65% of the box's 10Gbps NIC — not clearly
link-limited, but not far off either), the next question was whether the
bytes being pulled were actually useful.

- **Byte breakdown at chunk=8**: 66.8% of wire bytes were Legacy image
  data.
- **Measured match rate** (live sample, 18,355 DESI rows across 4
  partitions): only **10.0%** have a Legacy image match within 1
  arcsec; **90.0%** are spectrum-only.
- **Read `lsdb`'s crossmatch source** (`crossmatch_catalog_data.py`,
  `kdtree_match.py`): the KDTree match algorithm reads the *full* right
  (Legacy) partition — including the large `image` struct column — for
  every candidate row in an overlapping pixel pair, *before* running the
  distance filter. The ~90% of rows that don't match within radius get
  their image bytes fetched and then silently dropped by
  `AbstractCrossmatchAlgorithm._create_crossmatch_df`.
- **Net finding**: roughly **60% of total wire bytes** (66.8% × ~90%)
  were being spent downloading Legacy images that get discarded.
- **Tried and rejected**: dotted sub-column projection
  (`columns=["spectrum.flux", "spectrum.lambda", "spectrum.mask"]`
  instead of the whole `spectrum` struct) to drop the unused `ivar`/
  `lsf_sigma` fields. Confirmed by direct measurement this does **not**
  reduce wire bytes — HATS/parquet reads the whole nested column's
  row-group regardless of which sub-fields are kept afterward. In-memory
  schema pruning only, no wire savings. Negative result, not pursued
  further.
- **Not pursued**: `MFU_busy` (compute efficiency while *not* stalled) sat
  at ~16-18% across every run measured, against a 989 TFLOP/s peak — a
  separate, wire-independent lever (kernel/model-shape efficiency, not
  data pipeline). Flash attention, fused RMSNorm, fused rotary embeddings
  and QKV packing are already on in this config; no GPU-level profiling
  (torch profiler / nsys) was done this session to look further.

## 6. `OuterKdTreeCrossmatch`: recovering the wasted 60% for free

Since the unmatched Legacy image bytes are already being read into memory
before being discarded, the fix is to stop discarding them rather than to
fetch less — zero extra network cost either way.

`outer_crossmatch.py`'s `OuterKdTreeCrossmatch` subclasses `KdTreeCrossmatch`
and extends `how="left"` row assembly to also emit unmatched right-side
(Legacy) rows as image-only records (NA spectrum/DESI columns), instead of
silently dropping them. Two real bugs were caught by testing against live
data rather than shipping on inspection alone:

- **Margin-cache double-counting**: the right catalog's margin cache
  (candidate rows borrowed from a neighboring partition, included only to
  catch boundary-crossing matches) had to be excluded via spatial-index
  filtering (`healpix_to_spatial_index` bounds check against the row's own
  native pixel) — otherwise a margin row gets emitted here *and* again
  when its home partition is processed.
- **`pandas.NA` vs `None`**: a null `object_id` (pyarrow string) scalar
  comes through as `pandas.NA`, not Python `None`, both via raw pandas
  access and via `map_rows`. The original `decode_crossmatch_row` used
  `is None`, which would have silently decoded these rows' `object_id` as
  the literal string `"<NA>"` rather than falling back to
  `object_id_legacy` or raising. Fixed by switching to `pd.isna()` (a
  strict superset of `is None`, so plain-`None` test fixtures are
  unaffected).

**Verified on real data**: 6,703 rows split exactly into 1,403 matched +
4,333 spectrum-only + 967 image-only — zero loss, zero duplication, zero
rows with neither modality. `ObjectSequencer.build()` succeeds on all
sampled image-only objects.

Scope is deliberately narrow: this only recovers Legacy rows *within pixel
pairs the crossmatch already visits* (near some DESI pointing). A Legacy
partition with no DESI coverage nearby is never fetched by this join at
all — that's the majority of unmatched Legacy imagery, and recovering it
would need a genuinely separate, full-wire-cost stream (not built).

Commit: `62cce63`.

## 7. Re-tuning `partitions_per_chunk` after the outer join

`OuterKdTreeCrossmatch` recovers real extra data, but each recovered image
row is real extra weight (~277KB/row) buffered per chunk. This pushed
`chunk=8` into the same RAM-pressure regression `chunk=16` hit in section
4. Re-tuned with two equally-mature real-run readings (~552-575s each):

| | chunk=8 + outer-join | chunk=4 + outer-join |
|---|---|---|
| stall_share | 49.8% | **32.8%** |
| rows/s | 2,324 | **2,997** |
| MFU | 8.90% | **12.13%** |
| E_values/MiB | 66,218 | **89,458** |
| p99 step | 7.75s | **3.50s** |
| slow steps (>60s) | 1 | **0** |
| RAM (box-wide) | 529GB | **349GB** |

`chunk=4` won on *every* metric, not just RAM — confirms this wasn't a
tradeoff, `chunk=8` was genuinely past the regression point once the outer
join's extra data is counted. `chunk=2` was not tested (stopped once the
win was decisive); worth checking if further tuning is ever revisited.

Commit: `696e359` (`_PARTITIONS_PER_CHUNK = 4`).

## 8. Current state

Final settings as of this log: `map_rows`-based decode, per-worker
lightweight `dask.distributed` prefetch client, `partitions_per_chunk=4`,
`OuterKdTreeCrossmatch` for the crossmatch stream. Confirmed with a
wandb-logged run of this exact configuration
([`astropt3-loader-verify/runs/jsoi53bq`](https://wandb.ai/smith42/astropt3-loader-verify/runs/jsoi53bq),
separate project, scratch checkpoint path — not production), read at a
comparable maturity (540s) to the section-7 numbers:

| | section 7 reading | wandb-confirmed reading |
|---|---|---|
| stall_share | 32.8% | 35.4% |
| rows/s | 2,997 | 2,881 |
| MFU | 12.13% | 11.64% |
| E_values/MiB | 89,458 | 87,992 |

Consistent within normal run-to-run variance — the section-7 numbers hold.

**End-to-end from the original ~85% stall_share baseline (section 1) to
here: stall_share 85% → 35%, rows/s ~1,000 → ~2,900 (~2.9x), MFU 2.6% →
11.6% (~4.5x).**

## 9. v3 adoption bench: CrossMatchStream vs manual crossmatch (2026-09-24)

Setup: paired fresh runs, 2,000 steps/arm, DP=2 per arm on 2×H200 each
(both arms live simultaneously — shared link by design). Same working tree
except the stream: the v2 arm ran from a detached worktree of HEAD
(`../AstroPTv3-v2bench`, same retry fix applied) shadowed via PYTHONPATH,
because the training venv's editable `astropt3` points at the main tree.
Steady state = steps 21–2000. The v3 arms are
`lsdb.streams.CrossMatchStream` (astronomy-commons/lsdb#1584) at two
`count_fraction_threshold` settings.

| arm | link conditions | non-pad tokens/s | rows/s | img values/s | spec values/s | img token share |
|---|---|---|---|---|---|---|
| v2 r1 | paired | 100,644 | 2,498 | 26.6M | 14.0M | 71.7% |
| v3 thr=0.5 r1 | paired | 77,006 | 2,342 | 16.4M | 15.6M | 58.4% |
| v2 r2 | ~24% contended | 117,769 | 2,923 | 31.1M | 16.4M | 71.7% |
| v3 thr=0.0 | solo | 110,584 | 2,715 | 29.5M | 15.1M | 72.2% |

Findings:

1. **Mechanism tax ≥ 6% (floor; contention-corrected ~10%).** Even with
   the skip off, CrossMatchStream is slower than streaming a pre-
   crossmatched catalog: every drawn pixel pays `PixelSearch` + a fresh
   crossmatch graph build + the density estimate synchronously on the
   loader's critical path, where v2 builds the crossmatch graph once. The
   record mix is identical (72.2% vs 71.7% image share; per-modality value
   rates track) — pure overhead, semantics verified including
   OuterKdTreeCrossmatch's unmatched-right recovery.
2. **The skip at 0.5 is strictly harmful here.** Legacy bytes −31%, but
   tokens/s −~20% more and image share 72% → 58%: the loader is
   latency-bound, not bandwidth-bound, skipped pixels still pay the
   per-pixel fixed costs, and the skip discards the image-heavy rows.
   (0.1 previously showed no effect either direction.) Threshold set to
   0.0 on adoption.
3. **Repeatability**: v2-r2 rows/s 2,923 ≈ §8's solo 2,881–2,997.
   Pairing costs ~15% (v2 paired 100.6k vs r2 117.8k tokens/s).

**Decision:** adopt v3 at thr=0.0 — for the multi-catalog capability and
ADR 0015 upstream alignment — accepting the measured tax at 70M scale,
where 0.36s steps leave the GPU maximally exposed to loader latency; at
larger model sizes step time grows and the tax should shrink. Two asks
filed upstream on #1584: precompute/reuse per-pixel crossmatch graphs, and
move the skip decision off the critical path (which would make the skip a
pure win whenever bytes bind).

Reliability note from the same bench: a transient `HfHubHTTPError` can be
swallowed by fsspec's parquet reader (cat_ranges results are gathered with
`return_exceptions=True` but consumed without exception checks) and
surfaces as `TypeError: can't concat X to bytes` (block merge) or
`'X' object is not subscriptable` (footer slicing). The first killed the
original v3 run at step 46,767; the second killed a bench arm at step 302.
`_retryable` now classifies the whole family; regression tests cover both
shapes plus a live-fsspec message canary. Upstream issue drafted against
fsspec (root fix: check `is_exception` in `_transfer_ranges`).


## 11. Trimodal HSC trials: 1k and 10k steps (2026-09-28)

Third-catalog validation for roadmap item 3: `hsc_images` (HSC PDR3 Wide,
matched-only left join on the DESI × Legacy `CrossMatchStream`) trained
end-to-end on 70M with wandb online. Fresh cosine schedules; DP=2 × 8
loading workers; checkpoints every 1000 steps.

- **1k trial** (https://wandb.ai/smith42/astropt3/runs/u53mgt1i): 1000/1000
  steps, zero Arrow errors after the `Array2DExtensionType` unregister fix.
- **10k trial** (https://wandb.ai/smith42/astropt3/runs/qmnq2gd1):
  10,000/10,000 steps, checkpoint 10000, zero errors. Median losses
  early→late: overall 381→188, HSC 1110→886, Legacy images 99.6→−88.7,
  spectra −24.5→−216. HSC token share 5.87% (present in 86% of
  rank-steps; matched-only join, ~8% as many HSC spans as Legacy spans).
  Throughput 66.6k non-pad tokens/s vs 110.6k two-catalog upstream v3 —
  the gap is p99 HSC partition stalls (257 steps >5s; §10 shows leaf
  projection cannot fix them; the fix is upstream leaf-level range reads).

Configs: `configs/nanotron/astropt3-70m-v3-trimodal-trial{,-10k}.yaml`.
Implementation and the Arrow fix: PR #33.

## 12. JetFormer noise and convergence diagnosis (2026-07-14)

Measured on the step-20000 checkpoint of the low-LR 70M run
(`astropt3-70m-jetformer-lowlr`, wandb `y3oak0l0`, DeltaAI GH200), using
the HF-converted checkpoint + real val shards. Two independent problems:

1. **Optimisation drift.** `grad_norm` sat at 1000–1700 against
   `clip_grad=1.0` for the entire second half; `images_loss` bottomed at
   −0.76 (step 16k) then climbed back to +1.0 by 20k — the sampled
   checkpoint was past the best point. Negative loss is a tell: the flow
   can push `NLL_GMM(z) − logdet` below zero by inflating logdet without
   improving samples (a plausible gradient-blow-up source; the
   OLMo-style grad-norm growth persisted at 10× lower LR, so it is not an
   LR-knob issue).
2. **Uncalibrated noise generation at any temperature.** In model output
   space (noise isolated by a 3×3 high-pass): per-channel noise std ~5×
   too small at T=0.2 and ~15–20× too large at T=1.0 — no temperature
   matches the real amplitude; T=1 noise is also spatially anti-correlated
   where real noise is PSF-correlated. Teacher-forced `log_sigma` is NOT
   saturated (0% at the ceiling) and the mixture is not collapsed — the
   shallow 4-step flow reaches low z-space NLL while its inverse amplifies
   and whitens noise; exact z-likelihood does not constrain sample quality.

The transformer body was fine — teacher-forced GMM-mode reconstructions
looked reasonable; the sampling/variance path was the broken part.
Priorities recorded at the time: instrument the loss per term (GMM-NLL vs
logdet separately) and regularise `|logdet|`; stop memorising sensor noise
(whiten/down-weight by the known noise model, or denoise the target and
re-add calibrated noise at sample time); do not expect temperature to
produce realistic noise. Full detail (tables, spatial-texture stats,
ruled-out head defects, reproduction): git history at
`astro/docs/jetformer_noise_diagnosis.md` (deleted 2026-09-28).

## 13. Standard throughput benchmark validation (issue #45)

`scripts/benchmark_throughput.py` bounds the real nanotron training loop
without changing the YAML horizon or LR/noise schedules; usage and clock
definitions are in [`docs/training.md` §8](docs/training.md#8-standard-throughput-benchmark).

Functional checks (2026-10-20) ran on idle H200 NVL GPUs of a shared host, 2
warmup + 5 measured steps, with the tiny YAML's 50-step horizon unchanged. DP1
used that YAML as is (no loader workers); DP2 and TP2 used copies with that
topology, one loader worker per rank and console logging every third step.
All streamed live LegacySurvey data; cache state unknown.

| topology | step records | measured non-pad tokens | returned payload bytes | result |
|---|---:|---:|---:|---|
| DP1 / TP1 | 7 | 15,300 | 8,708,962,755 | complete |
| DP2 / TP1 | 7 per rank | 30,600 | 13,745,150,931 | complete |
| DP1 / TP2 | 7 per rank | 15,300 | 17,417,925,510 | complete |

TP2 counts tokens once but both replicas' downloads (2 × 8,708,962,755 B).
Warmup was ~44–47 s, almost all of it the first batch, while the measured
windows were 0.1–0.7 s: **these are execution/accounting checks, not
throughput or scaling measurements** (the MFU estimates, 0.01–0.07% on a
538k-parameter model, mean nothing). No checkpoints were written, and
launching into an existing output directory failed before any write.
Payload counts only what returned before training ended on each rank: one
earlier DP1 check ended with a 178 MB fetch in flight and recorded 57 fetches
instead of 58, in both the in-run and the regenerated report.

Environment: Python 3.13.15, torch 2.8.0+cu128, flash-attn 2.8.3, driver
615.71.09, LSDB `5102a8a9`; `rankN.json` records the rest. Not validated live:
multi-node, mixed DP+TP, DESI attribution, remote checkpoint staging, long
windows.

## Open items / not pursued

- **`chunk=2`**: not tested after the section-7 re-tune; the win at 4 was
  decisive enough to stop there.
- **Compute-side MFU (`MFU_busy` ~16-18%)**: a real, wire-independent
  lever, not investigated this session. Needs GPU-level profiling
  (torch profiler / nsys), not data-pipeline telemetry.
- **Legacy imagery outside DESI's footprint**: the majority of unmatched
  Legacy images (partitions with no nearby DESI coverage at all) are still
  never fetched by this stream. Recovering them needs a genuinely separate
  stream at full wire cost — a real design decision (more corpus coverage
  vs. more bandwidth spent), not attempted here.
- **lsdb docstring accuracy**: `InfiniteStream`/`CatalogStream`'s docstring
  claims background prefetching happens generically ("derived from
  `client` object"), which overstates the `client=None` case (fully
  synchronous). Worth a small upstream issue/PR; not filed.
