# Training AstroPTv3 models with nanotron

*The operational guide: environments, live survey data, launching, and
checkpoint/resume. Background on the model is in
[`architecture.md`](architecture.md). Every recipe uses YAML-configured
catalog streams.*

## 1. Environments

| env | where | contents | used for |
|-----|-------|----------|----------|
| `uv sync --extra dev` | anywhere | torch (CPU ok), transformers, lsdb | unit tests |
| GPU venv | training machine | torch + **flash-attn** + nanotron (editable `nanotron/`) + astro (editable `astro/`) + psutil | training, GPU tests, conversion |

flash-attn wheels are the constraint for the GPU venv: pick a torch version
with a prebuilt wheel for your CUDA (never compile it on a shared box).

```bash
uv venv gpuenv --python 3.13
uv pip install torch==2.8.0 \
  <flash_attn wheel from GitHub releases> \
  -e nanotron -e astro psutil
```

Verify the env on a GPU box: `pytest -m gpu tests/test_nanotron_gpu.py`.

## 2. Live pilot data

There is **no prep step** and no local corpus. `data/nanotron_loader.py` opens
Multimodal Universe HATS catalogs via LSDB inside each DataLoader worker,
decodes rows into the shared record shape, then sequences and packs them for
nanotron. The default recipe joins DESI, SDSS, HSC, galaxies-with-hats, and PROVABGS
separately to each of Legacy Survey North and DR10 South; other YAMLs can
select different supported modality/catalog combinations (§9).
Depending on overlap, records carry different subsets of the configured
modalities. Recovered unmatched records come from partitions visited by the
join, **not** every survey partition. The active YAML and loader are the
sources for the stream implementation; fetch/prefetch settings are experimental.

Facts to know:

- **Network is a hard dependency.** The Hub must remain accessible; there is
  no local training-corpus fallback.
- **Workers and memory.** `num_loading_workers` controls concurrent readers
  per DP rank; each may buffer large image partitions. Monitor box-wide RSS
  and lower the worker count after an observed OOM.
- **No cursors or coverage guarantee.** Each DP rank × worker owns a
  disjoint round-robin share of the anchor's HATS partitions (shared only when
  consumers outnumber partitions); retries and resumes can revisit records.
- **Provenance.** `uv.lock` records the resolved LSDB dependency, and `[data]`
  startup lines log its version and catalog information when available.

On resume, checkpoints restore weights/optimizer/scheduler/RNG, **not** the
LSDB record position. Recognized transport/storage errors discard the failed
iterator and reopen under bounded backoff; decode/validation and unknown
errors fail immediately.

## 3. Configs

Full nanotron run configs live in `astro/configs/nanotron/`: the Pythia ladder
`astropt3-{70m,160m,410m,1b,1p4b,2p8b,6p9b,12b}.yaml` plus single-GPU smoke
recipes `astropt3-{1m,10m}.yaml`. Every rung trains the same 57 modalities from
the same dual-anchor stream: 12 calibrated image bands, one display RGB
product, two spectra and 42 scalars. Rungs differ in model size, cluster
topology, batch and LR. The convenience launcher's default is
**`astropt3-70m.yaml`**, run at DP=1 with at most 4 loader workers (TP=1 rungs
only). Every ladder rung echoes each draw 4 times.
Model, catalog list, join policy, worker count and schedule are explicit YAML,
not injected by the script (see §9).

`catalog_stream` requires an explicit `anchors` list and a shared partner
`catalogs` list. Unknown settings are rejected. AstroPT selects the ownership
and split masks; LSDB applies footprint filtering and matching, pinned by
`uv.lock`; AstroPT applies calibration and quality gates.

Governing knobs, top to bottom:

```yaml
general:
  ignore_sanity_checks: true   # REQUIRED with DP>1: per-rank modality tensor
                               # shapes differ; the DP input check would crash
model:
  model_config:
    is_astropt3_config: true   # dispatches to AstroPT3ForTraining
    _use_doc_masking: true     # position_ids restarts = document boundaries
parallelism:
  pp: 1                        # asserted — do not change
  tp_mode: ALL_REDUCE          # modality modules are TP-replicated
  dp: <n>
tokens:
  sequence_length: 4096
  micro_batch_size: 16
  batch_accumulation_per_replica: 1
checkpoints:
  checkpoint_schedule: pythia  # steps 1,2,4,...,512 + every interval
  checkpoint_interval: 1000
  resume_checkpoint_path: null # set to the checkpoints dir to resume
```

Use the actual run YAML for batch sizes and parallelism.

## 4. Launching

Single GPU (from the repo root; choose an available device):

```bash
bash astro/scripts/run_train.bash GPU_INDEX       # default recipe, echo 4
bash astro/scripts/run_train.bash GPU_INDEX 1     # disable echoing
bash astro/scripts/run_train.bash GPU_INDEX 2 --config-file my-recipe.yaml
```

This launches `astro/scripts/train.py`. Set `ASTROPT3_ENV` to the GPU venv;
the local launcher defaults to `.venv-astropt3` in the repository. It uses the
installed LSDB/HATS versions, which must match `astro/uv.lock` and support the
multi-catalog/min_catalogs API; train.py checks this before creating outputs.

Multi-GPU on one node uses the same YAML-configured loader:

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 \
torchrun --nproc_per_node=<gpus> --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
    nanotron/run_train.py --config-file astro/configs/nanotron/<config>.yaml
```

Multi-node via slurm:

```bash
sbatch --nodes=<N> astro/scripts/launch_slurm.sbatch astro/configs/nanotron/<config>.yaml
# dry run: sbatch --nodes=<N> --export=ALL,DRY_RUN_STEPS=100 astro/scripts/launch_slurm.sbatch <config>
```

The launcher sources `$ASTROPT3_ENV` (default `../astroPTv3_gpuenv`) and
rendezvous on the first node. **Always dry-run first**, checking
tokens/s/GPU, MFU, and memory, before committing a real run.

Do not set `HF_DATASETS_OFFLINE=1` — training streams from the Hub. §6
describes the model-side evaluation tools.

## 5. Checkpoints and resume

Checkpoint dirs `{1,2,4,...,512,1000,...}` hold model weights (bf16),
optimizer + LR-scheduler state, RNG states, `model_config.json`, `config.yaml`;
`latest.txt` is written last, so any step dir it covers is complete.

Resume:

```yaml
checkpoints:
  resume_checkpoint_path: <checkpoints dir>   # reads latest.txt
```

The run restores model/optimizer/scheduler state and the RNG state on the
same TP/DP topology. The LSDB stream is cursorless: resume opens a fresh
stream and records may be revisited immediately. There is no exact-sequence continuation, no replay
audit, and no worker-count constraint on resume.

Resume needs a recipe whose modalities match the checkpoint's. The runner
reads the checkpoint's saved YAML only for the echo factor; catalogs come from
the launch YAML.

`lm_loss` (`train/loss` in the single-GPU launcher) is the optimized objective;
`grad_norm` is measured before clipping. Diagnostics such as `spectra_loss` and
`image_family_loss` pool all scored targets across the step's microbatches and
DP ranks. Each has a `*_loss_count`: zero means no loss is logged, not a loss
of zero. Modality losses are joint NLL per target; image and spectrum family
losses divide each patch NLL by its input dimension before pooling. Scalar
family losses pool the unscaled target NLLs. These diagnostics need not
reconstruct `lm_loss`; exact-likelihood losses can be negative.

## 6. Evaluation status

Source-backed validation/probe checkpoint sweeps are deferred.
`astropt3.eval` keeps model-side functions — `evaluate` (loss on provided
batches), `embed_objects` + `ridge_r2` (probe on supplied objects),
`scalar_head_metrics`, and sampling/rendering. `scripts/generate.py` remains
a separate live-sample CLI; it is **not** a fixed validation suite or an
automated checkpoint sweep. Generation uses the same configured stream:

```bash
python astro/scripts/generate.py --checkpoint <hf_dir> \
    --config-file astro/configs/nanotron/<recipe>.yaml
```

The checkpoint's modalities must have compatible outputs in that recipe. One
configured draw supplies templates. Ranking counts each image payload once,
regardless of filter count; `--rows` selects positions in that ordering.
Catalogs are never inferred from the model. Use `--band hsc_g` for an
unconditional single-band sample, `--mode spectra-to-images --band hsc_g`
to generate that band from spectra, or `--mode image-to-spectra --band hsc_g`
to generate spectra from that band alone. The template must carry the selected source.

## 7. Verification gates

1. `uv run pytest -m 'not gpu and not network'` for offline CPU checks in
   `astro/`. Plain `uv run pytest` also selects network-marked tests, which
   may fail when the Hub is unavailable.
2. `uv run python scripts/count_params.py` for current size totals.

For a bounded live check with Hub access:
`uv run pytest -m network tests/test_lsdb_stream.py`.

## 8. Standard throughput benchmark

Measures one real training YAML on the hardware it is launched on, using the
training environment and the usual torchrun arguments:

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 \
torchrun --nproc_per_node=<YAML dp*tp on this node> \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --log-dir <logs> --tee 3 \
    astro/scripts/benchmark_throughput.py \
    --config-file astro/configs/nanotron/<config>.yaml \
    --warmup-steps 10 --steps 200 --output /shared/benchmarks/<new-run>
```

`--output` must be a new directory, on storage every rank can see. The script
runs nanotron's own trainer and loader for warmup + measured steps on top of
any resumed step. The YAML horizon, LR/noise schedules, model, data and
parallelism stay unchanged; checkpoints, eval, profiling, uploads and
trackers are disabled and the overrides recorded. Explicit local
initialization/resume is honoured, and an unresolvable resume fails rather than
starting from random weights. Supported: one AstroPT streaming stage, PP=CP=EP=1.
`--bandwidth-cap` and `--cache-state` only record context; nothing is throttled
or flushed. On one GPU, a TP=1 recipe runs its per-GPU workload at DP=1 (same
model, batch shape and loader workers; recorded as an override).

Outputs: `baseline.yaml`/`effective.yaml`/`resolved.yaml`, `run.json` (window
and overrides), `rankN.json` (status, topology, host, GPU/CPU/RAM, software,
code revisions, loader settings), raw `steps.rankN.jsonl` and
`bytes.rankN.{main,wW}.jsonl`, and `result.json`, which the terminal
summarises. Regenerate the report after a crash with
`python astro/scripts/bench_report.py --telemetry <run> --out <run>/result.json`.
A failed rank, a missing or extra step, or a missing byte log gives a
`partial` report, no headline rates, and a nonzero exit.

**Reading it.** Every step ends with a CUDA sync and a world barrier, so the
measured window is synchronized wall time including loader stalls; compare runs
under this same policy. Throughput is non-padding tokens over that window,
counted once per TP group and summed over DP, then divided by world size for
the per-GPU rate. The first batch (stream open and first fetch) lands in warmup and is
reported as `first_step_loader_wait_s`. Modality mix and packing travel with the
rate: a different mix is a different workload, and row counts are
presentations, not unique objects. Payload bytes count every returned HF
payload, duplicates and prefetch included, by source, from script start to the
end of training on each rank. That rate is not link capacity. Windows under 200 steps
or 60 s are flagged as limited evidence. CPU tests do not establish
distributed correctness: validate an unexercised topology live first.

**MFU** is an estimate: measured non-padding and loss tokens (TP counted once)
priced with nanotron's own FLOPs/token, which assumes full-sequence attention
and so is an upper bound, divided by measured seconds times the sum of every
GPU's rated dense BF16 peak. Peaks come from an exact-name table
(`DENSE_BF16_PEAKS`; H200 NVL 835.5 vs SXM 989.5 TFLOP/s). An unlisted
GPU, a MIG slice or a non-BF16 run reports MFU as `null`, never zero.
`rankN.json` also records each up physical NIC's negotiated link speed. The
terminal shows the fastest per host; it is not internet or HF-route bandwidth.

## 9. YAML-driven N-modality training and shuffled echoing

Use the launch commands in §4. The local runner requires one GPU and one data
stage; YAML owns the training settings.
Each launch creates a new output directory. The Python CLI, `astro/scripts/train.py`,
accepts `--output`, `--resume-from CHECKPOINT_ROOT`, and `--max-steps` for startup
checks without changing the configured learning-rate schedule.

### Catalogs and modalities

Copy the [default YAML](../configs/nanotron/astropt3-70m.yaml).
`model.model_config.modalities` and
the `catalog_stream` source lists (`anchors` and their shared `catalogs`) are independent;
modality `record_keys` match catalog outputs. A catalog supplies `record_key`
and/or `scalar_columns` (**record key → flat catalog column**). Keep existing
token IDs frozen; new names need non-colliding IDs and sufficient `vocab_size`.

North and [DR10 South](legacy-south.md) each anchor an LSDB star with SDSS DR17, DESI, HSC and galaxies-with-hats outer joins and a PROVABGS left join at 1″. The [pinned Galaxies train collection](https://huggingface.co/datasets/Smith42/galaxies-with-hats/tree/1a2b1ebbbe06c67a15f49385643ab711c5824a0d/train) has a 10″ margin for cross-partition matches. `min_catalogs: 3` selects each anchor's sky where at least two outer partners have coverage. It does not count the alternative anchor, left-only partners, or modalities on a row. Missing modalities are omitted; right-only rows stay separate, and rows with no usable modalities are dropped.

For anchor-only streaming, set `catalogs: []` and `min_catalogs: 1`. When
omitted, `partitions_per_chunk` defaults to 1 for anchor-only streams and 4
for joins. Native Nanotron uses one catalog reader group per DP replica and
broadcasts packed batches to the other TP ranks in that replica.

Scalars need registered transforms (`data/scalar_registry.py`). Values outside `VALID_RANGES` or missing or failing `QUALITY_GATES` flags omit the scalar span. Gated scalars (e.g. DESI `Z`, gated on `ZWARN`) need flag columns from the same catalog. North and South have separate reddening and photometry scalars; galaxies-with-hats supplies a display RGB image and Galaxy Zoo vote fractions.

**Ownership is by sky, not a global object crossmatch.** The first anchor owns its eligible footprint. Each subsequent anchor gets its eligible footprint minus earlier claims and a one-cell order-14 guard around them. The guard prevents a partner within 1″ of two centers on opposite sides from entering through both paths. This protects both attached partners and outer-only rows without an index or cross-worker state. The cost is genuine South objects lost in shared sky: in the [pinned audit](dual-anchor-audit.md) 31 of 46 eligible South objects in the overlap cone have no North counterpart within 1″. Do not interpret this as a complete unique-object union. Records carry `source_ids`, source-qualified `object_id`, `anchor_path` (sky owner), and `anchor` (null for outer-only records).

**Mixing:** `draw_weight` counts partition draws per worker cycle, not objects or tokens. The ladder uses 1 North : 2 South to give the larger eligible South sky more draws while guaranteeing North one in every three. This is not a measured optimum. The reader seed rotates the starting path; LSDB shuffles partitions and rows. Both stars share one Dask client per worker, but can each hold a prefetched frame; size workers for available memory. Benchmark JSON reports `anchor_counts`, `fresh_anchor_counts`, and `replayed_anchor_counts` in the measured window. These count object exposures and nonpadding tokens by owning path, including outer-only objects. See the [bounded pilot](dual-anchor-pilot.md) for measured exposures and limitations.

**Splits:** recipes use `split: train`; generation defaults to `validation` for those recipes. Every source uses the same fixed 154 of 3,072 all-sky order-4 HEALPix cells for validation. A one-order-10-cell collar around them is excluded from training to separate adjacent cutouts and 1″ matches, including partners drawn without an anchor. The [North-only control](../configs/controls/astropt3-70m-north.yaml) uses the same split. Its model cannot resume as a dual-anchor model: South adds four separate band heads.

### Echoing, accounting, and resume

`data_stages[].data.dataset.echo_factor` counts **total uses**, including the
original: a positive integer, **4** in the ladder recipes, **1** when omitted. Precedence:
CLI > checkpoint factor on resume > recipe/default. It is saved in checkpoint YAML.

- **Factor 1:** normal streaming packing, including across draw boundaries.
- **Factor N > 1:** each worker packs and caches one finite LSDB/HEALPix draw,
  including its padded tail, then shuffles exactly N references to each batch.
  Every exposure is a separate optimizer update, including single-modality
  batches. With gradient accumulation (2.8B–12B), exposures of one batch can share
  an optimizer update, and the logged per-step `echo_index` is the step's last
  micro-batch's; fresh/replayed totals stay exact. Under DP each rank echoes
  its own draws with a rank-seeded shuffle; telemetry records echo counters per
  rank (`steps.rankN.jsonl`).
- Later exposures reshuffle whole modality spans with private RNG; orders can
  repeat. Targets, positions, patch order, model noise, and schedules stay unchanged.
- The cache adds **one packed draw per worker**, on top of normal prefetch, and
  increases startup latency. Echoes do not fetch data; prefetch can still fetch
  ahead. Stopping mid-draw can leave uneven exposure counts.
- Remove obsolete `ar_replicas` / `replica_placement` keys from older YAMLs.

Telemetry reports processed and `fresh_*` / `replayed_*` counts and zero-based
`echo_index`. **Fresh means first use of a delivered batch, not unique objects,
reads, or bytes.** Rates share a wall-time denominator, excluding ten warmup
updates after start/resume but including loading/checkpoint stalls. W&B receives
only metrics/config: paths, provenance, code, data, and machine metadata stay local.

Resume with the matching `--config-file`, not the default. Model settings and
modalities are validated; weights/optimizer/scheduler/RNG restore, but **no data
cursor or pending echoes are saved**. Ordinary Nanotron launches use their
supplied YAML.

[Data echoing](https://arxiv.org/abs/1907.05550) motivates reuse, not a guaranteed
AstroPT speedup or quality gain. Factor 4 is policy, not a measured optimum.
Compare wall-time-to-held-out-quality and fresh-data budgets, not just updates/s.
