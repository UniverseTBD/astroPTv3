# Training AstroPTv3 models with nanotron

*The operational guide: environments, live survey data, launching, and
checkpoint/resume. Background on the model is in
[`architecture.md`](architecture.md). The initial experimental LSDB cutover
is recorded in [ADR 0015](adr/0015-lsdb-infinite-stream-training.md);
subsequent run configs can also use DESI×LegacySurvey crossmatching. The old
contract's `train_smoke` gate remains absent on this branch.*

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
nanotron. Image-only configs read LegacySurvey North; crossmatch-enabled
configs also include DESI EDR SV3 spectra. Depending on survey overlap,
records can carry an image, a spectrum, or both. Recovered image-only rows
come from LegacySurvey partitions visited by the join, **not** from every
LegacySurvey partition. The active YAML and loader are the sources for the
stream implementation; its fetch and prefetch settings are experimental.

Facts to know:

- **Network is a hard dependency.** The Hub must remain accessible; there is
  no local training-corpus fallback.
- **Workers and memory.** `num_loading_workers` controls concurrent readers
  per DP rank; each may buffer large image partitions. Monitor box-wide RSS
  and lower the worker count after an observed OOM.
- **No cursors or coverage guarantee.** Consumer seeds reduce identical
  draws, but consumers may overlap; retries and resumes can revisit records.
- **Provenance.** `uv.lock` records the resolved LSDB dependency, and `[data]`
  startup lines log its version and catalog information when available.

On resume, checkpoints restore weights/optimizer/scheduler/RNG, **not** the
LSDB record position. Recognized transport/storage errors discard the failed
iterator and reopen under bounded backoff; decode/validation and unknown
errors fail immediately.

## 3. Configs

Full nanotron run configs live in `astro/configs/nanotron/`. For the imaging
+ spectra pilot, the dataset block includes:

```yaml
data_stages:
- data:
    dataset:
      is_astropt3_streaming: true
      crossmatch_desi: true         # include DESI spectra where present
      # ar_replicas: 1              # optional: distinct AR factorisations
      # replica_placement: decorrelated
    num_loading_workers: 8          # concurrent LSDB readers per DP rank
    seed: 42
```

The catalog projections and stream policy live in
`astropt3.data.nanotron_loader`. Without `crossmatch_desi: true`, the source
is LegacySurvey-only. The retired knobs (`data_root`, `match_index`,
synthetic fractions, `object_id_log`) are not supported.

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

Use the actual run YAML for batch sizes and parallelism; the historical
recipe and size-ladder rationale live in
[`../EXPERIMENTS.md`](../EXPERIMENTS.md) Part I.

## 4. Launching

Single node (from the repo root):

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

Do not set `HF_DATASETS_OFFLINE=1` — training streams from the Hub. The
source-backed evaluation sidecar (`run_probe_sweep.py`) and `EVAL_GPU`
co-launch hook are removed; §6 describes the remaining model-side tools.

## 5. Checkpoints and resume

Checkpoint dirs `{1,2,4,...,512,1000,...}` hold model weights (bf16),
optimizer + LR-scheduler state, RNG states, `model_config.json`;
`latest.txt` is written last, so any step dir it covers is complete.

Resume:

```yaml
checkpoints:
  resume_checkpoint_path: <checkpoints dir>   # reads latest.txt
```

The run restores model/optimizer/scheduler/RNG **and nothing else**. The
LSDB stream is cursorless: resume opens a fresh stream and records may be
revisited immediately. There is no exact-sequence continuation, no replay
audit, and no worker-count constraint on resume.

Watch the logs: `lm_loss`, per-modality `images_loss` and (when DESI is
included) `spectra_loss`, `tokens_per_sec_per_gpu`, `model_tflops_per_gpu`,
and memory lines. Exact-likelihood losses can be negative.

## 6. Evaluation status

Source-backed validation/probe checkpoint sweeps are deferred.
`astropt3.eval` keeps model-side functions — `evaluate` (loss on provided
batches), `embed_objects` + `ridge_r2` (probe on supplied objects),
`scalar_head_metrics`, and sampling/rendering. `scripts/generate.py` remains
a separate live-sample CLI; it is **not** a fixed validation suite or an
automated checkpoint sweep.

## 7. Verification gates

1. `uv run pytest -m 'not gpu and not network'` for offline CPU checks in
   `astro/`. Plain `uv run pytest` also selects network-marked tests, which
   may fail when the Hub is unavailable.
2. `uv run python scripts/count_params.py` for current size totals.
3. The former synthetic-dependent `train_smoke` gate is suspended; these
   checks do **not** complete that phase gate.

For a bounded live check with Hub access:
`uv run pytest -m network tests/test_lsdb_stream.py`.

## 8. Standard throughput benchmark

Use the **training environment** and the same torchrun arguments as training.
One invocation measures one YAML; it does not resize the workload for your
hardware. The output directory must be new and, for multi-node launches,
on storage shared by **all** ranks. A fresh run ID is broadcast and checked
against the manifest on every rank before any rank-local output is written.

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 \
torchrun --nproc_per_node=<YAML dp*tp on this node> \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
    astro/scripts/benchmark_throughput.py \
    --config-file astro/configs/nanotron/<config>.yaml \
    --warmup-steps 10 --steps 200 --output /shared/benchmarks/<new-run>
```

The counts are examples, not representative-window guarantees. For the
50-step tiny YAML, use e.g. `--warmup-steps 2 --steps 5`. Supply an external
cap description with `--bandwidth-cap '...'` and known cache state with
`--cache-state '...'`; both otherwise say `unknown`. These options **record**
context; they never apply throttling or flush caches. Repeats are manual.
Use your existing launcher/job wall-time limit, including for live checks.

### What runs and what is saved

The script uses nanotron's `DistributedTrainer`, `get_dataloader`, and
unchanged training loop. It executes warmup + measured optimizer steps
**in addition to** any resumed step. The YAML training horizon, LR/noise
schedules, model, modalities, worker count, batching and parallelism remain
unchanged. A run that would exceed the remaining horizon fails. Initial,
periodic and final checkpoint writes, evaluation, profiling, uploads and
external tracker calls are disabled; all overrides are recorded. Explicit
local initialization/resume is preserved and the resolved checkpoint is
recorded. A missing explicit resume fails rather than starting randomly.
Stage remote checkpoints locally first; the initial runner supports one
AstroPT streaming stage and PP=CP=EP=1, with DP/TP selected by the YAML.

- `baseline.yaml`: original YAML including comments; `effective.yaml`:
  benchmark overrides applied; `resolved.yaml`: nanotron's resolved defaults.
- `run.json`: schema version, requested window, baseline and overrides.
- `rankN.json`: status, actual rank/topology/device, host CPU/RAM, software,
  code revisions/dirty state, initialization and loader/catalog context.
- `trainer.rankN.log`, `steps.rankN.jsonl`, `bytes.rankN.{main,wW}.jsonl`,
  `probe.rankN.{main,wW}.json`: raw local diagnostics and instrumentation.
- `result.json`: versioned summary; the terminal prints the primary rates.

After a crash or launcher termination, regenerate the report once processes
have exited (no collection service is needed):

```bash
python astro/scripts/bench_report.py --telemetry /shared/benchmarks/<run> \
    --out /shared/benchmarks/<run>/result.json
```

Failed/missing ranks, missing or duplicate steps, malformed logs, or absent
byte probes produce a **partial** report and nonzero reporting exit status,
not a successful benchmark. Interrupted runs retain their diagnostics.

### Clock boundaries and interpretation

Startup is runner `main()` entry through the synchronized training boundary,
including imports, setup/resume, and a single retained first-batch fetch.
`first_batch_s` separately measures iterator creation through that fetch;
it is a **subset of startup**, not an additional training step. That batch
is consumed and counted once in training. Warmup begins after startup;
measured timing begins at the end of the last warmup step (or immediately
when warmup is zero) and ends after the final measured step.

With telemetry enabled, CUDA synchronization and a world barrier close every
step. Continuous rank-local wall clocks include streaming stalls, packing,
compute, console logging and inter-step telemetry work. Throughput uses the
largest rank's **whole measured-window duration**, not an average of step
rates or a sum of slowest-rank step durations. Per-step distributions remain
diagnostics. Synchronization/instrumentation impose overhead; compare runs
using this same policy. No GPU theoretical-FLOPs lookup is needed.

Useful tokens exclude padding. Logical tokens and per-modality loss tokens
are summed over DP and counted only once across TP (TP0), then divided by
**actual GPU world size** for per-GPU rates. Modality mix and packing efficiency
travel with the rate: a different mix is a different workload. Object counts
are presentations, **not unique astronomical coverage**.

Payload bytes count every returned HF range/stream-read payload across all
ranks/workers, including duplicated downloads and prefetch, with Legacy/DESI
source attribution. They exclude HTTP overhead and failed/unreturned partial
transfers. The observed payload rate divides whole-run bytes by the maximum
rank duration from runner entry through loader shutdown (including explicit
fetch-thread/worker joins, bounded by your launcher wall-time limit). It is **not connection
capacity**, nor bytes consumed exclusively by measured steps. Cache state is
unknown unless supplied. A short window (<200 measured steps or <60 seconds)
is flagged as limited evidence, not promoted to a stable hardware comparison.

Multi-node and any unexercised DP/TP topology require a bounded live validation
on suitable hardware before claiming support; CPU arithmetic tests alone do
not establish distributed training correctness.
