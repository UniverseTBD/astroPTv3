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
or flushed. On one GPU, `astropt3-70m-dp1.yaml` runs the 70M recipe's per-GPU
workload (DP=1; same model, batch shape and loader workers).

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
