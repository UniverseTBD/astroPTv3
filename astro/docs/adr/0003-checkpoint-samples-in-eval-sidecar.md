# ADR 0003: Per-checkpoint sample panels in the eval sidecar

- **Status:** Implemented
- **Date:** 2026-07-16
- **References:**
  - `astro/EXPERIMENTS.md` Part I Phase 4 — "Eval is fully outside the trainer" (the hard
    principle this ADR inherits)
  - `astro/src/astropt3/generation.py` — the sampling machinery
    (`generate` jetformer-only, `reconstruct` both tokenisers)
  - `astro/scripts/run_probe_sweep.py` — the polling sidecar this lands in
  - `astro/src/astropt3/eval/samples.py` — the shared implementation

## Question

For shakeout/test runs (and, being cheap, all runs), we want to *see* what
the model generates — a rendered output image and spectrum — at each (or a
thinned subset of) training checkpoints, logged to wandb so a panel shows
evolution over training. Where should this run, and into which wandb run
should it log?

## Context

- Sampling machinery already exists: `astropt3.generation.generate`
  (autoregressive GIVT sampling; jetformer-only — it needs the GMM heads and
  invertible flows; O(T²) with no KV cache, ~175 forwards for image+spectra)
  and `reconstruct` (one teacher-forced forward; works for affine
  checkpoints too). `scripts/generate.py` already inverted tokens to
  flux/wavelength space and rendered PNGs, as a manual one-off.
- Evaluation never runs in the trainer (PLAN Phase 4): the sidecar
  `run_probe_sweep.py` polls a run's checkpoint dir (gated on `latest.txt`),
  converts each step to HF, and runs `val_loss` + `linear_probe` on a spare
  GPU. Before this ADR it wrote JSONL only — no wandb.
- The nanotron trainer owns a live wandb run whose internal step counter is
  monotonic; wandb rejects/misorders logs at steps below a run's current
  step. The sidecar necessarily lags training.
- Per-checkpoint comparability requires holding everything but the weights
  fixed: same template record(s), same sampling seed, at every step.

## Decision drivers

- Never block or destabilize the training step (eval-outside-trainer, and
  the nanotron fork must stay thin).
- Panels must be comparable across checkpoints — differences should reflect
  the model, not the sampling noise or the template.
- One rendering implementation — `scripts/generate.py` and the sweep must
  not drift apart.
- Work for both tokenisers: `generate` raises for affine checkpoints, so a
  fallback is required.
- Cheap enough to leave on by default ("possibly all runs if the step is
  cheap"): seconds per checkpoint on the sidecar's spare GPU.

## Options considered

### Option A — Trainer-embedded sampling callback

Sample from the live model inside the nanotron trainer at checkpoint time,
log to the trainer's wandb run. **Rejected:** violates the
eval-outside-trainer principle, blocks the training step on O(T²)
autoregressive sampling, and thickens the fork (rendering + inversion code
would need mirroring or importing into nanotron ranks).

### Option B — Sidecar resumes the trainer's wandb run

Generate in the sidecar but log into the trainer's run via its run id.
**Rejected:** the sidecar lags training, so its logs land at steps below
the run's current step — fighting wandb's monotonic internal step counter —
and sweep restarts would be coupled to knowing the trainer's run id.

### Option C — Sidecar owns a separate eval wandb run (chosen)

The sweep initializes its own run (project `astropt3`, `job_type: eval`,
deterministic id `eval-{checkpoints-dir name}` so restarts resume it) and
declares `checkpoint_step` as a `define_metric` step-metric — every panel
and scalar plots against the training step regardless of when the sidecar
processed it. Eval scalars (val loss, per-modality losses, probe R²) are
mirrored alongside the sample panels, making the run the single eval
dashboard; the JSONL stays the authoritative done-step record.

### Option D — Status quo (manual `scripts/generate.py`)

Run generation by hand against interesting checkpoints. **Rejected as the
default:** no automatic evolution panel; exactly the manual toil this
feature removes. The script survives as a one-off tool on the same shared
implementation.

## Decision

**Sample and render in the eval sidecar, per processed checkpoint, into the
sidecar's own wandb run (Option C).**

1. **Placement:** `run_probe_sweep.py` samples after val-loss/probe for
   every checkpoint it processes, thinnable via the pythia predicate
   (`should_checkpoint(step, --samples-every)`: every pow2 ≤ 512 — dense
   where the model changes fast — plus every Nth step). Default: every
   checkpoint. `--sample-records none` disables.
2. **Determinism:** template record(s) fixed for the whole sweep (loaded
   once; `--sample-records` indices into `--data-root`), and a fresh
   generator seeded with `--seed` per (record, mode) — panels differ only
   through the model weights.
3. **Mode gating:** jetformer checkpoints get `unconditional` +
   `image-to-spectra` (`default_modes`); affine checkpoints fall back to
   `reconstruct` (the only mode `generate` cannot serve). Truth panels are
   always drawn, labelled "truth (reference)" for unconditional samples
   (where the record is a reference, not a conditioning target).
4. **wandb:** opt-in `--wandb`; separate eval run as in Option C; images
   under `samples/{mode}/{modality}/{object_id}`, scalars under `val/*` and
   `probe/r2`, all keyed on `checkpoint_step`.
5. **Shared implementation:** the template-loading / sampling / inversion /
   PNG code moved from `scripts/generate.py` into `astropt3.eval.samples`;
   both the script and the sweep are thin drivers over it. CPU tests cover
   the module offline (`tests/test_generate.py`).
6. **Launcher hook:** `EVAL_GPU=<id>` on the launch scripts backgrounds the
   sidecar (`--watch --until-step <train_steps> --wandb`) on a spare GPU,
   deriving paths from the training config; the trainer↔sidecar interface
   remains the filesystem only (`latest.txt`). Unset ⇒ current behavior.

## Consequences

### Positive

- Evolution panels for free on every sweep — image and spectrum samples per
  checkpoint, comparable across steps, next to the loss/probe curves.
- Eval scalars finally land in wandb instead of hand-plotted JSONL.
- One rendering implementation; `generate.py` and the sweep cannot drift.
- Trainer and fork untouched; tests stay CPU/offline; no new dependencies
  (wandb and matplotlib were already core deps).

### Negative / tradeoffs

- Steps already recorded in `probe_results.jsonl` before this feature never
  get retro-sampled (the JSONL is the done-set); re-log by deleting their
  lines or using a fresh `--out-dir` — HF conversions are cached, so
  re-eval is cheap.
- Affine image panels are qualitative only: per-patch standardization
  discards each patch's mean/std, so the physical inverse cannot be exact.
- Two wandb runs per training run (trainer + eval) that a viewer must
  mentally join; no `group=` linking yet (see Open issues).
- Unconditional sampling cost grows O(T²)·n if sequences get longer (KV
  cache remains deferred per `generation.py`'s ponytail note).
- Spectra render in model patch space — no inverse flux normalization
  exists for spectra (see ADR 0002's superseded-context note on the future
  spectra-norm ADR).

## Open issues

- **wandb grouping:** set `group=` on both the trainer and eval runs (needs
  the run name plumbed to both sides) so dashboards join them automatically.
- **Retro-sampling:** a small backfill mode that re-renders samples for
  already-evaluated steps from the cached HF conversions, without
  re-running val/probe.

## References

- `astro/src/astropt3/eval/samples.py` — `default_modes`, `sample_template`,
  `render_sampled_tokens`, `sample_checkpoint`.
- `astro/scripts/run_probe_sweep.py` — CLI (`--sample-*`, `--wandb`),
  wandb run setup, per-step logging.
- `astro/scripts/generate.py` — thin one-off driver over the module.
- `astro/scripts/launch_jetformer_70m.sh`,
  `astro/scripts/uhhpc_launch_jetformer_70m.sh` — `EVAL_GPU=` sidecar
  co-launch.
- `astro/src/astropt3/checkpoint_schedule.py` — the thinning predicate.
- [ADR 0001](0001-jetformer-inverse-variance-loss.md), 
  ADR 0002 — house format precedents.
