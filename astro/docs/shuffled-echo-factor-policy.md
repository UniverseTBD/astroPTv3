# Shuffled-echo factor policy: primary-source check

## Scope and conclusion

Independently checked the full text of Choi et al., *Faster Neural Network
Training with Data Echoing*: [v1](https://arxiv.org/html/1907.05550v1), cited by
issue #47, and [final v3](https://arxiv.org/html/1907.05550v3), 8 May 2020.
Public bounded `curl` fetches succeeded; artifacts:
`/tmp/astropt3-data-echoing-1907.05550v1.html` and
`/tmp/astropt3-data-echoing-1907.05550v3.html`. The paper was checked independently
of the prior local assessment, which remains untouched. A separate configuration
and live-baseline inspection is recorded below.

**Decision: implement shuffled batch reuse within each already-loaded
HEALPix draw, without another experiment campaign.** Keep modality-span
reshuffling on subsequent exposures and retain the existing first-presentation
shuffle. The user subsequently selected **factor 4 in the new four-modality
YAML default**; the legacy schema fallback remains 1. Modality count and echo
factor are independent settings. The paper supplies evidence for factors 2/4,
not an untuned AstroPT default or quality guarantee. V3's real 3.25× speedup
uses **example echoing at factor 5**, not whole-batch echoing at 4.

## Factor evidence retained in final v3

- **Meaning and timing:** `e` counts total uses, including the original, not
  additional repeats. [V1 §1](https://arxiv.org/html/1907.05550v1#S1.p4) defines it as
  “the number of times each intermediate output gets used.”
  [V3 §2, Eq. (1)](https://arxiv.org/html/1907.05550v3#S2) retains
  `R = t_upstream / t_downstream` and time per fresh upstream step plus `e`
  downstream steps as `max(t_upstream, e × t_downstream)`. This assumes parallel
  overlap and negligible echo overhead. Extra updates are time-free only while
  `e ≤ R`; this is not a statement about their statistical usefulness. The
  theoretical maximum plausible speedup is `R`, if repeats equal fresh data.
- **Factor 2:** [§3.1, Fig. 3](https://arxiv.org/html/1907.05550v3#S3.SS1)
  tests batch and example echoing across all five workloads. Fresh-example
  requirements fall except for ResNet-50 batch echoing, which is approximately
  unchanged. Example echoing is generally better; echoing before augmentation
  is better than after. This is the broadest evidence for a starting experiment.
- **Factor 4:** [V3 §3.6, Fig. 8](https://arxiv.org/html/1907.05550v3#S3.SS6)
  uses 4 for **all variants, including batch echoing**, on Transformer/LM1B and
  ResNet-50/ImageNet. All match or beat baseline predictive performance at a
  fixed **fresh-example budget**, not a fixed update budget. They retune using
  500 Transformer trials and 100 ResNet trials. Thus 4 has direct batch-echoing
  evidence, but not an untuned-default guarantee. These conclusions and tuning
  budgets are unchanged from [v1 §3.6](https://arxiv.org/html/1907.05550v1#S3.SS6).
  Under the shared §3 pipeline, batch echoes retain the upstream example shuffle
  buffer but do not reshuffle repeated batches or reapply image augmentation.
- **Factor 8:** [§3.3, Fig. 5](https://arxiv.org/html/1907.05550v3#S3.SS3)
  concerns **example echoing**, not batch echoing. For Transformer/LM1B,
  batch size 1024, the maximum useful factor is “somewhere between 4 and 8”;
  beyond that, fresh-example requirements worsen relative to smaller factors.
  Even 16 beats baseline; at batch size 4096 the useful limit exceeds 16.
  These are workload/batch-size-dependent results, not a recommendation for 8.
- **Non-vision and tuning:** [V3 §3](https://arxiv.org/html/1907.05550v3#S3)
  includes Transformer language modeling on LM1B and Common Crawl. Transformers
  use **Nesterov momentum and constant learning rate**, not AdamW. Each
  experiment independently tunes learning rate, momentum and applicable
  schedule parameters (ResNet linear decay; SSD warmup/exponential decay).
  V3 adds specifics: ResNet's decay duration is tuned; ImageNet's learning rate
  falls by 1,000× over that duration (§3, footnote 8). Ordinary Transformer
  searches use 100 non-divergent trials, repeated five times.
  [§4](https://arxiv.org/html/1907.05550v3#S4) notes Common Crawl training
  covers less than one epoch, so the evidence is not limited to small datasets.

## Final v3: the measured 3.25× experiment

[V3 §3.2 and Fig. 4](https://arxiv.org/html/1907.05550v3#S3.SS2) replace
[v1 §3.2/Fig. 4's assumed-ratio projections](https://arxiv.org/html/1907.05550v1#S3.SS2)
with an actual wall-clock experiment:

- **Workload/bottleneck:** ResNet-50 on ImageNet streamed from cloud storage;
  transfer latency dominates, with measured `R ≈ 6`, fluctuating with network
  transfer rate. This ratio is specific to their pipeline.
- **Exact variant/factor:** factors 1–5; the caption explicitly says “Example
  echoing before augmentation.” The text reports “a speedup factor of 3.25 for
  echoing factor 5.” This is not evidence of 3.25× speedup for batch echoing or
  factor 4.
- **Augmentation/shuffling:** the shared [§3 setup](https://arxiv.org/html/1907.05550v3#S3.p6)
  places a 100,000-example ImageNet shuffle buffer **after example echoing**, in
  place of the baseline buffer. Subsequent augmentation redraws resize/crop,
  horizontal reflection and color distortion. Default batch size is 1024;
  ResNet uses Nesterov momentum, linear learning-rate decay, and batch
  normalization with virtual batch size 32. §3.2 states no override of these
  shared settings.
- **Tuning and timed endpoint:** separately tune each factor, choose settings
  reaching **75.3% validation accuracy fastest**, then rerun the selected
  settings **five times per factor**, recording fresh examples and **wall time
  to 75% accuracy**. Thus the headline is time-to-quality, not throughput at a
  fixed update count. Their general ResNet-50 search budget is 50 non-divergent
  trials; §3.2 does not specify a different budget. Search cost is not included
  in the reported selected-run time-to-target speedup.
- **Measured overhead:** wall-time reduction is slightly smaller than the
  fresh-example reduction because repeating data has overhead. The result is
  therefore stronger evidence than v1's hypothetical wall-time curves, but
  still not a universal factor or speedup guarantee.

Reporting caveat: [Appendix A, Fig. 10](https://arxiv.org/html/1907.05550v3#A1.F10)
uses a generic caption describing five independent hyperparameter searches,
whereas §3.2 explicitly describes five reruns of selected settings. The protocol
above follows the experiment-specific §3.2; the paper does not reconcile that
wording.

## Applying this to the described AstroPT behavior (inference)

Repeating whole packed CPU batches for separate optimizer updates is **batch
echoing**. Redrawing within-object modality-span order is an augmentation-like
change to autoregressive conditioning, not fresh objects/targets and not the
paper's cross-example or cross-batch shuffle buffer. [V3 §2](https://arxiv.org/html/1907.05550v3#S2)
explains augmentation can make repeats more distinct;
[§3.5, Fig. 7](https://arxiv.org/html/1907.05550v3#S3.SS5) instead mixes repeated
examples/batches across time. It does not establish that modality reshuffling
provides equivalent diversity or makes 4/8 safe for AdamW.

The implemented reuse additionally shuffles repeated **batches across updates**,
within the finite current draw. This follows §3.5's motivation without claiming
to reproduce its exact shuffle-buffer pipeline. It does not mix examples into
new batches or cross draw boundaries. Fixed fresh data means more updates;
fixed updates means less fresh data. The implementation does not multiply
training duration or change the optimizer, LR/noise schedules, or checkpoint
cadence. At a partial-draw stop, not every batch has exhausted its repetitions,
so fresh counts must be observed rather than inferred from step / factor.

## AstroPT implementation and configuration

The earlier CLI-only adapter did not save its factor in Nanotron YAML. That gap
is now closed: [`AstroPT3StreamingDatasetsArgs`](../../nanotron/src/nanotron/config/astropt3_config.py)
contains a validated positive-integer `echo_factor`, default 1. The
[`YAML-driven runner`](../scripts/train_three_catalogs.py) and ordinary
[`run_train.py`](../../nanotron/run_train.py) launches both feed that setting to
[`build_astropt3_dataloader`](../src/astropt3/data/nanotron_loader.py).
Astronomy-specific ordering stays in `astro/`, not the Nanotron trainer.
The new [`astropt3-70m-four-modal.yaml`](../configs/nanotron/astropt3-70m-four-modal.yaml)
is the launcher's default. Its `catalog_stream` specification and model modality
list generalize the loader and reporting beyond four modes; see the
[training guide](training.md#9-yaml-driven-n-modality-training-and-shuffled-echoing).

```yaml
data_stages:
  - data:
      dataset:
        is_astropt3_streaming: true
        echo_factor: 4  # new four-modality recipe; dataclass fallback remains 1
```

- **Saved source of truth:** the adapter resolves explicit CLI > checkpoint
  factor on resume > baseline/default before writing effective/resolved YAML.
  Nanotron also saves the field in each checkpoint's `config.yaml`. Missing
  legacy fields mean 1. An omitted launcher argument no longer forces 1.
  Ordinary Nanotron launches honor their supplied run YAML; loading a saved
  checkpoint config retains its setting.
- **Draw ownership:** for factors above 1, each loader worker packs and caches
  one finite LSDB/HEALPix draw, including its padded tail. It shuffles a multiset
  with exactly N references to each batch, emits every reference, then advances
  draws. Empty draws emit nothing. Python's global RNG and model RNG are not
  consumed. The source epoch and run/rank/worker seed determine draw shuffling.
- **Span ownership:** first actual exposure keeps the original modality order;
  subsequent ones redraw whole spans. Objects, within-modality patches, physical
  targets, document positions, and row-major payload alignment remain unchanged.
  Single-modality batches repeat too. Mappings are owned; cached tensors are
  read-only. Factor 1 retains the existing cross-draw streaming packer.
- **Update/accounting boundary:** factors above 1 require accumulation=1 and
  world_size=dp_size=1. Each exposure is a separate optimizer update. Per-batch
  occurrence metadata is stripped before device transfer even when telemetry is
  disabled. Fresh/replayed counts reflect actual occurrences, not global step
  arithmetic. Fresh is newly presented packed data, not unique objects or bytes.
- **Resource boundary:** caches hold one packed draw per worker, in addition to
  normal LSDB/DataLoader prefetch. Full-draw packing costs startup latency and
  more resident memory than the former one-batch cache. Echoing itself does not
  fetch catalogues; normal asynchronous prefetch can still fetch ahead.
- **Resume boundary:** restore the factor, not pending batches, repeat counts,
  or an LSDB cursor. Resuming opens a fresh stream and can revisit objects.

## Evidence limits

A read-only snapshot of the active four-survey baseline, steps 12695–13694,
found 3.850 s/update, 89.27% time-weighted loader wait, and a 0.420 s median
ready-step duration (944 of 1,000 steps had <10 ms wait). These uncontrolled
measurements motivate reuse but do not independently measure `R` or establish
echo speedup/quality. Source: the local
[four-survey step log](../../../astroPTv3_telemetry/four-modal-30k-20261105T040243Z-2611048/telemetry/steps.rank0.jsonl).

The user chose implementation, not the earlier proposed 1/2/4 experiment
campaign. No additional training experiments or live-run changes accompany this
work. Offline tests check scheduling, payload alignment, accounting, and saved
configuration. The later user-launched four-survey experiment is separate and
is not restarted by the YAML/N-modality refactor. That refactor has not had a
new live/GPU validation run; no AstroPT speedup or learning-efficiency claim
is made, and AdamW held-out convergence remains unestablished.
