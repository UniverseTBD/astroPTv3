# JetFormer flow

The sole jetformer flow (`src/astropt3/flows.py`, `JetFormerFlow`), ported from
the cross-patch pilot (see [lineage](jetformer-flow-lineage.md)). Each patch
modality gets `jetformer_flow_steps` affine couplings. Each coupling is
conditioned by `jetformer_flow_depth` pre-norm Transformer blocks of width
`jetformer_flow_hidden`, with full attention over one object's one modality
span (never across packed objects; `flow_object_ids`).

## Width rule

There is no published scaling rule. JetFormer held its flow fixed (~450M)
while growing the trunk. Ours is a convention: 8 stages, depth 4, 8 heads, and

    jetformer_flow_hidden = min(hidden_size // 4, 512)

Flow parameters scale as width², like a trunk layer. With the five patch
modalities (Legacy, HSC and galaxies-with-hats images; DESI and SDSS spectra)
the flows are ~33–50% of trunk parameters up to 1B; the 512 cap (JetFormer's
flow size) holds them at ~519M from 1B upward. The 768-wide galaxies-with-hats
patches make its flow the largest. `tests/test_jetformer_flow.py` enforces the
rule on every `configs/nanotron/*.yaml`. `scripts/count_params.py` reports
flows in their own column and excludes them from the nominal-size check.

| Trunk | hidden_size | Flow width | Five flows |
|---|---:|---:|---:|
| 70M | 512 | 128 | 35.3M |
| 160M | 768 | 192 | 76.5M |
| 410M | 1024 | 256 | 133.5M |
| 1B and up | ≥2048 | 512 | 518.6M |

## Gaussian factor-out

Each patch modality accepts `d_ar`, an integer in `[1, input_size]` (default:
`input_size`). The flow still transforms the complete patch. Only
`z[..., :d_ar]` enters its input encoder and autoregressive GMM head; the
remaining coordinates have a fixed standard-normal prior. Scalars always
use their full width. For example, HSC's `input_size: 320` with `d_ar: 64`
assigns 256 coordinates to the prior.

The per-token exact likelihood is

    GMM_NLL(z[:d_ar]) + 0.5 * sum(z[d_ar:]² + log(2π)) - logdet

Both terms train the flow end to end. Factor-out does not remove the noise
entropy cost; it changes which model component must explain it. The full-width
default evaluates the same loss as full-dimensional GMM prediction.

HF outputs expose `loss_components[modality]` with `ar`, `prior`, and `logdet`
means. Nanotron logs `<modality>_ar_loss`, `<modality>_prior_loss`, and
`<modality>_logdet_loss`; total modality NLL is AR + prior − logdet. Absent
Nanotron modalities log zero. Family balancing is unchanged.

Generation draws the AR part from the GMM and the tail from N(0, I), then
inverts the complete latent span. Temperature affects only the GMM; point
sampling (`argmax=True`) and reconstruction use a zero prior mean. The
training noise curriculum perturbs embedded latents only.

Changing `d_ar` changes encoder/head shapes; train each width from scratch.
Conversion and same-config checkpoint resume preserve the split. For a bounded
fixed-draw comparison, use `scripts/benchmark_factor_out.py`; short-run results
are not evidence of improved full-training linear-probe or scalar-head accuracy.

## Limits

- CP=1 (asserted): the flow attends over whole object spans, which context
  parallelism would split. DP and TP are supported; flow parameters are
  TP-replicated like the other modality modules.
- MFU FLOPs (`modality_flops_per_token`) price attention over
  `max_positions`, an upper bound on span length (images use 144 of 361), so
  flow FLOPs and MFU are overstated.
- Permutations are derived from seed 42 + stage index, not stored; changing
  that rule breaks existing checkpoints.
