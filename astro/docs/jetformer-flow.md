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

Flow parameters scale as width², like a trunk layer. The training recipes
have one flow per calibrated band, one for the 768-wide RGB display patches,
and one per spectrum source. `tests/test_jetformer_flow.py` enforces the
width rule on every `configs/nanotron/*.yaml`. `scripts/count_params.py`
reports the HF size configs' flows separately from the nominal-size check.

## Noise curricula

Set `pre_flow_noise_max` on individual patch modalities in `modalities`
(HF YAML) or `model.model_config.modalities` (Nanotron YAML). Omitted entries
use zero; scalars cannot use pre-flow noise. During training, Gaussian noise
is added to normalized patch values before the flow. Its standard deviation
cosine-decays from `pre_flow_noise_max` to `pre_flow_noise_max · 3/64`,
using the floor-to-peak ratio from [JetFormer's image–text recipe](https://arxiv.org/html/2411.19722v2#S4).
The noised flow pass supplies the latents and logdet for the likelihood.

The [70M Nanotron recipe](../configs/nanotron/astropt3-70m.yaml) also adds
Gaussian noise with constant σ = 0.1 to the patch latents fed into the
causal decoder. This second draw does not alter the likelihood targets
or logdet. `jetformer_noise_max` and `jetformer_noise_min` control the
linear latent-input schedule; 70M sets them equal. Neither curriculum
adds noise during evaluation or generation.

Choose σ in the normalized units each flow sees. The 70M recipe starts
Legacy North at `(0.21, 0.37, 1.08)` for `g, r, z` and HSC at
`(0.15, 0.20, 0.32, 0.49, 0.93)` for `g, r, i, z, y`. These estimates use
robust pixel scatter in the outer regions of 23 Legacy North and 26 HSC
training cutouts, not a dedicated blank-sky measurement. Legacy South has
no added noise. Galaxies-with-hats RGB uses one shared starting σ of `0.04`.

DESI and SDSS start at σ = 0.1 in normalized flux units, based on robust
scatter between adjacent bins in archived training spectra. The SDSS estimate
is provisional (three spectra). The schedule also perturbs filled masked
bins and tail padding during training.

For the [#52 ablation](https://github.com/UniverseTBD/astroPTv3/issues/52), hold
the `isqm4s6c` recipe, seed, training horizon, and validation samples fixed.
To isolate pre-flow noise, keep the latent-noise settings identical and set
`pre_flow_noise_max` to zero in the control. Compare clean total NLL,
source vs. sky token NLL on fixed cutouts, and linear-probe and scalar-head metrics.

## Limits

- CP=1 (asserted): the flow attends over whole object spans, which context
  parallelism would split. DP and TP are supported; flow parameters are
  TP-replicated like the other modality modules.
- MFU FLOPs (`modality_flops_per_token`) price attention over
  `max_positions`, an upper bound on span length, so flow FLOPs and MFU can be
  overstated for shorter spans.
- Permutations are derived from seed 42 + stage index, not stored; changing
  that rule breaks existing checkpoints.
