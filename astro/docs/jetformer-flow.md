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
use zero; scalars cannot use pre-flow noise. The [70M Nanotron recipe](../configs/nanotron/astropt3-70m.yaml)
uses pre-flow noise only, for both images and spectra.

During training, independent Gaussian noise is added to normalized patch
values before the flow, with `σ = pre_flow_noise_max · (1 + cos(π · step / train_steps)) / 2`.
The same flow pass supplies the latents used for embedding and likelihood
and the logdet used in the loss. Evaluation and generation always use clean
inputs. `jetformer_noise_max` and `jetformer_noise_min` still control the
independent linear latent-input noise; set both to zero for pre-flow only.

Choose σ in each modality's normalized units. For calibrated images, estimate
per-band noise on held-out blank-sky patches and pick a representative scale
for the modality (a single σ is shared across bands). The 70M recipe starts
DESI and SDSS at σ = 0.1 in normalized flux units, based on robust scatter
between adjacent bins in archived training spectra. The SDSS estimate is
provisional (three spectra). The schedule also perturbs filled masked bins
and tail padding during training.

For the [#52 ablation](https://github.com/UniverseTBD/astroPTv3/issues/52), hold
the `isqm4s6c` recipe, seed, training horizon, and validation samples fixed.
Compare latent-only (baseline) and pre-flow-only runs. After
σ reaches zero, compare clean total NLL, source vs. sky token NLL on fixed
cutouts, and linear-probe and scalar-head metrics.

## Limits

- CP=1 (asserted): the flow attends over whole object spans, which context
  parallelism would split. DP and TP are supported; flow parameters are
  TP-replicated like the other modality modules.
- MFU FLOPs (`modality_flops_per_token`) price attention over
  `max_positions`, an upper bound on span length, so flow FLOPs and MFU can be
  overstated for shorter spans.
- Permutations are derived from seed 42 + stage index, not stored; changing
  that rule breaks existing checkpoints.
