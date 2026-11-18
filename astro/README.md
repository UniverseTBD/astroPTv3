<div align="center">

<img src="assets/shoggoth_telescope_sticker_2.png" alt="astroPT_shoggoth" width="300"/>

*There is some [deep lore about our logo](https://doi.org/10.4000/12m9y).*

</div>

# AstroPTv3

AstroPTv3 is a from-scratch suite of astronomical foundation
model configurations spanning 70M–12B parameters. Its Pythia-mirrored sizes
and checkpoint schedule support a research question: **how does learning from
continuous sky-survey measurements change as model size and training progress
increase?** The size ladder is an experimental design, not a claim that every
size has been trained or evaluated.

## Lineage

AstroPTv3 is the multimodal sister line of
[AstroPT](https://github.com/Smith42/astroPT) (nanoGPT lineage; the
[original paper](https://arxiv.org/abs/2405.14930), the AION tokeniser
experiments, and the scaling/probing programme live there). Its
JetFormer/GIVT tokeniser was ported from AstroPT's `sogol_branch`. The
two projects co-evolve and share findings — normalization conventions,
evaluation practice, checkpoint schedules — while their codebases stay
separate. The full AstroPTv3 docs (plan, experiments, guides, and ADRs)
are hosted in the
[AstroPTv3 section of AstroPT's Read the Docs site](https://astropt.readthedocs.io/en/latest/astropt3.html).

## Astronomy pilot

The [default 70M recipe](configs/nanotron/astropt3-70m.yaml) streams
[Multimodal Universe](https://huggingface.co/collections/UniverseTBD/multimodal-universe-hats)
Legacy Survey North and DR10 South images, DESI and SDSS spectra, HSC Wide images, and
[galaxies-with-hats](https://huggingface.co/datasets/Smith42/galaxies-with-hats)
RGB images with nine scalars (redshift, photo-z stellar mass, SFR, sSFR and five
Galaxy Zoo vote fractions); every rung of the Pythia ladder
(`astropt3-{70m…12b}.yaml`) trains the same modalities. All training recipes use YAML-configured catalog streams and
modality lists; N-modal training is the normal path. New physical survey
calibrations still require code.

North and South each anchor an independent join in sky covered by the anchor
and at least two outer partners. North owns shared eligible sky. Unmatched
outer-join records remain separate objects. **This is not a complete
survey pass.** Models learn from whichever configured modalities a record
carries. See the [training guide](docs/training.md) for the experimental stream
and echoing constraints.

## From measurements to predictions

Legacy Survey North images are 3×152×152 flux cubes; DR10 South images are
4×160×160. The sequencer crops to 96×96, physically normalizes each band,
and forms 144 patch tokens (192 values for North, 256 for South). DESI spectra are normalized onto an AB-flux scale and form
31 patches of 256 bins with wavelength positions. There is **no per-patch
standardization**: the flow-based likelihood needs the record-to-token
mapping to retain patch information. Optional catalog scalars can become
one-token spans when included in the config.

A small 64-id vocabulary marks modality spans; it is **not** a text
vocabulary. The SmolLM3 decoder predicts the next continuous patch using a
per-modality JetFormer flow and Gaussian-mixture likelihood head. Whole
astronomical sources are packed into training sequences without splitting an
object. The nanotron fork (`../nanotron`) runs pretraining; the transformers
implementation in `src/astropt3/` supports checkpoint conversion, generation,
and probing. See [architecture](docs/architecture.md) for the model contract
and [the lab book](EXPERIMENTS.md) (charter/roadmap, fixed decisions,
experiment log) for the size ladder.

## Set up and verify

From this directory:

```bash
uv sync --extra dev
uv run pytest -m 'not gpu and not network'  # offline CPU checks
uv run python scripts/count_params.py       # nominal-size tolerance
```

Live data checks need Hub access:

```bash
uv run pytest -m network tests/test_lsdb_stream.py
```

Training needs nanotron, flash-attn, a GPU, and live catalog access. From the
repository root:

```bash
bash astro/scripts/run_train.bash GPU_INDEX
```

See [training.md](docs/training.md) for custom recipes, echo overrides, resource
limits, and resume instructions. There is no source-backed evaluation sweep;
model-side evaluation and sample rendering are in `src/astropt3/eval/` and
`scripts/generate.py`.

For a bounded, YAML-based training benchmark with standard JSON results, see
[throughput benchmarking](docs/training.md#8-standard-throughput-benchmark).
The [loader experiment log](EXPERIMENTS.md) records measured throughput and
coverage limits.
