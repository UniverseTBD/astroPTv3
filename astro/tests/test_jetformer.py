"""jetformer tokeniser: flow + GMM head (JetFormer/GIVT-style); flow internals in test_jetformer_flow.py."""

import pytest
import torch
from legacy_fixture import record_stream
from test_model import single_band_collator as single_band_collator
from test_model import single_band_config as single_band_config
from test_model import single_band_sequencer as single_band_sequencer

from astropt3.data.packing import ObjectSequencer, PackedCollator
from astropt3.modalities import GMMHead, gmm_nll


@pytest.fixture(scope="module")
def jet_batch(single_band_sequencer, single_band_collator):
    return single_band_collator([single_band_sequencer.build(r) for r in record_stream(2)])


def test_gmm_nll_matches_torch_distributions():
    torch.manual_seed(0)
    n, k, d = 7, 4, 5
    logits_pi = torch.randn(n, k)
    mu = torch.randn(n, k, d)
    log_sigma = torch.randn(n, k, d).clamp(-2, 1)
    y = torch.randn(n, d)

    mix = torch.distributions.MixtureSameFamily(
        torch.distributions.Categorical(logits=logits_pi),
        torch.distributions.Independent(
            torch.distributions.Normal(mu, log_sigma.exp()), 1
        ),
    )
    assert torch.allclose(gmm_nll(y, logits_pi, mu, log_sigma), -mix.log_prob(y), atol=1e-5)


def test_gmm_head_shapes():
    head = GMMHead(64, 64, k=4)
    logits_pi, mu, log_sigma = head(torch.randn(11, 64))
    assert logits_pi.shape == (11, 4)
    assert mu.shape == (11, 4, 64)
    assert log_sigma.shape == (11, 4, 64)
    assert log_sigma.min() >= -7.0 and log_sigma.max() <= 2.0


def test_jetformer_skips_per_patch_standardization(single_band_config):
    """jetformer tokens must invert back to flux: no per-patch standardization."""
    from legacy_fixture import make_record

    from astropt3.data.band_registry import physical_normalize
    from astropt3.tokenization import antispiralise, patchify_image

    record = make_record(3, image_only_fraction=0.0)
    jet_seq = ObjectSequencer(single_band_config).build(record)
    from astropt3.data.packing import IMAGE_CROP

    flux = torch.as_tensor(record["image"]["flux"])
    h, w = flux.shape[-2:]
    top, left = (h - IMAGE_CROP) // 2, (w - IMAGE_CROP) // 2
    g = record["image"]["band"].index("des-g")
    flux = flux[g : g + 1, top : top + IMAGE_CROP, left : left + IMAGE_CROP]
    expected = patchify_image(physical_normalize(flux, ["des-g"]), 8)
    # undo the config's spiral patch order before comparing to raster patchify
    patches = jet_seq.values["legacy_north_g"]
    actual = antispiralise(patches) if single_band_config.spiral else patches
    assert torch.allclose(actual, expected)
    # spectra tokens are the physically normalized (ADR 0007) mask-zeroed
    # flux — invertible, but not per-patch standardized
    from astropt3.data.spectral import spectral_normalize

    sflux = torch.as_tensor(record["spectrum"]["flux"])
    lam = torch.as_tensor(record["spectrum"]["lambda"])
    assert torch.allclose(
        jet_seq.values["spectra"].flatten()[: len(sflux)],
        spectral_normalize(sflux, lam),
    )


def test_noise_curriculum_sigma_endpoints(single_band_config):
    from astropt3 import AstroPT3Model

    model = AstroPT3Model(single_band_config)
    model.set_jet_noise_frac(0.0)
    assert model._jet_noise_sigma() == pytest.approx(single_band_config.jetformer_noise_max)
    model.set_jet_noise_frac(1.0)
    assert model._jet_noise_sigma() == pytest.approx(single_band_config.jetformer_noise_min)
    model.set_jet_noise_frac(-3.0)  # clamped
    assert model._jet_noise_sigma() == pytest.approx(single_band_config.jetformer_noise_max)


def test_noise_curriculum_perturbs_train_only(single_band_config, jet_batch):
    from astropt3 import AstroPT3Model

    torch.manual_seed(0)
    model = AstroPT3Model(single_band_config)
    model.set_jet_noise_frac(0.0)  # sigma = noise_max

    # Training mode: noise on the embedded z copy makes forwards stochastic,
    # but the loss stays finite (targets/logdet are clean).
    model.train()
    out_a = model(**jet_batch)
    out_b = model(**jet_batch)
    assert torch.isfinite(out_a.loss) and torch.isfinite(out_b.loss)
    assert not torch.allclose(out_a.loss, out_b.loss)

    # Eval mode: noise off regardless of frac -> deterministic.
    model.eval()
    with torch.no_grad():
        eval_a = model(**jet_batch)
        eval_b = model(**jet_batch)
    assert torch.allclose(eval_a.loss, eval_b.loss)


def test_smoke_training_learns(single_band_config):
    """Local AdamW loop (astropt3.train_smoke is retired, ADR 0015)."""
    from astropt3.modeling_astropt3 import AstroPT3Model

    config = single_band_config
    torch.manual_seed(0)
    model = AstroPT3Model(config)
    model.train()
    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.requires_grad and p.dim() >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.1},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=1e-3,
        betas=(0.9, 0.95),
    )
    sequencer = ObjectSequencer(config)
    collator = PackedCollator(config, seq_len=896)
    losses = []
    for step in range(40):
        batch = collator([sequencer.build(r) for r in record_stream(2)])
        model.set_jet_noise_frac(step / 40)
        out = model(**batch)
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(out.loss.item())
    # Likelihood loss (NLL - logdet) can cross zero, so assert an absolute
    # drop rather than the affine gate's ratio check.
    assert losses[-1] < losses[0] - 10.0, f"{losses[0]:.4f} -> {losses[-1]:.4f}"
