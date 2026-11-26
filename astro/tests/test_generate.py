"""Sampling/generation from a jetformer checkpoint (astropt3.generation),
and the shared sample-rendering layer on top of it (astropt3.eval.samples)."""

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from legacy_fixture import make_record

from astropt3.config_io import load_model_config
from astropt3.configuration_astropt3 import LEGACY_SOUTH_MODALITIES, AstroPT3Config
from astropt3.data.packing import ObjectSequencer
from astropt3.eval.samples import (
    build_template,
    render_sampled_tokens,
    sample_checkpoint,
    sample_template,
)
from astropt3.generation import generate, reconstruct, sample_gmm
from astropt3.modeling_astropt3 import AstroPT3Model

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "model" / "test-tiny.yaml"


@pytest.fixture(scope="module")
def jet_config():
    config, _ = load_model_config(CONFIG)
    # Keep sampling tests small; recipe-level packing covers all image bands.
    config.modalities = [m for m in config.modalities if m.get("band") in (None, "des-g")]
    return config


@pytest.fixture(scope="module")
def smoke_model(jet_config, tmp_path_factory):
    """A briefly-trained tiny jetformer model, save/load-roundtripped."""
    from legacy_fixture import record_stream
    from transformers import AutoModel

    from astropt3.data.packing import ObjectSequencer, PackedCollator

    torch.manual_seed(0)
    model = AstroPT3Model(jet_config)
    model.train()
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.1},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=1e-3,
        betas=(0.9, 0.95),
    )
    sequencer = ObjectSequencer(jet_config)
    collator = PackedCollator(jet_config, seq_len=896)
    for batch in (
        collator([sequencer.build(r) for r in record_stream(2)]) for _ in range(5)
    ):
        out = model(**batch)
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        opt.step()
    save_dir = tmp_path_factory.mktemp("ckpt") / "jet"
    model.save_pretrained(save_dir)
    return AutoModel.from_pretrained(save_dir).eval()


@pytest.fixture(scope="module")
def template(jet_config):
    return build_template(ObjectSequencer(jet_config), make_record(3, image_only_fraction=0.0), "unconditional")


def test_sample_gmm_argmax_is_mixture_mean():
    torch.manual_seed(0)
    logits_pi = torch.randn(5, 3)
    mu = torch.randn(5, 3, 4)
    log_sigma = torch.randn(5, 3, 4)
    got = sample_gmm(logits_pi, mu, log_sigma, argmax=True)
    pi = torch.softmax(logits_pi, dim=-1)
    assert torch.allclose(got, (pi.unsqueeze(-1) * mu).sum(-2))


def test_unconditional_shapes(smoke_model, template):
    g = torch.Generator().manual_seed(0)
    out = generate(smoke_model, template, {"legacy_north_g", "spectra"}, n=2, generator=g)
    assert set(out) == {"legacy_north_g", "spectra"}
    assert out["legacy_north_g"].shape == (2, 144, 64)
    assert out["spectra"].shape == (2, 122, 64)
    assert all(torch.isfinite(v).all() for v in out.values())


def test_seeding_reproducible(smoke_model, template):
    # spectra-only (122 tokens): the full-span draw above is the expensive one
    def draw(seed):
        g = torch.Generator().manual_seed(seed)
        return generate(smoke_model, template, {"spectra"}, n=2, generator=g)["spectra"]

    a = draw(0)
    assert torch.allclose(draw(0), a)  # same seed reproduces
    assert not torch.allclose(draw(1), a)  # different seed differs


def test_argmax_is_deterministic_without_generator(smoke_model, template):
    a = generate(smoke_model, template, {"spectra"}, n=1, argmax=True)
    b = generate(smoke_model, template, {"spectra"}, n=1, argmax=True)
    assert torch.allclose(a["spectra"], b["spectra"])


def test_image_to_spectra_teacher_forces_images(smoke_model, template):
    g = torch.Generator().manual_seed(0)
    out = generate(smoke_model, template, {"spectra"}, n=1, generator=g)
    assert set(out) == {"spectra"}  # teacher-forced spans are not returned
    assert out["spectra"].shape == (1, 122, 64)


def test_generate_rejects_missing_span(smoke_model, jet_config):
    image_only = ObjectSequencer(jet_config).build(
        make_record(0, image_only_fraction=1.1)
    )
    with pytest.raises(ValueError, match="spectra"):
        generate(smoke_model, image_only, {"spectra"})


def test_reconstruct_shapes(smoke_model, template):
    preds = reconstruct(smoke_model, template)
    assert preds["legacy_north_g"].shape == (144, 64)
    assert preds["spectra"].shape == (122, 64)
    assert all(torch.isfinite(v).all() for v in preds.values())


def test_sample_template_modes(smoke_model, template):
    recon = sample_template(smoke_model, template, "reconstruct")
    assert recon["legacy_north_g"].shape == (1, 144, 64)
    assert recon["spectra"].shape == (1, 122, 64)

    # spectra-only draws: the full-span unconditional cost is already paid
    # by test_unconditional_shapes
    def draw(seed):
        g = torch.Generator().manual_seed(seed)
        return sample_template(
            smoke_model, template, "image-to-spectra", n=1, generator=g
        )

    a = draw(0)
    assert set(a) == {"spectra"}  # teacher-forced images are not returned
    assert a["spectra"].shape == (1, 122, 64)
    assert torch.allclose(draw(0)["spectra"], a["spectra"])  # same seed reproduces

    with pytest.raises(ValueError, match="unknown mode"):
        sample_template(smoke_model, template, "nope")


def test_render_sampled_tokens_writes_pngs(smoke_model, template, tmp_path):
    record = make_record(3, image_only_fraction=0.0)
    sampled = sample_template(smoke_model, template, "reconstruct")
    for show_truth in (True, False):
        pngs = render_sampled_tokens(
            smoke_model,
            record,
            template,
            sampled,
            out_dir=tmp_path / f"truth_{show_truth}",
            tag="t",
            show_truth=show_truth,
        )
        assert set(pngs) == {"legacy_north_g", "spectra"}
        assert all(p.exists() and p.stat().st_size > 0 for p in pngs.values())


def test_spiral_checkpoint_decodes_pixel_identical_to_raster(jet_config, tmp_path):
    """ADR 0004: the inverse path antispiralises iff the loaded config says
    spiral, so a spiral checkpoint's image panels are pixel-identical to a
    raster checkpoint's decode of the same underlying image."""
    from types import SimpleNamespace

    import matplotlib.pyplot as plt
    import numpy as np

    from astropt3 import AstroPT3Config
    from astropt3.tokenization import antispiralise

    spiral_config = AstroPT3Config(**{**jet_config.to_dict(), "spiral": True})
    raster_config = AstroPT3Config(**{**jet_config.to_dict(), "spiral": False})
    record = make_record(3, image_only_fraction=0.0)
    spiral_template = ObjectSequencer(spiral_config).build(record)
    raster_template = ObjectSequencer(raster_config).build(record)
    # forward half: the sequencer spiralises image tokens iff config.spiral
    assert not torch.equal(
        spiral_template.values["legacy_north_g"], raster_template.values["legacy_north_g"]
    )
    assert torch.equal(
        antispiralise(spiral_template.values["legacy_north_g"]),
        raster_template.values["legacy_north_g"],
    )
    assert torch.equal(
        spiral_template.values["spectra"], raster_template.values["spectra"]
    )

    # inverse half, through the shared samples path: rendering each
    # template's own tokens under its own config draws the same image
    pixels = {}
    for label, config, template in [
        ("spiral", spiral_config, spiral_template),
        ("raster", raster_config, raster_template),
    ]:
        pngs = render_sampled_tokens(
            SimpleNamespace(config=config),  # render only reads model.config
            record,
            template,
            {"legacy_north_g": template.values["legacy_north_g"].unsqueeze(0)},
            out_dir=tmp_path / label,
            tag="t",
            show_truth=True,
        )
        pixels[label] = plt.imread(pngs["legacy_north_g"])
    assert np.array_equal(pixels["spiral"], pixels["raster"])


def test_sequencer_rejects_spiral_override(jet_config):
    """ADR 0004: patch order comes from the checkpoint's config only — the
    old per-call kwarg is gone, so a contradicting override fails loud."""
    with pytest.raises(TypeError):
        cast(Any, ObjectSequencer)(jet_config, spiral=True)
    assert ObjectSequencer(jet_config).spiral is jet_config.spiral


def test_sample_checkpoint_end_to_end(smoke_model, tmp_path):
    ckpt = tmp_path / "ckpt"
    smoke_model.save_pretrained(ckpt)
    record = make_record(3, image_only_fraction=0.0)
    oid = ObjectSequencer(smoke_model.config).build(record).object_id
    pngs = sample_checkpoint(
        ckpt,
        [record],
        modes=["reconstruct", "image-to-spectra"],
        n=1,
        out_dir=tmp_path / "samples",
        device="cpu",
    )
    assert set(pngs) == {
        f"reconstruct/legacy_north_g/{oid}",
        f"reconstruct/spectra/{oid}",
        f"image-to-spectra/spectra/{oid}",
    }
    assert all(Path(p).exists() and Path(p).stat().st_size > 0 for p in pngs.values())


def test_sample_checkpoint_generates_one_south_band(jet_config, tmp_path):
    config = AstroPT3Config(**{**jet_config.to_dict(), "modalities": [*jet_config.modalities, *LEGACY_SOUTH_MODALITIES], "vocab_size": None})
    ckpt = tmp_path / "south"
    AstroPT3Model(config).save_pretrained(ckpt)
    image = {"band": ["des-g", "des-r", "des-i", "des-z"], "flux": np.ones((4, 160, 160), dtype=np.float32)}
    alone = {"object_id": "south-only", "legacy_south_image": image}
    matched = {**alone, "object_id": "south-match", "spectrum": make_record(3, image_only_fraction=0.0)["spectrum"]}
    with pytest.raises(ValueError, match="no image band"):
        build_template(ObjectSequencer(config), alone, "unconditional", band="legacy_north_g")
    for mode, context, target in (
        ("spectra-to-images", "spectra", "legacy_south_i"),
        ("image-to-spectra", "legacy_south_i", "spectra"),
    ):
        template = build_template(ObjectSequencer(config), matched, mode, band="legacy_south_i")
        assert set(template.masks) == {context, target}
        assert template.masks[context].nonzero()[0] < template.masks[target].nonzero()[0]
    pngs = sample_checkpoint(ckpt, [alone, matched], modes=["spectra-to-images", "image-to-spectra"],
                             band="legacy_south_i", n=1, out_dir=tmp_path / "samples", device="cpu")
    assert set(pngs) == {"spectra-to-images/legacy_south_i/south-match", "image-to-spectra/spectra/south-match"}
    assert all(Path(p).stat().st_size > 0 for p in pngs.values())


def test_render_sampled_tokens_inverts_rgb_rescale_and_size(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from astropt3.eval import samples as samples_mod

    rgb = {
        "name": "galaxies",
        "family": "image",
        "source": "galaxies_with_hats",
        "record_keys": ["galaxy_image"],
        "image_format": "rgb",
        "token_ids": [23, 24, 25],
        "input_size": 192,
        "patch_size": 8,
        "max_positions": 256,
        "image_crop": 0,
        "image_resize": 128,
    }
    config = AstroPT3Config(modalities=[rgb], spiral=True)
    flux = torch.linspace(0.1, 0.9, 3 * 256 * 256, dtype=torch.float32).reshape(3, 256, 256)
    record = {"object_id": "g-rgb", "galaxy_image": {"band": ["rgb-r", "rgb-g", "rgb-b"], "flux": flux}}
    template = ObjectSequencer(config).build(record)

    captured = {}
    orig_save = samples_mod.save_image_png

    def spy_save(values, path, title, truth=None, truth_label="truth"):
        captured["values"] = values
        captured["truth"] = truth
        return orig_save(values, path, title, truth=truth, truth_label=truth_label)

    monkeypatch.setattr(samples_mod, "save_image_png", spy_save)
    pngs = render_sampled_tokens(
        SimpleNamespace(config=config),
        record,
        template,
        {"galaxies": template.values["galaxies"].unsqueeze(0)},
        out_dir=tmp_path,
        tag="rgb",
        show_truth=True,
    )
    expected = torch.nn.functional.avg_pool2d(flux, 2).numpy()
    assert captured["values"].shape == (1, 3, 128, 128)
    assert captured["truth"].shape == (3, 128, 128)
    np.testing.assert_allclose(captured["values"][0], expected, atol=1e-6)
    np.testing.assert_allclose(captured["truth"], expected, atol=1e-6)
    assert pngs["galaxies"].stat().st_size > 0

    rgb_r = {
        "name": "galaxies_r",
        "family": "image",
        "source": "galaxies_with_hats",
        "record_keys": ["galaxy_image"],
        "image_format": "rgb",
        "band": "rgb-r",
        "token_ids": [217, 218, 219],
        "input_size": 64,
        "patch_size": 8,
        "max_positions": 144,
        "image_crop": 0,
        "image_resize": 96,
    }
    config_r = AstroPT3Config(modalities=[rgb_r], spiral=True)
    template_r = ObjectSequencer(config_r).build(record)
    pngs_r = render_sampled_tokens(
        SimpleNamespace(config=config_r),
        record,
        template_r,
        {"galaxies_r": template_r.values["galaxies_r"].unsqueeze(0)},
        out_dir=tmp_path,
        tag="rgb_r",
        show_truth=True,
    )
    expected_r = torch.nn.functional.interpolate(flux[0:1].unsqueeze(0), size=(96, 96), mode="area").squeeze(0).numpy()
    assert captured["values"].shape == (1, 1, 96, 96)
    assert captured["truth"].shape == (1, 96, 96)
    np.testing.assert_allclose(captured["values"][0], expected_r, atol=1e-6)
    np.testing.assert_allclose(captured["truth"], expected_r, atol=1e-6)
    assert pngs_r["galaxies_r"].stat().st_size > 0


def test_save_spectra_png_guards_shared_ylim_against_single_bin_outliers(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    from astropt3.eval.samples import save_spectra_png

    lam = np.linspace(3600.0, 9824.0, 7781)
    truth = np.linspace(2.0, 20.0, len(lam))
    truth[3000:3020] = 60.0  # narrow emission line in truth
    samples = np.tile(np.linspace(1.5, 22.0, len(lam)), (4, 1))
    samples[0, [10, 200, 5000]] = 500_000.0  # rare tail spikes
    samples[2, [100, 4000]] = -250_000.0

    captured = {}
    orig_close = plt.close

    def capture_close(fig):
        captured["ylim"] = fig.axes[0].get_ylim()
        return orig_close(fig)

    monkeypatch.setattr(plt, "close", capture_close)
    out = tmp_path / "spectra.png"
    save_spectra_png(samples, lam, out, "spike-guard", truth=truth)
    assert out.exists() and out.stat().st_size > 0
    lo, hi = captured["ylim"]
    assert -10.0 < lo < 5.0
    assert 55.0 < hi < 100.0  # emission line preserved, 500k spike does not flatten panels


def test_generate_and_reconstruct_ignore_masked_spectral_garbage(smoke_model):
    seq = ObjectSequencer(smoke_model.config)

    def make_masked(garbage):
        rec = make_record(5, image_only_fraction=0.0)
        spec = {k: v.copy() for k, v in rec["spectrum"].items()}
        spec["mask"][128:256] = True
        spec["flux"][128:256] = garbage
        spec["flux"][500:520] = 0.0
        rec["spectrum"] = spec
        return build_template(seq, rec, "unconditional")

    t_pos = make_masked(1e6)
    t_neg = make_masked(-1e6)

    gen_spec = generate(smoke_model, t_pos, {"spectra"}, n=2, generator=torch.Generator().manual_seed(0))
    assert gen_spec["spectra"].shape == (2, 122, 64)
    assert torch.isfinite(gen_spec["spectra"]).all()
    rec_out = reconstruct(smoke_model, t_pos)
    assert rec_out["spectra"].shape == (122, 64)
    assert torch.isfinite(rec_out["spectra"]).all()

    img_pos = generate(smoke_model, t_pos, {"legacy_north_g"}, n=1, argmax=True)["legacy_north_g"]
    img_neg = generate(smoke_model, t_neg, {"legacy_north_g"}, n=1, argmax=True)["legacy_north_g"]
    torch.testing.assert_close(img_pos, img_neg, atol=0.0, rtol=0.0)
