"""Sampling/generation from a jetformer checkpoint (astropt3.generation),
and the shared sample-rendering layer on top of it (astropt3.eval.samples)."""

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from legacy_fixture import make_record

from astropt3.config_io import load_model_config
from astropt3.configuration_astropt3 import LEGACY_SOUTH_IMAGES, AstroPT3Config
from astropt3.data.packing import ObjectSequencer
from astropt3.eval.samples import (
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
    # image_only_fraction=0 so the template has both spans
    return ObjectSequencer(jet_config).build(make_record(3, image_only_fraction=0.0))


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
    out = generate(smoke_model, template, {"images", "spectra"}, n=2, generator=g)
    assert set(out) == {"images", "spectra"}
    assert out["images"].shape == (2, 144, 192)
    assert out["spectra"].shape == (2, 31, 256)
    assert all(torch.isfinite(v).all() for v in out.values())


def test_seeding_reproducible(smoke_model, template):
    # spectra-only (31 tokens): the full-span draw above is the expensive one
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
    assert out["spectra"].shape == (1, 31, 256)


def test_generate_rejects_missing_span(smoke_model, jet_config):
    image_only = ObjectSequencer(jet_config).build(
        make_record(0, image_only_fraction=1.1)
    )
    with pytest.raises(ValueError, match="spectra"):
        generate(smoke_model, image_only, {"spectra"})


def test_reconstruct_shapes(smoke_model, template):
    preds = reconstruct(smoke_model, template)
    assert preds["images"].shape == (144, 192)
    assert preds["spectra"].shape == (31, 256)
    assert all(torch.isfinite(v).all() for v in preds.values())


def test_sample_template_modes(smoke_model, template):
    recon = sample_template(smoke_model, template, "reconstruct")
    assert recon["images"].shape == (1, 144, 192)
    assert recon["spectra"].shape == (1, 31, 256)

    # spectra-only draws: the full-span unconditional cost is already paid
    # by test_unconditional_shapes
    def draw(seed):
        g = torch.Generator().manual_seed(seed)
        return sample_template(
            smoke_model, template, "image-to-spectra", n=1, generator=g
        )

    a = draw(0)
    assert set(a) == {"spectra"}  # teacher-forced images are not returned
    assert a["spectra"].shape == (1, 31, 256)
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
        assert set(pngs) == {"images", "spectra"}
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
        spiral_template.values["images"], raster_template.values["images"]
    )
    assert torch.equal(
        antispiralise(spiral_template.values["images"]),
        raster_template.values["images"],
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
            {"images": template.values["images"].unsqueeze(0)},
            out_dir=tmp_path / label,
            tag="t",
            show_truth=True,
        )
        pixels[label] = plt.imread(pngs["images"])
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
        f"reconstruct/images/{oid}",
        f"reconstruct/spectra/{oid}",
        f"image-to-spectra/spectra/{oid}",
    }
    assert all(Path(p).exists() and Path(p).stat().st_size > 0 for p in pngs.values())


def test_sample_checkpoint_accepts_south_templates(jet_config, tmp_path):
    config = AstroPT3Config(**{**jet_config.to_dict(), "modalities": [*jet_config.modalities, LEGACY_SOUTH_IMAGES]})
    ckpt = tmp_path / "south"
    AstroPT3Model(config).save_pretrained(ckpt)
    image = {"band": ["des-g", "des-r", "des-i", "des-z"], "flux": np.ones((4, 160, 160), dtype=np.float32)}
    alone = {"object_id": "south-only", "legacy_south_image": image}
    matched = {**alone, "object_id": "south-match", "spectrum": make_record(3, image_only_fraction=0.0)["spectrum"]}
    pngs = sample_checkpoint(ckpt, [alone, matched], modes=["reconstruct", "spectra-to-images"],
                             n=1, out_dir=tmp_path / "samples", device="cpu")
    assert set(pngs) == {
        "reconstruct/legacy_south_images/south-only", "reconstruct/legacy_south_images/south-match",
        "reconstruct/spectra/south-match", "spectra-to-images/legacy_south_images/south-match",
    }
