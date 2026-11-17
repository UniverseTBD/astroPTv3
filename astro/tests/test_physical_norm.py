"""Physical band-registry image normalization (data/band_registry.py)."""

import math

import pytest
import torch

from astropt3.data.band_registry import (
    _DIV_FACTOR,
    BAND_REGISTRY,
    clamp_flux,
    physical_inverse,
    physical_normalize,
    rescale_factors,
)

DES_BANDS = ["des-g", "des-r", "des-z"]


def test_registry_constants():
    # DES is the reference scale -> identity rescale.
    assert rescale_factors("des-g") == (1.0, 1.0)
    for band in BAND_REGISTRY:
        zpscale, pxscale = rescale_factors(band)
        assert math.isfinite(zpscale) and zpscale > 0, band
        assert math.isfinite(pxscale) and pxscale > 0, band
        cf = clamp_flux(band)
        assert math.isfinite(cf) and cf > 0, band


@pytest.mark.parametrize("band", BAND_REGISTRY)
def test_zero_point_conversion_preserves_ab_magnitude(band):
    zp, _, _ = BAND_REGISTRY[band]
    # A 22.5 AB-mag source is one nanomaggy, regardless of the native zero point.
    native_flux = 10 ** ((zp - 22.5) / 2.5)
    zpscale, _ = rescale_factors(band)
    assert native_flux * zpscale == pytest.approx(1.0)


def test_hsc_native_dn_normalization_and_inverse():
    bands = [f"hsc-{band}" for band in "grizy"]
    flux = torch.ones(5, 2, 2, dtype=torch.float64)
    nmgy = 10 ** ((22.5 - 27.0) / 2.5) * (0.262 / 0.168) ** 2
    expected = torch.full_like(flux, math.asinh(nmgy / _DIV_FACTOR))
    actual = physical_normalize(flux, bands)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(physical_inverse(actual, bands), flux)


def test_normalize_des_is_arcsinh_over_divisor():
    torch.manual_seed(0)
    flux = torch.randn(3, 8, 8) * 0.05  # nMgy-scale, below the ~398 nMgy clamp
    out = physical_normalize(flux, DES_BANDS)
    assert torch.allclose(out, torch.arcsinh(flux / _DIV_FACTOR))
    assert torch.isfinite(out).all()
    # output is bounded by the compressed bright ceiling
    ceiling = math.asinh(clamp_flux("des-g") / _DIV_FACTOR)
    assert out.abs().max() <= ceiling


def test_raw_bands_pass_through():
    flux = torch.randn(3, 4, 4) * 100
    out = physical_normalize(flux, ["rgb-r", "rgb-g", "rgb-b"])
    assert out is flux
    assert physical_inverse(flux, ["rgb-r", "rgb-g", "rgb-b"]) is flux


def test_unknown_band_raises():
    flux = torch.randn(1, 4, 4)
    with pytest.raises(NotImplementedError, match="euclid-g"):
        physical_normalize(flux, ["euclid-g"])
    with pytest.raises(NotImplementedError, match="euclid-g"):
        physical_inverse(flux, ["euclid-g"])


def test_empty_band_list_passes_through():
    # all(...) is vacuously True on [] -> nothing to key the physics on
    flux = torch.randn(3, 4, 4)
    assert physical_normalize(flux, []) is flux


def test_roundtrip_below_clamp_is_exact_and_clamp_is_lossy():
    torch.manual_seed(0)
    flux = torch.randn(3, 8, 8).double() * 10  # well below the DES ~398 nMgy clamp
    back = physical_inverse(physical_normalize(flux, DES_BANDS), DES_BANDS)
    assert torch.allclose(back, flux, atol=1e-9)

    bright = torch.full((3, 2, 2), 2 * clamp_flux("des-g"), dtype=torch.float64)
    clipped = physical_inverse(physical_normalize(bright, DES_BANDS), DES_BANDS)
    assert torch.allclose(clipped, torch.full_like(bright, clamp_flux("des-g")))


def test_non_default_divisor_changes_output_and_roundtrips():
    torch.manual_seed(0)
    flux = torch.randn(3, 8, 8).double() * 0.05
    default = physical_normalize(flux, DES_BANDS)
    moved = physical_normalize(flux, DES_BANDS, divisor=0.5)
    assert not torch.allclose(default, moved)  # the knob actually acts
    back = physical_inverse(moved, DES_BANDS, divisor=0.5)
    assert torch.allclose(back, flux, atol=1e-9)


def test_sequencer_uses_config_divisor(tiny_config):
    """config.image_norm_divisor must reach the sequencer's normalization."""
    from legacy_fixture import make_record

    from astropt3 import AstroPT3Config
    from astropt3.data.packing import IMAGE_CROP, ObjectSequencer
    from astropt3.tokenization import antispiralise, patchify_image

    record = make_record(3)
    moved_config = AstroPT3Config(
        **{**tiny_config.to_dict(), "image_norm_divisor": 0.5}
    )
    seq = ObjectSequencer(moved_config).build(record)
    flux = torch.as_tensor(record["image"]["flux"])
    h, w = flux.shape[-2:]
    top, left = (h - IMAGE_CROP) // 2, (w - IMAGE_CROP) // 2
    flux = flux[..., top : top + IMAGE_CROP, left : left + IMAGE_CROP]
    expected = patchify_image(
        physical_normalize(flux, record["image"]["band"], divisor=0.5), 8
    )
    # undo the config's spiral patch order before comparing to raster patchify
    actual = (
        antispiralise(seq.values["images"])
        if getattr(moved_config, "spiral", False)
        else seq.values["images"]
    )
    assert torch.allclose(actual, expected)
    default_seq = ObjectSequencer(tiny_config).build(record)
    assert not torch.allclose(seq.values["images"], default_seq.values["images"])
