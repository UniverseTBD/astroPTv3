"""Physical DESI spectra normalization (data/spectral.py, ADR 0007)."""

import numpy as np
import pytest
import torch

from astropt3.data.spectral import (
    _DIV_FACTOR,
    CLAMP_FNU_NMGY,
    DESI_LAMBDA_BINS,
    DESI_LAMBDA_GRID,
    DESI_LAMBDA_MIN,
    DESI_LAMBDA_STEP,
    FNU_NMGY_PER_FLAM,
    spectral_inverse,
    spectral_normalize,
)

LAM = DESI_LAMBDA_GRID.float()


def test_grid_and_conversion_constants():
    # the verified pilot_v2 format: 3600-9824 A, exactly 0.8 A x 7781 bins
    assert DESI_LAMBDA_BINS == 7781
    assert DESI_LAMBDA_GRID[0] == DESI_LAMBDA_MIN == 3600.0
    assert DESI_LAMBDA_GRID[-1] == 9824.0
    assert DESI_LAMBDA_STEP == 0.8
    # f_nu = f_lambda * lambda^2 / c in the AB/maggie system: ~9.19e-8 per A^2
    assert pytest.approx(9.19e-8, rel=1e-3) == FNU_NMGY_PER_FLAM
    # unit sanity from the ADR: f_lambda = 1 DESI unit at 5500 A is ~2.8 nMgy,
    # i.e. m_AB ~ 21.4 — a typical DESI galaxy fiber magnitude
    fnu = 1.0 * 5500.0**2 * FNU_NMGY_PER_FLAM
    assert 22.5 - 2.5 * np.log10(fnu) == pytest.approx(21.4, abs=0.5)


def test_roundtrip_is_exact_and_zero_stays_zero():
    torch.manual_seed(0)
    flux = torch.randn(DESI_LAMBDA_BINS, dtype=torch.float64) * 10
    lam = DESI_LAMBDA_GRID
    back = spectral_inverse(spectral_normalize(flux, lam), lam)
    assert torch.allclose(back, flux, atol=1e-9)
    # mask-zeroed pixels must stay 0 through the map
    assert spectral_normalize(torch.zeros_like(flux), lam).abs().max() == 0


def test_non_default_divisor_changes_output_and_roundtrips():
    torch.manual_seed(0)
    flux = torch.randn(DESI_LAMBDA_BINS, dtype=torch.float64) * 10
    lam = DESI_LAMBDA_GRID
    default = spectral_normalize(flux, lam)
    moved = spectral_normalize(flux, lam, divisor=3.16)
    assert not torch.allclose(default, moved)  # the knob actually acts
    back = spectral_inverse(moved, lam, divisor=3.16)
    assert torch.allclose(back, flux, atol=1e-9)


def test_sdss_log_grid_roundtrips_and_rejects_wrong_steps():
    for start, bins in ((3800.0, 3800), (3553.0, 4664)):  # SDSS-I/II and DR17 BOSS/eBOSS
        lam = (
            10.0 ** (torch.log10(torch.tensor(start, dtype=torch.float64)) + 1e-4 * torch.arange(bins, dtype=torch.float64))
        ).float()
        flux = torch.randn(len(lam), dtype=torch.float64)
        tokens = spectral_normalize(flux, lam, source="sdss")
        assert torch.allclose(spectral_inverse(tokens, lam, source="sdss"), flux, atol=1e-9)
    with pytest.raises(NotImplementedError, match="SDSS"):
        spectral_normalize(flux, lam + torch.linspace(0, 10, len(lam)), source="sdss")
    with pytest.raises(NotImplementedError, match="unknown spectrum source"):
        spectral_normalize(flux, lam, source="unknown")


def test_unknown_grid_raises():
    flux = torch.randn(100)
    with pytest.raises(NotImplementedError, match="DESI"):
        spectral_normalize(flux, torch.linspace(3600, 9824, 100))
    shifted = LAM + 5.0  # right length, wrong wavelengths
    with pytest.raises(NotImplementedError, match="DESI"):
        spectral_inverse(torch.randn(DESI_LAMBDA_BINS), shifted)


def test_synthetic_tokens_land_in_the_o1_regime():
    """Synthetic DESI-unit flux must produce O(1) tokens (ADR 0007 regime)."""
    from legacy_fixture import make_record

    record = make_record(3, image_only_fraction=0.0)
    flux = torch.as_tensor(record["spectrum"]["flux"])
    lam = torch.as_tensor(record["spectrum"]["lambda"])
    tokens = spectral_normalize(flux, lam)
    assert torch.isfinite(tokens).all()
    assert 0.1 < tokens.abs().median() < 5.0
    assert tokens.abs().max() < 10.0


def test_sequencer_uses_config_divisor(tiny_config):
    """config.spectra_norm_divisor must reach the sequencer's normalization."""
    from legacy_fixture import make_record

    from astropt3 import AstroPT3Config
    from astropt3.data.packing import ObjectSequencer
    from astropt3.tokenization import patchify_spectrum

    record = make_record(3, image_only_fraction=0.0, spectrum_only_fraction=1.0)
    moved_config = AstroPT3Config(
        **{**tiny_config.to_dict(), "spectra_norm_divisor": 3.16}
    )
    seq = ObjectSequencer(moved_config).build(record)
    flux = torch.as_tensor(record["spectrum"]["flux"])
    lam = torch.as_tensor(record["spectrum"]["lambda"])
    expected, _ = patchify_spectrum(
        spectral_normalize(flux, lam, divisor=3.16), lam, 64
    )
    n = len(flux)  # tail padding is zeroed and masked out of loss
    assert torch.allclose(seq.values["spectra"].flatten()[:n], expected.flatten()[:n])
    default_seq = ObjectSequencer(tiny_config).build(record)
    assert not torch.allclose(seq.values["spectra"], default_seq.values["spectra"])
    assert _DIV_FACTOR == 10.0


def test_spectral_clamp_bounds_extreme_flux_and_tail_tokens():
    import math

    lam = DESI_LAMBDA_GRID
    ceiling = math.asinh(CLAMP_FNU_NMGY / _DIV_FACTOR)
    max_flam = CLAMP_FNU_NMGY / (lam**2 * FNU_NMGY_PER_FLAM)

    # Extreme raw flux clamps symmetrically in normalize -> inverse.
    bright = torch.full((DESI_LAMBDA_BINS,), 1e6, dtype=torch.float64)
    tokens = spectral_normalize(bright, lam)
    torch.testing.assert_close(tokens, torch.full_like(tokens, ceiling))
    torch.testing.assert_close(spectral_inverse(tokens, lam), max_flam)
    torch.testing.assert_close(spectral_inverse(-tokens, lam), -max_flam)

    # Rare flow/GMM tail draws (|token| >> ceiling) clamp before sinh() and stay finite.
    spikes = torch.zeros(DESI_LAMBDA_BINS, dtype=torch.float64)
    spikes[0] = 20.0
    spikes[1] = -25.0
    decoded = spectral_inverse(spikes, lam)
    assert torch.isfinite(decoded).all()
    torch.testing.assert_close(decoded[0], max_flam[0])
    torch.testing.assert_close(decoded[1], -max_flam[1])
    assert (decoded.abs() <= max_flam + 1e-9).all()
