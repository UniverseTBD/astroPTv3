"""Physical per-band constants and normalization for the image modality.

Ported from galactiktok (branch ``feat/norm``,
``models/image_transformer/band_registry.py`` + the tokenizer's
``_normalize``/``_physical_factors``/``decode`` methods). Source of truth for
the constants is the **surveys' own documentation**, not any external model.
Each band carries a magnitude zeropoint, pixel scale, and a bright-star
magnitude ``m_bright``; the pipeline uses these to rescale raw survey flux to
a common LegacySurvey nanomaggies scale and to clamp survey-flagged bright
pixels.

Everything is keyed to LegacySurvey: nanomaggies with AB zeropoint 22.5 and
0.262 arcsec/pixel (``m_AB = 22.5 - 2.5*log10(flux_nmgy)``).
The corrected zero-point conversion changes historical HSC/JWST normalization;
checkpoints trained with the reversed factor need their original preprocessing
or retraining. LegacySurvey's identity conversion is unchanged.

Records with bands that are all in :data:`RAW_BANDS` pass through untouched;
records mixing RAW and registry bands raise (not a real case for us — records
are all-``des-*`` or all-``rgb-*``).

Constants and citations
------------------------
DES / LegacySurvey (``des-g/r/i/z``)
    zeropoint 22.5, pixel scale 0.262 arcsec/pix (reference scale, identity
    rescale). ``m_bright = 16.0`` from the LegacySurvey ``MEDIUM`` bright-star
    mask (Gaia DR2 ``phot_g_mean_mag < 16``), the full extent of masked stars.
    https://www.legacysurvey.org/dr9/external/ (BRIGHT = G<13, MEDIUM = G<16).

HSC PDR3 (``hsc-g/r/i/z/y``)
    zeropoint 27.0, pixel scale 0.168 arcsec/pix (hsc-release coadd docs).
    ``m_bright = 18.0`` all bands: PDR3 defines the mask "for stars brighter
    than 18mag in each HSC broad-band filter".
    https://hsc-release.mtk.nao.ac.jp/doc/index.php/bright-star-masks__pdr3/
    https://hsc-release.mtk.nao.ac.jp/doc/index.php/faq__pdr3/ (27 mag/DN).

JWST NIRCam, GOODS-South (``jwst-f090w`` ... ``jwst-f444w``)
    zeropoint 28.9. Pixel scale 0.02 (short-wave f090..f200) / 0.04 (long-wave
    f277..f444) arcsec/pix per the ADR.  ``m_bright`` is the NIRCam full-frame
    saturation magnitude (K-mag Vega, G2V star, ~80% full well) per filter,
    from the NIRCam sensitivity/saturation tables.
    http://ircamera.as.arizona.edu/nircam/in_sensitivity.php
    GOODS-South is a deep field with no bright stars, so this clamp is a safety
    ceiling that effectively never fires. Two accepted approximations: the
    quoted magnitude is K-mag Vega of a G2V star (not in-filter AB), and the
    0.02/0.04 pixel scales are the ADR values — verify against the parquet
    ``image.scale`` field, since the 7 bands share one 96x96 grid and so may
    share a single common pixel scale.
"""

from __future__ import annotations

import math

import torch

LEGACYSURVEY_REFERENCE_ZP = 22.5
LEGACYSURVEY_REFERENCE_SCALE = 0.262  # arcsec/pixel

# arcsinh range-compression divisor (nanomaggies). See the galactiktok
# band_registry ADR: 0.01 nMgy puts the linear->log knee in the faint/sky
# regime so noise stays linear and galaxy light is log-compressed.
_DIV_FACTOR = 0.01

# band -> (zeropoint, pixel_scale_arcsec_per_pix, m_bright)
BAND_REGISTRY: dict[str, tuple[float, float, float]] = {
    # DES / LegacySurvey — reference scale, identity rescale.
    "des-g": (22.5, 0.262, 16.0),
    "des-r": (22.5, 0.262, 16.0),
    "des-i": (22.5, 0.262, 16.0),
    "des-z": (22.5, 0.262, 16.0),
    # HSC PDR3.
    "hsc-g": (27.0, 0.168, 18.0),
    "hsc-r": (27.0, 0.168, 18.0),
    "hsc-i": (27.0, 0.168, 18.0),
    "hsc-z": (27.0, 0.168, 18.0),
    "hsc-y": (27.0, 0.168, 18.0),
    # JWST NIRCam short-wave (0.02"/pix).
    "jwst-f090w": (28.9, 0.02, 15.35),
    "jwst-f115w": (28.9, 0.02, 15.47),
    "jwst-f150w": (28.9, 0.02, 15.63),
    "jwst-f200w": (28.9, 0.02, 15.11),
    # JWST NIRCam long-wave (0.04"/pix).
    "jwst-f277w": (28.9, 0.04, 15.48),
    "jwst-f356w": (28.9, 0.04, 14.76),
    "jwst-f444w": (28.9, 0.04, 13.82),
}


# Bands passed through with NO physical normalization (no rescale / clamp /
# arcsinh). RGB composites are already display-stretched 8-bit imagery, not
# calibrated flux, so the physical pipeline does not apply. Named by channel
# (rgb-*), not by file format, since the encoding (PNG/JPG) is irrelevant once
# decoded to pixels. Deliberately separate from BAND_REGISTRY: unknown bands
# (e.g. euclid-*) must still raise, only these pass through.
RAW_BANDS: frozenset[str] = frozenset({"rgb-r", "rgb-g", "rgb-b"})


def rescale_factors(band: str) -> tuple[float, float]:
    """Return ``(zpscale, pxscale)`` mapping raw flux to LegacySurvey nanomaggies.

    ``zpscale = 10**((22.5 - zp)/2.5)`` (zeropoint alignment),
    ``pxscale = (0.262 / pixel_scale)**2`` (per-pixel flux is proportional to
    pixel area; this rescales to a 0.262" reference pixel).
    """
    zp, pixel_scale, _ = BAND_REGISTRY[band]
    zpscale = 10 ** ((LEGACYSURVEY_REFERENCE_ZP - zp) / 2.5)
    pxscale = (LEGACYSURVEY_REFERENCE_SCALE / pixel_scale) ** 2
    return zpscale, pxscale


def clamp_flux(band: str) -> float:
    """Per-pixel bright clamp in nanomaggies: ``10**((22.5 - m_bright)/2.5)``."""
    _, _, m_bright = BAND_REGISTRY[band]
    return 10 ** ((LEGACYSURVEY_REFERENCE_ZP - m_bright) / 2.5)


def physical_factors(
    bands: list[str],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel ``(rescale, clamp_flux)`` tensors shaped ``(C, 1, 1)``.

    The shape broadcasts over channel-first flux cubes ``(C, H, W)`` and
    batches thereof ``(..., C, H, W)``. ``rescale = zpscale * pxscale`` maps
    raw flux to nanomaggies; ``clamp`` is the per-band bright ceiling. Raises
    ``NotImplementedError`` for any band without a physical entry (e.g.
    ``euclid-*``).
    """
    missing = [b for b in bands if b not in BAND_REGISTRY]
    if missing:
        raise NotImplementedError(
            f"physical normalization not supported for bands {missing!r}; "
            "add them to band_registry.BAND_REGISTRY"
        )
    rescale = []
    clamp = []
    for b in bands:
        zpscale, pxscale = rescale_factors(b)
        rescale.append(zpscale * pxscale)
        clamp.append(clamp_flux(b))
    shape = (len(bands), 1, 1)
    return (
        torch.tensor(rescale, device=device, dtype=dtype).reshape(shape),
        torch.tensor(clamp, device=device, dtype=dtype).reshape(shape),
    )


def physical_normalize(
    flux: torch.Tensor, bands: list[str], divisor: float = _DIV_FACTOR
) -> torch.Tensor:
    """Rescale to nanomaggies -> clamp bright pixels (lossy) -> arcsinh.

    The physical, invertible-up-to-the-clamp forward normalization. Raw bands
    (RGB composites) pass through untouched; an empty band list is vacuously
    all-RAW and also passes through (nothing to key the physics on).
    ``divisor`` is the arcsinh knee in nanomaggies — pass the checkpoint's
    ``config.image_norm_divisor`` so data and inverse stay in the regime the
    model was trained on.

    Output is flux denominated in units of the divisor (default 0.01 nMgy =
    10 pMgy): below the knee ``arcsinh(x/d) ≈ x/d`` reads as flux in tens of
    picomaggies, above it as log-flux. Keeping tokens O(1) (noise ~0.1, DES
    ceiling ~11.3) matters for the jetformer exact-likelihood path, which
    feeds these values to the flow/GMM unstandardized.
    """
    if all(b in RAW_BANDS for b in bands):
        return flux
    rescale, clamp = physical_factors(bands, flux.device, flux.dtype)
    flux = flux * rescale
    flux = flux.clamp(min=-clamp, max=clamp)
    return torch.arcsinh(flux / divisor)


def physical_inverse(
    flux: torch.Tensor, bands: list[str], divisor: float = _DIV_FACTOR
) -> torch.Tensor:
    """Invert :func:`physical_normalize`: sinh -> un-rescale.

    The clamp is lossy: bright pixels above the ceiling are not recoverable
    (accepted, see the galactiktok ADR). Model output is clamped to the
    compressed ceiling first so sinh cannot overflow — the encoder clamps to
    the same physical ceiling, so this is the symmetric representable range,
    not extra information loss. ``divisor`` must match the forward pass
    (the checkpoint's ``config.image_norm_divisor``).
    """
    if all(b in RAW_BANDS for b in bands):
        return flux
    rescale, clamp = physical_factors(bands, flux.device, flux.dtype)
    ceiling = torch.arcsinh(clamp / divisor)
    flux = flux.clamp(min=-ceiling, max=ceiling)
    flux = torch.sinh(flux) * divisor
    return flux / rescale


if __name__ == "__main__":
    for b in BAND_REGISTRY:
        zpscale, pxscale = rescale_factors(b)
        assert math.isfinite(zpscale) and zpscale > 0, b
        assert math.isfinite(pxscale) and pxscale > 0, b
        cf = clamp_flux(b)
        assert math.isfinite(cf) and cf > 0, b
    # DES is the reference scale -> identity rescale.
    assert rescale_factors("des-g") == (1.0, 1.0)
    print(f"ok: {len(BAND_REGISTRY)} bands")
