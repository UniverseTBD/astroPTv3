"""Fixed physical normalization for the ADR 0008 scalar modalities.

Mirrors ``band_registry.py`` and honours the ADR 0007 discipline: every
transform is fixed, invertible, physically motivated, and carries no fitted
constants — a checkpoint is never coupled to the corpus statistics it was
trained on. Unknown scalar names raise.

- ``Z``          -> ``log(1 + z)``: the standard photometric-redshift working
                    variable; errors become naturally fractional, so a
                    predicted sigma reads as the literature's ``dz/(1+z)``.
- ``ebv``        -> ``ebv / 0.1``: fixed knee putting typical Galactic
                    extinction at O(1) (band-registry philosophy).
- ``photometry`` -> ``arcsinh(f / 0.01 nMgy)`` per band: literally the band
                    registry's image transform, so aperture photometry and
                    image pixels live in the same units.
"""

from __future__ import annotations

import torch

from .band_registry import _DIV_FACTOR

# ebv knee: typical Galactic E(B-V) is a few hundredths of a magnitude
_EBV_DIV = 0.1

GWH_FRACTION_FIELDS = (
    "smooth-or-featured_smooth_fraction",
    "smooth-or-featured_featured-or-disk_fraction",
    "smooth-or-featured_artifact_fraction",
    "disk-edge-on_yes_fraction",
    "disk-edge-on_no_fraction",
    "has-spiral-arms_yes_fraction",
    "has-spiral-arms_no_fraction",
    "bar_strong_fraction",
    "bar_weak_fraction",
    "bar_no_fraction",
    "bulge-size_dominant_fraction",
    "bulge-size_large_fraction",
    "bulge-size_moderate_fraction",
    "bulge-size_small_fraction",
    "bulge-size_none_fraction",
    "how-rounded_round_fraction",
    "how-rounded_in-between_fraction",
    "how-rounded_cigar-shaped_fraction",
    "edge-on-bulge_boxy_fraction",
    "edge-on-bulge_none_fraction",
    "edge-on-bulge_rounded_fraction",
    "spiral-winding_tight_fraction",
    "spiral-winding_medium_fraction",
    "spiral-winding_loose_fraction",
    "spiral-arm-count_1_fraction",
    "spiral-arm-count_2_fraction",
    "spiral-arm-count_3_fraction",
    "spiral-arm-count_4_fraction",
    "spiral-arm-count_more-than-4_fraction",
    "spiral-arm-count_cant-tell_fraction",
    "merging_none_fraction",
    "merging_minor-disturbance_fraction",
    "merging_major-disturbance_fraction",
    "merging_merger_fraction",
)
_GWH_SCALAR_NAMES = {f"gwh_{field}" for field in GWH_FRACTION_FIELDS}


def _photometry_fwd(x: torch.Tensor) -> torch.Tensor:
    return torch.arcsinh(x / _DIV_FACTOR)


def _photometry_inv(x: torch.Tensor) -> torch.Tensor:
    return torch.sinh(x) * _DIV_FACTOR


# name -> (forward, inverse); both operate elementwise on tensors
SCALAR_TRANSFORMS = {
    "Z": (torch.log1p, torch.expm1),
    "sdss_Z": (torch.log1p, torch.expm1),
    "provabgs_Z_HP": (torch.log1p, torch.expm1),
    "provabgs_LOG_MSTAR": (lambda x: (x - 10.0) / 2.0, lambda x: x * 2.0 + 10.0),
    "provabgs_Z_MW": (
        lambda x: torch.arcsinh(x / 0.01),
        lambda x: torch.sinh(x) * 0.01,
    ),
    "provabgs_TAGE_MW": (lambda x: x / 10.0, lambda x: x * 10.0),
    "provabgs_AVG_SFR": (torch.asinh, torch.sinh),
    "ebv": (lambda x: x / _EBV_DIV, lambda x: x * _EBV_DIV),
    "photometry": (_photometry_fwd, _photometry_inv),
    # ADR 0014 A8 — free scalars from rows already on the wire. Knees are
    # published units, never corpus fits; the percentiles in A8 are shown only
    # to evidence that each lands O(1).
    #
    # fiberflux is nMgy like photometry, so it takes the IDENTICAL band-registry
    # knee — aperture flux and image pixels stay in one unit system.
    "fiberflux": (_photometry_fwd, _photometry_inv),
    # psfdepth is a PSF-flux inverse variance (p1-p99 ~49-916). log10 of 1+x
    # tolerates the 0 that means "no coverage"; -2.5 centres the corpus O(1).
    "psfdepth": (
        lambda x: torch.log10(1.0 + x) - 2.5,
        lambda x: torch.pow(10.0, x + 2.5) - 1.0,
    ),
    # seeing FWHM in arcsec, 1.05-2.24 measured; a 1.5" offset is enough
    "psf_fwhm": (lambda x: x - 1.5, lambda x: x + 1.5),
    # AB magnitudes, 17.3-25.5 measured against HSC's 22.5 DUD cut
    "hsc_cmodel_mag": (lambda x: (x - 21.5) / 2.0, lambda x: x * 2.0 + 21.5),
    # binary 0/1 star-galaxy flag; same map as the gwh fractions
    "hsc_extendedness": (lambda x: 2.0 * x - 1.0, lambda x: (x + 1.0) / 2.0),
    # second moments in px^2: heavy-tailed (p50 0.28, p99 24.7) and shape12 is
    # SIGNED, so arcsinh — it keeps both tails and the sign, which log cannot.
    "hsc_shape": (lambda x: torch.arcsinh(x / 0.3), lambda x: torch.sinh(x) * 0.3),
    # PSF moments: same units, tighter knee. A8 records that this target is
    # narrow enough (p1-p99 0.094-0.147) to be beaten by predicting a constant.
    "hsc_psf_shape": (lambda x: torch.arcsinh(x / 0.1), lambda x: torch.sinh(x) * 0.1),
    # galaxies-with-hats physical scalars. redshift and photo-z stellar mass reuse
    # the Z and PROVABGS log-mass maps; MPA-JHU SFR/sSFR are already log10 and land
    # O(1) with fixed offsets (sSFR p50 -11.1, p99 -9.5).
    "gwh_redshift": (torch.log1p, torch.expm1),
    "gwh_mass_med_photoz": (lambda x: (x - 10.0) / 2.0, lambda x: x * 2.0 + 10.0),
    "gwh_total_sfr_median": (lambda x: x, lambda x: x),
    "gwh_total_ssfr_median": (lambda x: x + 11.0, lambda x: x - 11.0),
}

# Catalog fill values that mean "not measured" for a scalar (photo-z and
# MPA-JHU use -99); such a value drops the span like a missing one.
MISSING_SENTINELS = {
    "gwh_mass_med_photoz": -99.0,
    "gwh_total_sfr_median": -99.0,
    "gwh_total_ssfr_median": -99.0,
}

_GWH_FRACTION = (lambda x: 2.0 * x - 1.0, lambda x: (x + 1.0) / 2.0)


def _transforms(name: str):
    if name in SCALAR_TRANSFORMS:
        return SCALAR_TRANSFORMS[name]
    if name in _GWH_SCALAR_NAMES:
        return _GWH_FRACTION
    raise NotImplementedError(
        f"no scalar normalization for {name!r}; add it to "
        "scalar_registry.SCALAR_TRANSFORMS"
    )


def scalar_normalize(name: str, value: torch.Tensor) -> torch.Tensor:
    return _transforms(name)[0](value)


def scalar_inverse(name: str, value: torch.Tensor) -> torch.Tensor:
    return _transforms(name)[1](value)


if __name__ == "__main__":
    for name in SCALAR_TRANSFORMS:
        x = torch.tensor([0.0, 0.03, 0.7, 1.5, 42.0])
        rt = scalar_inverse(name, scalar_normalize(name, x))
        assert torch.allclose(rt, x, atol=1e-5), name
    try:
        scalar_normalize("sSFR", torch.tensor(1.0))
    except NotImplementedError as error:
        assert "sSFR" in str(error)
    else:
        raise AssertionError("unknown scalar must raise")
    print(f"ok: {len(SCALAR_TRANSFORMS)} scalar transforms round-trip")
