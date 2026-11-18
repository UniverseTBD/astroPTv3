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

import math

import torch

from .band_registry import _DIV_FACTOR

# ebv knee: typical Galactic E(B-V) is a few hundredths of a magnitude
_EBV_DIV = 0.1

# Galaxy Zoo (Zoobot) question -> kept answers. One joint modality per question,
# ``gwh_gz_<question>``, over ``<question>_<answer>_fraction``. A question's
# fractions sum to 1, so its null answer (no/none/artifact/cant-tell; else the
# last) is dropped: it is 1 - sum(kept), and keeping it would hand the diagonal
# GMM a degenerate direction. Unreached follow-up questions are NaN upstream,
# which is the ordinary absent span.
GZ_QUESTIONS = {
    "smooth-or-featured": ("smooth", "featured-or-disk"),
    "disk-edge-on": ("yes",),
    "has-spiral-arms": ("yes",),
    "bar": ("strong", "weak"),
    "bulge-size": ("dominant", "large", "moderate", "small"),
    "how-rounded": ("round", "in-between"),
    "edge-on-bulge": ("boxy", "rounded"),
    "spiral-winding": ("tight", "medium"),
    "spiral-arm-count": ("1", "2", "3", "4", "more-than-4"),
    "merging": ("minor-disturbance", "major-disturbance", "merger"),
}
_GZ_NAMES = {f"gwh_gz_{question}" for question in GZ_QUESTIONS}


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
    # Issue #53 triage (ADR 0014 A9). Log-scale catalogue values reuse the
    # maps above; sizes, widths and dispersions are positive, so log10 with a
    # fixed unit offset; AB magnitudes take a fixed (m - m0) / 2.
    "gwh_fibre_sfr_median": (lambda x: x, lambda x: x),
    "gwh_fibre_ssfr_median": (lambda x: x + 11.0, lambda x: x - 11.0),
    "gwh_ssfr_med_photoz": (lambda x: x + 11.0, lambda x: x - 11.0),
    "gwh_logMH": (lambda x: (x - 10.0) / 2.0, lambda x: x * 2.0 + 10.0),
    "gwh_W50": (lambda x: torch.log10(x) - 2.0, lambda x: torch.pow(10.0, x + 2.0)),  # km/s
    "gwh_est_petro_th50": (torch.log10, lambda x: torch.pow(10.0, x)),  # arcsec
    "gwh_petro_radius": (torch.log10, lambda x: torch.pow(10.0, x)),  # th50, th90 arcsec
    "gwh_sersic_n": (torch.log10, lambda x: torch.pow(10.0, x)),
    "gwh_sersic_ba": (lambda x: 2.0 * x - 1.0, lambda x: (x + 1.0) / 2.0),
    "gwh_nsa_mag": (lambda x: (x - 17.0) / 2.0, lambda x: x * 2.0 + 17.0),  # GALEX F, N, SDSS u, i
    "gwh_petro_ba": (lambda x: 2.0 * x - 1.0, lambda x: (x + 1.0) / 2.0),  # b/a at th50, th90
    "sdss_vdisp": (lambda x: torch.log10(x) - 2.0, lambda x: torch.pow(10.0, x + 2.0)),  # km/s
    "provabgs_MAG_W1": (lambda x: (x - 18.0) / 2.0, lambda x: x * 2.0 + 18.0),
}

# Open valid interval per scalar; any value outside drops the span. Log maps
# need x > 0; -99 is the photo-z / MPA-JHU "not measured" fill; the rest are
# the ADR 0014 A8 predicates and published caps (SDSS VDISP saturates at 850).
_ABOVE_FILL = (-99.0, math.inf)
_POSITIVE = (0.0, math.inf)
_AB_MAG = (10.0, 30.0)
VALID_RANGES = {
    "gwh_mass_med_photoz": _ABOVE_FILL,
    "gwh_total_sfr_median": _ABOVE_FILL,
    "gwh_total_ssfr_median": _ABOVE_FILL,
    "gwh_fibre_sfr_median": _ABOVE_FILL,
    "gwh_fibre_ssfr_median": _ABOVE_FILL,
    "gwh_ssfr_med_photoz": _ABOVE_FILL,
    "gwh_W50": _POSITIVE,
    "gwh_est_petro_th50": _POSITIVE,
    "gwh_petro_radius": _POSITIVE,
    "gwh_sersic_n": _POSITIVE,
    "gwh_nsa_mag": _AB_MAG,
    "psfdepth": _POSITIVE,
    "psf_fwhm": (0.0, 5.0),
    "hsc_cmodel_mag": _AB_MAG,
    "sdss_vdisp": (0.0, 850.0),
    "provabgs_MAG_W1": _AB_MAG,
    "gwh_petro_ba": _POSITIVE,
    "gz10_label": (-0.5, 9.5),
}

# Scalar -> {record key of a quality flag or error: largest passing value}.
# The flag must come from the scalar's own catalogue (CatalogStream checks), and
# a missing flag fails: an unknown quality must not look reliable.
QUALITY_GATES = {
    "Z": {"ZWARN": 0.0},
    "sdss_Z": {"sdss_ZWARNING": 0.0},
    "sdss_vdisp": {"sdss_ZWARNING": 0.0},
    "hsc_cmodel_mag": {f"hsc_{band}_cmodel_magerr": 0.1 for band in "grizy"},
    "gwh_total_sfr_median": {"gwh_total_sfr_flag": 0.0},
    "gwh_total_ssfr_median": {"gwh_total_ssfr_flag": 0.0},
}

_GZ_FRACTION = (lambda x: 2.0 * x - 1.0, lambda x: (x + 1.0) / 2.0)

# Categorical scalars: one record key holding a class index -> its one-hot
# vector (input_size == classes); the inverse is the argmax. A density over
# class indices would invent an ordering, so these take a CategoricalHead.
CATEGORICAL = {"gz10_label": 10}  # Galaxy10 DECaLS morphology classes 0-9


def _one_hot(classes: int):
    return (
        lambda x: torch.nn.functional.one_hot(x[..., 0].round().long(), classes).to(x.dtype),
        lambda x: x.argmax(dim=-1, keepdim=True).to(x.dtype),
    )


def scalar_valid(name: str, values, record: dict) -> bool:
    """Whether a scalar's raw values pass its range and quality predicates."""
    try:
        numbers = [float(value) for value in values]
    except (TypeError, ValueError) as error:
        raise ValueError(f"modality {name!r} has a non-numeric value") from error
    low, high = VALID_RANGES.get(name, (-math.inf, math.inf))
    if not all(math.isfinite(x) and low < x < high for x in numbers):
        return False
    if name in CATEGORICAL and any(x != round(x) for x in numbers):
        return False
    return all(record.get(flag) is not None and float(record[flag]) <= limit
               for flag, limit in QUALITY_GATES.get(name, {}).items())


def _transforms(name: str):
    if name in SCALAR_TRANSFORMS:
        return SCALAR_TRANSFORMS[name]
    if name in _GZ_NAMES:
        return _GZ_FRACTION
    if name in CATEGORICAL:
        return _one_hot(CATEGORICAL[name])
    raise NotImplementedError(
        f"no scalar normalization for {name!r}; add it to "
        "scalar_registry.SCALAR_TRANSFORMS"
    )


def scalar_normalize(name: str, value: torch.Tensor) -> torch.Tensor:
    return _transforms(name)[0](value)


def scalar_inverse(name: str, value: torch.Tensor) -> torch.Tensor:
    return _transforms(name)[1](value)


if __name__ == "__main__":
    for name, classes in CATEGORICAL.items():
        labels = torch.arange(classes, dtype=torch.float32).unsqueeze(-1)
        assert torch.equal(scalar_inverse(name, scalar_normalize(name, labels)), labels), name
    for name in [*SCALAR_TRANSFORMS, *_GZ_NAMES]:
        x = torch.tensor([0.03, 0.7, 1.5, 42.0])
        rt = scalar_inverse(name, scalar_normalize(name, x))
        assert torch.allclose(rt, x, atol=1e-5), name
    try:
        scalar_normalize("sSFR", torch.tensor(1.0))
    except NotImplementedError as error:
        assert "sSFR" in str(error)
    else:
        raise AssertionError("unknown scalar must raise")
    print(f"ok: {len(SCALAR_TRANSFORMS)} scalar transforms round-trip")
